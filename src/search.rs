use crate::board::Board;
use crate::eval::{evaluate, PIECE_VALUES};
use crate::movegen::{generate_legal_moves, generate_noisy_moves};
use crate::tt::{TTFlag, TranspositionTable};
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType};
use std::time::Instant;

pub const INFINITY: i32 = 1_000_000;
pub const MATE_SCORE: i32 = 100_000;
pub const MAX_PLY: usize = 64;

pub struct Searcher {
    pub tt: TranspositionTable,
    pub nodes: u64,
    pub stop: bool,
    start_time: Instant,
    time_limit_ms: Option<u128>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
}

fn score_to_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_SCORE - 100 {
        score + ply as i32
    } else if score <= -MATE_SCORE + 100 {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_SCORE - 100 {
        score - ply as i32
    } else if score <= -MATE_SCORE + 100 {
        score + ply as i32
    } else {
        score
    }
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_mb),
            nodes: 0,
            stop: false,
            start_time: Instant::now(),
            time_limit_ms: None,
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
    }

    pub fn search(&mut self, board: &mut Board, max_depth: u8, time_ms: Option<u128>) -> Move {
        self.nodes = 0;
        self.stop = false;
        self.start_time = Instant::now();
        self.time_limit_ms = time_ms;

        let mut best_move = Move::NULL;

        for depth in 1..=max_depth {
            let score = self.negamax(board, depth, 0, -INFINITY, INFINITY, true);
            if self.stop {
                break;
            }

            if let Some(entry) = self.tt.probe(board.hash) {
                if entry.best_move != Move::NULL {
                    best_move = entry.best_move;
                }
            }

            let elapsed = self.start_time.elapsed().as_millis().max(1);
            let nps = (self.nodes as u128 * 1000) / elapsed;

            let pv = self.extract_pv(board, depth);
            let pv_str = pv.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" ");

            if score.abs() > MATE_SCORE - 100 {
                let mate_dist = (MATE_SCORE - score.abs() + 1) / 2;
                let sign = if score > 0 { 1 } else { -1 };
                println!(
                    "info depth {} score mate {} nodes {} nps {} time {} pv {}",
                    depth,
                    mate_dist * sign,
                    self.nodes,
                    nps,
                    elapsed,
                    pv_str
                );
            } else {
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                    depth, score, self.nodes, nps, elapsed, pv_str
                );
            }
        }

        if best_move == Move::NULL {
            let moves = generate_legal_moves(board);
            if moves.count > 0 {
                best_move = moves.moves[0];
            }
        }

        best_move
    }

    fn extract_pv(&self, board: &mut Board, depth: u8) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut undos = Vec::new();

        for _ in 0..depth {
            if let Some(entry) = self.tt.probe(board.hash) {
                if entry.best_move == Move::NULL {
                    break;
                }
                let moves = generate_legal_moves(board);
                if !moves.as_slice().contains(&entry.best_move) {
                    break;
                }
                let m = entry.best_move;
                pv.push(m);
                undos.push(board.make_move(m));
            } else {
                break;
            }
        }

        for (&m, undo) in pv.iter().zip(undos).rev() {
            board.undo_move(m, undo);
        }

        pv
    }

    fn check_time(&mut self) {
        if let Some(limit) = self.time_limit_ms {
            if (self.nodes & 2047) == 0 && self.start_time.elapsed().as_millis() >= limit {
                self.stop = true;
            }
        }
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        mut depth: u8,
        ply: u8,
        mut alpha: i32,
        beta: i32,
        is_pv: bool,
    ) -> i32 {
        self.check_time();
        if self.stop {
            return 0;
        }

        if ply > 0 && board.is_repetition() {
            return 0;
        }

        if (ply as usize) >= MAX_PLY {
            return evaluate(board);
        }

        let in_check = board.in_check();
        if in_check {
            depth += 1;
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        self.nodes += 1;

        let alpha_orig = alpha;
        let mut tt_move = Move::NULL;

        if let Some(entry) = self.tt.probe(board.hash) {
            let tt_score = score_from_tt(entry.score, ply);
            if entry.depth >= depth && ply > 0 && !is_pv {
                match entry.flag {
                    TTFlag::Exact => return tt_score,
                    TTFlag::LowerBound => alpha = alpha.max(tt_score),
                    TTFlag::UpperBound => {
                        if tt_score <= alpha {
                            return tt_score;
                        }
                    }
                }
                if alpha >= beta {
                    return tt_score;
                }
            }
            tt_move = entry.best_move;
        }

        let static_eval = evaluate(board);

        if !is_pv && !in_check {
            if depth <= 3 && static_eval - 85 * (depth as i32) >= beta {
                return static_eval;
            }

            if depth >= 3 && static_eval >= beta && board.has_non_pawn_material(board.side_to_move) {
                let r = 2 + depth / 4;
                let undo = board.make_null_move();
                let score = -self.negamax(board, depth.saturating_sub(r + 1), ply + 1, -beta, -beta + 1, false);
                board.undo_null_move(undo);

                if score >= beta {
                    return if score >= MATE_SCORE - 100 { beta } else { score };
                }
            }
        }

        let mut moves = generate_legal_moves(board);
        if moves.count == 0 {
            if in_check {
                return -MATE_SCORE + ply as i32;
            }
            return 0;
        }

        self.order_moves(board, &mut moves, tt_move, ply);

        let mut best_score = -INFINITY;
        let mut best_move = Move::NULL;
        let mut moves_searched = 0;

        for i in 0..moves.count {
            let m = moves.moves[i];
            let is_capture = board.piece_on[m.to()] != Piece::None || m.move_type() == MoveType::EnPassant;
            let is_quiet = !is_capture && m.move_type() != MoveType::Promotion;

            let undo = board.make_move(m);

            let score = if moves_searched == 0 {
                -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, is_pv)
            } else {
                let mut reduced = depth - 1;
                if moves_searched >= 3 && depth >= 3 && is_quiet {
                    let r = 1 + (moves_searched >= 6) as u8 + (depth >= 6) as u8;
                    reduced = depth.saturating_sub(1 + r).max(1);
                }

                let mut s = -self.negamax(board, reduced, ply + 1, -alpha - 1, -alpha, false);
                if s > alpha && reduced < depth - 1 {
                    s = -self.negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha, false);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, true);
                }
                s
            };

            board.undo_move(m, undo);
            moves_searched += 1;

            if self.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                if is_quiet && (ply as usize) < MAX_PLY {
                    if self.killers[ply as usize][0] != m {
                        self.killers[ply as usize][1] = self.killers[ply as usize][0];
                        self.killers[ply as usize][0] = m;
                    }
                    let us = board.side_to_move as usize;
                    let from = m.from() as usize;
                    let to = m.to() as usize;
                    self.history[us][from][to] += (depth as i32) * (depth as i32);
                    if self.history[us][from][to] > 16000 {
                        self.history[us][from][to] /= 2;
                    }
                }
                break;
            }
        }

        let flag = if best_score <= alpha_orig {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        self.tt.store(board.hash, score_to_tt(best_score, ply), depth, flag, best_move);
        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: u8) -> i32 {
        self.check_time();
        if self.stop {
            return 0;
        }

        self.nodes += 1;

        let in_check = board.in_check();

        let mut moves = if in_check {
            let legal = generate_legal_moves(board);
            if legal.count == 0 {
                return -MATE_SCORE + ply as i32;
            }
            legal
        } else {
            let stand_pat = evaluate(board);
            if stand_pat >= beta {
                return beta;
            }
            alpha = alpha.max(stand_pat);
            generate_noisy_moves(board)
        };

        self.order_moves(board, &mut moves, Move::NULL, ply);

        for i in 0..moves.count {
            let m = moves.moves[i];
            let undo = board.make_move(m);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            board.undo_move(m, undo);

            if self.stop {
                return 0;
            }

            if score >= beta {
                return beta;
            }
            alpha = alpha.max(score);
        }

        alpha
    }

    fn score_move(&self, board: &Board, m: Move, tt_move: Move, ply: u8) -> i32 {
        if m == tt_move {
            return 2_000_000;
        }

        let captured = board.piece_on[m.to()];
        if captured != Piece::None {
            let victim = PIECE_VALUES[captured.piece_type() as usize];
            let attacker = PIECE_VALUES[board.piece_on[m.from()].piece_type() as usize];
            return 1_000_000 + victim * 10 - attacker;
        }

        if (ply as usize) < MAX_PLY {
            if m == self.killers[ply as usize][0] {
                return 900_000;
            }
            if m == self.killers[ply as usize][1] {
                return 800_000;
            }
        }

        let us = board.side_to_move as usize;
        self.history[us][m.from() as usize][m.to() as usize]
    }

    fn order_moves(&self, board: &Board, list: &mut MoveList, tt_move: Move, ply: u8) {
        let mut scores = [0i32; 256];
        for i in 0..list.count {
            scores[i] = self.score_move(board, list.moves[i], tt_move, ply);
        }

        for i in 0..list.count {
            for j in (i + 1)..list.count {
                if scores[j] > scores[i] {
                    scores.swap(i, j);
                    list.moves.swap(i, j);
                }
            }
        }
    }
}
