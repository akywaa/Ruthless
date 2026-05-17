use crate::board::Board;
use crate::eval::{evaluate, PIECE_VALUES};
use crate::movegen::{generate_legal_moves, generate_noisy_moves};
use crate::see::see;
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
    soft_time_ms: Option<u128>,
    hard_time_ms: Option<u128>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    counter_moves: [[Move; 64]; 64],
    conthist: [[[i32; 64]; 64]; 12],
    played_moves: [Move; MAX_PLY],
    played_pieces: [Piece; MAX_PLY],
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

#[inline(always)]
fn update_history(val: &mut i32, bonus: i32) {
    let clamped = bonus.clamp(-1600, 1600);
    *val += clamped - (*val * clamped.abs()) / 16384;
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_mb),
            nodes: 0,
            stop: false,
            start_time: Instant::now(),
            soft_time_ms: None,
            hard_time_ms: None,
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            counter_moves: [[Move::NULL; 64]; 64],
            conthist: [[[0; 64]; 64]; 12],
            played_moves: [Move::NULL; MAX_PLY],
            played_pieces: [Piece::None; MAX_PLY],
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.counter_moves = [[Move::NULL; 64]; 64];
        self.conthist = [[[0; 64]; 64]; 12];
        self.played_moves = [Move::NULL; MAX_PLY];
        self.played_pieces = [Piece::None; MAX_PLY];
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        max_depth: u8,
        soft_time: Option<u128>,
        hard_time: Option<u128>,
    ) -> Move {
        let legal_moves = generate_legal_moves(board);
        if legal_moves.count == 0 {
            return Move::NULL;
        }
        if legal_moves.count == 1 {
            return legal_moves.moves[0];
        }

        self.nodes = 0;
        self.stop = false;
        self.start_time = Instant::now();
        self.soft_time_ms = soft_time;
        self.hard_time_ms = hard_time;

        let mut best_move = legal_moves.moves[0];
        let mut prev_best_move = Move::NULL;
        let mut score = 0;
        let mut prev_score = 0;
        let mut stable_iterations = 0;

        for depth in 1..=max_depth {
            if depth >= 4 {
                let mut delta = 20;
                let mut alpha = (score - delta).max(-INFINITY);
                let mut beta = (score + delta).min(INFINITY);

                loop {
                    score = self.negamax(board, depth, 0, alpha, beta, true, Move::NULL);
                    if self.stop {
                        break;
                    }

                    if score <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = (alpha - delta).max(-INFINITY);
                    } else if score >= beta {
                        beta = (beta + delta).min(INFINITY);
                    } else {
                        break;
                    }

                    delta += delta / 2;
                    if delta > 1000 {
                        alpha = -INFINITY;
                        beta = INFINITY;
                    }
                }
            } else {
                score = self.negamax(board, depth, 0, -INFINITY, INFINITY, true, Move::NULL);
            }

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

            if best_move == prev_best_move {
                stable_iterations += 1;
            } else {
                stable_iterations = 0;
            }

            if let Some(soft_limit) = self.soft_time_ms {
                let mut time_scale = 1.0f32;

                if best_move != prev_best_move && depth >= 6 {
                    time_scale *= 1.5;
                }
                if score < prev_score - 30 && depth >= 6 {
                    time_scale *= 1.3;
                }
                if stable_iterations >= 4 && depth >= 8 {
                    time_scale *= 0.7;
                }

                let dynamic_soft = ((soft_limit as f32) * time_scale) as u128;
                if elapsed >= dynamic_soft || elapsed * 2 >= self.hard_time_ms.unwrap_or(u128::MAX) {
                    break;
                }
            }

            prev_best_move = best_move;
            prev_score = score;
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
        if let Some(hard_limit) = self.hard_time_ms {
            if (self.nodes & 2047) == 0 && self.start_time.elapsed().as_millis() >= hard_limit {
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
        excluded_move: Move,
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
        let mut tt_score = 0;
        let mut tt_depth = 0;
        let mut tt_flag = TTFlag::Exact;
        let mut has_tt = false;

        if let Some(entry) = self.tt.probe(board.hash) {
            has_tt = true;
            tt_score = score_from_tt(entry.score, ply);
            tt_depth = entry.depth;
            tt_flag = entry.flag;
            tt_move = entry.best_move;

            if excluded_move == Move::NULL && entry.depth >= depth && ply > 0 && !is_pv {
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
        }

        let static_eval = evaluate(board);

        if !is_pv && !in_check {
            if depth <= 3 && static_eval - 85 * (depth as i32) >= beta {
                return static_eval;
            }

            if depth >= 3 && static_eval >= beta && board.has_non_pawn_material(board.side_to_move) {
                let r = 2 + depth / 4;
                let undo = board.make_null_move();
                let score = -self.negamax(board, depth.saturating_sub(r + 1), ply + 1, -beta, -beta + 1, false, Move::NULL);
                board.undo_null_move(undo);

                if score >= beta {
                    return if score >= MATE_SCORE - 100 { beta } else { score };
                }
            }
        }

        let mut extension = 0;

        if depth >= 7
            && has_tt
            && tt_move != Move::NULL
            && excluded_move == Move::NULL
            && (ply as usize) < MAX_PLY
            && !in_check
            && tt_depth >= depth - 3
            && (tt_flag == TTFlag::Exact || tt_flag == TTFlag::LowerBound)
            && tt_score.abs() < MATE_SCORE - 100
        {
            let singular_beta = tt_score - (depth as i32) * 2;
            let singular_depth = (depth - 1) / 2;

            let score = self.negamax(
                board,
                singular_depth,
                ply,
                singular_beta - 1,
                singular_beta,
                false,
                tt_move,
            );

            if score < singular_beta {
                extension = 1;
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

        let lmp_threshold = 3 + 3 * (depth as usize) * (depth as usize);
        let futility_margin = 120 * (depth as i32);
        let futility_pruning =
            !is_pv && !in_check && depth <= 3 && static_eval + futility_margin <= alpha;

        let mut quiet_moves = [Move::NULL; 64];
        let mut quiet_count = 0;

        for i in 0..moves.count {
            let m = moves.moves[i];

            if m == excluded_move {
                continue;
            }

            let is_capture = board.piece_on[m.to()] != Piece::None || m.move_type() == MoveType::EnPassant;
            let is_quiet = !is_capture && m.move_type() != MoveType::Promotion;

            if !is_pv && !in_check && moves_searched > 0 {
                // Late Move Pruning
                if is_quiet && depth <= 4 && moves_searched >= lmp_threshold {
                    continue;
                }

                // Futility Pruning
                if is_quiet && futility_pruning {
                    continue;
                }

                // Prune quiet moves with negative SEE
                if is_quiet && depth <= 3 && !see(board, m, -30 * (depth as i32)) {
                    continue;
                }
            }

            if is_quiet && quiet_count < 64 {
                quiet_moves[quiet_count] = m;
                quiet_count += 1;
            }

            if (ply as usize) < MAX_PLY {
                self.played_moves[ply as usize] = m;
                self.played_pieces[ply as usize] = board.piece_on[m.from()];
            }

            let undo = board.make_move(m);

            let score = if moves_searched == 0 {
                -self.negamax(board, depth - 1 + extension, ply + 1, -beta, -alpha, is_pv, Move::NULL)
            } else {
                let mut reduced = depth - 1;
                if moves_searched >= 3 && depth >= 3 && is_quiet {
                    let r = 1 + (moves_searched >= 6) as u8 + (depth >= 6) as u8;
                    reduced = depth.saturating_sub(1 + r).max(1);
                }

                let mut s = -self.negamax(board, reduced, ply + 1, -alpha - 1, -alpha, false, Move::NULL);
                if s > alpha && reduced < depth - 1 {
                    s = -self.negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha, false, Move::NULL);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, true, Move::NULL);
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

                    let prev_move = if ply > 0 {
                        self.played_moves[(ply - 1) as usize]
                    } else {
                        Move::NULL
                    };
                    let prev_piece = if ply > 0 {
                        self.played_pieces[(ply - 1) as usize]
                    } else {
                        Piece::None
                    };

                    if prev_move != Move::NULL {
                        self.counter_moves[prev_move.from() as usize][prev_move.to() as usize] = m;
                    }

                    let bonus = ((depth as i32) * (depth as i32)).min(1600);
                    let us = board.side_to_move as usize;

                    update_history(&mut self.history[us][m.from() as usize][m.to() as usize], bonus);
                    if prev_piece != Piece::None {
                        update_history(
                            &mut self.conthist[prev_piece as usize][prev_move.to() as usize][m.to() as usize],
                            bonus,
                        );
                    }

                    for j in 0..quiet_count.saturating_sub(1) {
                        let qm = quiet_moves[j];
                        update_history(&mut self.history[us][qm.from() as usize][qm.to() as usize], -bonus);
                        if prev_piece != Piece::None {
                            update_history(
                                &mut self.conthist[prev_piece as usize][prev_move.to() as usize][qm.to() as usize],
                                -bonus,
                            );
                        }
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

        if excluded_move == Move::NULL {
            self.tt.store(board.hash, score_to_tt(best_score, ply), depth, flag, best_move);
        }
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

            if !in_check && !see(board, m, 0) {
                continue;
            }

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
        if captured != Piece::None || m.move_type() == MoveType::EnPassant {
            let victim = if m.move_type() == MoveType::EnPassant {
                PIECE_VALUES[PieceType::Pawn as usize]
            } else {
                PIECE_VALUES[captured.piece_type() as usize]
            };
            let attacker = PIECE_VALUES[board.piece_on[m.from()].piece_type() as usize];
            let mvv_lva = victim * 10 - attacker;

            if see(board, m, 0) {
                return 1_000_000 + mvv_lva;
            } else {
                return -500_000 + mvv_lva;
            }
        }

        if (ply as usize) < MAX_PLY {
            if m == self.killers[ply as usize][0] {
                return 900_000;
            }
            if m == self.killers[ply as usize][1] {
                return 800_000;
            }
        }

        let prev_move = if ply > 0 && (ply as usize) < MAX_PLY {
            self.played_moves[(ply - 1) as usize]
        } else {
            Move::NULL
        };

        if prev_move != Move::NULL
            && m == self.counter_moves[prev_move.from() as usize][prev_move.to() as usize]
        {
            return 700_000;
        }

        let us = board.side_to_move as usize;
        let mut score = self.history[us][m.from() as usize][m.to() as usize];

        if ply > 0 && (ply as usize) < MAX_PLY {
            let prev_piece = self.played_pieces[(ply - 1) as usize];
            if prev_piece != Piece::None {
                score += self.conthist[prev_piece as usize][prev_move.to() as usize][m.to() as usize];
            }
        }

        score
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
