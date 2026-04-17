use crate::board::Board;
use crate::eval::{evaluate, PIECE_VALUES};
use crate::movegen::{generate_legal_moves, generate_noisy_moves};
use crate::tt::{TTFlag, TranspositionTable};
use crate::types::{Move, MoveList, Piece};
use std::time::Instant;

pub const INFINITY: i32 = 1_000_000;
pub const MATE_SCORE: i32 = 100_000;

pub struct Searcher {
    pub tt: TranspositionTable,
    pub nodes: u64,
    pub stop: bool,
    start_time: Instant,
    time_limit_ms: Option<u128>,
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_mb),
            nodes: 0,
            stop: false,
            start_time: Instant::now(),
            time_limit_ms: None,
        }
    }

    pub fn search(&mut self, board: &mut Board, max_depth: u8, time_ms: Option<u128>) -> Move {
        self.nodes = 0;
        self.stop = false;
        self.start_time = Instant::now();
        self.time_limit_ms = time_ms;

        let mut best_move = Move::NULL;

        for depth in 1..=max_depth {
            let score = self.negamax(board, depth, 0, -INFINITY, INFINITY);
            if self.stop && depth > 1 {
                break;
            }

            if let Some(entry) = self.tt.probe(board.hash) {
                if entry.best_move != Move::NULL {
                    best_move = entry.best_move;
                }
            }

            let elapsed = self.start_time.elapsed().as_millis().max(1);
            let nps = (self.nodes as u128 * 1000) / elapsed;

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
                    best_move
                );
            } else {
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                    depth, score, self.nodes, nps, elapsed, best_move
                );
            }

            if self.stop {
                break;
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

    fn check_time(&mut self) {
        if let Some(limit) = self.time_limit_ms {
            if (self.nodes & 2047) == 0 && self.start_time.elapsed().as_millis() >= limit {
                self.stop = true;
            }
        }
    }

    fn negamax(&mut self, board: &mut Board, depth: u8, ply: u8, mut alpha: i32, beta: i32) -> i32 {
        self.check_time();
        if self.stop {
            return 0;
        }

        if ply > 0 && board.is_repetition() {
            return 0;
        }

        let in_check = board.in_check();
        let depth = if in_check { depth + 1 } else { depth };

        if depth == 0 {
            return self.quiescence(board, alpha, beta);
        }

        self.nodes += 1;

        let alpha_orig = alpha;
        let mut tt_move = Move::NULL;

        if let Some(entry) = self.tt.probe(board.hash) {
            if entry.depth >= depth && ply > 0 {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => alpha = alpha.max(entry.score),
                    TTFlag::UpperBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
            tt_move = entry.best_move;
        }

        let mut moves = generate_legal_moves(board);
        if moves.count == 0 {
            if in_check {
                return -MATE_SCORE + ply as i32;
            }
            return 0;
        }

        self.order_moves(board, &mut moves, tt_move);

        let mut best_score = -INFINITY;
        let mut best_move = Move::NULL;

        for i in 0..moves.count {
            let m = moves.moves[i];
            let undo = board.make_move(m);
            let score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha);
            board.undo_move(m, undo);

            if self.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            alpha = alpha.max(score);
            if alpha >= beta {
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

        self.tt.store(board.hash, best_score, depth, flag, best_move);
        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        self.check_time();
        if self.stop {
            return 0;
        }

        self.nodes += 1;
        let stand_pat = evaluate(board);
        if stand_pat >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat);

        let mut moves = generate_noisy_moves(board);
        self.order_moves(board, &mut moves, Move::NULL);

        for i in 0..moves.count {
            let m = moves.moves[i];
            let undo = board.make_move(m);
            let score = -self.quiescence(board, -beta, -alpha);
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

    fn score_move(&self, board: &Board, m: Move, tt_move: Move) -> i32 {
        if m == tt_move {
            return 1_000_000;
        }
        let captured = board.piece_on[m.to()];
        if captured != Piece::None {
            let victim = PIECE_VALUES[captured.piece_type() as usize];
            let attacker = PIECE_VALUES[board.piece_on[m.from()].piece_type() as usize];
            return 100_000 + victim * 10 - attacker;
        }
        0
    }

    fn order_moves(&self, board: &Board, list: &mut MoveList, tt_move: Move) {
        let mut scores = [0i32; 256];
        for i in 0..list.count {
            scores[i] = self.score_move(board, list.moves[i], tt_move);
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
