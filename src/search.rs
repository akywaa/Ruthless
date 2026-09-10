use crate::board::Board;
use crate::eval::evaluate;
use crate::movegen::generate_legal_moves;
use crate::movepick::MovePicker;
use crate::see::see;
use crate::tt::{TTFlag, TranspositionTable};
use crate::types::{Move, MoveType, Piece, PieceType};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

pub const INFINITY: i32 = 32_500;
pub const MATE_SCORE: i32 = 32_000;
pub const MAX_PLY: usize = 128;
pub const CORR_ENTRIES: usize = 16384;
pub const CORR_BUCKETS: usize = 8;

static LMR: OnceLock<[[i32; 64]; 64]> = OnceLock::new();

fn init_lmr() -> [[i32; 64]; 64] {
    let mut table = [[0; 64]; 64];
    for d in 1..64 {
        for m in 1..64 {
            let base = ((d as f64).ln() * (m as f64).ln()) / 2.4;
            table[d][m] = (base as i32).max(1);
        }
    }
    table
}

#[inline(always)]
fn lmr(depth: usize, move_count: usize) -> i32 {
    let table = LMR.get_or_init(init_lmr);
    table[depth.min(63)][move_count.min(63)]
}

pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    pub nodes: u64,
    pub stop: Arc<AtomicBool>,
    pub thread_id: usize,
    pub num_threads: usize,
    pub shared_nodes: Arc<AtomicU64>,
    pub soft_stop_votes: Arc<AtomicUsize>,
    start_time: Instant,
    soft_time_ms: Option<u128>,
    hard_time_ms: Option<u128>,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    pawn_history: Box<[[[i32; 64]; 12]; 512]>,
    noisy_history: Box<[[[[i32; 2]; 6]; 64]; 12]>,
    counter_moves: [[Move; 64]; 64],
    conthist: Box<[[[[[[i32; 64]; 64]; 12]; 2]; 2]; 4]>,
    pawn_corr: Box<[[[i16; CORR_ENTRIES]; 2]; CORR_BUCKETS]>,
    non_pawn_corr: Box<[[[[i16; CORR_ENTRIES]; 2]; 2]; CORR_BUCKETS]>,
    cont_corr: Box<[[[[i16; 64]; 12]; 64]; 12]>,
    played_moves: [Move; MAX_PLY],
    played_pieces: [Piece; MAX_PLY],
    prev_in_check: [bool; MAX_PLY],
    prev_is_capture: [bool; MAX_PLY],
    eval_stack: [i32; MAX_PLY],
    root_move_nodes: [u64; 256],
    root_best_idx: usize,
    pub root_best_move: Move,
    pub root_score: i32,
    pub completed_depth: u8,
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

#[inline(always)]
fn update_corr(val: &mut i16, bonus: i32) {
    let clamped = bonus.clamp(-1600, 1600);
    *val += (clamped - (*val as i32 * clamped.abs()) / 16384) as i16;
}

fn alloc_box_zeroed<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    }
}

impl Searcher {
    pub fn new(
        tt: Arc<TranspositionTable>,
        stop: Arc<AtomicBool>,
        thread_id: usize,
        num_threads: usize,
        shared_nodes: Arc<AtomicU64>,
        soft_stop_votes: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            tt,
            nodes: 0,
            stop,
            thread_id,
            num_threads,
            shared_nodes,
            soft_stop_votes,
            start_time: Instant::now(),
            soft_time_ms: None,
            hard_time_ms: None,
            killers: [[Move::NULL; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            pawn_history: alloc_box_zeroed(),
noisy_history: alloc_box_zeroed(),
counter_moves: [[Move::NULL; 64]; 64],
conthist: alloc_box_zeroed(),
pawn_corr: alloc_box_zeroed(),
non_pawn_corr: alloc_box_zeroed(),
cont_corr: alloc_box_zeroed(),
            played_moves: [Move::NULL; MAX_PLY],
            played_pieces: [Piece::None; MAX_PLY],
            prev_in_check: [false; MAX_PLY],
            prev_is_capture: [false; MAX_PLY],
            eval_stack: [0; MAX_PLY],
            root_move_nodes: [0; 256],
            root_best_idx: 0,
            root_best_move: Move::NULL,
            root_score: 0,
            completed_depth: 0,
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[Move::NULL; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.counter_moves = [[Move::NULL; 64]; 64];
unsafe {
    std::ptr::write_bytes(self.pawn_history.as_mut(), 0, 1);
    std::ptr::write_bytes(self.noisy_history.as_mut(), 0, 1);
    std::ptr::write_bytes(self.conthist.as_mut(), 0, 1);
    std::ptr::write_bytes(self.pawn_corr.as_mut(), 0, 1);
    std::ptr::write_bytes(self.non_pawn_corr.as_mut(), 0, 1);
    std::ptr::write_bytes(self.cont_corr.as_mut(), 0, 1);
}
        self.played_moves = [Move::NULL; MAX_PLY];
        self.played_pieces = [Piece::None; MAX_PLY];
        self.prev_in_check = [false; MAX_PLY];
        self.prev_is_capture = [false; MAX_PLY];
        self.eval_stack = [0; MAX_PLY];
        self.root_move_nodes = [0; 256];
        self.root_best_idx = 0;
        self.root_best_move = Move::NULL;
        self.root_score = 0;
        self.completed_depth = 0;
    }

    #[inline(always)]
    fn raw_eval(&self, board: &Board) -> i32 {
        if let Some(entry) = self.tt.probe(board.tt_hash) {
            if entry.raw_eval != crate::tt::RAW_EVAL_NONE {
                return entry.raw_eval as i32;
            }
        }
        evaluate(board)
    }

    #[inline(always)]
    fn corrected_eval(&self, board: &Board, ply: u8, raw: i32) -> i32 {
        let side = board.side_to_move as usize;
        let bucket = (board.halfmove_clock as usize / 16).min(CORR_BUCKETS - 1);

        let p_idx = (board.pawn_hash as usize) & (CORR_ENTRIES - 1);
        let w_np_idx = (board.non_pawn_hash[0] as usize) & (CORR_ENTRIES - 1);
        let b_np_idx = (board.non_pawn_hash[1] as usize) & (CORR_ENTRIES - 1);

        let pawn_term = self.pawn_corr[bucket][side][p_idx] as i32 * 12;
        let np_term = (self.non_pawn_corr[bucket][0][side][w_np_idx] as i32
            + self.non_pawn_corr[bucket][1][side][b_np_idx] as i32)
            * 9;

        let mut cont_term = 0i32;
        let ply_idx = ply as usize;
        if ply_idx >= 1 {
            let p1_piece = self.played_pieces[ply_idx - 1];
            let p1_move = self.played_moves[ply_idx - 1];

            if p1_piece != Piece::None && p1_move != Move::NULL {
                if ply_idx >= 2 {
                    let p2_piece = self.played_pieces[ply_idx - 2];
                    let p2_move = self.played_moves[ply_idx - 2];
                    if p2_piece != Piece::None && p2_move != Move::NULL {
                        cont_term += self.cont_corr[p2_piece as usize][p2_move.to() as usize]
                            [p1_piece as usize][p1_move.to() as usize] as i32;
                    }
                }
                if ply_idx >= 4 {
                    let p4_piece = self.played_pieces[ply_idx - 4];
                    let p4_move = self.played_moves[ply_idx - 4];
                    if p4_piece != Piece::None && p4_move != Move::NULL {
                        cont_term += self.cont_corr[p4_piece as usize][p4_move.to() as usize]
                            [p1_piece as usize][p1_move.to() as usize] as i32;
                    }
                }
            }
        }
        cont_term *= 7;

        let total_bonus = (pawn_term + np_term + cont_term) / (8 * 64);
        // Scale down the history bonus so it doesn't overpower the new NNUE scale
        let clamped_bonus = (total_bonus / 3).clamp(-70, 70);

        let mut eval = raw + clamped_bonus;

        // Scale evaluation towards draw near 50-move rule
        eval = eval * (200 - board.halfmove_clock as i32) / 200;

        eval.clamp(-MATE_SCORE + 100, MATE_SCORE - 100)
    }

    const CONT_OFFSETS: [usize; 4] = [1, 2, 4, 6];

    #[inline(always)]
    fn get_conthist(&self, ply: u8, m: Move) -> i32 {
        let mut score = 0;
        let ply_idx = ply as usize;
        for (layer, &offset) in Self::CONT_OFFSETS.iter().enumerate() {
            if ply_idx >= offset {
                let prev_idx = ply_idx - offset;
                let prev_piece = self.played_pieces[prev_idx];
                let prev_move = self.played_moves[prev_idx];
                if prev_piece != Piece::None && prev_move != Move::NULL {
                    let chk = self.prev_in_check[prev_idx] as usize;
                    let cap = self.prev_is_capture[prev_idx] as usize;
                    score += self.conthist[layer][chk][cap][prev_piece as usize][prev_move.to() as usize][m.to() as usize];
                }
            }
        }
        score
    }

    #[inline(always)]
    fn draw_score(&self) -> i32 {
        (self.nodes as i32 & 3) - 2
    }

    #[inline(always)]
    fn quiet_history_score(&self, board: &Board, ply: u8, m: Move) -> i32 {
        let us = board.side_to_move as usize;
        let piece = board.piece_on[m.from()];
        let p_idx = (board.pawn_hash as usize) & 511;

        self.history[us][m.from() as usize][m.to() as usize]
            + self.pawn_history[p_idx][piece as usize][m.to() as usize]
            + self.get_conthist(ply, m)
    }

    #[inline(always)]
    fn noisy_history_score(&self, board: &Board, m: Move) -> i32 {
        let moving_pc = board.piece_on[m.from()];
        let victim_pt = match m.move_type() {
            MoveType::EnPassant => PieceType::Pawn,
            MoveType::Promotion => m.promo_type(),
            _ => board.piece_on[m.to()].piece_type(),
        };
        let pawn_threats = board.opponent_pawn_threats();
        let to_threatened = pawn_threats.contains(m.to()) as usize;

        self.noisy_history[moving_pc as usize][m.to() as usize][victim_pt as usize][to_threatened]
    }

    fn update_conthist(&mut self, ply: u8, m: Move, bonus: i32) {
        let ply_idx = ply as usize;
        for (layer, &offset) in Self::CONT_OFFSETS.iter().enumerate() {
            if ply_idx >= offset {
                let prev_idx = ply_idx - offset;
                let prev_piece = self.played_pieces[prev_idx];
                let prev_move = self.played_moves[prev_idx];
                if prev_piece != Piece::None && prev_move != Move::NULL {
                    let chk = self.prev_in_check[prev_idx] as usize;
                    let cap = self.prev_is_capture[prev_idx] as usize;
                    update_history(
                        &mut self.conthist[layer][chk][cap][prev_piece as usize][prev_move.to() as usize][m.to() as usize],
                        bonus,
                    );
                }
            }
        }
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
        self.stop.store(false, Ordering::Relaxed);
        self.start_time = Instant::now();
        self.tt.new_search();
        self.soft_time_ms = soft_time;
        self.hard_time_ms = hard_time;

        let mut best_move = legal_moves.moves[0];
        let mut prev_best_move = Move::NULL;
        let mut score = 0;
        let mut prev_score = 0;
        let mut stable_iterations = 0;

        for depth in 1..=max_depth {
            self.root_move_nodes = [0; 256];
            if depth >= 4 {
                let mut full_window = false;
                let mut delta = 25;
                let mut alpha = (score - delta).max(-INFINITY);
                let mut beta = (score + delta).min(INFINITY);

                loop {
                    score = self.negamax(board, depth, 0, alpha, beta, true, Move::NULL, false);
                    if self.stop.load(Ordering::Relaxed) {
                        break;
                    }

                    if score <= alpha {
                        if full_window { break; }
                        beta = (alpha + beta) / 2;
                        alpha = (alpha - delta).max(-INFINITY);
                        delta = delta.saturating_add(delta * 2 / 3);
                        if alpha <= -MATE_SCORE {
                            alpha = -INFINITY;
                            beta = INFINITY;
                            full_window = true;
                        }
                    } else if score >= beta {
                        if full_window { break; }
                        beta = (beta + delta).min(INFINITY);
                        delta = delta.saturating_add(delta * 2 / 3);
                        if beta >= MATE_SCORE {
                            alpha = -INFINITY;
                            beta = INFINITY;
                            full_window = true;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                score = self.negamax(board, depth, 0, -INFINITY, INFINITY, true, Move::NULL, false);
            }

            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            if self.root_best_move != Move::NULL {
                best_move = self.root_best_move;
            }

            let elapsed = self.start_time.elapsed().as_millis().max(1);
            let total_nodes = self.shared_nodes.load(Ordering::Relaxed) + (self.nodes & 2047);
            let nps = (total_nodes as u128 * 1000) / elapsed;

            let pv = self.extract_pv(board, depth);
            let pv_str = pv.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" ");

            if score.abs() > MATE_SCORE - 100 {
                let mate_dist = (MATE_SCORE - score.abs() + 1) / 2;
                let sign = if score > 0 { 1 } else { -1 };
                println!(
                    "info depth {} score mate {} nodes {} nps {} time {} pv {}",
                    depth,
                    mate_dist * sign,
                    total_nodes,
                    nps,
                    elapsed,
                    pv_str
                );
            } else {
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {} pv {}",
                    depth, score, total_nodes, nps, elapsed, pv_str
                );
            }

            if best_move == prev_best_move {
                stable_iterations += 1;
            } else {
                stable_iterations = 0;
            }

            let eval_stable = (prev_score - score).abs() < 15;

            if let Some(soft_limit) = self.soft_time_ms {
                let score_diff = (prev_score - score).clamp(-100, 100) as f32;
                let score_trend = if score < -100 { 1.25 } else { (0.85 + 0.03 * score_diff).clamp(0.8, 1.3) };
                let pv_factor = (1.25 - 0.05 * (stable_iterations as f32)).max(0.7);
                let eval_factor = if eval_stable { 0.85 } else { 1.15 };

                let root_total: u64 = self.root_move_nodes.iter().sum();
                let best_share = if root_total > 0 {
                    self.root_move_nodes[self.root_best_idx.min(255)] as f32 / root_total as f32
                } else {
                    0.5
                };
                let node_factor = if best_share > 0.70 {
                    0.75
                } else if best_share < 0.40 {
                    1.25
                } else {
                    1.0
                };

                let dynamic_soft = ((soft_limit as f32) * score_trend * pv_factor * eval_factor * node_factor) as u128;

                if elapsed >= dynamic_soft || elapsed >= self.hard_time_ms.unwrap_or(u128::MAX) {
                    let votes = self.soft_stop_votes.fetch_add(1, Ordering::AcqRel) + 1;
                    let majority = (self.num_threads * 65).div_ceil(100);
                    if votes >= majority || elapsed >= dynamic_soft || elapsed >= self.hard_time_ms.unwrap_or(u128::MAX) {
                        self.stop.store(true, Ordering::Relaxed);
                    }
                    break;
                }
            }

            prev_best_move = best_move;
            prev_score = score;

            self.root_best_move = best_move;
            self.root_score = score;
            self.completed_depth = depth;
        }

        best_move
    }

    pub fn search_helper(&mut self, board: &mut Board, max_depth: u8, soft_time: Option<u128>) {
        self.nodes = 0;
        self.soft_time_ms = soft_time;
        let mut score = 0;
        let start_depth = 1 + (self.thread_id % 2) as u8;
        let target_depth = (max_depth as usize + 8).min(MAX_PLY) as u8;

        for depth in start_depth..=target_depth {
            self.root_move_nodes = [0; 256];
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            if depth >= 4 {
                let mut full_window = false;
                let mut delta = 25;
                let mut alpha = (score - delta).max(-INFINITY);
                let mut beta = (score + delta).min(INFINITY);

                loop {
                    score = self.negamax(board, depth, 0, alpha, beta, false, Move::NULL, false);
                    if self.stop.load(Ordering::Relaxed) {
                        break;
                    }

                    if score <= alpha {
                        if full_window { break; }
                        beta = (alpha + beta) / 2;
                        alpha = (alpha - delta).max(-INFINITY);
                        delta = delta.saturating_add(delta * 2 / 3);
                        if alpha <= -MATE_SCORE {
                            alpha = -INFINITY;
                            beta = INFINITY;
                            full_window = true;
                        }
                    } else if score >= beta {
                        if full_window { break; }
                        beta = (beta + delta).min(INFINITY);
                        delta = delta.saturating_add(delta * 2 / 3);
                        if beta >= MATE_SCORE {
                            alpha = -INFINITY;
                            beta = INFINITY;
                            full_window = true;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                score = self.negamax(board, depth, 0, -INFINITY, INFINITY, false, Move::NULL, false);
            }

            self.root_score = score;
            self.completed_depth = depth;
            if let Some(entry) = self.tt.probe(board.tt_hash) {
                if entry.best_move != Move::NULL {
                    self.root_best_move = entry.best_move;
                }
            }

            if let Some(soft_limit) = self.soft_time_ms {
                let elapsed = self.start_time.elapsed().as_millis();
                if depth >= 6 && elapsed >= soft_limit {
                    let votes = self.soft_stop_votes.fetch_add(1, Ordering::AcqRel) + 1;
                    let majority = (self.num_threads * 65).div_ceil(100);
                    if votes >= majority {
                        self.stop.store(true, Ordering::Relaxed);
                    }
                    break;
                }
            }
        }
    }

    fn extract_pv(&self, board: &mut Board, depth: u8) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut undos = Vec::new();

        for _ in 0..depth {
            if let Some(entry) = self.tt.probe(board.tt_hash) {
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
    if self.nodes > 0 && (self.nodes & 2047) == 0 {
        self.shared_nodes.fetch_add(2048, Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_millis();

        // Only the hard limit is allowed to abort mid-search; the soft limit
        // is enforced between iterations so a partial depth is never discarded.
        if let Some(hard_limit) = self.hard_time_ms {
            if elapsed >= hard_limit {
                self.stop.store(true, Ordering::Relaxed);
            }
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
        cut_node: bool,
    ) -> i32 {
        self.check_time();
        if self.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if ply > 0 && board.is_draw() {
            return self.draw_score();
        }

        if (ply as usize) >= MAX_PLY {
            return evaluate(board);
        }

        let in_check = board.in_check();
        if in_check && (ply as usize) < MAX_PLY - 1 && excluded_move == Move::NULL {
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

        if let Some(entry) = self.tt.probe(board.tt_hash) {
            has_tt = true;
            tt_score = score_from_tt(entry.score, ply);
            tt_depth = entry.depth;
            tt_flag = entry.flag;
            tt_move = entry.best_move;

            if excluded_move == Move::NULL && entry.depth >= depth && ply > 0 && !is_pv {
                match entry.flag {
                    TTFlag::Exact => return tt_score,
                    TTFlag::LowerBound if tt_score >= beta => return tt_score,
                    TTFlag::UpperBound if tt_score <= alpha => return tt_score,
                    _ => {}
                }
            }
        }

        if excluded_move == Move::NULL && depth >= 2 && tt_move == Move::NULL && (is_pv || cut_node) {
            depth -= 1;
        }

        let raw_eval = self.raw_eval(board);
        let static_eval = self.corrected_eval(board, ply, raw_eval);
        if (ply as usize) < MAX_PLY {
            self.eval_stack[ply as usize] = static_eval;
        }

        let improving = if in_check || ply < 2 {
            false
        } else {
            static_eval > self.eval_stack[(ply - 2) as usize]
        };

        if !is_pv && !in_check {
            // Reverse futility pruning
            let rfp_margin = (80 - 20 * improving as i32) * (depth as i32);
            if depth <= 9 && static_eval - rfp_margin >= beta {
                return static_eval;
            }

            // Razoring
            if depth <= 3 && static_eval + 300 + 150 * (depth as i32) <= alpha {
                let qscore = self.quiescence(board, alpha, beta, ply);
                if qscore <= alpha {
                    return qscore;
                }
            }

            // Null move pruning
            if excluded_move == Move::NULL
                && depth >= 3
                && static_eval >= beta
                && board.has_non_pawn_material(board.side_to_move)
                && (ply == 0 || self.played_moves[(ply - 1) as usize] != Move::NULL)
            {
                let r = 3 + depth / 3 + ((static_eval - beta) / 200).clamp(0, 3) as u8;
                if (ply as usize) < MAX_PLY {
                    self.played_moves[ply as usize] = Move::NULL;
                    self.played_pieces[ply as usize] = Piece::None;
                }
                let undo = board.make_null_move();
                let score = -self.negamax(
                    board,
                    depth.saturating_sub(r),
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    Move::NULL,
                    !cut_node,
                );
                board.undo_null_move(undo);

                if score >= beta {
                    return if score >= MATE_SCORE - 100 { beta } else { score };
                }
            }
        }

        // ProbCut
        if !is_pv && !in_check && excluded_move == Move::NULL && depth >= 5 && beta.abs() < MATE_SCORE - 100 {
            let probcut_beta = beta + 200;
            let mut probcut_picker = MovePicker::new_qsearch(Move::NULL);

            while let Some(m) = probcut_picker.next(
                board,
                &self.history,
                &self.pawn_history,
                &self.conthist,
                &self.noisy_history,
                ply,
                &self.played_pieces,
                &self.played_moves,
                &self.prev_in_check,
                &self.prev_is_capture,
            ) {
                if !see(board, m, probcut_beta - static_eval) {
                    continue;
                }

                let undo = board.make_move(m);
                let mut score = -self.quiescence(board, -probcut_beta, -probcut_beta + 1, ply + 1);

                if score >= probcut_beta {
                    score = -self.negamax(
                        board,
                        depth - 4,
                        ply + 1,
                        -probcut_beta,
                        -probcut_beta + 1,
                        false,
                        Move::NULL,
                        !cut_node,
                    );
                }

                board.undo_move(m, undo);

                if score >= probcut_beta {
                    return score;
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
            let singular_margin = (depth as i32) * 2;
            let singular_beta = tt_score - singular_margin;
            let singular_depth = (depth - 1) / 2;

            let score = self.negamax(
                board,
                singular_depth,
                ply,
                singular_beta - 1,
                singular_beta,
                false,
                tt_move,
                cut_node,
            );

            if self.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score < singular_beta {
                extension = 1;
                if !is_pv && score + 20 < singular_beta && ply < self.completed_depth {
                    extension = 2;
                }
            } else if singular_beta >= beta {
                return singular_beta;
            } else if !is_pv && tt_score >= beta {
                extension = -1;
            } else if !is_pv && cut_node {
                extension = -1;
            }
        }

        let mut best_score = -INFINITY;
        let mut best_move = Move::NULL;
        let mut moves_searched = 0;

        let lmp_threshold = 2 + (depth as usize) * (depth as usize) / (1 + (!improving as usize) * 2);
        let futility_margin = 70 + 75 * (depth as i32);
        let futility_pruning = !is_pv
            && !in_check
            && depth <= 6
            && alpha.abs() < MATE_SCORE - 100
            && (static_eval + futility_margin <= alpha);

        let mut quiet_moves = [Move::NULL; 64];
        let mut quiet_count = 0;
        let mut noisy_moves = [Move::NULL; 32];
        let mut noisy_count = 0;

        let prev_move = if ply > 0 && (ply as usize) < MAX_PLY {
            self.played_moves[(ply - 1) as usize]
        } else {
            Move::NULL
        };
        let counter_move = if prev_move != Move::NULL {
            self.counter_moves[prev_move.from() as usize][prev_move.to() as usize]
        } else {
            Move::NULL
        };

        let killers = if (ply as usize) < MAX_PLY {
            self.killers[ply as usize]
        } else {
            [Move::NULL; 2]
        };

        let mut picker = MovePicker::new(tt_move, killers, counter_move);

        while let Some(m) = picker.next(
            board,
            &self.history,
            &self.pawn_history,
            &self.conthist,
            &self.noisy_history,
            ply,
            &self.played_pieces,
            &self.played_moves,
            &self.prev_in_check,
            &self.prev_is_capture,
        ) {
            if m == excluded_move {
                continue;
            }

            let is_capture = board.piece_on[m.to()] != Piece::None || m.move_type() == MoveType::EnPassant;
            let is_quiet = !is_capture && m.move_type() != MoveType::Promotion;
            let hist = if is_quiet {
                self.quiet_history_score(board, ply, m)
            } else {
                self.noisy_history_score(board, m)
            };

            if !is_pv && !in_check && moves_searched > 0 {
                if is_quiet && depth <= 10 && moves_searched >= lmp_threshold {
                    continue;
                }

                if is_quiet
                    && futility_pruning
                    && m != killers[0]
                    && m != killers[1]
                    && m != counter_move
                {
                    continue;
                }

                if is_quiet && depth <= 6 && !see(board, m, -35 * (depth as i32) * (depth as i32)) {
                    continue;
                }

                if is_quiet && depth <= 7 {
                    let threshold = -1200 * (depth as i32) - if improving { 800 } else { 0 };
                    if hist < threshold {
                        continue;
                    }
                }

                if !is_quiet && depth <= 6 && !see(board, m, -120 * (depth as i32)) {
                    continue;
                }
            }

            if is_quiet && quiet_count < 64 {
                quiet_moves[quiet_count] = m;
                quiet_count += 1;
            } else if !is_quiet && noisy_count < 32 {
                noisy_moves[noisy_count] = m;
                noisy_count += 1;
            }

            if (ply as usize) < MAX_PLY {
                self.played_moves[ply as usize] = m;
                self.played_pieces[ply as usize] = board.piece_on[m.from()];
                self.prev_in_check[ply as usize] = in_check;
                self.prev_is_capture[ply as usize] = is_capture;
            }

            let nodes_before = if ply == 0 { self.nodes } else { 0 };
            let is_see_ge_zero = !is_quiet && see(board, m, 0);

            let undo = board.make_move(m);
            let gives_check = board.in_check();

            let ext = if m == tt_move { extension } else { 0 };

            let score = if moves_searched == 0 {
                let next_depth = (depth as i32 - 1 + ext).max(0) as u8;
                -self.negamax(board, next_depth, ply + 1, -beta, -alpha, is_pv, Move::NULL, false)
            } else {
                let mut r = 0;

                if depth >= 3
                    && moves_searched >= 1
                    && (!is_pv || is_quiet)
                    && (is_quiet || moves_searched >= 2)
                {
                    r = lmr(depth as usize, moves_searched);

                    if !is_quiet {
                        r = r * 33 / 64;
                        if !is_see_ge_zero {
                            r += 2;
                        }
                        r -= (hist / 8192).clamp(-2, 2);
                    }

                    if !improving {
                        r += 1;
                    }

                    if cut_node {
                        r += 2;
                    }

                    if is_quiet {
                        if m == killers[0] || m == killers[1] || m == counter_move {
                            r -= 2;
                        }
                        r -= (hist / 8192).clamp(-2, 2);
                    }

                    if is_pv {
                        r -= 1;
                    }

                    if self.thread_id > 0 && ((moves_searched + self.thread_id) % 2 == 0) {
                        r += 1;
                    }

                    // Never reduce checks: critical for tactics.
                    if gives_check {
                        r = 0;
                    }

                    r = r.clamp(0, depth as i32 - 2);
                }

                let reduced = (depth as i32 - 1 - r).max(0) as u8;

                let mut s = -self.negamax(board, reduced, ply + 1, -alpha - 1, -alpha, false, Move::NULL, true);

                if s > alpha && reduced < depth - 1 {
                    s = -self.negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha, false, Move::NULL, !cut_node);
                }

                if s > alpha && s < beta {
                    s = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, true, Move::NULL, false);
                }

                s
            };

            board.undo_move(m, undo);

            if ply == 0 {
                self.root_move_nodes[moves_searched] = self.nodes - nodes_before;
            }
            moves_searched += 1;

            if self.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
                if ply == 0 {
                    self.root_best_move = m;
                    self.root_best_idx = moves_searched - 1;
                }
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                let bonus = ((depth as i32) * (depth as i32)).min(1600);

                let moving_pc = board.piece_on[m.from()] as usize;
                let victim_pt = match m.move_type() {
                    MoveType::EnPassant => PieceType::Pawn,
                    MoveType::Promotion => m.promo_type(),
                    _ => board.piece_on[m.to()].piece_type(),
                } as usize;

                let threats = board.opponent_threats();
                let to_threatened = threats.contains(m.to()) as usize;

                if is_quiet {
                    if (ply as usize) < MAX_PLY {
                        if self.killers[ply as usize][0] != m {
                            self.killers[ply as usize][1] = self.killers[ply as usize][0];
                            self.killers[ply as usize][0] = m;
                        }

                        if prev_move != Move::NULL {
                            self.counter_moves[prev_move.from() as usize][prev_move.to() as usize] = m;
                        }
                    }

                    let us = board.side_to_move as usize;
                    let p_idx = (board.pawn_hash as usize) & 511;

                    update_history(&mut self.history[us][m.from() as usize][m.to() as usize], bonus);
                    if moving_pc < 12 {
                        update_history(&mut self.pawn_history[p_idx][moving_pc][m.to() as usize], bonus);
                    }
                    self.update_conthist(ply, m, bonus);

                    for j in 0..quiet_count {
                        let qm = quiet_moves[j];
                        if qm == m || qm == Move::NULL { continue; }

                        let q_pc = board.piece_on[qm.from()] as usize;
                        if q_pc < 12 {
                            update_history(&mut self.history[us][qm.from() as usize][qm.to() as usize], -bonus);
                            update_history(&mut self.pawn_history[p_idx][q_pc][qm.to() as usize], -bonus);
                            self.update_conthist(ply, qm, -bonus);
                        }
                    }

                    if noisy_count > 0 {
                        for j in 0..noisy_count {
                            let nm = noisy_moves[j];
                            if nm == m || nm == Move::NULL { continue; }

                            let n_pc = board.piece_on[nm.from()] as usize;
                            let n_victim_pt = match nm.move_type() {
                                MoveType::EnPassant => PieceType::Pawn,
                                MoveType::Promotion => nm.promo_type(),
                                _ => board.piece_on[nm.to()].piece_type(),
                            } as usize;

                            if n_pc < 12 && n_victim_pt < 6 {
                                let n_to_threatened = threats.contains(nm.to()) as usize;
                                update_history(&mut self.noisy_history[n_pc][nm.to() as usize][n_victim_pt][n_to_threatened], -bonus);
                            }
                        }
                    }
                } else {
                    if moving_pc < 12 && victim_pt < 6 {
                        update_history(&mut self.noisy_history[moving_pc][m.to() as usize][victim_pt][to_threatened], bonus);
                    }

                    for j in 0..noisy_count {
                        let nm = noisy_moves[j];
                        if nm == m || nm == Move::NULL { continue; }

                        let n_pc = board.piece_on[nm.from()] as usize;
                        let n_victim_pt = match nm.move_type() {
                            MoveType::EnPassant => PieceType::Pawn,
                            MoveType::Promotion => nm.promo_type(),
                            _ => board.piece_on[nm.to()].piece_type(),
                        } as usize;

                        if n_pc < 12 && n_victim_pt < 6 {
                            let n_to_threatened = threats.contains(nm.to()) as usize;
                            update_history(&mut self.noisy_history[n_pc][nm.to() as usize][n_victim_pt][n_to_threatened], -bonus);
                        }
                    }
                }
                break;
            }
        }

        if moves_searched == 0 {
            if excluded_move != Move::NULL {
                return alpha;
            }
            if in_check {
                return -MATE_SCORE + ply as i32;
            }
            return 0;
        }

        let flag = if best_score <= alpha_orig {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        if excluded_move == Move::NULL {
            self.tt.store(board.tt_hash, score_to_tt(best_score, ply), depth, flag, best_move, raw_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16);

            let tt_move_quiet = best_move == Move::NULL
                || (board.piece_on[best_move.to()] == Piece::None
                    && best_move.move_type() != MoveType::Promotion
                    && best_move.move_type() != MoveType::EnPassant);

            if !in_check
                && tt_move_quiet
                && !(flag == TTFlag::LowerBound && best_score <= static_eval)
                && !(flag == TTFlag::UpperBound && best_score >= static_eval)
            {
                let bonus = ((best_score - static_eval) * (depth as i32)).clamp(-1200, 1200);
                let side = board.side_to_move as usize;
                let bucket = (board.halfmove_clock as usize / 16).min(CORR_BUCKETS - 1);

                let p_idx = (board.pawn_hash as usize) & (CORR_ENTRIES - 1);
                let w_np_idx = (board.non_pawn_hash[0] as usize) & (CORR_ENTRIES - 1);
                let b_np_idx = (board.non_pawn_hash[1] as usize) & (CORR_ENTRIES - 1);

                update_corr(&mut self.pawn_corr[bucket][side][p_idx], bonus);
                update_corr(&mut self.non_pawn_corr[bucket][0][side][w_np_idx], bonus);
                update_corr(&mut self.non_pawn_corr[bucket][1][side][b_np_idx], bonus);

                let ply_idx = ply as usize;
                if ply_idx >= 1 {
                    let p1_piece = self.played_pieces[ply_idx - 1];
                    let p1_move = self.played_moves[ply_idx - 1];

                    if p1_piece != Piece::None && p1_move != Move::NULL {
                        if ply_idx >= 2 {
                            let p2_piece = self.played_pieces[ply_idx - 2];
                            let p2_move = self.played_moves[ply_idx - 2];
                            if p2_piece != Piece::None && p2_move != Move::NULL {
                                update_corr(
                                    &mut self.cont_corr[p2_piece as usize][p2_move.to() as usize]
                                        [p1_piece as usize][p1_move.to() as usize],
                                    bonus,
                                );
                            }
                        }
                        if ply_idx >= 4 {
                            let p4_piece = self.played_pieces[ply_idx - 4];
                            let p4_move = self.played_moves[ply_idx - 4];
                            if p4_piece != Piece::None && p4_move != Move::NULL {
                                update_corr(
                                    &mut self.cont_corr[p4_piece as usize][p4_move.to() as usize]
                                        [p1_piece as usize][p1_move.to() as usize],
                                    bonus,
                                );
                            }
                        }
                    }
                }
            }
        }
        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: u8) -> i32 {
        self.check_time();
        if self.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if ply > 0 && (board.is_repetition() || board.halfmove_clock >= 100) {
            return self.draw_score();
        }

        if (ply as usize) >= MAX_PLY {
            return evaluate(board);
        }

        let alpha_orig = alpha;
        let in_check = board.in_check();
        let mut tt_move = Move::NULL;

        if let Some(entry) = self.tt.probe(board.tt_hash) {
            tt_move = entry.best_move;
            let tt_score = score_from_tt(entry.score, ply);

            match entry.flag {
                TTFlag::Exact => return tt_score,
                TTFlag::LowerBound if tt_score >= beta => return tt_score,
                TTFlag::UpperBound if tt_score <= alpha => return tt_score,
                _ => {}
            }
        }

        self.nodes += 1;

        let mut stand_pat = -INFINITY;
        let mut raw_eval = 0;
        if !in_check {
            raw_eval = self.raw_eval(board);
            stand_pat = self.corrected_eval(board, ply, raw_eval);
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
        }

        let raw_eval_to_store = if in_check {
            crate::tt::RAW_EVAL_NONE
        } else {
            raw_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        };

        if in_check {
            let moves = generate_legal_moves(board);
            if moves.count == 0 {
                return -MATE_SCORE + ply as i32;
            }
            let mut best_score = -INFINITY;
            let mut best_move = Move::NULL;

            for &m in moves.as_slice() {
                let undo = board.make_move(m);
                let score = -self.quiescence(board, -beta, -alpha, ply + 1);
                board.undo_move(m, undo);

                if self.stop.load(Ordering::Relaxed) {
                    return 0;
                }

                if score > best_score {
                    best_score = score;
                    best_move = m;
                }

                if score >= beta {
                    self.tt.store(board.tt_hash, score_to_tt(score, ply), 0, TTFlag::LowerBound, m, raw_eval_to_store);
                    return score;
                }
                alpha = alpha.max(score);
            }

            let flag = if best_score >= beta {
                TTFlag::LowerBound
            } else if best_score > alpha_orig {
                TTFlag::Exact
            } else {
                TTFlag::UpperBound
            };
            self.tt.store(board.tt_hash, score_to_tt(best_score, ply), 0, flag, best_move, raw_eval_to_store);
            return best_score;
        }

        let mut picker = MovePicker::new_qsearch(tt_move);
        let mut best_score = stand_pat;
        let mut best_move = Move::NULL;

        while let Some(m) = picker.next(
            board,
            &self.history,
            &self.pawn_history,
            &self.conthist,
            &self.noisy_history,
            ply,
            &self.played_pieces,
            &self.played_moves,
            &self.prev_in_check,
            &self.prev_is_capture,
        ) {
            let is_promo = m.move_type() == MoveType::Promotion;
            if !is_promo {
                let cap_pt = match m.move_type() {
                    MoveType::EnPassant => PieceType::Pawn,
                    _ => board.piece_on[m.to()].piece_type(),
                };
                if cap_pt == PieceType::None {
                    continue;
                }
                let gain = crate::eval::PIECE_VALUES[cap_pt as usize];
                if stand_pat + gain + 200 < alpha {
                    continue;
                }
                if !see(board, m, 0) {
                    continue;
                }
            }

            let undo = board.make_move(m);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            board.undo_move(m, undo);

            if self.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score >= beta {
                self.tt.store(board.tt_hash, score_to_tt(score, ply), 0, TTFlag::LowerBound, m, raw_eval_to_store);
                return score;
            }
            alpha = alpha.max(score);
        }

        let flag = if best_score >= beta {
            TTFlag::LowerBound
        } else if best_score > alpha_orig {
            TTFlag::Exact
        } else {
            TTFlag::UpperBound
        };
        self.tt.store(board.tt_hash, score_to_tt(best_score, ply), 0, flag, best_move, raw_eval_to_store);

        best_score
    }
}
