use std::mem::MaybeUninit;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::movegen::{generate_noisy_pseudo, generate_quiet_pseudo};
use crate::see::see;
use crate::search::{ContHist, MAX_PLY};
use crate::types::{Move, MoveList, MoveType, Piece, PieceType};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Stage {
    TTMove,
    GenerateNoisy,
    GoodNoisy,
    Killers,
    CounterMove,
    GenerateQuiets,
    Quiets,
    BadNoisy,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    killer_idx: usize,
    counter_move: Move,
    moves: MoveList,
    /// Written lazily for `cur_idx..moves.count` by the scoring pass, so the
    /// constructor does not have to zero a kilobyte on every node.
    scores: [MaybeUninit<i32>; 256],
    cur_idx: usize,
    /// At most 64 pieces can deliver a losing capture, so this buffer is much
    /// smaller than a full MoveList and is never touched in qsearch.
    bad_noisy: [Move; 64],
    bad_noisy_count: usize,
    qsearch: bool,
    checkers: Bitboard,
    pinned: Bitboard,
    king_info_ready: bool,
    /// Opponent threats in the node's position, computed at most once and
    /// shared by move scoring and the search's history updates.
    threats: Option<Bitboard>,
    /// Check squares for this node, computed at most once and shared by every
    /// move scored here and by the search's pruning decisions.
    check_squares: Option<[Bitboard; 6]>,
    /// Whether the move returned by the last `next()` call is a capture with
    /// SEE >= 0. Cached here so the search does not have to run SEE a second
    /// time on the same move.
    pub last_see_ge_zero: bool,
    /// Whether the last returned move came from the good-noisy stage, where SEE
    /// was already proven to be >= 0.
    pub last_good_noisy: bool,
}

impl MovePicker {
    pub fn new(tt_move: Move, killers: [Move; 2], counter_move: Move) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers,
            killer_idx: 0,
            counter_move,
            moves: MoveList::new(),
            scores: [MaybeUninit::uninit(); 256],
            cur_idx: 0,
            bad_noisy: [Move::NULL; 64],
            bad_noisy_count: 0,
            qsearch: false,
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
            king_info_ready: false,
            threats: None,
            check_squares: None,
            last_see_ge_zero: true,
            last_good_noisy: false,
        }
    }

    /// Opponent threats for the position this picker was created for.
    #[inline(always)]
    pub fn threats(&mut self, board: &Board) -> Bitboard {
        match self.threats {
            Some(t) => t,
            None => {
                let t = board.opponent_threats();
                self.threats = Some(t);
                t
            }
        }
    }

    /// Whether `m` gives a direct check, using per-node precomputed squares.
    /// The squares are kept by reference so the 6-bitboard table is not copied
    /// out of the picker on every call.
    #[inline(always)]
    pub fn gives_check(&mut self, board: &Board, m: Move) -> bool {
        if self.check_squares.is_none() {
            self.check_squares = Some(board.check_squares());
        }
        let squares = self.check_squares.as_ref().unwrap();
        board.gives_check_with(m, squares)
    }

    pub fn new_qsearch(tt_move: Move) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers: [Move::NULL; 2],
            killer_idx: 0,
            counter_move: Move::NULL,
            moves: MoveList::new(),
            scores: [MaybeUninit::uninit(); 256],
            cur_idx: 0,
            bad_noisy: [Move::NULL; 64],
            bad_noisy_count: 0,
            qsearch: true,
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
            king_info_ready: false,
            threats: None,
            check_squares: None,
            last_see_ge_zero: true,
            last_good_noisy: false,
        }
    }

    #[inline(always)]
    fn ensure_king_info(&mut self, board: &Board) {
        if !self.king_info_ready {
            self.checkers = board.checkers();
            self.pinned = board.pinned_pieces(board.side_to_move);
            self.king_info_ready = true;
        }
    }

    #[inline(always)]
    fn is_legal(&mut self, board: &Board, m: Move) -> bool {
        self.ensure_king_info(board);
        board.is_legal_with(m, self.checkers, self.pinned)
    }

    pub fn next(
        &mut self,
        board: &Board,
        history: &crate::search::MainHistory,
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &ContHist,
        noisy_history: &[[[[i32; 2]; 6]; 64]; 12],
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
    ) -> Option<Move> {
        self.last_good_noisy = false;
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateNoisy;
                    let m = self.tt_move;
                    let is_noisy = board.piece_on[m.to()] != Piece::None
                        || m.move_type() == MoveType::Promotion
                        || m.move_type() == MoveType::EnPassant;

                    if (!self.qsearch || is_noisy)
                        && self.is_pseudo_legal_any(board, m)
                        && self.is_legal(board, m)
                    {
                        // SEE is not evaluated for the TT move; assume a good
                        // capture so LMR stays conservative.
                        self.last_see_ge_zero = true;
                        return Some(m);
                    }
                }
                Stage::GenerateNoisy => {
                    self.moves.count = 0;
                    generate_noisy_pseudo(board, &mut self.moves);
                    self.score_noisy(board, noisy_history);
                    self.cur_idx = 0;
                    self.stage = Stage::GoodNoisy;
                }
                Stage::GoodNoisy => {
                    while self.cur_idx < self.moves.count {
                        let m = self.pick_best();
                        if m == self.tt_move {
                            continue;
                        }

                        // Underpromotions are a full-depth concern; searching all
                        // four in the quiescence tree only costs nodes.
                        if self.qsearch
                            && m.move_type() == MoveType::Promotion
                            && m.promo_type() != PieceType::Queen
                        {
                            continue;
                        }

                        // Fixed SEE 0 split. A history-dependent threshold was
                        // proposed but not applied: bad captures are already
                        // pruned by capture history in the search loop.
                        if see(board, m, 0) {
                            if self.is_legal(board, m) {
                                self.last_see_ge_zero = true;
                                self.last_good_noisy = true;
                                return Some(m);
                            }
                        } else if !self.qsearch && self.bad_noisy_count < self.bad_noisy.len() {
                            self.bad_noisy[self.bad_noisy_count] = m;
                            self.bad_noisy_count += 1;
                        }
                    }

                    if self.qsearch {
                        self.stage = Stage::Done;
                    } else {
                        self.stage = Stage::Killers;
                    }
                }
                Stage::Killers => {
                    while self.killer_idx < 2 {
                        let k = self.killers[self.killer_idx];
                        self.killer_idx += 1;
                        if k != Move::NULL
                            && k != self.tt_move
                            && !(self.killer_idx == 2 && k == self.killers[0])
                            && self.is_pseudo_legal(board, k)
                            && self.is_legal(board, k)
                        {
                            self.last_see_ge_zero = true;
                            return Some(k);
                        }
                    }
                    self.stage = Stage::CounterMove;
                }
                Stage::CounterMove => {
                    self.stage = Stage::GenerateQuiets;
                    let cm = self.counter_move;
                    if cm != Move::NULL
                        && cm != self.tt_move
                        && cm != self.killers[0]
                        && cm != self.killers[1]
                        && self.is_pseudo_legal(board, cm)
                        && self.is_legal(board, cm)
                    {
                        self.last_see_ge_zero = true;
                        return Some(cm);
                    }
                }
                Stage::GenerateQuiets => {
                    self.moves.count = 0;
                    generate_quiet_pseudo(board, &mut self.moves);
                    self.score_quiets(
                        board,
                        history,
                        pawn_history,
                        conthist,
                        ply,
                        played_pieces,
                        played_moves,
                    );
                    self.cur_idx = 0;
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    while self.cur_idx < self.moves.count {
                        let m = self.pick_best();
                        if m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move {
                            continue;
                        }
                        if self.is_legal(board, m) {
                            self.last_see_ge_zero = true;
                            return Some(m);
                        }
                    }
                    self.cur_idx = 0;
                    self.stage = Stage::BadNoisy;
                }
                Stage::BadNoisy => {
                    while self.cur_idx < self.bad_noisy_count {
                        let m = self.bad_noisy[self.cur_idx];
                        self.cur_idx += 1;
                        if m != self.tt_move && self.is_legal(board, m) {
                            self.last_see_ge_zero = false;
                            return Some(m);
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    fn pick_best(&mut self) -> Move {
        let mut best_score = unsafe { self.scores[self.cur_idx].assume_init() };
        let mut best_idx = self.cur_idx;

        for i in (self.cur_idx + 1)..self.moves.count {
            let s = unsafe { self.scores[i].assume_init() };
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }

        self.scores.swap(self.cur_idx, best_idx);
        self.moves.swap(self.cur_idx, best_idx);

        let m = self.moves.get(self.cur_idx);
        self.cur_idx += 1;
        m
    }

    fn score_noisy(&mut self, board: &Board, noisy_history: &[[[[i32; 2]; 6]; 64]; 12]) {
        let threats = self.threats(board);

        for i in 0..self.moves.count {
            let m = self.moves.get(i);
            let attacker = board.piece_on[m.from()];
            let captured = board.piece_on[m.to()];
            let victim_pt = match m.move_type() {
                MoveType::EnPassant => PieceType::Pawn,
                MoveType::Promotion => m.promo_type(),
                _ => captured.piece_type(),
            };

            let to_threatened = threats.contains(m.to()) as usize;

            let attacker_idx = attacker as usize;
            let victim_idx = victim_pt as usize;

            // Guard against broken moves (Piece::None attacker/victim) that
            // would index out of bounds in PIECE_VALUES and noisy_history.
            if attacker_idx < 12 && victim_idx < 6 {
                let mvv_lva = PIECE_VALUES[victim_idx] * 12 - PIECE_VALUES[attacker.piece_type() as usize];
                let hist = noisy_history[attacker_idx][m.to() as usize][victim_idx][to_threatened];
                let mut score = mvv_lva + hist / 32;

                if m.move_type() == MoveType::Promotion && m.promo_type() == PieceType::Queen {
                    score += 3000;
                }
                if self.gives_check(board, m) {
                    score += 2000;
                }

                self.scores[i].write(score);
            } else {
                self.scores[i].write(0);
            }
        }
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        history: &crate::search::MainHistory,
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &ContHist,
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
    ) {
        let us = board.side_to_move;
        let pawn_threats = board.opponent_pawn_threats();
        let p_idx = (board.pawn_hash as usize) & 511;
        let offsets = [1usize, 2, 4, 6];
        let ply_idx = ply as usize;

        for i in 0..self.moves.count {
            let m = self.moves.get(i);
            let from = m.from();
            let to = m.to();
            let piece = board.piece_on[from];
            let pt = piece.piece_type();
            let from_threat = pawn_threats.contains(from) as usize;
            let to_threat = pawn_threats.contains(to) as usize;

            let mut score = history[us as usize][from_threat][to_threat][from as usize][to as usize]
                + pawn_history[p_idx][piece as usize][to as usize];

            for (layer, &offset) in offsets.iter().enumerate() {
                if ply_idx >= offset {
                    let prev_idx = ply_idx - offset;
                    let prev_piece = played_pieces[prev_idx];
                    let prev_move = played_moves[prev_idx];
                    if prev_piece != Piece::None
                        && prev_move != Move::NULL
                        && (piece as usize) < 12
                    {
                        score += conthist[layer][prev_piece as usize][prev_move.to() as usize]
                            [piece as usize][to as usize] as i32;
                    }
                }
            }

            let is_threatened_to = to_threat == 1;

            if self.gives_check(board, m) && !is_threatened_to {
                score += 2000;
            }

            if from_threat == 1 && !is_threatened_to {
                score += 2000;
            }

            if is_threatened_to {
                score -= if pt != PieceType::Pawn { 2500 } else { 1000 };
            }

            self.scores[i].write(score);
        }
    }

    #[inline(always)]
    fn is_pseudo_legal(&self, board: &Board, m: Move) -> bool {
        m.move_type() != MoveType::EnPassant
            && board.piece_on[m.to()] == Piece::None
            && board.is_pseudo_legal(m)
    }

    #[inline(always)]
    fn is_pseudo_legal_any(&self, board: &Board, m: Move) -> bool {
        board.is_pseudo_legal(m)
    }
}
