use std::mem::MaybeUninit;


use crate::bitboard::Bitboard;
use crate::board::{Board, noisy_victim_index};
use crate::eval::PIECE_VALUES;
use crate::movegen::{generate_noisy_pseudo, generate_quiet_pseudo};
use crate::search::{ContHist, MAX_PLY};
use crate::see::see;
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
    // filled lazily when scoring; the split keeps the best-score scan at a
    // 4-byte stride
    scores: [MaybeUninit<i32>; 256],
    hists: [MaybeUninit<i32>; 256],
    cur_idx: usize,
    // bad captures buffer
    bad_noisy: [MaybeUninit<(Move, i32)>; 64],
    bad_noisy_count: usize,
    pawn_threats: Option<Bitboard>,
    qsearch: bool,
    in_check: bool,
    checkers: Bitboard,
    pinned: Bitboard,
    checkers_ready: bool,
    pinned_ready: bool,
    // cached check squares
    check_squares: Option<[Bitboard; 6]>,
    // cached see >= 0 for last move
    pub last_see_ge_zero: bool,
    // true if last move came from good noisy
    pub last_good_noisy: bool,
    pub last_hist: i32,
    pub last_has_hist: bool,
    delta_threshold: i32,
    skip_quiets: bool,
}

impl MovePicker {
#[inline(always)]
    pub fn new(tt_move: Move, killers: [Move; 2], counter_move: Move, in_check: bool) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers,
            killer_idx: 0,
            counter_move,
            moves: MoveList::new(),
            scores: [MaybeUninit::uninit(); 256],
            hists: [MaybeUninit::uninit(); 256],
            cur_idx: 0,
            bad_noisy: [MaybeUninit::uninit(); 64],
            bad_noisy_count: 0,
            pawn_threats: None,
            qsearch: false,
            in_check,
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
            checkers_ready: false,
            pinned_ready: false,
            check_squares: None,
            last_see_ge_zero: false,
            last_good_noisy: false,
            last_hist: 0,
            last_has_hist: false,
            delta_threshold: i32::MIN,
            skip_quiets: false,
        }
    }

    #[inline(always)]
    pub fn init_in_place(
        slot: &mut MaybeUninit<Self>,
        tt_move: Move,
        killers: [Move; 2],
        counter_move: Move,
        in_check: bool,
    ) -> &mut Self {
        unsafe {
            let p = slot.as_mut_ptr();
            std::ptr::addr_of_mut!((*p).stage).write(if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy });
            std::ptr::addr_of_mut!((*p).tt_move).write(tt_move);
            std::ptr::addr_of_mut!((*p).killers).write(killers);
            std::ptr::addr_of_mut!((*p).killer_idx).write(0);
            std::ptr::addr_of_mut!((*p).counter_move).write(counter_move);
            std::ptr::addr_of_mut!((*p).moves.count).write(0);
            std::ptr::addr_of_mut!((*p).cur_idx).write(0);
            std::ptr::addr_of_mut!((*p).bad_noisy_count).write(0);
            std::ptr::addr_of_mut!((*p).pawn_threats).write(None);
            std::ptr::addr_of_mut!((*p).qsearch).write(false);
            std::ptr::addr_of_mut!((*p).in_check).write(in_check);
            std::ptr::addr_of_mut!((*p).checkers).write(Bitboard::EMPTY);
            std::ptr::addr_of_mut!((*p).pinned).write(Bitboard::EMPTY);
            std::ptr::addr_of_mut!((*p).checkers_ready).write(false);
            std::ptr::addr_of_mut!((*p).pinned_ready).write(false);
            std::ptr::addr_of_mut!((*p).check_squares).write(None);
            std::ptr::addr_of_mut!((*p).last_see_ge_zero).write(false);
            std::ptr::addr_of_mut!((*p).last_good_noisy).write(false);
            std::ptr::addr_of_mut!((*p).last_hist).write(0);
            std::ptr::addr_of_mut!((*p).last_has_hist).write(false);
            std::ptr::addr_of_mut!((*p).delta_threshold).write(i32::MIN);
            std::ptr::addr_of_mut!((*p).skip_quiets).write(false);
            slot.assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn init_qsearch_in_place(slot: &mut MaybeUninit<Self>, tt_move: Move, in_check: bool) -> &mut Self {
        unsafe {
            let p = slot.as_mut_ptr();
            std::ptr::addr_of_mut!((*p).stage).write(if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy });
            std::ptr::addr_of_mut!((*p).tt_move).write(tt_move);
            std::ptr::addr_of_mut!((*p).killers).write([Move::NULL; 2]);
            std::ptr::addr_of_mut!((*p).killer_idx).write(0);
            std::ptr::addr_of_mut!((*p).counter_move).write(Move::NULL);
            std::ptr::addr_of_mut!((*p).moves.count).write(0);
            std::ptr::addr_of_mut!((*p).cur_idx).write(0);
            std::ptr::addr_of_mut!((*p).bad_noisy_count).write(0);
            std::ptr::addr_of_mut!((*p).pawn_threats).write(None);
            std::ptr::addr_of_mut!((*p).qsearch).write(true);
            std::ptr::addr_of_mut!((*p).in_check).write(in_check);
            std::ptr::addr_of_mut!((*p).checkers).write(Bitboard::EMPTY);
            std::ptr::addr_of_mut!((*p).pinned).write(Bitboard::EMPTY);
            std::ptr::addr_of_mut!((*p).checkers_ready).write(false);
            std::ptr::addr_of_mut!((*p).pinned_ready).write(false);
            std::ptr::addr_of_mut!((*p).check_squares).write(None);
            std::ptr::addr_of_mut!((*p).last_see_ge_zero).write(false);
            std::ptr::addr_of_mut!((*p).last_good_noisy).write(false);
            std::ptr::addr_of_mut!((*p).last_hist).write(0);
            std::ptr::addr_of_mut!((*p).last_has_hist).write(false);
            std::ptr::addr_of_mut!((*p).delta_threshold).write(i32::MIN);
            std::ptr::addr_of_mut!((*p).skip_quiets).write(false);
            slot.assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn gives_check(&mut self, board: &Board, m: Move) -> bool {
        if self.check_squares.is_none() {
            self.check_squares = Some(board.check_squares());
        }
        let squares = self.check_squares.as_ref().unwrap();
        board.gives_check_with(m, squares)
    }

    #[inline(always)]
    pub fn pawn_threats(&mut self, board: &Board) -> Bitboard {
        *self.pawn_threats.get_or_insert_with(|| board.opponent_pawn_threats())
    }

    #[inline(always)]
    pub fn new_qsearch(tt_move: Move, in_check: bool) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers: [Move::NULL; 2],
            killer_idx: 0,
            counter_move: Move::NULL,
            moves: MoveList::new(),
            scores: [MaybeUninit::uninit(); 256],
            hists: [MaybeUninit::uninit(); 256],
            cur_idx: 0,
            bad_noisy: [MaybeUninit::uninit(); 64],
            bad_noisy_count: 0,
            pawn_threats: None,
            qsearch: true,
            in_check,
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
            checkers_ready: false,
            pinned_ready: false,
            check_squares: None,
            last_see_ge_zero: false,
            last_good_noisy: false,
            last_hist: 0,
            last_has_hist: false,
            delta_threshold: i32::MIN,
            skip_quiets: false,
        }
    }

    #[inline(always)]
    fn ensure_checkers(&mut self, board: &Board) {
        if !self.checkers_ready {
            if self.in_check {
                self.checkers = board.checkers();
            } else {
                self.checkers = Bitboard::EMPTY;
            }
            self.checkers_ready = true;
        }
    }

    #[inline(always)]
    fn ensure_pinned(&mut self, board: &Board) {
        if !self.pinned_ready {
            self.pinned = board.pinned_pieces(board.side_to_move);
            self.pinned_ready = true;
        }
    }

    #[inline(always)]
    fn is_legal(&mut self, board: &Board, m: Move) -> bool {
        self.ensure_checkers(board);
        let from = m.from();
        let piece = board.piece_on[from];
        // king moves are checked directly and en passant has its own occupancy
        // check inside is_legal_with, so only regular moves can be pinned
        if piece.piece_type() == PieceType::King || m.move_type() == MoveType::EnPassant {
            return board.is_legal_with(m, self.checkers, Bitboard::EMPTY);
        }
        // a pin can only matter when the piece is on a line with our king
        if !self.pinned_ready && Self::aligned_with_king(from, board.king_square(board.side_to_move)) {
            self.ensure_pinned(board);
        }
        board.is_legal_with(m, self.checkers, self.pinned)
    }

    #[inline(always)]
    fn aligned_with_king(from: crate::types::Square, ksq: crate::types::Square) -> bool {
        from.rank() == ksq.rank()
            || from.file() == ksq.file()
            || (from.rank() as i32 - ksq.rank() as i32).abs() == (from.file() as i32 - ksq.file() as i32).abs()
    }

    #[inline(always)]
    fn is_legal_tt(&mut self, board: &Board, m: Move) -> bool {
        let from = m.from();
        let piece = board.piece_on[from];
        if piece == Piece::None {
            return false;
        }
        if piece.piece_type() == PieceType::King || m.move_type() == MoveType::EnPassant {
            self.ensure_checkers(board);
            self.ensure_pinned(board);
            return board.is_legal_with(m, self.checkers, self.pinned);
        }

        self.ensure_checkers(board);
        let checkers = self.checkers;
        if checkers.count() > 1 {
            return false;
        }
        if self.pinned_ready {
            return board.is_legal_with(m, checkers, self.pinned);
        }
        let ksq = board.king_square(board.side_to_move);
        if Self::aligned_with_king(from, ksq) {
            self.ensure_pinned(board);
            return board.is_legal_with(m, self.checkers, self.pinned);
        }
        board.is_legal_with(m, checkers, Bitboard::EMPTY)
    }

    #[inline(always)]
    pub fn set_delta_threshold(&mut self, threshold: i32) {
        self.delta_threshold = threshold;
    }

    #[inline(always)]
    pub fn skip_quiets(&mut self) {
        self.skip_quiets = true;
    }
    pub fn next(
        &mut self,
        board: &Board,
        history: &crate::search::MainHistory,
        pawn_history: &[[[i16; 64]; 12]; 512],
        conthist: &ContHist,
        noisy_history: &[[[[i16; 2]; 6]; 64]; 12],
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
    ) -> Option<Move> {
        self.last_good_noisy = false;
        self.last_has_hist = false;
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateNoisy;
                    let m = self.tt_move;
                    if m != Move::NULL {
                        let is_qsearch_ok = if self.qsearch {
                            if m.move_type() == MoveType::Promotion {
                                m.promo_type() == PieceType::Queen
                            } else {
                                board.piece_on[m.to()] != Piece::None || m.move_type() == MoveType::EnPassant
                            }
                        } else {
                            true
                        };

                        if is_qsearch_ok
                            && self.is_pseudo_legal_any(board, m)
                            && self.is_legal_tt(board, m)
                            && (!self.qsearch || see(board, m, 0))
                        {
                            // assume good see for tt move
                            self.last_see_ge_zero = true;
                            return Some(m);
                        }
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
                        let (m, hist) = self.pick_best();
                        if m == self.tt_move {
                            continue;
                        }

                        // skip underpromotions in qsearch
                        if self.qsearch && m.move_type() == MoveType::Promotion && m.promo_type() != PieceType::Queen {
                            continue;
                        }

                        if self.qsearch && m.move_type() != MoveType::Promotion {
                            let cap_pt = match m.move_type() {
                                MoveType::EnPassant => PieceType::Pawn,
                                _ => board.piece_on[m.to()].piece_type(),
                            };
                            if cap_pt == PieceType::None || crate::eval::PIECE_VALUES[cap_pt as usize] < self.delta_threshold {
                                continue;
                            }
                        }

                        // split captures at see 0
                        if see(board, m, 0) {
                            if self.is_legal(board, m) {
                                self.last_see_ge_zero = true;
                                self.last_good_noisy = true;
                                self.last_hist = hist;
                                self.last_has_hist = true;
                                return Some(m);
                            }
                        } else if !self.qsearch && self.bad_noisy_count < self.bad_noisy.len() {
                            self.bad_noisy[self.bad_noisy_count].write((m, hist));
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
                            && self.is_legal_tt(board, k)
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
                        && self.is_legal_tt(board, cm)
                    {
                        self.last_see_ge_zero = true;
                        return Some(cm);
                    }
                }
                Stage::GenerateQuiets => {
                    self.moves.count = 0;
                    generate_quiet_pseudo(board, &mut self.moves);
                    self.score_quiets(board, history, pawn_history, conthist, ply, played_pieces, played_moves);
                    self.cur_idx = 0;
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    if self.skip_quiets {
                        while self.cur_idx < self.moves.count {
                            let m = self.moves.get(self.cur_idx);
                            let hist = unsafe { self.hists[self.cur_idx].assume_init() };
                            self.cur_idx += 1;
                            if m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move {
                                continue;
                            }
                            if self.gives_check(board, m) && self.is_legal(board, m) {
                                self.last_see_ge_zero = true;
                                self.last_hist = hist;
                                self.last_has_hist = true;
                                return Some(m);
                            }
                        }
                        self.cur_idx = 0;
                        self.stage = Stage::BadNoisy;
                        continue;
                    }
                    while self.cur_idx < self.moves.count {
                        let (m, hist) = self.pick_best();
                        if m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move {
                            continue;
                        }
                        if self.is_legal(board, m) {
                            self.last_see_ge_zero = true;
                            self.last_hist = hist;
                            self.last_has_hist = true;
                            return Some(m);
                        }
                    }
                    self.cur_idx = 0;
                    self.stage = Stage::BadNoisy;
                }
                Stage::BadNoisy => {
                    while self.cur_idx < self.bad_noisy_count {
                        let (m, hist) = unsafe { self.bad_noisy[self.cur_idx].assume_init() };
                        self.cur_idx += 1;
                        if m != self.tt_move && self.is_legal(board, m) {
                            self.last_see_ge_zero = false;
                            self.last_hist = hist;
                            self.last_has_hist = true;
                            return Some(m);
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    #[inline(always)]
    fn pick_best(&mut self) -> (Move, i32) {
        let mut best_score = unsafe { self.scores[self.cur_idx].assume_init() };
        let mut best_idx = self.cur_idx;

        let mut i = self.cur_idx + 1;
        while i < self.moves.count {
            let s = unsafe { self.scores[i].assume_init() };
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
            i += 1;
        }

        if best_idx != self.cur_idx {
            self.scores.swap(self.cur_idx, best_idx);
            self.hists.swap(self.cur_idx, best_idx);
            self.moves.swap(self.cur_idx, best_idx);
        }

        let m = self.moves.get(self.cur_idx);
        let hist = unsafe { self.hists[self.cur_idx].assume_init() };
        self.cur_idx += 1;
        (m, hist)
    }

    fn score_noisy(&mut self, board: &Board, noisy_history: &[[[[i16; 2]; 6]; 64]; 12]) {
        if self.moves.count == 0 {
            return;
        }

        // qsearch discards captures below the delta threshold later anyway, so
        // drop them before scoring; the compaction keeps the original order.
        // underpromotions are never searched in qsearch either.
        if self.qsearch && self.delta_threshold > i32::MIN {
            let threshold = self.delta_threshold;
            let mut write = 0usize;
            for read in 0..self.moves.count {
                let m = self.moves.get(read);
                let keep = if m.move_type() == MoveType::Promotion {
                    m.promo_type() == PieceType::Queen
                } else {
                    let cap_pt = match m.move_type() {
                        MoveType::EnPassant => PieceType::Pawn,
                        _ => board.piece_on[m.to()].piece_type(),
                    };
                    cap_pt != PieceType::None && crate::eval::PIECE_VALUES[cap_pt as usize] >= threshold
                };
                if keep {
                    if write != read {
                        self.moves.set(write, m);
                    }
                    write += 1;
                }
            }
            self.moves.count = write;
            if write == 0 {
                return;
            }
        }

        // a single capture needs no ordering and qsearch never reads the
        // history value, so skip the expensive scoring entirely
        if self.qsearch && self.moves.count == 1 {
            self.scores[0].write(0);
            self.hists[0].write(0);
            return;
        }

        let pawn_threats = self.pawn_threats(board);

        for i in 0..self.moves.count {
            let m = self.moves.get(i);
            let attacker = board.piece_on[m.from()];
            let captured = board.piece_on[m.to()];
            let victim_pt = match m.move_type() {
                MoveType::EnPassant => PieceType::Pawn,
                MoveType::Promotion => m.promo_type(),
                _ => captured.piece_type(),
            };
            let hist_victim = noisy_victim_index(board, m);

            let to_threatened = pawn_threats.contains(m.to()) as usize;

            let attacker_idx = attacker as usize;
            let victim_idx = victim_pt as usize;

            // bounds check before indexing
            if attacker_idx < 12 && victim_idx < 6 {
                let mvv_lva = PIECE_VALUES[victim_idx] * 12 - PIECE_VALUES[attacker.piece_type() as usize];
                let hist = noisy_history[attacker_idx][m.to() as usize][hist_victim][to_threatened] as i32;
                let mut score = mvv_lva + hist / 4;

                if m.move_type() == MoveType::Promotion && m.promo_type() == PieceType::Queen {
                    score += 3000;
                }
                if self.gives_check(board, m) {
                    score += 2000;
                }

                self.scores[i].write(score);
                self.hists[i].write(hist);
            } else {
                self.scores[i].write(0);
                self.hists[i].write(0);
            }
        }
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        history: &crate::search::MainHistory,
        pawn_history: &[[[i16; 64]; 12]; 512],
        conthist: &ContHist,
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
    ) {
        if self.moves.count == 0 {
            return;
        }

        // the quiet stages skip the tt move without scoring it, so remove it
        // first; compaction preserves the order
        {
            let tt = self.tt_move;
            let mut write = 0usize;
            for read in 0..self.moves.count {
                let m = self.moves.get(read);
                if m == tt {
                    continue;
                }
                if write != read {
                    self.moves.set(write, m);
                }
                write += 1;
            }
            self.moves.count = write;
            if write == 0 {
                return;
            }
        }

        let us = board.side_to_move;
        let pawn_threats = self.pawn_threats(board);
        let p_idx = (board.pawn_hash as usize) & 511;
        let offsets = [1usize, 2, 4, 6];
        let ply_idx = ply as usize;

        let mut cont_slices: [&[[i16; 64]; 12]; 4] = [&conthist[0][0][0]; 4];
        let mut cont_len = 0usize;
        for (layer, &offset) in offsets.iter().enumerate() {
            if ply_idx >= offset {
                let prev_idx = ply_idx - offset;
                let prev_piece = played_pieces[prev_idx];
                let prev_move = played_moves[prev_idx];
                if prev_piece != Piece::None && prev_move != Move::NULL {
                    cont_slices[cont_len] = &conthist[layer][prev_piece as usize][prev_move.to() as usize];
                    cont_len += 1;
                }
            }
        }

        let check_squares = *self.check_squares.get_or_insert_with(|| board.check_squares());

        for i in 0..self.moves.count {
            let m = self.moves.get(i);
            let from = m.from();
            let to = m.to();
            let piece = board.piece_on[from];
            let pt = piece.piece_type();
            let from_threat = pawn_threats.contains(from) as usize;
            let to_threat = pawn_threats.contains(to) as usize;

            debug_assert!(from.is_valid() && to.is_valid() && (piece as usize) < 12);
            let mut hist = unsafe {
                *history
                    .get_unchecked(us as usize)
                    .get_unchecked(from_threat)
                    .get_unchecked(to_threat)
                    .get_unchecked(from as usize)
                    .get_unchecked(to as usize) as i32
                    + *pawn_history.get_unchecked(p_idx).get_unchecked(piece as usize).get_unchecked(to as usize) as i32
            };

            for layer in 0..cont_len {
                hist += unsafe { *cont_slices[layer].get_unchecked(piece as usize).get_unchecked(to as usize) } as i32;
            }

            let mut score = hist;
            let is_threatened_to = to_threat == 1;

            // quiet moves are never promotions, so only castling needs the
            // slower direct-check path
            let gives_check = if m.move_type() == MoveType::Castling {
                board.gives_direct_check(m)
            } else {
                check_squares[pt as usize].contains(to)
            };
            if !is_threatened_to && gives_check {
                score += 2000;
            }

            if from_threat == 1 && !is_threatened_to {
                score += 2000;
            }

            if is_threatened_to {
                score -= if pt != PieceType::Pawn { 2500 } else { 1000 };
            }

            self.scores[i].write(score);
            self.hists[i].write(hist);
        }
    }

    #[inline(always)]
    fn is_pseudo_legal(&self, board: &Board, m: Move) -> bool {
        m.move_type() != MoveType::EnPassant && board.piece_on[m.to()] == Piece::None && board.is_pseudo_legal(m)
    }

    #[inline(always)]
    fn is_pseudo_legal_any(&self, board: &Board, m: Move) -> bool {
        board.is_pseudo_legal(m)
    }
}
