use crate::alloc::HugeBox;
use crate::attacks::{between, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::nnue::{Accumulator, FeatureDelta, ThreatAccumulator, ThreatList};
use crate::search::MAX_PLY;
use crate::types::{COLOR_NB, Color, Move, MoveType, PIECE_NB, Piece, PieceType, SQUARE_NB, Square};
use crate::zobrist::{castling_key, ep_key, fiftymove_key, piece_key, side_key};

pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const CASTLING_RIGHTS_MASK: [u8; SQUARE_NB] = {
    let mut table = [0xFFu8; SQUARE_NB];
    table[Square::A1 as usize] = !Board::CASTLE_WQ;
    table[Square::E1 as usize] = !(Board::CASTLE_WK | Board::CASTLE_WQ);
    table[Square::H1 as usize] = !Board::CASTLE_WK;
    table[Square::A8 as usize] = !Board::CASTLE_BQ;
    table[Square::E8 as usize] = !(Board::CASTLE_BK | Board::CASTLE_BQ);
    table[Square::H8 as usize] = !Board::CASTLE_BK;
    table
};

#[derive(Copy, Clone)]
pub struct UndoState {
    pub castling_rights: u8,
    pub ep_square: Square,
    pub halfmove_clock: u8,
    pub captured: Piece,
    pub hash: u64,
    pub pawn_hash: u64,
    pub non_pawn_hash: [u64; 2],
}

// stack size for per-ply frames. MAX_PLY + 4 for safety margin
const FRAME_STACK_SIZE: usize = MAX_PLY + 4;

// per-ply accumulator state for lazy evaluation
#[repr(C, align(64))]
#[derive(Clone)]
struct AccumulatorFrame {
    pub accumulator: Accumulator,
    pub delta: FeatureDelta,
    pub dirty: [bool; 2],
    pub refresh: [bool; 2],
    pub ksq: [Square; 2],
}
// per-ply threat state: accumulators, lists, and mirror state for one ply
#[repr(C, align(64))]
#[derive(Clone)]
struct ThreatFrame {
    pub threat: ThreatAccumulator,
    // index of frame with actual threat data
    pub threat_base: usize,
    pub threat_king_mirrored: [bool; 2],
    pub dirty: bool,
    pub last_move: Move,
    pub last_captured: Piece,
}
// the rook squares of a castle, keyed by the king's destination
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to {
        Square::G1 => (Square::H1, Square::F1),
        Square::C1 => (Square::A1, Square::D1),
        Square::G8 => (Square::H8, Square::F8),
        Square::C8 => (Square::A8, Square::D8),
        _ => unreachable!(),
    }
}

pub struct Board {
    pub pieces: [Bitboard; PIECE_NB],
    pub occupied_co: [Bitboard; COLOR_NB],
    pub occupied: Bitboard,
    pub piece_on: [Piece; SQUARE_NB],
    pub side_to_move: Color,
    pub castling_rights: u8,
    pub ep_square: Square,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    pub hash: u64,
    pub tt_hash: u64,
    pub pawn_hash: u64,
    pub non_pawn_hash: [u64; 2],
    pub history: [u64; 1024],
    pub hist_len: usize,
    // root index in history (2-fold after root, 3-fold before)
    pub root_ply: usize,
    // per-ply accumulator frames for lazy evaluation
    acc_frames: HugeBox<[AccumulatorFrame; FRAME_STACK_SIZE]>,
    // per-ply threat frames. Indexed by acc_depth (same depth as accumulator)
    threat_frames: HugeBox<[ThreatFrame; FRAME_STACK_SIZE]>,
    // current depth in the threat/accumulator stacks. Incremented on make_move,
    // decremented on undo_move/null_move.
    acc_depth: usize,
}
impl Clone for Board {
    fn clone(&self) -> Self {
        let mut new_frames: HugeBox<[AccumulatorFrame; FRAME_STACK_SIZE]> = HugeBox::zeroed();
        let mut new_threat_frames: HugeBox<[ThreatFrame; FRAME_STACK_SIZE]> = HugeBox::zeroed();
        for i in 0..=self.acc_depth {
            new_frames[i] = self.acc_frames[i].clone();
            new_threat_frames[i] = self.threat_frames[i].clone();
        }
        Self {
            pieces: self.pieces,
            occupied_co: self.occupied_co,
            occupied: self.occupied,
            piece_on: self.piece_on,
            side_to_move: self.side_to_move,
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            hash: self.hash,
            tt_hash: self.tt_hash,
            pawn_hash: self.pawn_hash,
            non_pawn_hash: self.non_pawn_hash,
            history: self.history,
            hist_len: self.hist_len,
            root_ply: self.root_ply,
            acc_frames: new_frames,
            threat_frames: new_threat_frames,
            acc_depth: self.acc_depth,
        }
    }
}

impl Board {
    pub const CASTLE_WK: u8 = 1 << 0;
    pub const CASTLE_WQ: u8 = 1 << 1;
    pub const CASTLE_BK: u8 = 1 << 2;
    pub const CASTLE_BQ: u8 = 1 << 3;

    pub fn new() -> Self {
        Self {
            pieces: [Bitboard::EMPTY; PIECE_NB],
            occupied_co: [Bitboard::EMPTY; COLOR_NB],
            occupied: Bitboard::EMPTY,
            piece_on: [Piece::None; SQUARE_NB],
            side_to_move: Color::White,
            castling_rights: 0,
            ep_square: Square::None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            tt_hash: 0,
            pawn_hash: 0,
            non_pawn_hash: [0; 2],
            history: [0; 1024],
            hist_len: 0,
            root_ply: 0,
            acc_frames: HugeBox::zeroed(),
            threat_frames: HugeBox::zeroed(),
            acc_depth: 0,
        }
    }

    // returns the current threat frame for the current ply
    #[inline(always)]
    fn threat_frame(&mut self) -> &mut ThreatFrame {
        &mut self.threat_frames[self.acc_depth]
    }

    // returns the current threat frame for the current ply (immutable)
    #[inline(always)]
    fn threat_frame_immut(&self) -> &ThreatFrame {
        &self.threat_frames[self.acc_depth]
    }

    // returns the current accumulator frame for the current ply
    #[inline(always)]
    fn acc_frame(&mut self) -> &mut AccumulatorFrame {
        &mut self.acc_frames[self.acc_depth]
    }

    // returns the current accumulator frame for the current ply (immutable)
    #[inline(always)]
    fn acc_frame_immut(&self) -> &AccumulatorFrame {
        &self.acc_frames[self.acc_depth]
    }

    // reset accumulator stack to root
    pub fn reset_accumulator_stack(&mut self) {
        self.acc_depth = 0;
        self.acc_frames[0].dirty = [true, true];
        self.acc_frames[0].refresh = [false, false];
        self.threat_frames[0].dirty = true;
    }

    // returns the current accumulator for the current ply, computing it if dirty
    #[inline(always)]
    pub fn accumulator(&mut self) -> &mut Accumulator {
        self.ensure_accumulator_ready();
        &mut self.acc_frame().accumulator
    }

    // returns the current accumulator for the current ply (immutable)
    #[inline(always)]
    pub fn accumulator_immut(&self) -> &Accumulator {
        &self.acc_frame_immut().accumulator
    }

    pub fn refresh_accumulator(&mut self) {
        let w_ksq = self.king_square(Color::White);
        let b_ksq = self.king_square(Color::Black);
        let piece_on = self.piece_on;
        let occ = self.occupied;

        let frame = self.acc_frame();
        frame.accumulator.refresh_side_occ(&piece_on, occ, w_ksq, Color::White);
        frame.accumulator.refresh_side_occ(&piece_on, occ, b_ksq, Color::Black);
        frame.dirty = [false, false];
        frame.refresh = [false, false];
        frame.ksq = [w_ksq, b_ksq];

        let d = self.acc_depth;
        let threat_frame = self.threat_frame();
        threat_frame.threat = ThreatAccumulator::new();
        threat_frame.threat_base = d;
        threat_frame.dirty = true;
        threat_frame.threat_king_mirrored = [w_ksq.file() >= 4, b_ksq.file() >= 4];
        threat_frame.last_move = Move::NULL;
        threat_frame.last_captured = Piece::None;
        self.refresh_threats();
    }

    #[inline(always)]
    pub fn ensure_accumulator_ready(&mut self) {
        let d = self.acc_depth;
        if !self.acc_frames[d].dirty[0] && !self.acc_frames[d].dirty[1] {
            return;
        }

        for side in 0..2 {
            if !self.acc_frames[d].dirty[side] {
                continue;
            }

            let mut anc = d;
            while anc > 0 && self.acc_frames[anc].dirty[side] && !self.acc_frames[anc].refresh[side] {
                anc -= 1;
            }

            if self.acc_frames[anc].dirty[side] {
                let color = Color::from_index(side);
                let ksq = self.king_square(color);
                let piece_on = self.piece_on;
                let occ = self.occupied;
                self.acc_frames[d].accumulator.refresh_side_occ(&piece_on, occ, ksq, color);
                self.acc_frames[d].dirty[side] = false;
            } else {
                for k in (anc + 1)..=d {
                    let delta = self.acc_frames[k].delta;
                    let (left, right) = self.acc_frames.split_at_mut(k);
                    right[0].accumulator.apply_delta_from_side(&left[k - 1].accumulator, side, &delta);
                    right[0].dirty[side] = false;
                }
            }
        }
    }

    // full rebuild of threat accumulator
    pub fn refresh_threats(&mut self) {
        let Some(ml) = crate::nnue::network().multilayer.as_deref() else {
            // compute king mirrored state before getting mutable frame
            let w_ksq = self.king_square(Color::White);
            let b_ksq = self.king_square(Color::Black);
            let threat_king_mirrored = [w_ksq.file() >= 4, b_ksq.file() >= 4];
            let d = self.acc_depth;
            let frame = self.threat_frame();
            frame.threat = ThreatAccumulator::new();
            frame.threat_base = d;
            frame.dirty = false;
            frame.threat_king_mirrored = threat_king_mirrored;
            frame.last_move = Move::NULL;
            frame.last_captured = Piece::None;
            return;
        };
        let Some(weights) = ml.threat_weights.as_deref() else {
            // compute king mirrored state before getting mutable frame
            let w_ksq = self.king_square(Color::White);
            let b_ksq = self.king_square(Color::Black);
            let threat_king_mirrored = [w_ksq.file() >= 4, b_ksq.file() >= 4];
            let frame = self.threat_frame();
            frame.dirty = false;
            frame.threat_king_mirrored = threat_king_mirrored;
            frame.last_move = Move::NULL;
            frame.last_captured = Piece::None;
            return;
        };

        // collect threat lists for both sides before mutably borrowing the frame
        let mut lists = [ThreatList::new(), ThreatList::new()];
        let mut mirrored = [false; 2];
        for side in 0..2 {
            let ksq = self.king_square(Color::from_index(side));
            mirrored[side] = ksq.file() >= 4;
        }
        crate::nnue::collect_threats_both_unsorted(self, [true, true], &mut lists);

        let d = self.acc_depth;
        let frame = self.threat_frame();
        frame.threat_base = d;

        for side in 0..2 {
            frame.threat.refresh_side(side, &lists[side], weights);
            frame.threat_king_mirrored[side] = mirrored[side];
        }
        frame.dirty = false;
        frame.last_move = Move::NULL;
        frame.last_captured = Piece::None;
    }

    #[inline(always)]
    fn update_threat_frame_step(&mut self, src_idx: usize, dst_idx: usize, m: Move, captured: Piece, weights: &crate::nnue::ThreatWeights) {
        self.threat_frames[dst_idx].threat_base = dst_idx;
        let src_base = self.threat_frames[src_idx].threat_base;
        let mirrored = [self.king_square(Color::White).file() >= 4, self.king_square(Color::Black).file() >= 4];
        let prev_mirrored = self.threat_frames[src_idx].threat_king_mirrored;
        let need_inc = [mirrored[0] == prev_mirrored[0], mirrored[1] == prev_mirrored[1]];

        let mut subs = [ThreatList::new(), ThreatList::new()];
        let mut adds = [ThreatList::new(), ThreatList::new()];

        if need_inc[0] || need_inc[1] {
            let changed = self.move_changed_squares(m);
            let occ_after = self.occupied;
            let occ_before = self.occupancy_before(m, captured);

            let mut scan = crate::nnue::SharedThreatScan::new(self);
            crate::nnue::append_changed_square_threats_both(self, &scan, changed, need_inc, &mut adds);
            crate::nnue::collect_xray_threat_diffs_both(self, &scan, changed, occ_before, occ_after, need_inc, &mut subs, &mut adds);
            self.toggle_pieces_backward(m, captured);
            scan.refresh_pieces(self);
            crate::nnue::append_changed_square_threats_both(self, &scan, changed, need_inc, &mut subs);
            self.toggle_pieces_forward(m, captured);
        }

        // prefetch removed for benchmark
        for side in 0..2 {
            if need_inc[side] {
                let (left, right) = self.threat_frames.split_at_mut(dst_idx);
                right[0]
                    .threat
                    .apply_lists_from(&left[src_base].threat, side, weights, subs[side].as_slice(), adds[side].as_slice());
            } else {
                let mut full = ThreatList::new();
                crate::nnue::collect_threats(self, Color::from_index(side), &mut full);
                self.threat_frames[dst_idx].threat.refresh_side(side, &full, weights);
            }
            self.threat_frames[dst_idx].threat_king_mirrored[side] = mirrored[side];
        }
        self.threat_frames[dst_idx].dirty = false;
    }

    #[inline(always)]
    pub fn ensure_threats_ready(&mut self) {
        let d = self.acc_depth;
        if !self.threat_frames[d].dirty {
            return;
        }

        let Some(ml) = crate::nnue::network().multilayer.as_deref() else {
            let w_ksq = self.king_square(Color::White);
            let b_ksq = self.king_square(Color::Black);
            let frame = &mut self.threat_frames[d];
            frame.threat = ThreatAccumulator::new();
            frame.threat_base = d;
            frame.dirty = false;
            frame.threat_king_mirrored = [w_ksq.file() >= 4, b_ksq.file() >= 4];
            return;
        };
        let Some(weights) = ml.threat_weights.as_deref() else {
            let w_ksq = self.king_square(Color::White);
            let b_ksq = self.king_square(Color::Black);
            let frame = &mut self.threat_frames[d];
            frame.threat_base = d;
            frame.dirty = false;
            frame.threat_king_mirrored = [w_ksq.file() >= 4, b_ksq.file() >= 4];
            return;
        };

        let mut anc = d;
        let mut non_null_count = 0usize;
        while anc > 0 && self.threat_frames[anc].dirty {
            if self.threat_frames[anc].last_move != Move::NULL {
                non_null_count += 1;
            }
            anc -= 1;
        }

        if self.threat_frames[anc].dirty || non_null_count >= 3 {
            self.refresh_threats();
            return;
        }

        if non_null_count == 0 {
            let prev_base = self.threat_frames[anc].threat_base;
            let prev_mirrored = self.threat_frames[anc].threat_king_mirrored;
            let frame = &mut self.threat_frames[d];
            frame.threat_base = prev_base;
            frame.threat_king_mirrored = prev_mirrored;
            frame.dirty = false;
            return;
        }

        if d == anc + 1 {
            let m = self.threat_frames[d].last_move;
            let cap = self.threat_frames[d].last_captured;
            self.update_threat_frame_step(anc, d, m, cap, weights);
            return;
        }

        for k in ((anc + 1)..=d).rev() {
            let m = self.threat_frames[k].last_move;
            if m != Move::NULL {
                let cap = self.threat_frames[k].last_captured;
                self.toggle_pieces_backward(m, cap);
            }
        }

        for k in (anc + 1)..=d {
            let m = self.threat_frames[k].last_move;
            if m == Move::NULL {
                let (left, right) = self.threat_frames.split_at_mut(k);
                right[0].threat_base = left[k - 1].threat_base;
                right[0].threat_king_mirrored = left[k - 1].threat_king_mirrored;
                right[0].dirty = false;
            } else {
                let cap = self.threat_frames[k].last_captured;
                self.toggle_pieces_forward(m, cap);
                self.update_threat_frame_step(k - 1, k, m, cap, weights);
            }
        }
    }

    // returns the current threat accumulator for the current ply
    #[inline(always)]
    pub fn threat(&mut self) -> &mut ThreatAccumulator {
        self.ensure_threats_ready();
        let d = self.acc_depth;
        let base = self.threat_frames[d].threat_base;
        &mut self.threat_frames[base].threat
    }

    // returns the current threat accumulator for the current ply (immutable)
    #[inline(always)]
    pub fn threat_immut(&self) -> &ThreatAccumulator {
        let d = self.acc_depth;
        let base = self.threat_frames[d].threat_base;
        &self.threat_frames[base].threat
    }

    // squares changed by move
    pub(crate) fn move_changed_squares(&self, m: Move) -> Bitboard {
        let mut changed = Bitboard::from_square(m.from()) | Bitboard::from_square(m.to());
        match m.move_type() {
            MoveType::EnPassant => {
                changed |= Bitboard::from_square(Square::from_coords(m.to().file(), m.from().rank()));
            }
            MoveType::Castling => {
                let (rook_from, rook_to) = castling_rook_squares(m.to());
                changed |= Bitboard::from_square(rook_from) | Bitboard::from_square(rook_to);
            }
            _ => {}
        }
        changed
    }

    #[inline(always)]
    pub(crate) fn occupancy_before(&self, m: Move, captured: Piece) -> Bitboard {
        let mut occ = self.occupied;
        if captured == Piece::None {
            occ.clear(m.to());
        }
        occ.set(m.from());
        match m.move_type() {
            MoveType::EnPassant => {
                occ.set(Square::from_coords(m.to().file(), m.from().rank()));
            }
            MoveType::Castling => {
                let (rook_from, rook_to) = castling_rook_squares(m.to());
                occ.clear(rook_to);
                occ.set(rook_from);
            }
            _ => {}
        }
        occ
    }

    #[inline(always)]
    fn put_piece_no_hash(&mut self, piece: Piece, sq: Square) {
        self.pieces[piece].set(sq);
        self.occupied_co[piece.color()].set(sq);
        self.occupied.set(sq);
        self.piece_on[sq] = piece;
    }

    #[inline(always)]
    fn remove_piece_no_hash(&mut self, sq: Square) -> Piece {
        let piece = self.piece_on[sq];
        if piece != Piece::None {
            self.pieces[piece].clear(sq);
            self.occupied_co[piece.color()].clear(sq);
            self.occupied.clear(sq);
            self.piece_on[sq] = Piece::None;
        }
        piece
    }

    #[inline(always)]
    fn toggle_pieces_backward(&mut self, m: Move, captured: Piece) {
        let from = m.from();
        let to = m.to();
        match m.move_type() {
            MoveType::Normal => {
                let moved = self.remove_piece_no_hash(to);
                self.put_piece_no_hash(moved, from);
                if captured != Piece::None {
                    self.put_piece_no_hash(captured, to);
                }
            }
            MoveType::Castling => {
                let moved = self.remove_piece_no_hash(to);
                self.put_piece_no_hash(moved, from);
                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self.remove_piece_no_hash(rook_to);
                self.put_piece_no_hash(rook, rook_from);
            }
            MoveType::EnPassant => {
                let moved = self.remove_piece_no_hash(to);
                self.put_piece_no_hash(moved, from);
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.put_piece_no_hash(Piece::new(!moved.color(), PieceType::Pawn), cap_sq);
            }
            MoveType::Promotion => {
                let promo = self.remove_piece_no_hash(to);
                self.put_piece_no_hash(Piece::new(promo.color(), PieceType::Pawn), from);
                if captured != Piece::None {
                    self.put_piece_no_hash(captured, to);
                }
            }
        }
    }

    #[inline(always)]
    fn toggle_pieces_forward(&mut self, m: Move, captured: Piece) {
        let from = m.from();
        let to = m.to();
        match m.move_type() {
            MoveType::Normal => {
                let moved = self.remove_piece_no_hash(from);
                if captured != Piece::None {
                    self.remove_piece_no_hash(to);
                }
                self.put_piece_no_hash(moved, to);
            }
            MoveType::Castling => {
                let moved = self.remove_piece_no_hash(from);
                self.put_piece_no_hash(moved, to);
                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self.remove_piece_no_hash(rook_from);
                self.put_piece_no_hash(rook, rook_to);
            }
            MoveType::EnPassant => {
                let moved = self.remove_piece_no_hash(from);
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.remove_piece_no_hash(cap_sq);
                self.put_piece_no_hash(moved, to);
            }
            MoveType::Promotion => {
                let moved = self.remove_piece_no_hash(from);
                if captured != Piece::None {
                    self.remove_piece_no_hash(to);
                }
                self.put_piece_no_hash(Piece::new(moved.color(), m.promo_type()), to);
            }
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut board = Self::new();
        board.acc_depth = 0;
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty FEN string".to_string());
        }

        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("Invalid number of ranks in FEN".to_string());
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - (rank_idx as u8);
            let mut file: u8 = 0;

            for ch in rank_str.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    file = file.saturating_add(digit as u8);
                    if file > 8 {
                        return Err("Too many squares in a FEN rank".to_string());
                    }
                } else {
                    if file >= 8 {
                        return Err("Invalid file index in FEN".to_string());
                    }
                    let sq = Square::from_coords(file, rank);
                    let piece = match ch {
                        'P' => Piece::WhitePawn,
                        'N' => Piece::WhiteKnight,
                        'B' => Piece::WhiteBishop,
                        'R' => Piece::WhiteRook,
                        'Q' => Piece::WhiteQueen,
                        'K' => Piece::WhiteKing,
                        'p' => Piece::BlackPawn,
                        'n' => Piece::BlackKnight,
                        'b' => Piece::BlackBishop,
                        'r' => Piece::BlackRook,
                        'q' => Piece::BlackQueen,
                        'k' => Piece::BlackKing,
                        _ => return Err(format!("Invalid piece character: {}", ch)),
                    };
                    if piece.piece_type() == PieceType::Pawn && (rank == 0 || rank == 7) {
                        return Err("Pawn on invalid rank in FEN".to_string());
                    }
                    board.put_piece(piece, sq);
                    file += 1;
                }
            }
        }

        if parts.len() > 1 {
            board.side_to_move = match parts[1] {
                "w" => Color::White,
                "b" => Color::Black,
                _ => return Err("Invalid side to move in FEN".to_string()),
            };
        }

        if parts.len() > 2 {
            board.castling_rights = 0;
            for ch in parts[2].chars() {
                match ch {
                    'K' => board.castling_rights |= Self::CASTLE_WK,
                    'Q' => board.castling_rights |= Self::CASTLE_WQ,
                    'k' => board.castling_rights |= Self::CASTLE_BK,
                    'q' => board.castling_rights |= Self::CASTLE_BQ,
                    '-' => break,
                    _ => return Err(format!("Invalid castling character: {}", ch)),
                }
            }
        }

        // drop castling rights if king/rook not on home square
        let mut rights = board.castling_rights;
        if board.piece_on[Square::E1] != Piece::WhiteKing {
            rights &= !(Self::CASTLE_WK | Self::CASTLE_WQ);
        }
        if board.piece_on[Square::H1] != Piece::WhiteRook {
            rights &= !Self::CASTLE_WK;
        }
        if board.piece_on[Square::A1] != Piece::WhiteRook {
            rights &= !Self::CASTLE_WQ;
        }
        if board.piece_on[Square::E8] != Piece::BlackKing {
            rights &= !(Self::CASTLE_BK | Self::CASTLE_BQ);
        }
        if board.piece_on[Square::H8] != Piece::BlackRook {
            rights &= !Self::CASTLE_BK;
        }
        if board.piece_on[Square::A8] != Piece::BlackRook {
            rights &= !Self::CASTLE_BQ;
        }
        board.castling_rights = rights;

        if parts.len() > 3 {
            board.ep_square = match parts[3] {
                "-" => Square::None,
                s if s.len() == 2 => Square::from_str(s).unwrap_or(Square::None),
                _ => Square::None,
            };

            if board.ep_square.is_valid() {
                let us = board.side_to_move;
                let them = !us;
                let cap_rank = match us {
                    Color::White => 4,
                    Color::Black => 3,
                };
                let cap_sq = Square::from_coords(board.ep_square.file(), cap_rank);
                let our_pawns = board.pieces[Piece::new(us, PieceType::Pawn)];
                let attackers = pawn_attacks(them, board.ep_square) & our_pawns;

                if board.piece_on[cap_sq] != Piece::new(them, PieceType::Pawn) || attackers.is_empty() {
                    board.ep_square = Square::None;
                }
            }
        }

        if parts.len() > 4 {
            // clamp halfmove clock to u8
            board.halfmove_clock = parts[4].parse::<u32>().unwrap_or(0).min(255) as u8;
        }

        if parts.len() > 5 {
            board.fullmove_number = parts[5].parse::<u16>().unwrap_or(1).max(1);
        }

        if board.pieces[Piece::WhiteKing].count() != 1 || board.pieces[Piece::BlackKing].count() != 1 {
            return Err("FEN must contain exactly one king per side".to_string());
        }

        // side not to move can't be in check
        if board.is_square_attacked(board.king_square(!board.side_to_move), board.side_to_move) {
            return Err("FEN has the side not to move in check".to_string());
        }

        board.hash = board.compute_hash();
        board.refresh_tt_hash();
        board.pawn_hash = board.compute_pawn_hash();
        board.non_pawn_hash = board.compute_non_pawn_hash();
        board.refresh_accumulator();
        Ok(board)
    }

    #[inline(always)]
    pub fn refresh_tt_hash(&mut self) {
        self.tt_hash = self.hash ^ fiftymove_key(self.halfmove_clock / 8);
    }

    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in 0..64 {
            let p = self.piece_on[sq];
            if p != Piece::None {
                h ^= piece_key(p, Square::new(sq as u8));
            }
        }
        if self.side_to_move == Color::Black {
            h ^= side_key();
        }
        h ^= castling_key(self.castling_rights);
        if self.ep_square.is_valid() {
            h ^= ep_key(self.ep_square.file());
        }
        h
    }

    #[inline(always)]
    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        self.pieces[piece].set(sq);
        self.occupied_co[piece.color()].set(sq);
        self.occupied.set(sq);
        self.piece_on[sq] = piece;
        self.hash ^= piece_key(piece, sq);
        if piece.piece_type() == PieceType::Pawn {
            self.pawn_hash ^= piece_key(piece, sq);
        } else {
            self.non_pawn_hash[piece.color() as usize] ^= piece_key(piece, sq);
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, sq: Square) -> Piece {
        let piece = self.piece_on[sq];
        if piece != Piece::None {
            self.pieces[piece].clear(sq);
            self.occupied_co[piece.color()].clear(sq);
            self.occupied.clear(sq);
            self.piece_on[sq] = Piece::None;
            self.hash ^= piece_key(piece, sq);
            if piece.piece_type() == PieceType::Pawn {
                self.pawn_hash ^= piece_key(piece, sq);
            } else {
                self.non_pawn_hash[piece.color() as usize] ^= piece_key(piece, sq);
            }
        }
        piece
    }

    pub fn compute_pawn_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in 0..64 {
            let p = self.piece_on[sq];
            if p != Piece::None && p.piece_type() == PieceType::Pawn {
                h ^= piece_key(p, Square::new(sq as u8));
            }
        }
        h
    }

    pub fn compute_non_pawn_hash(&self) -> [u64; 2] {
        let mut h = [0u64; 2];
        for sq in 0..64 {
            let p = self.piece_on[sq];
            if p != Piece::None && p.piece_type() != PieceType::Pawn {
                h[p.color() as usize] ^= piece_key(p, Square::new(sq as u8));
            }
        }
        h
    }

    #[inline(always)]
    pub fn king_square(&self, color: Color) -> Square {
        let piece = Piece::new(color, PieceType::King);
        self.pieces[piece].lsb()
    }

    #[inline(always)]
    pub fn is_square_attacked(&self, sq: Square, by_color: Color) -> bool {
        let opp_pawns = self.pieces[Piece::new(by_color, PieceType::Pawn)];
        if !(pawn_attacks(!by_color, sq) & opp_pawns).is_empty() {
            return true;
        }

        let opp_knights = self.pieces[Piece::new(by_color, PieceType::Knight)];
        if !(knight_attacks(sq) & opp_knights).is_empty() {
            return true;
        }

        let opp_bishops_queens = self.pieces[Piece::new(by_color, PieceType::Bishop)] | self.pieces[Piece::new(by_color, PieceType::Queen)];
        if !(bishop_attacks(sq, self.occupied) & opp_bishops_queens).is_empty() {
            return true;
        }

        let opp_rooks_queens = self.pieces[Piece::new(by_color, PieceType::Rook)] | self.pieces[Piece::new(by_color, PieceType::Queen)];
        if !(rook_attacks(sq, self.occupied) & opp_rooks_queens).is_empty() {
            return true;
        }

        let opp_king = self.pieces[Piece::new(by_color, PieceType::King)];
        !(king_attacks(sq) & opp_king).is_empty()
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        let ksq = self.king_square(self.side_to_move);
        self.is_square_attacked(ksq, !self.side_to_move)
    }

    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        let ksq = self.king_square(self.side_to_move);
        let them = !self.side_to_move;
        let pawns = self.pieces[Piece::new(them, PieceType::Pawn)];
        let knights = self.pieces[Piece::new(them, PieceType::Knight)];
        let bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)] | self.pieces[Piece::new(them, PieceType::Queen)];
        let rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)] | self.pieces[Piece::new(them, PieceType::Queen)];
        let king = self.pieces[Piece::new(them, PieceType::King)];
        (pawn_attacks(self.side_to_move, ksq) & pawns)
            | (knight_attacks(ksq) & knights)
            | (bishop_attacks(ksq, self.occupied) & bishops_queens)
            | (rook_attacks(ksq, self.occupied) & rooks_queens)
            | (king_attacks(ksq) & king)
    }

    pub fn pinned_pieces(&self, color: Color) -> Bitboard {
        let ksq = self.king_square(color);
        let them = !color;
        let bq = self.pieces[Piece::new(them, PieceType::Bishop)] | self.pieces[Piece::new(them, PieceType::Queen)];
        let rq = self.pieces[Piece::new(them, PieceType::Rook)] | self.pieces[Piece::new(them, PieceType::Queen)];
        let mut sliders = (bishop_attacks(ksq, Bitboard::EMPTY) & bq) | (rook_attacks(ksq, Bitboard::EMPTY) & rq);
        let mut pinned = Bitboard::EMPTY;
        while !sliders.is_empty() {
            let sq = sliders.pop_lsb();
            let blockers = between(sq, ksq) & self.occupied;
            if blockers.count() == 1 {
                pinned |= blockers & self.occupied_co[color];
            }
        }
        pinned
    }

    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m == Move::NULL {
            return false;
        }

        let from = m.from();
        let to = m.to();
        let piece = self.piece_on[from];

        if piece == Piece::None || piece.color() != self.side_to_move {
            return false;
        }

        let us = self.side_to_move;
        let them = !us;
        let dest_piece = self.piece_on[to];

        if dest_piece != Piece::None && dest_piece.color() == us {
            return false;
        }

        match m.move_type() {
            MoveType::Normal => {
                let pt = piece.piece_type();
                match pt {
                    PieceType::Pawn => {
                        let (forward, start_rank, promo_rank) = match us {
                            Color::White => (8i8, 1u8, 7u8),
                            Color::Black => (-8i8, 6u8, 0u8),
                        };

                        // last rank pawn move must be promotion
                        if to.rank() == promo_rank {
                            return false;
                        }

                        if dest_piece == Piece::None {
                            if to as i8 == from as i8 + forward {
                                return true;
                            }
                            if from.rank() == start_rank
                                && to as i8 == from as i8 + forward * 2
                                && self.piece_on[Square::new((from as i8 + forward) as u8)] == Piece::None
                            {
                                return true;
                            }
                            false
                        } else {
                            pawn_attacks(us, from).contains(to)
                        }
                    }
                    PieceType::Knight => knight_attacks(from).contains(to),
                    PieceType::Bishop => bishop_attacks(from, self.occupied).contains(to),
                    PieceType::Rook => rook_attacks(from, self.occupied).contains(to),
                    PieceType::Queen => (bishop_attacks(from, self.occupied) | rook_attacks(from, self.occupied)).contains(to),
                    PieceType::King => king_attacks(from).contains(to),
                    PieceType::None => false,
                }
            }
            MoveType::Promotion => {
                if piece.piece_type() != PieceType::Pawn {
                    return false;
                }
                let promo_rank = if us == Color::White { 7 } else { 0 };
                if to.rank() != promo_rank {
                    return false;
                }

                let forward = if us == Color::White { 8i8 } else { -8i8 };
                if dest_piece == Piece::None {
                    to as i8 == from as i8 + forward
                } else {
                    pawn_attacks(us, from).contains(to)
                }
            }
            MoveType::EnPassant => {
                if piece.piece_type() != PieceType::Pawn || to != self.ep_square {
                    return false;
                }
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.piece_on[cap_sq] == Piece::new(them, PieceType::Pawn) && pawn_attacks(us, from).contains(to)
            }
            MoveType::Castling => {
                if piece.piece_type() != PieceType::King {
                    return false;
                }
                let occ = self.occupied;
                match (us, from, to) {
                    (Color::White, Square::E1, Square::G1) => {
                        (self.castling_rights & Self::CASTLE_WK) != 0
                            && (occ.0 & ((1u64 << Square::F1 as u8) | (1u64 << Square::G1 as u8))) == 0
                            && !self.is_square_attacked(Square::E1, Color::Black)
                            && !self.is_square_attacked(Square::F1, Color::Black)
                            && !self.is_square_attacked(Square::G1, Color::Black)
                    }
                    (Color::White, Square::E1, Square::C1) => {
                        (self.castling_rights & Self::CASTLE_WQ) != 0
                            && (occ.0 & ((1u64 << Square::B1 as u8) | (1u64 << Square::C1 as u8) | (1u64 << Square::D1 as u8))) == 0
                            && !self.is_square_attacked(Square::E1, Color::Black)
                            && !self.is_square_attacked(Square::D1, Color::Black)
                            && !self.is_square_attacked(Square::C1, Color::Black)
                    }
                    (Color::Black, Square::E8, Square::G8) => {
                        (self.castling_rights & Self::CASTLE_BK) != 0
                            && (occ.0 & ((1u64 << Square::F8 as u8) | (1u64 << Square::G8 as u8))) == 0
                            && !self.is_square_attacked(Square::E8, Color::White)
                            && !self.is_square_attacked(Square::F8, Color::White)
                            && !self.is_square_attacked(Square::G8, Color::White)
                    }
                    (Color::Black, Square::E8, Square::C8) => {
                        (self.castling_rights & Self::CASTLE_BQ) != 0
                            && (occ.0 & ((1u64 << Square::B8 as u8) | (1u64 << Square::C8 as u8) | (1u64 << Square::D8 as u8))) == 0
                            && !self.is_square_attacked(Square::E8, Color::White)
                            && !self.is_square_attacked(Square::D8, Color::White)
                            && !self.is_square_attacked(Square::C8, Color::White)
                    }
                    _ => false,
                }
            }
        }
    }

    #[inline(always)]
    pub fn gives_direct_check(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let ksq = self.king_square(!us);
        let from = m.from();
        let to = m.to();
        let occ = (self.occupied ^ Bitboard::from_square(from)) | Bitboard::from_square(to);

        if m.move_type() == MoveType::Castling {
            let (rook_from, rook_to) = match to {
                Square::G1 => (Square::H1, Square::F1),
                Square::C1 => (Square::A1, Square::D1),
                Square::G8 => (Square::H8, Square::F8),
                Square::C8 => (Square::A8, Square::D8),
                _ => return false,
            };
            let rook_occ = (occ ^ Bitboard::from_square(rook_from)) | Bitboard::from_square(rook_to);
            return rook_attacks(rook_to, rook_occ).contains(ksq);
        }

        let pt = if m.move_type() == MoveType::Promotion {
            m.promo_type()
        } else {
            self.piece_on[from].piece_type()
        };

        match pt {
            PieceType::Knight => knight_attacks(to).contains(ksq),
            PieceType::Bishop => bishop_attacks(to, occ).contains(ksq),
            PieceType::Rook => rook_attacks(to, occ).contains(ksq),
            PieceType::Queen => (bishop_attacks(to, occ) | rook_attacks(to, occ)).contains(ksq),
            PieceType::Pawn => pawn_attacks(us, to).contains(ksq),
            _ => false,
        }
    }

    // squares that give check per piece type
    #[inline(always)]
    pub fn check_squares(&self) -> [Bitboard; 6] {
        let us = self.side_to_move;
        let ksq = self.king_square(!us);
        let bishop = bishop_attacks(ksq, self.occupied);
        let rook = rook_attacks(ksq, self.occupied);

        let mut squares = [Bitboard::EMPTY; 6];
        squares[PieceType::Pawn as usize] = pawn_attacks(!us, ksq);
        squares[PieceType::Knight as usize] = knight_attacks(ksq);
        squares[PieceType::Bishop as usize] = bishop;
        squares[PieceType::Rook as usize] = rook;
        squares[PieceType::Queen as usize] = bishop | rook;
        squares
    }

    // fast direct check using check_squares
    #[inline(always)]
    pub fn gives_check_with(&self, m: Move, check_squares: &[Bitboard; 6]) -> bool {
        if m.move_type() == MoveType::Castling || m.move_type() == MoveType::Promotion {
            return self.gives_direct_check(m);
        }

        let pt = self.piece_on[m.from()].piece_type();
        pt != PieceType::None && check_squares[pt as usize].contains(m.to())
    }

    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        self.is_legal_with(m, self.checkers(), self.pinned_pieces(us))
    }

    // legality check reusing checkers and pinned masks
    #[inline(always)]
    pub fn is_legal_with(&self, m: Move, checkers: Bitboard, pinned: Bitboard) -> bool {
        let us = self.side_to_move;
        let them = !us;
        let from = m.from();
        let to = m.to();
        let moving_piece = self.piece_on[from];
        let ksq = self.king_square(us);

        if m.move_type() == MoveType::Castling {
            return true;
        }

        let num_checkers = checkers.count();

        if num_checkers > 1 {
            if moving_piece.piece_type() != PieceType::King {
                return false;
            }
        } else if num_checkers == 1 && moving_piece.piece_type() != PieceType::King {
            let checker_sq = checkers.lsb();
            let ep_takes_checker = m.move_type() == MoveType::EnPassant && Square::from_coords(to.file(), from.rank()) == checker_sq;
            if !ep_takes_checker && !((between(ksq, checker_sq) | checkers).contains(to)) {
                return false;
            }
        }

        if moving_piece.piece_type() == PieceType::King {
            let mut occ = self.occupied;
            occ.clear(ksq);
            occ.set(to);
            let opp_pawns = self.pieces[Piece::new(them, PieceType::Pawn)];
            let opp_knights = self.pieces[Piece::new(them, PieceType::Knight)];
            let opp_bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)] | self.pieces[Piece::new(them, PieceType::Queen)];
            let opp_rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)] | self.pieces[Piece::new(them, PieceType::Queen)];
            let opp_king = self.pieces[Piece::new(them, PieceType::King)];
            return (pawn_attacks(us, to) & opp_pawns).is_empty()
                && (knight_attacks(to) & opp_knights).is_empty()
                && (bishop_attacks(to, occ) & opp_bishops_queens).is_empty()
                && (rook_attacks(to, occ) & opp_rooks_queens).is_empty()
                && (king_attacks(to) & opp_king).is_empty();
        }

        if m.move_type() == MoveType::EnPassant {
            let cap_sq = Square::from_coords(to.file(), from.rank());
            if self.piece_on[cap_sq] != Piece::new(them, PieceType::Pawn) {
                return false;
            }

            let mut occ = self.occupied;
            occ.clear(from);
            occ.clear(cap_sq);
            occ.set(to);

            let opp_bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)] | self.pieces[Piece::new(them, PieceType::Queen)];
            let opp_rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)] | self.pieces[Piece::new(them, PieceType::Queen)];

            return (bishop_attacks(ksq, occ) & opp_bishops_queens).is_empty() && (rook_attacks(ksq, occ) & opp_rooks_queens).is_empty();
        }

        if pinned.contains(from) {
            let from_rank = (from.rank() as i32) - (ksq.rank() as i32);
            let from_file = (from.file() as i32) - (ksq.file() as i32);
            let to_rank = (to.rank() as i32) - (ksq.rank() as i32);
            let to_file = (to.file() as i32) - (ksq.file() as i32);
            if from_rank * to_file != from_file * to_rank {
                return false;
            }
        }

        true
    }

    #[inline(always)]
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        let knights = self.pieces[Piece::new(color, PieceType::Knight)];
        let bishops = self.pieces[Piece::new(color, PieceType::Bishop)];
        let rooks = self.pieces[Piece::new(color, PieceType::Rook)];
        let queens = self.pieces[Piece::new(color, PieceType::Queen)];
        !(knights | bishops | rooks | queens).is_empty()
    }

    #[inline(always)]
    pub fn non_pawn_material(&self) -> i32 {
        let knights = (self.pieces[Piece::WhiteKnight] | self.pieces[Piece::BlackKnight]).count() as i32;
        let bishops = (self.pieces[Piece::WhiteBishop] | self.pieces[Piece::BlackBishop]).count() as i32;
        let rooks = (self.pieces[Piece::WhiteRook] | self.pieces[Piece::BlackRook]).count() as i32;
        let queens = (self.pieces[Piece::WhiteQueen] | self.pieces[Piece::BlackQueen]).count() as i32;

        knights * 320 + bishops * 330 + rooks * 500 + queens * 900
    }

    pub fn draw_by_material(&self) -> bool {
        let pawns = self.pieces[Piece::WhitePawn] | self.pieces[Piece::BlackPawn];
        let rooks = self.pieces[Piece::WhiteRook] | self.pieces[Piece::BlackRook];
        let queens = self.pieces[Piece::WhiteQueen] | self.pieces[Piece::BlackQueen];
        if !(pawns | rooks | queens).is_empty() {
            return false;
        }

        let count = self.occupied.count();
        if count <= 2 {
            return true;
        }
        if count == 3 {
            let knights = self.pieces[Piece::WhiteKnight] | self.pieces[Piece::BlackKnight];
            let bishops = self.pieces[Piece::WhiteBishop] | self.pieces[Piece::BlackBishop];
            if !(knights | bishops).is_empty() {
                return true;
            }
        }
        if count == 4 {
            let w_bishops = self.pieces[Piece::WhiteBishop];
            let b_bishops = self.pieces[Piece::BlackBishop];
            if w_bishops.count() == 1 && b_bishops.count() == 1 {
                let light_squares = Bitboard(0x55AA55AA55AA55AAu64);
                let w_light = !(w_bishops & light_squares).is_empty();
                let b_light = !(b_bishops & light_squares).is_empty();
                if w_light == b_light {
                    return true;
                }
            }
        }
        false
    }

    #[inline(always)]
    pub fn is_draw(&self) -> bool {
        self.is_repetition()
            || (self.halfmove_clock >= 100
                && (!self.in_check() || crate::movegen::generate_legal_moves(self).count > 0))
            // material draw needs <= 4 pieces
            || (self.occupied.count() <= 4 && self.draw_by_material())
    }

    // pieces that can give discovered check
    #[inline(always)]
    pub fn discovered_check_candidates(&self) -> Bitboard {
        let us = self.side_to_move;
        let them = !us;
        let ksq = self.king_square(them);
        let mut candidates = Bitboard::EMPTY;

        let mut snipers = (bishop_attacks(ksq, Bitboard::EMPTY) & (self.pieces[Piece::new(us, PieceType::Bishop)] | self.pieces[Piece::new(us, PieceType::Queen)]))
            | (rook_attacks(ksq, Bitboard::EMPTY) & (self.pieces[Piece::new(us, PieceType::Rook)] | self.pieces[Piece::new(us, PieceType::Queen)]));

        while !snipers.is_empty() {
            let sniper = snipers.pop_lsb();
            let blockers = between(sniper, ksq) & self.occupied;
            if blockers.count() == 1 {
                candidates |= blockers & self.occupied_co[us];
            }
        }

        candidates
    }

    #[inline(always)]
    pub fn opponent_pawn_threats(&self) -> Bitboard {
        let them = !self.side_to_move;
        let pawns = self.pieces[Piece::new(them, PieceType::Pawn)];
        match them {
            Color::White => (pawns & !Bitboard(0x8080808080808080u64)) << 9 | (pawns & !Bitboard(0x0101010101010101u64)) << 7,
            Color::Black => (pawns & !Bitboard(0x8080808080808080u64)) >> 7 | (pawns & !Bitboard(0x0101010101010101u64)) >> 9,
        }
    }

    #[allow(dead_code)]
    pub fn opponent_threats(&self) -> Bitboard {
        let them = !self.side_to_move;
        let mut threats = self.opponent_pawn_threats();

        let mut knights = self.pieces[Piece::new(them, PieceType::Knight)];
        while !knights.is_empty() {
            threats |= knight_attacks(knights.pop_lsb());
        }

        let occ = self.occupied;
        let mut bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)] | self.pieces[Piece::new(them, PieceType::Queen)];
        while !bishops_queens.is_empty() {
            threats |= bishop_attacks(bishops_queens.pop_lsb(), occ);
        }

        let mut rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)] | self.pieces[Piece::new(them, PieceType::Queen)];
        while !rooks_queens.is_empty() {
            threats |= rook_attacks(rooks_queens.pop_lsb(), occ);
        }

        let ksq = self.king_square(them);
        threats |= king_attacks(ksq);

        threats
    }

    #[inline(always)]
    pub fn is_repetition(&self) -> bool {
        let count = self.hist_len;
        if count < 4 || self.halfmove_clock < 4 {
            return false;
        }

        let max_steps = (self.halfmove_clock as usize).min(count);
        let mut earlier_reps = 0;
        let mut i = 2;
        while i <= max_steps {
            let idx = count - i;
            if self.history[idx] == self.hash {
                // 2-fold in tree, 3-fold before root
                if idx >= self.root_ply || earlier_reps > 0 {
                    return true;
                }
                earlier_reps += 1;
            }
            i += 2;
        }

        false
    }

    pub fn upcoming_repetition(&self) -> bool {
        let count = self.hist_len;
        let max_steps = (self.halfmove_clock as usize).min(count);
        if max_steps < 3 {
            return false;
        }

        let current_key = self.hash;
        let mut index = count - 1;
        // hoist cuckoo lookup out of loop
        let cuckoo = crate::zobrist::cuckoo();

        let mut compared_ply = 3;
        while compared_ply <= max_steps {
            index -= 2;

            // exact xor difference between the current and an older position
            let diff = current_key ^ self.history[index];
            let mut c_idx = crate::zobrist::h1(diff);

            if cuckoo.keys[c_idx] != diff {
                c_idx = crate::zobrist::h2(diff);
                if cuckoo.keys[c_idx] != diff {
                    compared_ply += 2;
                    continue;
                }
            }

            let sq1 = cuckoo.sq_a[c_idx];
            let sq2 = cuckoo.sq_b[c_idx];
            if (crate::attacks::between(sq1, sq2) & self.occupied).is_empty() {
                let p1 = self.piece_on[sq1];
                let (from, to) = if p1 != crate::types::Piece::None && p1.color() == self.side_to_move {
                    (sq1, sq2)
                } else {
                    (sq2, sq1)
                };

                // build a pseudo-legal move and strictly verify it (king safety)
                let m = crate::types::Move::new(from, to, crate::types::PieceType::None, crate::types::MoveType::Normal);
                if self.is_pseudo_legal(m) && self.is_legal(m) {
                    // draw immediately in tree, needs prior repeat if before root
                    if index >= self.root_ply {
                        return true;
                    }
                    let repeated_key = self.history[index];
                    let mut j = index;
                    while j >= 2 {
                        j -= 2;
                        if self.history[j] == repeated_key {
                            return true;
                        }
                    }
                }
            }

            compared_ply += 2;
        }

        false
    }

    pub fn make_null_move(&mut self) -> UndoState {
        let undo = UndoState {
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            captured: Piece::None,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
            non_pawn_hash: self.non_pawn_hash,
        };
        self.history[self.hist_len] = self.hash;
        self.hist_len += 1;

        // write metadata to next ply frames (no clone needed)
        let next_d = self.acc_depth + 1;
        let cur_ksq = self.acc_frames[self.acc_depth].ksq;
        {
            let af = &mut self.acc_frames[next_d];
            af.delta = FeatureDelta::new();
            af.dirty = [true, true];
            af.refresh = [false, false];
            af.ksq = cur_ksq;
        }
        {
            let tf = &mut self.threat_frames[next_d];
            tf.dirty = true;
            tf.threat_base = next_d;
            tf.last_move = Move::NULL;
            tf.last_captured = Piece::None;
        }
        self.acc_depth = next_d;

        if self.ep_square.is_valid() {
            self.hash ^= ep_key(self.ep_square.file());
            self.ep_square = Square::None;
        }

        self.side_to_move = !self.side_to_move;
        self.hash ^= side_key();
        self.halfmove_clock = 0;
        self.tt_hash = self.hash ^ fiftymove_key(0);

        undo
    }

    pub fn undo_null_move(&mut self, undo: UndoState) {
        self.hist_len -= 1;
        self.acc_depth -= 1;
        self.side_to_move = !self.side_to_move;
        self.castling_rights = undo.castling_rights;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.refresh_tt_hash();
        self.non_pawn_hash = undo.non_pawn_hash;
    }

    pub fn make_move(&mut self, m: Move) -> UndoState {
        let from = m.from();
        let to = m.to();
        let move_type = m.move_type();
        let us = self.side_to_move;
        let them = !us;
        let moving_piece = self.piece_on[from];
        let is_king_move = moving_piece.piece_type() == PieceType::King;
        let w_ksq = self.king_square(Color::White);
        let b_ksq = self.king_square(Color::Black);

        let undo = UndoState {
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            captured: self.piece_on[to],
            hash: self.hash,
            pawn_hash: self.pawn_hash,
            non_pawn_hash: self.non_pawn_hash,
        };

        // write metadata to next ply frames (no clone needed)
        let next_d = self.acc_depth + 1;
        // use current king squares as initial values, will be updated after king move check
        let cur_ksq = self.acc_frames[self.acc_depth].ksq;
        {
            let af = &mut self.acc_frames[next_d];
            af.dirty = [true, true];
            af.refresh = [false, false];
            af.ksq = cur_ksq;
        }
        {
            let tf = &mut self.threat_frames[next_d];
            tf.dirty = true;
            tf.threat_base = next_d;
            tf.last_move = m;
            tf.last_captured = undo.captured;
        }
        self.history[self.hist_len] = self.hash;
        self.hist_len += 1;
        if self.ep_square.is_valid() {
            self.hash ^= ep_key(self.ep_square.file());
            self.ep_square = Square::None;
        }

        self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        if moving_piece.piece_type() == PieceType::Pawn || undo.captured != Piece::None {
            self.halfmove_clock = 0;
        }

        self.remove_piece(from);

        // refresh on castling or king bucket/mirror cross
        let refresh_us = is_king_move && (move_type == MoveType::Castling || crate::nnue::needs_accumulator_refresh(from, to, us));

        let (acc_w_ksq, acc_b_ksq) = if is_king_move {
            match us {
                Color::White => (to, b_ksq),
                Color::Black => (w_ksq, to),
            }
        } else {
            (w_ksq, b_ksq)
        };

        // update the accumulator frame with correct king squares and refresh flag
        self.acc_frames[next_d].ksq = [acc_w_ksq, acc_b_ksq];
        self.acc_frames[next_d].refresh[us as usize] = refresh_us;

        // skip our side if doing full refresh
        let skip = if refresh_us { Some(us) } else { None };
        let mut delta = FeatureDelta::new();

        delta.sub(moving_piece, from, acc_w_ksq, acc_b_ksq, skip);
        if move_type != MoveType::Promotion {
            delta.add(moving_piece, to, acc_w_ksq, acc_b_ksq, skip);
        }

        match move_type {
            MoveType::Normal => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                    delta.sub(undo.captured, to, acc_w_ksq, acc_b_ksq, skip);
                }
                self.put_piece(moving_piece, to);

                if moving_piece.piece_type() == PieceType::Pawn && ((from as i8) - (to as i8)).abs() == 16 {
                    let ep = Square::new(((from as u8) + (to as u8)) / 2);
                    let enemy_pawns = self.pieces[Piece::new(them, PieceType::Pawn)];
                    if !(pawn_attacks(us, ep) & enemy_pawns).is_empty() {
                        self.ep_square = ep;
                        self.hash ^= ep_key(ep.file());
                    }
                }
            }
            MoveType::Castling => {
                self.put_piece(moving_piece, to);

                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self.remove_piece(rook_from);
                self.put_piece(rook, rook_to);

                delta.sub(rook, rook_from, acc_w_ksq, acc_b_ksq, skip);
                delta.add(rook, rook_to, acc_w_ksq, acc_b_ksq, skip);
            }
            MoveType::EnPassant => {
                let cap_sq = Square::from_coords(to.file(), from.rank());
                let cap_pawn = self.remove_piece(cap_sq);
                delta.sub(cap_pawn, cap_sq, acc_w_ksq, acc_b_ksq, skip);

                self.put_piece(moving_piece, to);
            }
            MoveType::Promotion => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                    delta.sub(undo.captured, to, acc_w_ksq, acc_b_ksq, skip);
                }
                let promo_piece = Piece::new(us, m.promo_type());
                self.put_piece(promo_piece, to);
                delta.add(promo_piece, to, acc_w_ksq, acc_b_ksq, skip);
            }
        }

        self.acc_frames[next_d].delta = delta;
        self.acc_depth = next_d;

        let new_castling = self.castling_rights & CASTLING_RIGHTS_MASK[from as usize] & CASTLING_RIGHTS_MASK[to as usize];
        if new_castling != self.castling_rights {
            self.hash ^= castling_key(self.castling_rights);
            self.hash ^= castling_key(new_castling);
            self.castling_rights = new_castling;
        }

        if us == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        self.side_to_move = them;
        self.hash ^= side_key();
        self.refresh_tt_hash();

        undo
    }

    pub fn undo_move(&mut self, m: Move, undo: UndoState) {
        self.hist_len -= 1;
        self.acc_depth -= 1;
        self.side_to_move = !self.side_to_move;
        let us = self.side_to_move;

        let from = m.from();
        let to = m.to();
        let move_type = m.move_type();

        if us == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_sub(1);
        }

        let moved_piece = self.remove_piece_no_hash(to);

        match move_type {
            MoveType::Normal => {
                self.put_piece_no_hash(moved_piece, from);
                if undo.captured != Piece::None {
                    self.put_piece_no_hash(undo.captured, to);
                }
            }
            MoveType::Castling => {
                self.put_piece_no_hash(moved_piece, from);
                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self.remove_piece_no_hash(rook_to);
                self.put_piece_no_hash(rook, rook_from);
            }
            MoveType::EnPassant => {
                self.put_piece_no_hash(moved_piece, from);
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.put_piece_no_hash(Piece::new(!us, PieceType::Pawn), cap_sq);
            }
            MoveType::Promotion => {
                self.put_piece_no_hash(Piece::new(us, PieceType::Pawn), from);
                if undo.captured != Piece::None {
                    self.put_piece_no_hash(undo.captured, to);
                }
            }
        }

        self.castling_rights = undo.castling_rights;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.refresh_tt_hash();
        self.non_pawn_hash = undo.non_pawn_hash;
        // threat frame and accumulator frame are already restored by decrementing acc_depth
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(STARTING_FEN).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluate;
    use crate::movegen::{generate_legal_moves, generate_noisy_pseudo, generate_quiet_pseudo};

    fn pseudo_legal(board: &Board) -> crate::types::MoveList {
        let mut list = crate::types::MoveList::new();
        generate_noisy_pseudo(board, &mut list);
        generate_quiet_pseudo(board, &mut list);
        list
    }

    fn play(board: &mut Board, s: &str) {
        let moves = generate_legal_moves(board);
        for &m in moves.as_slice() {
            if m.to_string() == s {
                board.make_move(m);
                return;
            }
        }
        panic!("move not found: {}", s);
    }

    #[test]
    fn ep_square_and_king_validation() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_err());

        let mut board = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        play(&mut board, "e2e4");
        assert!(!board.ep_square.is_valid());
        assert_eq!(board.hash, Board::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").unwrap().hash);

        let mut board = Board::from_fen("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1").unwrap();
        play(&mut board, "e2e4");
        assert_eq!(board.ep_square, Square::E3);
    }

    #[test]
    fn pseudo_legal_rejects_unencoded_promotions() {
        let board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let quiet_push = Move::new(Square::A7, Square::A8, PieceType::None, MoveType::Normal);
        assert!(!board.is_pseudo_legal(quiet_push));

        let promo = Move::new(Square::A7, Square::A8, PieceType::Queen, MoveType::Promotion);
        assert!(board.is_pseudo_legal(promo));

        let board = Board::from_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1").unwrap();
        let quiet_push = Move::new(Square::A2, Square::A1, PieceType::None, MoveType::Normal);
        assert!(!board.is_pseudo_legal(quiet_push));

        // a castling-typed move must actually start from the king square
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let fake = Move::new(Square::H1, Square::G1, PieceType::None, MoveType::Castling);
        assert!(!board.is_pseudo_legal(fake));
    }

    #[test]
    fn fen_drops_castling_rights_without_pieces() {
        // no rook on a1 and no rook on h1: neither white right may survive
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1").unwrap();
        assert_eq!(board.castling_rights, 0);

        // kingside right is real (rook on h1), queenside isn't (no rook on a1)
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w KQ - 0 1").unwrap();
        assert_eq!(board.castling_rights, Board::CASTLE_WK);

        // king off its home square invalidates both rights for that side
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3K3R w KQ - 0 1").unwrap();
        assert_eq!(board.castling_rights, 0);
    }

    #[test]
    fn fen_rejects_side_not_to_move_in_check() {
        // black isn't to move but is attacked by the rook on e2
        assert!(Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1").is_err());
        // the side to move is allowed to be in check
        assert!(Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1").is_ok());
    }

    #[test]
    fn fen_rejects_pawns_on_back_ranks() {
        assert!(Board::from_fen("P3k3/8/8/8/8/8/8/4K3 w - - 0 1").is_err());
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/P3K3 w - - 0 1").is_err());
        assert!(Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").is_ok());
    }

    #[test]
    fn fifty_move_rule_does_not_mask_checkmate() {
        let board = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1").unwrap();
        assert!(board.in_check());
        assert!(generate_legal_moves(&board).count == 0);
        assert!(!board.is_draw());
    }

    #[test]
    fn repetition_needs_threefold_before_root() {
        let mut board = Board::default();
        for s in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            play(&mut board, s);
        }

        board.root_ply = board.hist_len;
        assert!(!board.is_repetition());

        board.root_ply = 0;
        assert!(board.is_repetition());
    }

    #[test]
    fn gives_direct_check_promotion_and_castling() {
        let board = Board::from_fen("8/4k1P1/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let knight_promo = generate_legal_moves(&board)
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.move_type() == MoveType::Promotion && m.promo_type() == PieceType::Knight)
            .unwrap();
        assert!(board.gives_direct_check(knight_promo));

        let castle = Board::from_fen("5k2/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        let castle_move = generate_legal_moves(&castle)
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.move_type() == MoveType::Castling)
            .unwrap();
        assert!(castle.gives_direct_check(castle_move));
    }

    #[test]
    fn check_squares_match_direct_check() {
        let check = |board: &Board, fen: &str| {
            let squares = board.check_squares();
            for &m in pseudo_legal(board).as_slice() {
                // castling checks come from the rook, skip
                if m.move_type() == MoveType::Castling {
                    continue;
                }
                assert_eq!(
                    board.gives_check_with(m, &squares),
                    board.gives_direct_check(m),
                    "check detection mismatch on {} in {}",
                    m,
                    fen
                );
            }
        };

        let promos = Board::from_fen("4k3/6P1/8/8/8/8/1p6/K7 w - - 0 1").unwrap();
        check(&promos, "promotions");

        let mut board = Board::default();
        let mut rng: u64 = 0x0BAD_F00D_1234_5678;
        for _ in 0..200 {
            check(&board, "random walk");
            let moves = generate_legal_moves(&board);
            if moves.count == 0 {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            board.make_move(moves.get((rng as usize) % moves.count));
            board.reset_accumulator_stack();
        }
    }

    #[test]
    fn debug_game_position() {
        let mut board = Board::default();
        for s in [
            "e2e4", "g8f6", "e4e5", "b8c6", "e5f6", "e7f6", "d2d4", "d7d5", "f1b5", "f8b4", "c1d2", "e8g8", "d2b4", "c6d4", "d1d4", "c8h3", "g2h3", "f8e8",
            "b5e8", "c7c5", "d4c5", "d8e8", "g1e2", "e8e6", "b1c3",
        ] {
            play(&mut board, s);
        }
        println!("side_to_move = {:?}", board.side_to_move);
        println!("eval (side to move) = {}", evaluate(&mut board));
        println!("is_draw = {}", board.is_draw());
        println!("upcoming_repetition = {}", board.upcoming_repetition());
        println!("is_repetition = {}", board.is_repetition());
        println!("halfmove = {}", board.halfmove_clock);

        let mut start = Board::default();
        println!("eval startpos (white to move) = {}", evaluate(&mut start));
        let mut noq = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1").unwrap();
        println!("eval white-no-queen (white to move) = {}", evaluate(&mut noq));
    }

    // compare incremental threats against full rebuild
    fn assert_threat_state(board: &mut Board, weights: &crate::nnue::ThreatWeights, ply: usize, stage: &str) {
        let mut fresh = crate::nnue::ThreatAccumulator::new();
        for side in 0..2 {
            let mut list = ThreatList::new();
            crate::nnue::collect_threats(board, Color::from_index(side), &mut list);
            fresh.refresh_side(side, &list, weights);
        }
        assert_eq!(fresh.vals, board.threat().vals, "{stage}: threat accumulator at ply {ply}");
    }

    #[test]
    fn threat_accumulator_roundtrip_random_walk() {
        // use active net weights or random ones in tests
        let synthetic;
        let weights: &crate::nnue::ThreatWeights = match crate::nnue::threat_weights() {
            Some(weights) => weights,
            None => {
                synthetic = crate::nnue::ThreatWeights::random(0x51EE_DD00);
                &synthetic
            }
        };

        let mut board = Board::default();
        board.refresh_threats();
        assert_threat_state(&mut board, weights, 0, "root");

        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut ply = 0;
        for _ in 0..1000 {
            let moves = generate_legal_moves(&board);
            if moves.count == 0 {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = moves.get((rng as usize) % moves.count);

            let undo = board.make_move(m);
            board.ensure_threats_ready();
            assert_threat_state(&mut board, weights, ply, "after make");

            board.undo_move(m, undo);
            board.ensure_threats_ready();
            assert_threat_state(&mut board, weights, ply, "after undo");
            ply += 1;
        }

        assert!(ply > 500, "the walk should reach a reasonable depth ({ply} plies)");
    }

    #[test]
    fn threat_accumulator_handles_king_bucket_changes() {
        let synthetic;
        let weights: &crate::nnue::ThreatWeights = match crate::nnue::threat_weights() {
            Some(weights) => weights,
            None => {
                synthetic = crate::nnue::ThreatWeights::random(0xF00D_1234);
                &synthetic
            }
        };

        // walk king across mirror line and bucket boundaries
        let mut board = Board::from_fen("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1").unwrap();
        board.ensure_threats_ready();
        assert_threat_state(&mut board, weights, 0, "root");

        // black shuffles its king back and forth while white crosses the mirror
        // line and two bucket boundaries.
        for (ply, text) in ["e1d1", "e8d8", "d1c1", "d8c8", "c1b1", "c8d8", "b1b2", "d8e8", "b2a2"].iter().enumerate() {
            play(&mut board, text);
            board.ensure_threats_ready();
            assert_threat_state(&mut board, weights, ply, text);
        }
    }

    #[test]
    fn accumulator_roundtrip_random_walk() {
        let mut board = Board::default();
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..400 {
            let moves = generate_legal_moves(&mut board);
            if moves.count == 0 {
                break;
            }
            for &m in moves.as_slice() {
                board.ensure_accumulator_ready();
                let before = board.accumulator().vals;
                let undo = board.make_move(m);
                board.ensure_accumulator_ready();
                let mut check = board.clone();
                check.refresh_accumulator();
                assert_eq!(board.accumulator().vals, check.accumulator().vals, "make mismatch on {}", m);
                board.undo_move(m, undo);
                board.ensure_accumulator_ready();
                assert_eq!(board.accumulator().vals, before, "undo mismatch on {}", m);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.count;
            board.make_move(moves.get(pick));
            board.reset_accumulator_stack();
        }
    }

    #[test]
    fn accumulator_roundtrip_bench_positions() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/8/1k1NK3/r7/2R2P1P/3n2P1/8/8 b - - 0 59",
            "8/3k4/3b3p/R1p1p2P/p1Pr4/P2P1r2/2K1R3/4B3 b - - 4 75",
        ];

        for fen in fens {
            let mut board = Board::from_fen(fen).unwrap();
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
            for _ in 0..24 {
                let moves = generate_legal_moves(&board);
                if moves.count == 0 {
                    break;
                }
                for &m in moves.as_slice() {
                    board.ensure_accumulator_ready();
                    let before = board.accumulator().vals;
                    let undo = board.make_move(m);
                    board.ensure_accumulator_ready();
                    let mut check = board.clone();
                    check.refresh_accumulator();
                    assert_eq!(board.accumulator().vals, check.accumulator().vals, "make mismatch on {} in {}", m, fen);
                    board.undo_move(m, undo);
                    board.ensure_accumulator_ready();
                    assert_eq!(board.accumulator().vals, before, "undo mismatch on {}", m);
                }

                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                board.make_move(moves.get((rng as usize) % moves.count));
                board.reset_accumulator_stack();
            }
        }
    }

    #[test]
    fn is_legal_matches_make_unmake() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp4p/p1pb4/4P3/8/1P3N2/PBP2PPP/R5K1 w - - 0 1",
            "1r4k1/3q1ppp/p3bn2/2pP4/2P1B3/1P6/P4PPP/2R1Q1K1 w - - 0 1",
        ];

        for fen in fens {
            let mut board = Board::from_fen(fen).unwrap();
            let moves = pseudo_legal(&board);
            assert!(!moves.as_slice().is_empty(), "no moves generated from {}", fen);

            for &m in moves.as_slice() {
                let us = board.side_to_move;
                let rule = board.is_legal(m);
                let undo = board.make_move(m);
                let our_ksq = board.king_square(us);
                let still_legal = !board.is_square_attacked(our_ksq, board.side_to_move);
                board.undo_move(m, undo);

                assert_eq!(rule, still_legal, "is_legal mismatch on {} in {}", m, fen);
            }
        }
    }

    #[test]
    fn perft() {
        let cases: [(&str, u32, u64); 3] = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197_281),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43_238),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97_862),
        ];

        for (fen, depth, expected) in cases {
            let mut board = Board::from_fen(fen).unwrap();
            let nodes = perft_impl(&mut board, depth);
            assert_eq!(nodes, expected, "perft mismatch for FEN {}", fen);
        }
    }

    fn perft_impl(board: &mut Board, depth: u32) -> u64 {
        let moves = generate_legal_moves(board);
        if depth <= 1 {
            return moves.count as u64;
        }
        let mut nodes = 0;
        for &m in moves.as_slice() {
            let undo = board.make_move(m);
            nodes += perft_impl(board, depth - 1);
            board.undo_move(m, undo);
        }
        nodes
    }
}
