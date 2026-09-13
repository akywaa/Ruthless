use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks, between};
use crate::bitboard::Bitboard;
use crate::nnue::Accumulator;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square, COLOR_NB, PIECE_NB, SQUARE_NB};
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
    pub non_pawn_hash: [u64; 2],
}

#[derive(Clone)]
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
    pub history: Vec<u64>,
    pub accumulator: Accumulator,
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
            history: Vec::with_capacity(256),
            accumulator: Accumulator::new(),
        }
    }

    pub fn refresh_accumulator(&mut self) {
        self.accumulator = Accumulator::new();
        let w_ksq = self.king_square(Color::White);
        let b_ksq = self.king_square(Color::Black);
        for sq in 0..64 {
            let piece = self.piece_on[sq];
            if piece != Piece::None {
                self.accumulator
                    .add_feature(piece, Square::new(sq as u8), w_ksq, b_ksq);
            }
        }
    }

    pub fn refresh_accumulator_side(&mut self, color: Color) {
        let piece_on = self.piece_on;
        let ksq = self.king_square(color);
        self.accumulator.refresh_side(&piece_on, ksq, color);
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut board = Self::new();
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
                    file += digit as u8;
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
            board.halfmove_clock = parts[4].parse().unwrap_or(0);
        }

        if parts.len() > 5 {
            board.fullmove_number = parts[5].parse().unwrap_or(1);
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

        let opp_bishops_queens = self.pieces[Piece::new(by_color, PieceType::Bishop)]
            | self.pieces[Piece::new(by_color, PieceType::Queen)];
        if !(bishop_attacks(sq, self.occupied) & opp_bishops_queens).is_empty() {
            return true;
        }

        let opp_rooks_queens = self.pieces[Piece::new(by_color, PieceType::Rook)]
            | self.pieces[Piece::new(by_color, PieceType::Queen)];
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
        let bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
        let rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
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
        let bq = self.pieces[Piece::new(them, PieceType::Bishop)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
        let rq = self.pieces[Piece::new(them, PieceType::Rook)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
        let mut sliders = (bishop_attacks(ksq, Bitboard::EMPTY) & bq)
            | (rook_attacks(ksq, Bitboard::EMPTY) & rq);
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
                        let (forward, start_rank) = match us {
                            Color::White => (8i8, 1u8),
                            Color::Black => (-8i8, 6u8),
                        };

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
                let occ = self.occupied;
                match (us, to) {
                    (Color::White, Square::G1) => {
                        (self.castling_rights & Self::CASTLE_WK) != 0
                            && (occ.0 & ((1u64 << Square::F1 as u8) | (1u64 << Square::G1 as u8))) == 0
                            && !self.is_square_attacked(Square::E1, Color::Black)
                            && !self.is_square_attacked(Square::F1, Color::Black)
                            && !self.is_square_attacked(Square::G1, Color::Black)
                    }
                    (Color::White, Square::C1) => {
                        (self.castling_rights & Self::CASTLE_WQ) != 0
                            && (occ.0 & ((1u64 << Square::B1 as u8) | (1u64 << Square::C1 as u8) | (1u64 << Square::D1 as u8))) == 0
                            && !self.is_square_attacked(Square::E1, Color::Black)
                            && !self.is_square_attacked(Square::D1, Color::Black)
                            && !self.is_square_attacked(Square::C1, Color::Black)
                    }
                    (Color::Black, Square::G8) => {
                        (self.castling_rights & Self::CASTLE_BK) != 0
                            && (occ.0 & ((1u64 << Square::F8 as u8) | (1u64 << Square::G8 as u8))) == 0
                            && !self.is_square_attacked(Square::E8, Color::White)
                            && !self.is_square_attacked(Square::F8, Color::White)
                            && !self.is_square_attacked(Square::G8, Color::White)
                    }
                    (Color::Black, Square::C8) => {
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

    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        let them = !us;
        let from = m.from();
        let to = m.to();
        let moving_piece = self.piece_on[from];
        let ksq = self.king_square(us);

        if m.move_type() == MoveType::Castling {
            return true;
        }

        let checkers = self.checkers();
        let num_checkers = checkers.count();

        if num_checkers > 1 {
            if moving_piece.piece_type() != PieceType::King {
                return false;
            }
        } else if num_checkers == 1 && moving_piece.piece_type() != PieceType::King {
            let checker_sq = checkers.lsb();
            let ep_takes_checker = m.move_type() == MoveType::EnPassant
                && Square::from_coords(to.file(), from.rank()) == checker_sq;
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
            let opp_bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)]
                | self.pieces[Piece::new(them, PieceType::Queen)];
            let opp_rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)]
                | self.pieces[Piece::new(them, PieceType::Queen)];
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

            let opp_bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)]
                | self.pieces[Piece::new(them, PieceType::Queen)];
            let opp_rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)]
                | self.pieces[Piece::new(them, PieceType::Queen)];

            return (bishop_attacks(ksq, occ) & opp_bishops_queens).is_empty()
                && (rook_attacks(ksq, occ) & opp_rooks_queens).is_empty();
        }

        if self.pinned_pieces(us).contains(from) {
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
        self.is_repetition() || self.halfmove_clock >= 100 || self.draw_by_material()
    }

    #[inline(always)]
    pub fn opponent_pawn_threats(&self) -> Bitboard {
        let them = !self.side_to_move;
        let pawns = self.pieces[Piece::new(them, PieceType::Pawn)];
        match them {
            Color::White => {
                (pawns & !Bitboard(0x8080808080808080u64)) << 9
                    | (pawns & !Bitboard(0x0101010101010101u64)) << 7
            }
            Color::Black => {
                (pawns & !Bitboard(0x8080808080808080u64)) >> 7
                    | (pawns & !Bitboard(0x0101010101010101u64)) >> 9
            }
        }
    }

    pub fn opponent_threats(&self) -> Bitboard {
        let them = !self.side_to_move;
        let mut threats = self.opponent_pawn_threats();

        let mut knights = self.pieces[Piece::new(them, PieceType::Knight)];
        while !knights.is_empty() {
            threats |= knight_attacks(knights.pop_lsb());
        }

        let occ = self.occupied;
        let mut bishops_queens = self.pieces[Piece::new(them, PieceType::Bishop)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
        while !bishops_queens.is_empty() {
            threats |= bishop_attacks(bishops_queens.pop_lsb(), occ);
        }

        let mut rooks_queens = self.pieces[Piece::new(them, PieceType::Rook)]
            | self.pieces[Piece::new(them, PieceType::Queen)];
        while !rooks_queens.is_empty() {
            threats |= rook_attacks(rooks_queens.pop_lsb(), occ);
        }

        let ksq = self.king_square(them);
        threats |= king_attacks(ksq);

        threats
    }

    #[inline(always)]
    pub fn is_repetition(&self) -> bool {
        let count = self.history.len();
        if count < 4 || self.halfmove_clock < 4 {
            return false;
        }

        let max_steps = (self.halfmove_clock as usize).min(count);
        let mut i = 2;
        while i <= max_steps {
            if self.history[count - i] == self.hash {
                return true;
            }
            i += 2;
        }

        false
    }

    pub fn upcoming_repetition(&self) -> bool {
        let count = self.history.len();
        let max_steps = (self.halfmove_clock as usize).min(count);
        if max_steps < 3 {
            return false;
        }

        let current_key = self.hash;
        let mut index = count - 1;
        let mut other = current_key ^ self.history[index] ^ side_key();

        let mut compared_ply = 3;
        while compared_ply <= max_steps {
            index -= 1;
            other ^= self.history[index] ^ self.history[index - 1] ^ side_key();
            index -= 1;

            if other == 0 {
                let diff = current_key ^ self.history[index];
                let mut c_idx = crate::zobrist::h1(diff);

                let cuckoo = crate::zobrist::cuckoo();
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
                    return true;
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
            non_pawn_hash: self.non_pawn_hash,
        };

        self.history.push(self.hash);

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
        self.history.pop();
        self.side_to_move = !self.side_to_move;
        self.castling_rights = undo.castling_rights;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
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
            non_pawn_hash: self.non_pawn_hash,
        };

        self.history.push(self.hash);

        if self.ep_square.is_valid() {
            self.hash ^= ep_key(self.ep_square.file());
            self.ep_square = Square::None;
        }

        self.halfmove_clock += 1;
        if moving_piece.piece_type() == PieceType::Pawn || undo.captured != Piece::None {
            self.halfmove_clock = 0;
        }

        self.remove_piece(from);

        if !is_king_move {
            self.accumulator.remove_feature(moving_piece, from, w_ksq, b_ksq);
        } else {
            // For the opponent, king bucket never changes; update incrementally
            self.accumulator.remove_feature_side(moving_piece, from, self.king_square(them), them);
        }

        match move_type {
            MoveType::Normal => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                    if !is_king_move {
                        self.accumulator.remove_feature(undo.captured, to, w_ksq, b_ksq);
                    } else {
                        self.accumulator.remove_feature_side(undo.captured, to, self.king_square(them), them);
                    }
                }
                self.put_piece(moving_piece, to);
                if !is_king_move {
                    self.accumulator.add_feature(moving_piece, to, w_ksq, b_ksq);
                } else {
                    self.accumulator.add_feature_side(moving_piece, to, self.king_square(them), them);
                }

                if moving_piece.piece_type() == PieceType::Pawn && ((from as i8) - (to as i8)).abs() == 16 {
                    self.ep_square = Square::new(((from as u8) + (to as u8)) / 2);
                    self.hash ^= ep_key(self.ep_square.file());
                }
            }
            MoveType::Castling => {
                self.put_piece(moving_piece, to);
                self.accumulator.add_feature_side(moving_piece, to, self.king_square(them), them);

                let (rook_from, rook_to) = match to {
                    Square::G1 => (Square::H1, Square::F1),
                    Square::C1 => (Square::A1, Square::D1),
                    Square::G8 => (Square::H8, Square::F8),
                    Square::C8 => (Square::A8, Square::D8),
                    _ => unreachable!(),
                };
                let rook = self.remove_piece(rook_from);
                self.put_piece(rook, rook_to);

                self.accumulator.remove_feature_side(rook, rook_from, self.king_square(them), them);
                self.accumulator.add_feature_side(rook, rook_to, self.king_square(them), them);
            }
            MoveType::EnPassant => {
                let cap_sq = Square::from_coords(to.file(), from.rank());
                let cap_pawn = self.remove_piece(cap_sq);
                self.accumulator.remove_feature(cap_pawn, cap_sq, w_ksq, b_ksq);

                self.put_piece(moving_piece, to);
                self.accumulator.add_feature(moving_piece, to, w_ksq, b_ksq);
            }
            MoveType::Promotion => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                    self.accumulator.remove_feature(undo.captured, to, w_ksq, b_ksq);
                }
                let promo_piece = Piece::new(us, m.promo_type());
                self.put_piece(promo_piece, to);
                self.accumulator.add_feature(promo_piece, to, w_ksq, b_ksq);
            }
        }

        if is_king_move {
            self.refresh_accumulator_side(us);
        }

        let new_castling = self.castling_rights & CASTLING_RIGHTS_MASK[from as usize] & CASTLING_RIGHTS_MASK[to as usize];
        if new_castling != self.castling_rights {
            self.hash ^= castling_key(self.castling_rights);
            self.hash ^= castling_key(new_castling);
            self.castling_rights = new_castling;
        }

        if us == Color::Black {
            self.fullmove_number += 1;
        }
        self.side_to_move = them;
        self.hash ^= side_key();
        self.refresh_tt_hash();

        undo
    }

    pub fn undo_move(&mut self, m: Move, undo: UndoState) {
        self.history.pop();
        self.side_to_move = !self.side_to_move;
        let us = self.side_to_move;

        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        let from = m.from();
        let to = m.to();
        let move_type = m.move_type();

        let moved_piece = self.remove_piece(to);

        match move_type {
            MoveType::Normal => {
                self.put_piece(moved_piece, from);
                if undo.captured != Piece::None {
                    self.put_piece(undo.captured, to);
                }
            }
            MoveType::Castling => {
                self.put_piece(moved_piece, from);
                let (rook_from, rook_to) = match to {
                    Square::G1 => (Square::H1, Square::F1),
                    Square::C1 => (Square::A1, Square::D1),
                    Square::G8 => (Square::H8, Square::F8),
                    Square::C8 => (Square::A8, Square::D8),
                    _ => unreachable!(),
                };
                let rook = self.remove_piece(rook_to);
                self.put_piece(rook, rook_from);
            }
            MoveType::EnPassant => {
                self.put_piece(moved_piece, from);
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.put_piece(Piece::new(!us, PieceType::Pawn), cap_sq);
            }
            MoveType::Promotion => {
                self.put_piece(Piece::new(us, PieceType::Pawn), from);
                if undo.captured != Piece::None {
                    self.put_piece(undo.captured, to);
                }
            }
        }

        self.castling_rights = undo.castling_rights;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.refresh_tt_hash();
        self.non_pawn_hash = undo.non_pawn_hash;

        let them = !us;
        let is_king_move = moved_piece.piece_type() == PieceType::King;

        if is_king_move {
            let ksq_them = self.king_square(them);
            match move_type {
                MoveType::Castling => {
                    let (rook_from, rook_to) = match to {
                        Square::G1 => (Square::H1, Square::F1),
                        Square::C1 => (Square::A1, Square::D1),
                        Square::G8 => (Square::H8, Square::F8),
                        Square::C8 => (Square::A8, Square::D8),
                        _ => unreachable!(),
                    };
                    let rook = self.piece_on[rook_from];
                    self.accumulator.add_feature_side(moved_piece, from, ksq_them, them);
                    self.accumulator.remove_feature_side(moved_piece, to, ksq_them, them);
                    self.accumulator.add_feature_side(rook, rook_from, ksq_them, them);
                    self.accumulator.remove_feature_side(rook, rook_to, ksq_them, them);
                }
                _ => {
                    self.accumulator.add_feature_side(moved_piece, from, ksq_them, them);
                    self.accumulator.remove_feature_side(moved_piece, to, ksq_them, them);
                    if undo.captured != Piece::None {
                        self.accumulator.add_feature_side(undo.captured, to, ksq_them, them);
                    }
                }
            }
            self.refresh_accumulator_side(us);
        } else {
            let w_ksq = self.king_square(Color::White);
            let b_ksq = self.king_square(Color::Black);
            match move_type {
                MoveType::EnPassant => {
                    let cap_sq = Square::from_coords(to.file(), from.rank());
                    let cap_pawn = Piece::new(them, PieceType::Pawn);
                    self.accumulator.remove_feature(moved_piece, to, w_ksq, b_ksq);
                    self.accumulator.add_feature(cap_pawn, cap_sq, w_ksq, b_ksq);
                    self.accumulator.add_feature(moved_piece, from, w_ksq, b_ksq);
                }
                MoveType::Promotion => {
                    let pawn = Piece::new(us, PieceType::Pawn);
                    self.accumulator.remove_feature(moved_piece, to, w_ksq, b_ksq);
                    if undo.captured != Piece::None {
                        self.accumulator.add_feature(undo.captured, to, w_ksq, b_ksq);
                    }
                    self.accumulator.add_feature(pawn, from, w_ksq, b_ksq);
                }
                _ => {
                    self.accumulator.remove_feature(moved_piece, to, w_ksq, b_ksq);
                    if undo.captured != Piece::None {
                        self.accumulator.add_feature(undo.captured, to, w_ksq, b_ksq);
                    }
                    self.accumulator.add_feature(moved_piece, from, w_ksq, b_ksq);
                }
            }
        }
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
    use crate::movegen::{generate_legal_moves, generate_noisy_pseudo, generate_quiet_pseudo};
    use crate::eval::evaluate;

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
    fn debug_game_position() {
        let mut board = Board::default();
        for s in [
            "e2e4", "g8f6", "e4e5", "b8c6", "e5f6", "e7f6", "d2d4", "d7d5",
            "f1b5", "f8b4", "c1d2", "e8g8", "d2b4", "c6d4", "d1d4", "c8h3",
            "g2h3", "f8e8", "b5e8", "c7c5", "d4c5", "d8e8", "g1e2", "e8e6", "b1c3",
        ] {
            play(&mut board, s);
        }
        println!("side_to_move = {:?}", board.side_to_move);
        println!("eval (side to move) = {}", evaluate(&board));
        println!("is_draw = {}", board.is_draw());
        println!("upcoming_repetition = {}", board.upcoming_repetition());
        println!("is_repetition = {}", board.is_repetition());
        println!("halfmove = {}", board.halfmove_clock);

        let start = Board::default();
        println!("eval startpos (white to move) = {}", evaluate(&start));
        let noq = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1").unwrap();
        println!("eval white-no-queen (white to move) = {}", evaluate(&noq));
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
                let before = board.accumulator.vals;
                let undo = board.make_move(m);
                let mut check = board.clone();
                check.refresh_accumulator();
                assert_eq!(board.accumulator.vals, check.accumulator.vals, "make mismatch on {}", m);
                board.undo_move(m, undo);
                assert_eq!(board.accumulator.vals, before, "undo mismatch on {}", m);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.count;
            board.make_move(moves.moves[pick]);
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
