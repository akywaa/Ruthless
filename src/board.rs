use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square, COLOR_NB, PIECE_NB, SQUARE_NB};

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
        }
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
                s if s.len() == 2 => {
                    let bytes = s.as_bytes();
                    let file = bytes[0].wrapping_sub(b'a');
                    let rank = bytes[1].wrapping_sub(b'1');
                    if file < 8 && rank < 8 {
                        Square::from_coords(file, rank)
                    } else {
                        Square::None
                    }
                }
                _ => Square::None,
            };
        }

        if parts.len() > 4 {
            board.halfmove_clock = parts[4].parse().unwrap_or(0);
        }

        if parts.len() > 5 {
            board.fullmove_number = parts[5].parse().unwrap_or(1);
        }

        Ok(board)
    }

    #[inline(always)]
    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        self.pieces[piece].set(sq);
        self.occupied_co[piece.color()].set(sq);
        self.occupied.set(sq);
        self.piece_on[sq] = piece;
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, sq: Square) -> Piece {
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

    pub fn make_move(&mut self, m: Move) -> UndoState {
        let from = m.from();
        let to = m.to();
        let move_type = m.move_type();
        let us = self.side_to_move;
        let them = !us;
        let moving_piece = self.piece_on[from];

        let undo = UndoState {
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            captured: self.piece_on[to],
        };

        self.ep_square = Square::None;
        self.halfmove_clock += 1;

        if moving_piece.piece_type() == PieceType::Pawn || undo.captured != Piece::None {
            self.halfmove_clock = 0;
        }

        self.remove_piece(from);

        match move_type {
            MoveType::Normal => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                }
                self.put_piece(moving_piece, to);

                if moving_piece.piece_type() == PieceType::Pawn && ((from as i8) - (to as i8)).abs() == 16 {
                    self.ep_square = Square::new(((from as u8) + (to as u8)) / 2);
                }
            }
            MoveType::Castling => {
                self.put_piece(moving_piece, to);
                let (rook_from, rook_to) = match to {
                    Square::G1 => (Square::H1, Square::F1),
                    Square::C1 => (Square::A1, Square::D1),
                    Square::G8 => (Square::H8, Square::F8),
                    Square::C8 => (Square::A8, Square::D8),
                    _ => unreachable!(),
                };
                let rook = self.remove_piece(rook_from);
                self.put_piece(rook, rook_to);
            }
            MoveType::EnPassant => {
                let cap_sq = Square::from_coords(to.file(), from.rank());
                self.remove_piece(cap_sq);
                self.put_piece(moving_piece, to);
            }
            MoveType::Promotion => {
                if undo.captured != Piece::None {
                    self.remove_piece(to);
                }
                let promo_piece = Piece::new(us, m.promo_type());
                self.put_piece(promo_piece, to);
            }
        }

        self.castling_rights &= CASTLING_RIGHTS_MASK[from as usize] & CASTLING_RIGHTS_MASK[to as usize];

        if us == Color::Black {
            self.fullmove_number += 1;
        }
        self.side_to_move = them;

        undo
    }

    pub fn undo_move(&mut self, m: Move, undo: UndoState) {
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
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(STARTING_FEN).unwrap()
    }
}