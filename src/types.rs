use std::ops::{Index, IndexMut, Not};

pub const SQUARE_NB: usize = 64;
pub const PIECE_NB: usize = 12;
pub const PIECE_TYPE_NB: usize = 6;
pub const COLOR_NB: usize = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::White,
            _ => Self::Black,
        }
    }
}

impl Not for Color {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

impl<T> Index<Color> for [T] {
    type Output = T;
    #[inline(always)]
    fn index(&self, color: Color) -> &Self::Output {
        &self[color as usize]
    }
}

impl<T> IndexMut<Color> for [T] {
    #[inline(always)]
    fn index_mut(&mut self, color: Color) -> &mut Self::Output {
        &mut self[color as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
    None = 6,
}

impl PieceType {
    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Pawn,
            1 => Self::Knight,
            2 => Self::Bishop,
            3 => Self::Rook,
            4 => Self::Queen,
            5 => Self::King,
            _ => Self::None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    WhitePawn = 0,
    WhiteKnight = 1,
    WhiteBishop = 2,
    WhiteRook = 3,
    WhiteQueen = 4,
    WhiteKing = 5,
    BlackPawn = 6,
    BlackKnight = 7,
    BlackBishop = 8,
    BlackRook = 9,
    BlackQueen = 10,
    BlackKing = 11,
    None = 12,
}

impl Piece {
    #[inline(always)]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        let val = (color as u8) * 6 + (piece_type as u8);
        unsafe { std::mem::transmute(val) }
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        Color::from_index((self as usize) / 6)
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::from_index((self as usize) % 6)
    }
}

impl<T> Index<Piece> for [T] {
    type Output = T;
    #[inline(always)]
    fn index(&self, piece: Piece) -> &Self::Output {
        &self[piece as usize]
    }
}

impl<T> IndexMut<Piece> for [T] {
    #[inline(always)]
    fn index_mut(&mut self, piece: Piece) -> &mut Self::Output {
        &mut self[piece as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Square {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    #[default]
    None = 64,
}

impl Square {
    #[inline(always)]
    pub const fn new(val: u8) -> Self {
        unsafe { std::mem::transmute(val) }
    }

    #[inline(always)]
    pub const fn from_coords(file: u8, rank: u8) -> Self {
        Self::new(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        (self as u8) & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        (self as u8) >> 3
    }
}

impl<T> Index<Square> for [T] {
    type Output = T;
    #[inline(always)]
    fn index(&self, sq: Square) -> &Self::Output {
        &self[sq as usize]
    }
}

impl<T> IndexMut<Square> for [T] {
    #[inline(always)]
    fn index_mut(&mut self, sq: Square) -> &mut Self::Output {
        &mut self[sq as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MoveType {
    Normal = 0,
    Castling = 1,
    EnPassant = 2,
    Promotion = 3,
}

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Self = Self(0);

    #[inline(always)]
    pub const fn new(from: Square, to: Square, promo: PieceType, move_type: MoveType) -> Self {
        let promo_bits = match promo {
            PieceType::Knight => 0,
            PieceType::Bishop => 1,
            PieceType::Rook => 2,
            PieceType::Queen => 3,
            _ => 0,
        };
        let val = (from as u16)
            | ((to as u16) << 6)
            | ((promo_bits as u16) << 12)
            | ((move_type as u16) << 14);
        Self(val)
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square::new((self.0 & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn promo_type(self) -> PieceType {
        match (self.0 >> 12) & 3 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            3 => PieceType::Queen,
            _ => PieceType::None,
        }
    }

    #[inline(always)]
    pub const fn move_type(self) -> MoveType {
        match (self.0 >> 14) & 3 {
            0 => MoveType::Normal,
            1 => MoveType::Castling,
            2 => MoveType::EnPassant,
            3 => MoveType::Promotion,
            _ => unreachable!(),
        }
    }
}