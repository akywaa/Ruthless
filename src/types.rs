use std::ops::{Index, IndexMut, Not};

pub const SQUARE_NB: usize = 64;
pub const PIECE_NB: usize = 12;
#[allow(dead_code)]
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
        if (self as usize) == 12 {
            Color::White
        } else {
            Color::from_index((self as usize) / 6)
        }
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        if (self as usize) == 12 {
            PieceType::None
        } else {
            PieceType::from_index((self as usize) % 6)
        }
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
        if val < 64 {
            unsafe { std::mem::transmute(val) }
        } else {
            Square::None
        }
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

    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        (self as u8) < 64
    }

    pub fn from_str(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 2 {
            return None;
        }
        let file = b[0].wrapping_sub(b'a');
        let rank = b[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Self::from_coords(file, rank))
        } else {
            None
        }
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.is_valid() {
            return write!(f, "-");
        }
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
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

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Move::NULL {
            return write!(f, "0000");
        }
        let promo = match self.move_type() {
            MoveType::Promotion => match self.promo_type() {
                PieceType::Knight => "n",
                PieceType::Bishop => "b",
                PieceType::Rook => "r",
                PieceType::Queen => "q",
                _ => "",
            },
            _ => "",
        };
        write!(f, "{}{}{}", self.from(), self.to(), promo)
    }
}

#[derive(Copy, Clone)]
pub struct MoveList {
    pub moves: [Move; 256],
    pub count: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            moves: [Move::NULL; 256],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        self.moves[self.count] = m;
        self.count += 1;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.count]
    }
}
