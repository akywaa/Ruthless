use crate::types::{Color, Piece, Square};

pub const HIDDEN_SIZE: usize = 128;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

#[repr(C, align(64))]
pub struct AccumulatorRaw {
    pub vals: [i16; HIDDEN_SIZE],
}

#[repr(C)]
pub struct Network {
    feature_weights: [AccumulatorRaw; 768],
    feature_bias: AccumulatorRaw,
    output_weights: [i16; 2 * HIDDEN_SIZE],
    output_bias: i16,
}

static NETWORK_BYTES: &[u8] = include_bytes!("../resources/ruthless.bin");

#[inline(always)]
pub fn network() -> &'static Network {
    unsafe { &*(NETWORK_BYTES.as_ptr() as *const Network) }
}

#[derive(Copy, Clone)]
pub struct Accumulator {
    pub vals: [[i16; HIDDEN_SIZE]; 2],
}

impl Accumulator {
    #[inline(always)]
    pub fn new() -> Self {
        let net = network();
        Self {
            vals: [net.feature_bias.vals, net.feature_bias.vals],
        }
    }

    #[inline(always)]
    pub fn add_feature(&mut self, piece: Piece, sq: Square) {
        let net = network();
        let (w_idx, b_idx) = feature_indices(piece, sq);

        let w_weights = &net.feature_weights[w_idx].vals;
        let b_weights = &net.feature_weights[b_idx].vals;

        for i in 0..HIDDEN_SIZE {
            self.vals[Color::White as usize][i] += w_weights[i];
            self.vals[Color::Black as usize][i] += b_weights[i];
        }
    }

    #[inline(always)]
    pub fn remove_feature(&mut self, piece: Piece, sq: Square) {
        let net = network();
        let (w_idx, b_idx) = feature_indices(piece, sq);

        let w_weights = &net.feature_weights[w_idx].vals;
        let b_weights = &net.feature_weights[b_idx].vals;

        for i in 0..HIDDEN_SIZE {
            self.vals[Color::White as usize][i] -= w_weights[i];
            self.vals[Color::Black as usize][i] -= b_weights[i];
        }
    }
}

#[inline(always)]
fn feature_indices(piece: Piece, sq: Square) -> (usize, usize) {
    let p_idx = piece as usize;
    let sq_idx = sq as usize;
    let white_idx = p_idx * 64 + sq_idx;

    let flipped_piece = match piece {
        Piece::WhitePawn => Piece::BlackPawn,
        Piece::WhiteKnight => Piece::BlackKnight,
        Piece::WhiteBishop => Piece::BlackBishop,
        Piece::WhiteRook => Piece::BlackRook,
        Piece::WhiteQueen => Piece::BlackQueen,
        Piece::WhiteKing => Piece::BlackKing,
        Piece::BlackPawn => Piece::WhitePawn,
        Piece::BlackKnight => Piece::WhiteKnight,
        Piece::BlackBishop => Piece::WhiteBishop,
        Piece::BlackRook => Piece::WhiteRook,
        Piece::BlackQueen => Piece::WhiteQueen,
        Piece::BlackKing => Piece::WhiteKing,
        Piece::None => Piece::None,
    } as usize;

    let flipped_sq = sq_idx ^ 56;
    let black_idx = flipped_piece * 64 + flipped_sq;

    (white_idx, black_idx)
}

#[inline(always)]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

#[inline(always)]
pub fn evaluate(acc: &Accumulator, side_to_move: Color) -> i32 {
    let net = network();
    let us = side_to_move as usize;
    let them = (!side_to_move) as usize;

    let mut output = 0i32;

    for i in 0..HIDDEN_SIZE {
        output += screlu(acc.vals[us][i]) * i32::from(net.output_weights[i]);
    }

    for i in 0..HIDDEN_SIZE {
        output += screlu(acc.vals[them][i]) * i32::from(net.output_weights[HIDDEN_SIZE + i]);
    }

    output /= i32::from(QA);
    output += i32::from(net.output_bias);
    output *= SCALE;
    output /= i32::from(QA) * i32::from(QB);

    output
}