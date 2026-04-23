use crate::types::{Color, Piece, Square};
use std::sync::OnceLock;

pub const INPUT_NB: usize = 768;
pub const L1_SIZE: usize = 256;
pub const SCALE: i32 = 64;

#[derive(Clone)]
pub struct Network {
    pub feature_weights: Vec<i16>,
    pub feature_bias: [i16; L1_SIZE],
    pub output_weights: [i16; L1_SIZE * 2],
    pub output_bias: i16,
}

static NETWORK: OnceLock<Network> = OnceLock::new();

#[inline(always)]
pub fn network() -> &'static Network {
    NETWORK.get_or_init(Network::init_default)
}

#[derive(Copy, Clone)]
pub struct Accumulator {
    pub vals: [[i16; L1_SIZE]; 2],
}

impl Accumulator {
    pub fn new() -> Self {
        let net = network();
        Self {
            vals: [net.feature_bias; 2],
        }
    }

    #[inline(always)]
    pub fn add_feature(&mut self, piece: Piece, sq: Square) {
        let net = network();
        let (w_idx, b_idx) = feature_indices(piece, sq);

        let w_offset = w_idx * L1_SIZE;
        let b_offset = b_idx * L1_SIZE;

        for i in 0..L1_SIZE {
            self.vals[Color::White as usize][i] += net.feature_weights[w_offset + i];
            self.vals[Color::Black as usize][i] += net.feature_weights[b_offset + i];
        }
    }

    #[inline(always)]
    pub fn remove_feature(&mut self, piece: Piece, sq: Square) {
        let net = network();
        let (w_idx, b_idx) = feature_indices(piece, sq);

        let w_offset = w_idx * L1_SIZE;
        let b_offset = b_idx * L1_SIZE;

        for i in 0..L1_SIZE {
            self.vals[Color::White as usize][i] -= net.feature_weights[w_offset + i];
            self.vals[Color::Black as usize][i] -= net.feature_weights[b_offset + i];
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
pub fn evaluate(acc: &Accumulator, side_to_move: Color) -> i32 {
    let net = network();
    let us = side_to_move as usize;
    let them = (!side_to_move) as usize;

    let mut sum = net.output_bias as i32;

    for i in 0..L1_SIZE {
        let val = acc.vals[us][i].clamp(0, 255) as i32;
        sum += val * (net.output_weights[i] as i32);
    }

    for i in 0..L1_SIZE {
        let val = acc.vals[them][i].clamp(0, 255) as i32;
        sum += val * (net.output_weights[L1_SIZE + i] as i32);
    }

    sum / SCALE
}

impl Network {
    fn init_default() -> Self {
        let mut feature_weights = vec![0i16; INPUT_NB * L1_SIZE];
        let feature_bias = [0i16; L1_SIZE];
        let mut output_weights = [0i16; L1_SIZE * 2];
        let output_bias = 0i16;

        let piece_base_vals: [i16; 6] = [100, 320, 330, 500, 900, 0];

        for p in 0..12 {
            let pt = p % 6;
            let is_white = p < 6;
            let val = piece_base_vals[pt];

            for sq in 0..64 {
                let idx = p * 64 + sq;
                let offset = idx * L1_SIZE;

                let bonus = match pt {
                    0 => {
                        let r = (sq / 8) as i16;
                        if is_white { r * 8 } else { (7 - r) * 8 }
                    }
                    1 | 2 => {
                        let f = (sq % 8) as i16;
                        let r = (sq / 8) as i16;
                        let center_dist = (3 - f).abs().max((4 - f).abs()) + (3 - r).abs().max((4 - r).abs());
                        16 - center_dist * 3
                    }
                    _ => 0,
                };

                let assigned_val = if is_white { val + bonus } else { -(val + bonus) };

                for neuron in 0..L1_SIZE {
                    if neuron == (idx % L1_SIZE) {
                        feature_weights[offset + neuron] = assigned_val / 4;
                    }
                }
            }
        }

        for i in 0..L1_SIZE {
            output_weights[i] = 16;
            output_weights[L1_SIZE + i] = -16;
        }

        Self {
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        }
    }
}
