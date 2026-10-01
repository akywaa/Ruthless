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

#[repr(C, align(64))]
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_add_avx2(&mut self.vals[Color::White as usize], w_weights);
                    vec_add_avx2(&mut self.vals[Color::Black as usize], b_weights);
                    return;
                }
            }
        }

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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_sub_avx2(&mut self.vals[Color::White as usize], w_weights);
                    vec_sub_avx2(&mut self.vals[Color::Black as usize], b_weights);
                    return;
                }
            }
        }

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

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return evaluate_avx2(acc, us, them, net);
            }
        }
    }

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    let a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let w_ptr = weights.as_ptr() as *const __m256i;
    for i in 0..(HIDDEN_SIZE / 16) {
        let va = _mm256_load_si256(a_ptr.add(i));
        let vw = _mm256_load_si256(w_ptr.add(i));
        _mm256_store_si256(a_ptr.add(i), _mm256_add_epi16(va, vw));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_sub_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    let a_ptr = acc.as_mut_ptr() as *mut __m256i;
    let w_ptr = weights.as_ptr() as *const __m256i;
    for i in 0..(HIDDEN_SIZE / 16) {
        let va = _mm256_load_si256(a_ptr.add(i));
        let vw = _mm256_load_si256(w_ptr.add(i));
        _mm256_store_si256(a_ptr.add(i), _mm256_sub_epi16(va, vw));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_avx2(acc: &Accumulator, us: usize, them: usize, net: &'static Network) -> i32 {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA);

    let mut sum_vec = _mm256_setzero_si256();

    forward_side_avx2(&acc.vals[us], &net.output_weights[0..HIDDEN_SIZE], zero, qa, &mut sum_vec);
    forward_side_avx2(&acc.vals[them], &net.output_weights[HIDDEN_SIZE..2 * HIDDEN_SIZE], zero, qa, &mut sum_vec);

    let low128 = _mm256_castsi256_si128(sum_vec);
    let high128 = _mm256_extracti128_si256(sum_vec, 1);
    let sum128 = _mm_add_epi32(low128, high128);
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_00_01));
    let mut output = _mm_cvtsi128_si32(sum32);

    output /= i32::from(QA);
    output += i32::from(net.output_bias);
    output *= SCALE;
    output /= i32::from(QA) * i32::from(QB);

    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_side_avx2(
    vals: &[i16; HIDDEN_SIZE],
    weights: &[i16],
    zero: std::arch::x86_64::__m256i,
    qa: std::arch::x86_64::__m256i,
    sum_vec: &mut std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;

    let v_ptr = vals.as_ptr() as *const __m256i;
    let w_ptr = weights.as_ptr() as *const __m256i;

    for i in 0..(HIDDEN_SIZE / 16) {
        let v = _mm256_load_si256(v_ptr.add(i));
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);

        let low_16 = _mm256_castsi256_si128(clamped);
        let high_16 = _mm256_extracti128_si256(clamped, 1);

        let y_low = _mm256_cvtepi16_epi32(low_16);
        let y_high = _mm256_cvtepi16_epi32(high_16);

        // SCReLU: y * y
        let sq_low = _mm256_mullo_epi32(y_low, y_low);
        let sq_high = _mm256_mullo_epi32(y_high, y_high);

        let w = _mm256_loadu_si256(w_ptr.add(i));
        let w_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
        let w_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));

        // (y * y) * weight
        let p_low = _mm256_mullo_epi32(sq_low, w_low);
        let p_high = _mm256_mullo_epi32(sq_high, w_high);

        *sum_vec = _mm256_add_epi32(*sum_vec, p_low);
        *sum_vec = _mm256_add_epi32(*sum_vec, p_high);
    }
}