use crate::board::Board;
use crate::types::{Color, Piece, Square};

pub const HIDDEN_SIZE: usize = 1024;
pub const NUM_INPUT_BUCKETS: usize = 10;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

#[repr(C, align(64))]
pub struct AccumulatorRaw {
    pub vals: [i16; HIDDEN_SIZE],
}

#[repr(C)]
pub struct Network {
    feature_weights: [AccumulatorRaw; 768 * NUM_INPUT_BUCKETS],
    feature_bias: AccumulatorRaw,
    output_weights: [[i16; 2 * HIDDEN_SIZE]; NUM_OUTPUT_BUCKETS],
    output_bias: [i16; NUM_OUTPUT_BUCKETS],
}

static NETWORK_BYTES: &[u8] = include_bytes!("../resources/ruthless.bin");

static NETWORK_ALIGNED: std::sync::OnceLock<&'static Network> = std::sync::OnceLock::new();

#[inline(always)]
pub fn network() -> &'static Network {
    *NETWORK_ALIGNED.get_or_init(|| {
        // include_bytes! gives no alignment guarantee; the AVX2 path needs a
        // 64-byte-aligned Network, so copy into an aligned, never-freed block.
        let layout = std::alloc::Layout::from_size_align(NETWORK_BYTES.len(), 64)
            .expect("invalid network layout");
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "out of memory allocating network");
        unsafe {
            std::ptr::copy_nonoverlapping(NETWORK_BYTES.as_ptr(), ptr, NETWORK_BYTES.len());
            &*(ptr as *const Network)
        }
    })
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
    pub fn add_feature_side(&mut self, piece: Piece, sq: Square, ksq: Square, color: Color) {
        let net = network();
        let idx = feature_index_side(piece, sq, ksq, color);
        let weights = &net.feature_weights[idx].vals;
        let side = color as usize;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_add_avx2(&mut self.vals[side], weights);
                    return;
                }
            }
        }

        for i in 0..HIDDEN_SIZE {
            self.vals[side][i] += weights[i];
        }
    }

    #[inline(always)]
    pub fn remove_feature_side(&mut self, piece: Piece, sq: Square, ksq: Square, color: Color) {
        let net = network();
        let idx = feature_index_side(piece, sq, ksq, color);
        let weights = &net.feature_weights[idx].vals;
        let side = color as usize;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    vec_sub_avx2(&mut self.vals[side], weights);
                    return;
                }
            }
        }

        for i in 0..HIDDEN_SIZE {
            self.vals[side][i] -= weights[i];
        }
    }

    #[inline(always)]
    pub fn add_feature(&mut self, piece: Piece, sq: Square, w_ksq: Square, b_ksq: Square) {
        self.add_feature_side(piece, sq, w_ksq, Color::White);
        self.add_feature_side(piece, sq, b_ksq, Color::Black);
    }

    #[inline(always)]
    pub fn remove_feature(&mut self, piece: Piece, sq: Square, w_ksq: Square, b_ksq: Square) {
        self.remove_feature_side(piece, sq, w_ksq, Color::White);
        self.remove_feature_side(piece, sq, b_ksq, Color::Black);
    }

    pub fn refresh_side(&mut self, piece_on: &[Piece; 64], ksq: Square, color: Color) {
        let net = network();
        let side = color as usize;
        self.vals[side] = net.feature_bias.vals;

        for sq in 0..64 {
            let piece = piece_on[sq];
            if piece != Piece::None {
                self.add_feature_side(piece, Square::new(sq as u8), ksq, color);
            }
        }
    }
}

#[inline(always)]
pub fn king_bucket(sq: Square) -> usize {
    let file = sq.file();
    let rank = sq.rank();
    let mirrored_file = if file > 3 { 7 - file } else { file };
    BUCKET_LAYOUT[(rank * 4 + mirrored_file) as usize]
}

#[inline(always)]
pub fn feature_index_side(piece: Piece, sq: Square, ksq: Square, color: Color) -> usize {
    let p_idx = piece as usize;
    let sq_idx = sq as usize;

    if color == Color::White {
        let flip = if ksq.file() > 3 { 7 } else { 0 };
        king_bucket(ksq) * 768 + p_idx * 64 + (sq_idx ^ flip)
    } else {
        let flipped_ksq = Square::new((ksq as u8) ^ 56);
        let flip = if flipped_ksq.file() > 3 { 7 } else { 0 };
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
        let flipped_sq = sq_idx ^ 56 ^ flip;
        king_bucket(flipped_ksq) * 768 + flipped_piece * 64 + flipped_sq
    }
}

#[inline(always)]
pub fn output_bucket(board: &Board) -> usize {
    ((board.occupied.count() as usize - 2) / 4).min(7)
}

#[inline(always)]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let net = network();
    let us = board.side_to_move as usize;
    let them = (!board.side_to_move) as usize;
    let bucket = output_bucket(board);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return evaluate_avx2(&board.accumulator, us, them, bucket, net);
            }
        }
    }

    let mut output = 0i32;

    for i in 0..HIDDEN_SIZE {
        output += screlu(board.accumulator.vals[us][i]) * i32::from(net.output_weights[bucket][i]);
    }

    for i in 0..HIDDEN_SIZE {
        output += screlu(board.accumulator.vals[them][i]) * i32::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
    }

    output /= i32::from(QA);
    output += i32::from(net.output_bias[bucket]);
    output *= SCALE;
    output /= i32::from(QA) * i32::from(QB);

    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = weights.as_ptr() as *const __m256i;
        for i in 0..(HIDDEN_SIZE / 16) {
            let va = _mm256_load_si256(a_ptr.add(i));
            let vw = _mm256_load_si256(w_ptr.add(i));
            _mm256_store_si256(a_ptr.add(i), _mm256_add_epi16(va, vw));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_sub_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = weights.as_ptr() as *const __m256i;
        for i in 0..(HIDDEN_SIZE / 16) {
            let va = _mm256_load_si256(a_ptr.add(i));
            let vw = _mm256_load_si256(w_ptr.add(i));
            _mm256_store_si256(a_ptr.add(i), _mm256_sub_epi16(va, vw));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_avx2(
    acc: &Accumulator,
    us: usize,
    them: usize,
    bucket: usize,
    net: &'static Network,
) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA);

        let mut sum_vec = _mm256_setzero_si256();

        forward_side_avx2(&acc.vals[us], &net.output_weights[bucket][0..HIDDEN_SIZE], zero, qa, &mut sum_vec);
        forward_side_avx2(&acc.vals[them], &net.output_weights[bucket][HIDDEN_SIZE..2 * HIDDEN_SIZE], zero, qa, &mut sum_vec);

        let low128 = _mm256_castsi256_si128(sum_vec);
        let high128 = _mm256_extracti128_si256(sum_vec, 1);
        let sum128 = _mm_add_epi32(low128, high128);
        let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_00_01));
        let mut output = _mm_cvtsi128_si32(sum32);

        output /= i32::from(QA);
        output += i32::from(net.output_bias[bucket]);
        output *= SCALE;
        output /= i32::from(QA) * i32::from(QB);

        output
    }
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

    unsafe {
        let v_ptr = vals.as_ptr() as *const __m256i;
        let w_ptr = weights.as_ptr() as *const __m256i;

        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_load_si256(v_ptr.add(i));
            let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);

            let low_16 = _mm256_castsi256_si128(clamped);
            let high_16 = _mm256_extracti128_si256(clamped, 1);

            let y_low = _mm256_cvtepi16_epi32(low_16);
            let y_high = _mm256_cvtepi16_epi32(high_16);

            let sq_low = _mm256_mullo_epi32(y_low, y_low);
            let sq_high = _mm256_mullo_epi32(y_high, y_high);

            let w = _mm256_loadu_si256(w_ptr.add(i));
            let w_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
            let w_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));

            let p_low = _mm256_mullo_epi32(sq_low, w_low);
            let p_high = _mm256_mullo_epi32(sq_high, w_high);

            *sum_vec = _mm256_add_epi32(*sum_vec, p_low);
            *sum_vec = _mm256_add_epi32(*sum_vec, p_high);
        }
    }
}
