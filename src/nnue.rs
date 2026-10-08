use crate::board::Board;
use crate::types::{Color, Piece, Square};

pub const HIDDEN_SIZE: usize = 512;
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                vec_add_avx2(&mut self.vals[side], weights);
                return;
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..HIDDEN_SIZE {
                self.vals[side][i] += weights[i];
            }
        }
    }

    #[inline(always)]
    pub fn remove_feature_side(&mut self, piece: Piece, sq: Square, ksq: Square, color: Color) {
        let net = network();
        let idx = feature_index_side(piece, sq, ksq, color);
        let weights = &net.feature_weights[idx].vals;
        let side = color as usize;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe {
                vec_sub_avx2(&mut self.vals[side], weights);
                return;
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..HIDDEN_SIZE {
                self.vals[side][i] -= weights[i];
            }
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
pub fn oriented_piece(piece: Piece, view: Color) -> usize {
    let pc_color = piece.color();
    let pt = piece.piece_type() as usize;
    if pc_color == view {
        pt
    } else {
        pt + 6
    }
}

#[inline(always)]
pub fn feature_index_side(piece: Piece, sq: Square, ksq: Square, view: Color) -> usize {
    let (oriented_sq, oriented_ksq) = if view == Color::White {
        (sq, ksq)
    } else {
        (Square::new((sq as u8) ^ 56), Square::new((ksq as u8) ^ 56))
    };

    let flip = if oriented_ksq.file() > 3 { 7 } else { 0 };
    let final_sq = (oriented_sq as usize) ^ flip;
    let piece_idx = oriented_piece(piece, view);

    king_bucket(oriented_ksq) * 768 + piece_idx * 64 + final_sq
}

#[inline(always)]
pub fn output_bucket(board: &Board) -> usize {
    ((board.occupied.count() as usize - 2) / 4).min(7)
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let net = network();
    let bucket = output_bucket(board);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            let us = board.side_to_move as usize;
            let them = us ^ 1;
            return evaluate_avx2(&board.accumulator, us, them, bucket, net);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut output = 0i64;

        // First weight half pairs with the side to move, the second half with the
        // opponent (dual-perspective training).
        let us = board.side_to_move as usize;
        let them = us ^ 1;
        for i in 0..HIDDEN_SIZE {
            output += screlu(board.accumulator.vals[us][i]) as i64 * i64::from(net.output_weights[bucket][i]);
        }

        for i in 0..HIDDEN_SIZE {
            output += screlu(board.accumulator.vals[them][i]) as i64 * i64::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
        }

        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);

        output as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;

    fn eval_ordering(board: &Board, swap: bool) -> i32 {
        let net = network();
        let us = board.side_to_move as usize;
        let them = (!board.side_to_move) as usize;
        let bucket = output_bucket(board);
        let (a, b) = if swap { (them, us) } else { (us, them) };
        let mut output = 0i64;
        for i in 0..HIDDEN_SIZE {
            output += screlu(board.accumulator.vals[a][i]) as i64 * i64::from(net.output_weights[bucket][i]);
        }
        for i in 0..HIDDEN_SIZE {
            output += screlu(board.accumulator.vals[b][i]) as i64 * i64::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
        }
        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);
        output as i32
    }

    #[test]
    fn debug_eval_orderings() {
        let cases = [
            ("K vs K w", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            ("K w vs K b", "4k3/8/8/8/8/8/8/4K3 b - - 0 1"),
            ("K+Q w", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
            ("startpos w", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("midgame white +R", "r1bqk2r/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
        ];
        for (name, fen) in cases {
            let board = Board::from_fen(fen).unwrap();
            println!(
                "{}: normal={} swapped={}",
                name,
                eval_ordering(&board, false),
                eval_ordering(&board, true)
            );
        }

        let net = network();
        let k = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let kq = Board::from_fen("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let changed = k.accumulator.vals[0]
            .iter()
            .zip(kq.accumulator.vals[0].iter())
            .filter(|(a, b)| a != b)
            .count();
        println!("hidden units changed by adding white Q: {}", changed);
        let mut ow_max = 0i32;
        let mut ow_gt100 = 0usize;
        for b in 0..NUM_OUTPUT_BUCKETS {
            for &w in net.output_weights[b].iter() {
                let a = (w as i32).abs();
                ow_max = ow_max.max(a);
                if a > 100 {
                    ow_gt100 += 1;
                }
            }
        }
        println!("output_weights: maxabs={} count>100={} total={}", ow_max, ow_gt100, NUM_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE);
        let mut fw_max = 0i32;
        let mut fw_mean = 0i64;
        let mut fw_count = 0i64;
        for f in 0..768 * NUM_INPUT_BUCKETS {
            for &w in net.feature_weights[f].vals.iter() {
                let a = (w as i32).abs();
                fw_max = fw_max.max(a);
                fw_mean += a as i64;
                fw_count += 1;
            }
        }
        println!("feature_weights: maxabs={} meanabs={}", fw_max, fw_mean / fw_count);
        println!("output_bias = {:?}", net.output_bias);
        println!("feature_bias[0..6] = {:?}", &net.feature_bias.vals[0..6]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar_random_walk() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut board = Board::default();
        let mut rng: u64 = 0xC0FF_EE00_1234_5678;
        for _ in 0..400 {
            let moves = generate_legal_moves(&mut board);
            if moves.count == 0 {
                break;
            }
            for &m in moves.as_slice() {
                let undo = board.make_move(m);
                let net = network();
                let bucket = output_bucket(&board);
                let us = board.side_to_move as usize;
                let them = us ^ 1;
                // Scalar reference in the same [side-to-move, non-side-to-move]
                // ordering and perspective used by the production evaluate().
                let mut out = 0i64;
                for i in 0..HIDDEN_SIZE {
                    out += screlu(board.accumulator.vals[us][i]) as i64 * i64::from(net.output_weights[bucket][i]);
                }
                for i in 0..HIDDEN_SIZE {
                    out += screlu(board.accumulator.vals[them][i]) as i64 * i64::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
                }
                out /= i64::from(QA);
                out += i64::from(net.output_bias[bucket]);
                out *= SCALE as i64;
                out /= i64::from(QA) * i64::from(QB);
                let expected = out as i32;
                let got = unsafe { evaluate_avx2(&board.accumulator, us, them, bucket, net) };
                assert_eq!(got, expected, "AVX2 mismatch after {}", m);
                board.undo_move(m, undo);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.count;
            board.make_move(moves.moves[pick]);
        }
    }
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
        let mut acc64 = _mm256_setzero_si256();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m256i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m256i;
            let mut acc = _mm256_setzero_si256();

            for chunk in 0..(HIDDEN_SIZE / 16) {
                let v = _mm256_load_si256(v_ptr.add(chunk));
                let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);

                // Split both clamped and weights into 128-bit halves with the
                // same sign-extension so lane i pairs with weight i.
                let low_16 = _mm256_castsi256_si128(clamped);
                let high_16 = _mm256_extracti128_si256(clamped, 1);
                let y_low = _mm256_cvtepi16_epi32(low_16);
                let y_high = _mm256_cvtepi16_epi32(high_16);

                let w = _mm256_loadu_si256(w_ptr.add(chunk));
                let w_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
                let w_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));

                // Single screlu*w product per lane fits in i32 (screlu <= QA^2,
                // |w| bounded). Sign-extend to i64 and accumulate there so the
                // running total cannot overflow i32.
                let p_low = _mm256_mullo_epi32(_mm256_mullo_epi32(y_low, y_low), w_low);
                let p_high = _mm256_mullo_epi32(_mm256_mullo_epi32(y_high, y_high), w_high);
                acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p_low)));
                acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p_high)));

                // Low 128 bits already consumed; do the high halves.
                let p_low_hi = _mm256_extracti128_si256(p_low, 1);
                acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(p_low_hi));
                let p_high_hi = _mm256_extracti128_si256(p_high, 1);
                acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(p_high_hi));
            }

            acc64 = _mm256_add_epi64(acc64, acc);
        };

        evaluate_side(&acc.vals[us], 0);
        evaluate_side(&acc.vals[them], HIDDEN_SIZE);

        let mut arr = [0i64; 4];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, acc64);

        let mut output = 0i64;
        for i in 0..4 {
            output += arr[i];
        }

        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);

        output as i32
    }
}
