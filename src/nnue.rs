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
            let net = &*(ptr as *const Network);

            #[cfg(target_arch = "x86_64")]
            {
                // The AVX2 output layer multiplies clipped activations (0..=QA)
                // by output weights in 16-bit lanes, which is only exact while
                // |weight| * QA fits into an i16. Reject a net that would
                // silently produce wrong evaluations instead.
                if has_avx2() {
                    let mut max_abs = 0i32;
                    for bucket in &net.output_weights {
                        for &w in bucket.iter() {
                            max_abs = max_abs.max((w as i32).abs());
                        }
                    }
                    assert!(
                        max_abs * i32::from(QA) <= i32::from(i16::MAX),
                        "network output weights exceed the 16-bit SIMD range"
                    );
                }
            }

            net
        }
    })
}

#[repr(C, align(64))]
#[derive(Copy, Clone, Debug)]
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
        add_feature_weights(&mut self.vals[color as usize], &net.feature_weights[idx].vals);
    }

    #[inline(always)]
    pub fn remove_feature_side(&mut self, piece: Piece, sq: Square, ksq: Square, color: Color) {
        let net = network();
        let idx = feature_index_side(piece, sq, ksq, color);
        sub_feature_weights(&mut self.vals[color as usize], &net.feature_weights[idx].vals);
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

/// AVX2 support, cached after the first runtime check. Builds compiled with
/// AVX2 enabled answer `true` at compile time, so the vector code keeps being
/// inlined exactly as before; portable builds dispatch at runtime instead of
/// falling back to the scalar loops on every AVX2 machine.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        use std::sync::atomic::{AtomicU8, Ordering};

        // 0 = not checked yet, 1 = unsupported, 2 = supported.
        static AVX2: AtomicU8 = AtomicU8::new(0);

        match AVX2.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let supported = std::arch::is_x86_feature_detected!("avx2");
                AVX2.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
                supported
            }
        }
    }
}

#[inline(always)]
fn add_feature_weights(vals: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        unsafe { vec_add_avx512(vals, weights) };
        return;
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            unsafe { vec_add_avx2(vals, weights) };
            return;
        }
        for (dst, &w) in vals.iter_mut().zip(weights.iter()) {
            *dst += w;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vec_add_neon(vals, weights) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for (dst, &w) in vals.iter_mut().zip(weights.iter()) {
            *dst += w;
        }
    }
}

#[inline(always)]
fn sub_feature_weights(vals: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        unsafe { vec_sub_avx512(vals, weights) };
        return;
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            unsafe { vec_sub_avx2(vals, weights) };
            return;
        }
        for (dst, &w) in vals.iter_mut().zip(weights.iter()) {
            *dst -= w;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vec_sub_neon(vals, weights) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for (dst, &w) in vals.iter_mut().zip(weights.iter()) {
            *dst -= w;
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
pub fn needs_accumulator_refresh(from: Square, to: Square, color: Color) -> bool {
    let (o_from, o_to) = if color == Color::White {
        (from, to)
    } else {
        (Square::new((from as u8) ^ 56), Square::new((to as u8) ^ 56))
    };

    king_bucket(o_from) != king_bucket(o_to) || (o_from.file() > 3) != (o_to.file() > 3)
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

#[inline(always)]
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

/// Scalar reference output layer. Always compiled: it is the fallback for
/// machines without AVX2 and the reference the SIMD kernels are tested against.
fn evaluate_scalar(
    acc: &Accumulator,
    us: usize,
    them: usize,
    bucket: usize,
    net: &'static Network,
) -> i32 {
    let mut output = 0i64;

    // First weight half pairs with the side to move, the second half with the
    // opponent (dual-perspective training).
    for i in 0..HIDDEN_SIZE {
        output += screlu(acc.vals[us][i]) as i64 * i64::from(net.output_weights[bucket][i]);
    }

    for i in 0..HIDDEN_SIZE {
        output += screlu(acc.vals[them][i]) as i64 * i64::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
    }

    output /= i64::from(QA);
    output += i64::from(net.output_bias[bucket]);
    output *= SCALE as i64;
    output /= i64::from(QA) * i64::from(QB);

    output as i32
}

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let net = network();
    let bucket = output_bucket(board);
    let acc = &board.accumulators[board.acc_index];
    let us = board.side_to_move as usize;
    let them = us ^ 1;
    evaluate_acc(acc, us, them, bucket, net)
}

#[inline(always)]
fn evaluate_acc(
    acc: &Accumulator,
    us: usize,
    them: usize,
    bucket: usize,
    net: &'static Network,
) -> i32 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        return unsafe { evaluate_avx512(acc, us, them, bucket, net) };
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            return unsafe { evaluate_avx2(acc, us, them, bucket, net) };
        }
        return evaluate_scalar(acc, us, them, bucket, net);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { evaluate_neon(acc, us, them, bucket, net) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        evaluate_scalar(acc, us, them, bucket, net)
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
        let acc = &board.accumulators[board.acc_index];
        let mut output = 0i64;
        for i in 0..HIDDEN_SIZE {
            output += screlu(acc.vals[a][i]) as i64 * i64::from(net.output_weights[bucket][i]);
        }
        for i in 0..HIDDEN_SIZE {
            output += screlu(acc.vals[b][i]) as i64 * i64::from(net.output_weights[bucket][HIDDEN_SIZE + i]);
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
        let changed = k.accumulators[k.acc_index].vals[0]
            .iter()
            .zip(kq.accumulators[kq.acc_index].vals[0].iter())
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
                let acc = &board.accumulators[board.acc_index];
                // Scalar reference in the same [side-to-move, non-side-to-move]
                // ordering and perspective used by the production evaluate().
                let expected = evaluate_scalar(acc, us, them, bucket, net);
                let got = unsafe { evaluate_avx2(acc, us, them, bucket, net) };
                assert_eq!(got, expected, "AVX2 mismatch after {}", m);
                board.undo_move(m, undo);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.count;
            board.make_move(moves.get(pick));
            // Collapse the accumulator stack; otherwise the walk overflows the
            // 256-ply stack and panics at board.rs make_move.
            board.reset_accumulator_stack();
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
        // Eight i32 lanes. Each lane only ever sums a few dozen products of at
        // most QA * QA * 128, so i32 cannot overflow here.
        let mut sum = _mm256_setzero_si256();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m256i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m256i;

            for chunk in 0..(HIDDEN_SIZE / 16) {
                let v = _mm256_load_si256(v_ptr.add(chunk));
                let x = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
                let w = _mm256_loadu_si256(w_ptr.add(chunk));

                // x is in 0..=QA and |w| * QA fits in an i16 (checked when the
                // network is loaded), so this 16-bit product is exact.
                let xw = _mm256_mullo_epi16(x, w);
                // madd sums adjacent pairs into i32 lanes: x * (x * w) = x^2 * w.
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, xw));
            }
        };

        evaluate_side(&acc.vals[us], 0);
        evaluate_side(&acc.vals[them], HIDDEN_SIZE);

        let mut arr = [0i32; 8];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut _, sum);

        let mut output = 0i64;
        for &lane in arr.iter() {
            output += i64::from(lane);
        }

        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);

        output as i32
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_add_avx512(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr() as *mut __m512i;
        let w_ptr = weights.as_ptr() as *const __m512i;
        for i in 0..(HIDDEN_SIZE / 32) {
            let va = _mm512_loadu_si512(a_ptr.add(i));
            let vw = _mm512_loadu_si512(w_ptr.add(i));
            _mm512_storeu_si512(a_ptr.add(i), _mm512_add_epi16(va, vw));
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_sub_avx512(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr() as *mut __m512i;
        let w_ptr = weights.as_ptr() as *const __m512i;
        for i in 0..(HIDDEN_SIZE / 32) {
            let va = _mm512_loadu_si512(a_ptr.add(i));
            let vw = _mm512_loadu_si512(w_ptr.add(i));
            _mm512_storeu_si512(a_ptr.add(i), _mm512_sub_epi16(va, vw));
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn evaluate_avx512(
    acc: &Accumulator,
    us: usize,
    them: usize,
    bucket: usize,
    net: &'static Network,
) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm512_setzero_si512();
        let qa = _mm512_set1_epi16(QA);
        let mut acc64 = _mm512_setzero_si512();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m512i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m512i;
            let mut l_acc = _mm512_setzero_si512();

            for chunk in 0..(HIDDEN_SIZE / 32) {
                let v = _mm512_loadu_si512(v_ptr.add(chunk));
                let clamped = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);

                let low_16 = _mm512_castsi512_si256(clamped);
                let high_16 = _mm512_extracti64x4_epi64::<1>(clamped);
                let y_low = _mm512_cvtepi16_epi32(low_16);
                let y_high = _mm512_cvtepi16_epi32(high_16);

                let w = _mm512_loadu_si512(w_ptr.add(chunk));
                let w_low = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w));
                let w_high = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(w));

                let p_low = _mm512_mullo_epi32(_mm512_mullo_epi32(y_low, y_low), w_low);
                let p_high = _mm512_mullo_epi32(_mm512_mullo_epi32(y_high, y_high), w_high);

                l_acc = _mm512_add_epi64(l_acc, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p_low)));
                l_acc = _mm512_add_epi64(
                    l_acc,
                    _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(p_low)),
                );
                l_acc = _mm512_add_epi64(
                    l_acc,
                    _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p_high)),
                );
                l_acc = _mm512_add_epi64(
                    l_acc,
                    _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(p_high)),
                );
            }

            acc64 = _mm512_add_epi64(acc64, l_acc);
        };

        evaluate_side(&acc.vals[us], 0);
        evaluate_side(&acc.vals[them], HIDDEN_SIZE);

        let res = _mm512_reduce_add_epi64(acc64);

        let mut output = res;
        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);

        output as i32
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_add_neon(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::aarch64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();
        for i in (0..HIDDEN_SIZE).step_by(8) {
            let va = vld1q_s16(a_ptr.add(i));
            let vw = vld1q_s16(w_ptr.add(i));
            vst1q_s16(a_ptr.add(i), vaddq_s16(va, vw));
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_sub_neon(acc: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    use std::arch::aarch64::*;
    unsafe {
        let a_ptr = acc.as_mut_ptr();
        let w_ptr = weights.as_ptr();
        for i in (0..HIDDEN_SIZE).step_by(8) {
            let va = vld1q_s16(a_ptr.add(i));
            let vw = vld1q_s16(w_ptr.add(i));
            vst1q_s16(a_ptr.add(i), vsubq_s16(va, vw));
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn evaluate_neon(
    acc: &Accumulator,
    us: usize,
    them: usize,
    bucket: usize,
    net: &'static Network,
) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let qa = vdupq_n_s16(QA);
        let zero = vdupq_n_s16(0);

        let mut sum0 = vdupq_n_s32(0);
        let mut sum1 = vdupq_n_s32(0);

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr();
            let w_ptr = net.output_weights[bucket][offset..].as_ptr();

            for chunk in (0..HIDDEN_SIZE).step_by(8) {
                let v = vld1q_s16(v_ptr.add(chunk));
                let clamped = vminq_s16(vmaxq_s16(v, zero), qa);

                let y_low = vmovl_s16(vget_low_s16(clamped));
                let y_high = vmovl_high_s16(clamped);

                let w = vld1q_s16(w_ptr.add(chunk));
                let w_low = vmovl_s16(vget_low_s16(w));
                let w_high = vmovl_high_s16(w);

                let sq_low = vmulq_s32(y_low, y_low);
                let sq_high = vmulq_s32(y_high, y_high);

                let p_low = vmulq_s32(sq_low, w_low);
                let p_high = vmulq_s32(sq_high, w_high);

                sum0 = vaddq_s32(sum0, p_low);
                sum1 = vaddq_s32(sum1, p_high);
            }
        };

        evaluate_side(&acc.vals[us], 0);
        evaluate_side(&acc.vals[them], HIDDEN_SIZE);

        let mut arr = [0i32; 8];
        vst1q_s32(arr.as_mut_ptr(), sum0);
        vst1q_s32(arr.as_mut_ptr().add(4), sum1);

        let mut output = 0i64;
        for i in 0..8 {
            output += arr[i] as i64;
        }

        output /= i64::from(QA);
        output += i64::from(net.output_bias[bucket]);
        output *= SCALE as i64;
        output /= i64::from(QA) * i64::from(QB);

        output as i32
    }
}
