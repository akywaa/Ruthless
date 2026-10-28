use crate::board::Board;
use crate::types::{Color, Piece, Square};

pub const HIDDEN_SIZE: usize = 512;
pub const NUM_INPUT_BUCKETS: usize = 16;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const FEATURES_PER_BUCKET: usize = 768;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

/// King input buckets, one entry per square of the mirrored half board, laid
/// out as 8 ranks of the four a-d files (a1 = index 0). A king on the e-h files
/// uses the mirrored entry of its file, so every rank appears twice per file.
/// This is the standard 16-bucket horizontal-mirroring table (`.16hm`):
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

/// Bucket layout of the previous 10-bucket network that is still embedded in
/// the binary. A 16-bucket network loads straight from its file; the embedded
/// one is expanded on load by mapping each of the 16 buckets back onto the
/// coarser bucket it used to share, see [`LEGACY_BUCKET_SOURCE`].
#[cfg(test)]
#[rustfmt::skip]
const LEGACY_BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

/// For every bucket of [`BUCKET_LAYOUT`], the bucket of [`LEGACY_BUCKET_LAYOUT`]
/// that held the same king squares. The legacy layout only ever merged squares
/// that the 16-bucket layout keeps apart, so expanding the old weights this way
/// reproduces the old evaluations exactly while the feature space stays 16hm.
const LEGACY_BUCKET_SOURCE: [usize; NUM_INPUT_BUCKETS] = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];

pub const LEGACY_NUM_INPUT_BUCKETS: usize = 10;

/// Size of the embedded 10-bucket network: weights, bias and output layer.
pub const LEGACY_NET_BYTES: usize =
    FEATURES_PER_BUCKET * LEGACY_NUM_INPUT_BUCKETS * HIDDEN_SIZE * 2 + HIDDEN_SIZE * 2 + NUM_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE * 2 + NUM_OUTPUT_BUCKETS * 2;

#[repr(C, align(64))]
pub struct AccumulatorRaw {
    pub vals: [i16; HIDDEN_SIZE],
}

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [AccumulatorRaw; FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS],
    pub feature_bias: AccumulatorRaw,
    pub output_weights: [[i16; 2 * HIDDEN_SIZE]; NUM_OUTPUT_BUCKETS],
    pub output_bias: [i16; NUM_OUTPUT_BUCKETS],
}

static NETWORK_BYTES: &[u8] = include_bytes!("../resources/ruthless.bin");

#[repr(C, align(64))]
struct ZeroWeights([i16; HIDDEN_SIZE]);

static ZERO_WEIGHTS: ZeroWeights = ZeroWeights([0; HIDDEN_SIZE]);

static NETWORK_ALIGNED: std::sync::OnceLock<&'static Network> = std::sync::OnceLock::new();

#[inline(always)]
pub fn network() -> &'static Network {
    *NETWORK_ALIGNED.get_or_init(init_network)
}

/// Copies one 512-lane weight row out of the little-endian network image.
#[inline(always)]
fn read_row(src: &[u8], dst: &mut [i16; HIDDEN_SIZE]) {
    debug_assert!(src.len() >= HIDDEN_SIZE * 2);
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            dst.as_mut_ptr() as *mut u8,
            HIDDEN_SIZE * 2,
        );
    }
    #[cfg(not(target_endian = "little"))]
    for (d, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
        *d = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
}

/// Expands the embedded 10-bucket network into the 16-bucket feature space.
fn load_legacy_network(bytes: &[u8], net: &mut Network) {
    assert!(
        bytes.len() >= LEGACY_NET_BYTES,
        "embedded network is truncated"
    );

    let mut offset = 0usize;
    for feature in 0..FEATURES_PER_BUCKET * LEGACY_NUM_INPUT_BUCKETS {
        let bucket = feature / FEATURES_PER_BUCKET;
        let rest = feature % FEATURES_PER_BUCKET;
        let src = &bytes[offset..offset + HIDDEN_SIZE * 2];
        for new_bucket in 0..NUM_INPUT_BUCKETS {
            if LEGACY_BUCKET_SOURCE[new_bucket] == bucket {
                read_row(src, &mut net.feature_weights[new_bucket * FEATURES_PER_BUCKET + rest].vals);
            }
        }
        offset += HIDDEN_SIZE * 2;
    }

    read_row(&bytes[offset..], &mut net.feature_bias.vals);
    offset += HIDDEN_SIZE * 2;

    for bucket in net.output_weights.iter_mut() {
        for half in 0..2 {
            let row = &mut bucket[half * HIDDEN_SIZE..(half + 1) * HIDDEN_SIZE];
            read_row(&bytes[offset..], row.try_into().unwrap());
            offset += HIDDEN_SIZE * 2;
        }
    }

    for (i, bias) in net.output_bias.iter_mut().enumerate() {
        *bias = i16::from_le_bytes([bytes[offset + i * 2], bytes[offset + i * 2 + 1]]);
    }
}

fn init_network() -> &'static Network {
    // The network is read at runtime: the embedded file holds a 10-bucket net
    // that is expanded here, while the 16-bucket network would not fit the
    // binary comfortably. Alignment matters for the SIMD kernels, so the block
    // is allocated on a 64-byte boundary and never freed.
    let layout = std::alloc::Layout::from_size_align(std::mem::size_of::<Network>(), 64)
        .expect("invalid network layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Network;
    assert!(!ptr.is_null(), "out of memory allocating network");

    unsafe {
        let net = &mut *ptr;
        load_legacy_network(NETWORK_BYTES, net);

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

        &*ptr
    }
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
    pub fn add_feature(&mut self, piece: Piece, sq: Square, w_ksq: Square, b_ksq: Square) {
        self.add_feature_side(piece, sq, w_ksq, Color::White);
        self.add_feature_side(piece, sq, b_ksq, Color::Black);
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

    #[inline(always)]
    pub fn apply_delta(&mut self, delta: &FeatureDelta) {
        let net = network();

        for side in 0..2 {
            let adds = &delta.adds[side];
            let subs = &delta.subs[side];
            let add_count = delta.add_count[side];
            let sub_count = delta.sub_count[side];
            if add_count == 0 && sub_count == 0 {
                continue;
            }

            // Nearly every move changes exactly one feature per point of view,
            // so that case runs on the two-op kernel instead of the general
            // four-op one, which would waste half its work on zero vectors.
            if add_count == 1 && sub_count == 1 {
                let add = &net.feature_weights[adds[0] as usize].vals;
                let sub = &net.feature_weights[subs[0] as usize].vals;
                apply_delta_weights_11(&mut self.vals[side], add, sub);
                continue;
            }

            let add1 = feature_weights_or_zero(net, adds[0]);
            let add2 = feature_weights_or_zero(net, adds[1]);
            let sub1 = feature_weights_or_zero(net, subs[0]);
            let sub2 = feature_weights_or_zero(net, subs[1]);

            apply_delta_weights(&mut self.vals[side], add1, add2, sub1, sub2);
        }
    }

    /// Reverses [`Accumulator::apply_delta`]. Used by `Board::undo_move`, which
    /// updates the single accumulator in place instead of copying a 2 KiB
    /// snapshot for every child node.
    #[inline(always)]
    pub fn apply_delta_inverse(&mut self, delta: &FeatureDelta) {
        let net = network();

        for side in 0..2 {
            let adds = &delta.adds[side];
            let subs = &delta.subs[side];
            let add_count = delta.add_count[side];
            let sub_count = delta.sub_count[side];
            if add_count == 0 && sub_count == 0 {
                continue;
            }

            if add_count == 1 && sub_count == 1 {
                let add = &net.feature_weights[adds[0] as usize].vals;
                let sub = &net.feature_weights[subs[0] as usize].vals;
                apply_delta_weights_11(&mut self.vals[side], sub, add);
                continue;
            }

            let add1 = feature_weights_or_zero(net, adds[0]);
            let add2 = feature_weights_or_zero(net, adds[1]);
            let sub1 = feature_weights_or_zero(net, subs[0]);
            let sub2 = feature_weights_or_zero(net, subs[1]);

            apply_delta_weights(&mut self.vals[side], sub1, sub2, add1, add2);
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
fn apply_delta_weights(
    vals: &mut [i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        unsafe { vec_delta_avx512(vals, add1, add2, sub1, sub2) };
        return;
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            unsafe { vec_delta_avx2(vals, add1, add2, sub1, sub2) };
            return;
        }
        scalar_delta(vals, add1, add2, sub1, sub2);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vec_delta_neon(vals, add1, add2, sub1, sub2) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_delta(vals, add1, add2, sub1, sub2);
}

#[inline(always)]
fn scalar_delta(
    vals: &mut [i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    for i in 0..HIDDEN_SIZE {
        vals[i] = vals[i]
            .wrapping_add(add1[i])
            .wrapping_add(add2[i])
            .wrapping_sub(sub1[i])
            .wrapping_sub(sub2[i]);
    }
}

/// Two-op accumulator update, used when a move touches exactly one feature per
/// point of view. The general kernel would add and subtract a zero vector in
/// that case, which doubles its vector work for no benefit.
#[inline(always)]
fn apply_delta_weights_11(
    vals: &mut [i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub: &[i16; HIDDEN_SIZE],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        unsafe { vec_delta_avx512_11(vals, add, sub) };
        return;
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            unsafe { vec_delta_avx2_11(vals, add, sub) };
            return;
        }
        scalar_delta_11(vals, add, sub);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vec_delta_neon_11(vals, add, sub) };
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar_delta_11(vals, add, sub);
}

#[inline(always)]
fn scalar_delta_11(vals: &mut [i16; HIDDEN_SIZE], add: &[i16; HIDDEN_SIZE], sub: &[i16; HIDDEN_SIZE]) {
    for i in 0..HIDDEN_SIZE {
        vals[i] = vals[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
    }
}

#[inline(always)]
pub fn king_bucket(sq: Square) -> usize {
    let file = sq.file();
    let rank = sq.rank();
    let mirrored_file = if file > 3 { 7 - file } else { file };
    BUCKET_LAYOUT[(rank * 4 + mirrored_file) as usize]
}

/// Bucket the given square had under the embedded 10-bucket layout. Only used
/// to check that expanding its weights kept the old evaluations intact.
#[cfg(test)]
pub fn legacy_king_bucket(sq: Square) -> usize {
    let file = sq.file();
    let rank = sq.rank();
    let mirrored_file = if file > 3 { 7 - file } else { file };
    LEGACY_BUCKET_LAYOUT[(rank * 4 + mirrored_file) as usize]
}

/// The accumulator only has to be rebuilt when the king changes bucket or
/// crosses the mirror line, because those are the only things that change the
/// input indices of every piece.
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

/// Marker for an unused feature slot in a [`FeatureDelta`]. The largest index
/// `feature_index_side` can produce is just under 7680, so u16 has room for it.
const NO_FEATURE: u16 = u16::MAX;

/// Feature changes of one move, grouped per point of view. A move touches at
/// most two features per side (a castling king and rook), so a fixed-size
/// buffer is enough and the whole delta can be applied in one memory pass.
/// Slots hold u16 indices with [`NO_FEATURE`] for "empty" instead of
/// `Option<usize>`: the delta is built and copied on every make/undo, and the
/// optional layout made it 160 bytes of stack traffic per move.
#[derive(Copy, Clone)]
pub struct FeatureDelta {
    adds: [[u16; 2]; 2],
    subs: [[u16; 2]; 2],
    add_count: [usize; 2],
    sub_count: [usize; 2],
}

impl FeatureDelta {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            adds: [[NO_FEATURE; 2]; 2],
            subs: [[NO_FEATURE; 2]; 2],
            add_count: [0; 2],
            sub_count: [0; 2],
        }
    }

    #[inline(always)]
    pub fn add(&mut self, piece: Piece, sq: Square, w_ksq: Square, b_ksq: Square, skip: Option<Color>) {
        if skip != Some(Color::White) {
            let view = Color::White as usize;
            self.adds[view][self.add_count[view]] =
                feature_index_side(piece, sq, w_ksq, Color::White) as u16;
            self.add_count[view] += 1;
        }
        if skip != Some(Color::Black) {
            let view = Color::Black as usize;
            self.adds[view][self.add_count[view]] =
                feature_index_side(piece, sq, b_ksq, Color::Black) as u16;
            self.add_count[view] += 1;
        }
    }

    #[inline(always)]
    pub fn sub(&mut self, piece: Piece, sq: Square, w_ksq: Square, b_ksq: Square, skip: Option<Color>) {
        if skip != Some(Color::White) {
            let view = Color::White as usize;
            self.subs[view][self.sub_count[view]] =
                feature_index_side(piece, sq, w_ksq, Color::White) as u16;
            self.sub_count[view] += 1;
        }
        if skip != Some(Color::Black) {
            let view = Color::Black as usize;
            self.subs[view][self.sub_count[view]] =
                feature_index_side(piece, sq, b_ksq, Color::Black) as u16;
            self.sub_count[view] += 1;
        }
    }
}

/// Weight vector of a delta slot, or the shared zero vector when the slot is
/// unused.
#[inline(always)]
fn feature_weights_or_zero(net: &'static Network, idx: u16) -> &'static [i16; HIDDEN_SIZE] {
    if idx == NO_FEATURE {
        &ZERO_WEIGHTS.0
    } else {
        &net.feature_weights[idx as usize].vals
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
    let acc = &board.accumulator;
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

    #[test]
    fn king_buckets_cover_the_new_layout() {
        let mut seen = [false; NUM_INPUT_BUCKETS];
        for s in 0..64u8 {
            let sq = Square::new(s);
            let bucket = king_bucket(sq);
            assert!(bucket < NUM_INPUT_BUCKETS);
            seen[bucket] = true;
            // Horizontal mirroring: both files of a mirrored pair share a bucket.
            let mirrored = Square::from_coords(7 - sq.file(), sq.rank());
            assert_eq!(bucket, king_bucket(mirrored), "mirror mismatch on {sq}");
        }
        assert!(seen.iter().all(|&used| used), "not all buckets are used");
    }

    #[test]
    fn expanding_legacy_weights_preserves_bucket_semantics() {
        for s in 0..64u8 {
            let sq = Square::new(s);
            assert_eq!(
                LEGACY_BUCKET_SOURCE[king_bucket(sq)],
                legacy_king_bucket(sq),
                "bucket {} of {sq} no longer maps to the legacy bucket",
                king_bucket(sq)
            );
        }
    }

    #[test]
    fn expanded_buckets_share_the_legacy_rows() {
        let net = network();
        for feature in 0..FEATURES_PER_BUCKET {
            for bucket in 0..NUM_INPUT_BUCKETS {
                let source = LEGACY_BUCKET_SOURCE[bucket];
                for other in 0..NUM_INPUT_BUCKETS {
                    if LEGACY_BUCKET_SOURCE[other] == source {
                        assert_eq!(
                            net.feature_weights[bucket * FEATURES_PER_BUCKET + feature].vals,
                            net.feature_weights[other * FEATURES_PER_BUCKET + feature].vals
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn accumulator_refresh_only_on_bucket_or_mirror_change() {
        // e1 -> d1 keeps bucket 3 but crosses the mirror line.
        assert!(needs_accumulator_refresh(Square::E1, Square::D1, Color::White));
        // a3 -> b3 stays inside bucket 8.
        assert!(!needs_accumulator_refresh(Square::A3, Square::B3, Color::White));
        // a1 -> a3 crosses from bucket 0 to bucket 8.
        assert!(needs_accumulator_refresh(Square::A1, Square::A3, Color::White));
        // The same boundaries from black's point of view.
        assert!(needs_accumulator_refresh(Square::E8, Square::D8, Color::Black));
        assert!(!needs_accumulator_refresh(Square::A6, Square::B6, Color::Black));
        assert!(needs_accumulator_refresh(Square::A8, Square::A6, Color::Black));
    }

    fn eval_ordering(board: &Board, swap: bool) -> i32 {
        let net = network();
        let us = board.side_to_move as usize;
        let them = (!board.side_to_move) as usize;
        let bucket = output_bucket(board);
        let (a, b) = if swap { (them, us) } else { (us, them) };
        let acc = &board.accumulator;
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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    fn avx512_matches_scalar_random_walk() {
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
                let acc = &board.accumulator;
                let expected = evaluate_scalar(acc, us, them, bucket, net);
                let got = unsafe { evaluate_avx512(acc, us, them, bucket, net) };
                assert_eq!(got, expected, "AVX-512 mismatch after {}", m);
                board.undo_move(m, undo);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.count;
            board.make_move(moves.get(pick));
            board.reset_accumulator_stack();
        }
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
                let acc = &board.accumulator;
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
unsafe fn vec_delta_avx2(
    vals: &mut [i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr() as *mut __m256i;
        let a1_ptr = add1.as_ptr() as *const __m256i;
        let a2_ptr = add2.as_ptr() as *const __m256i;
        let s1_ptr = sub1.as_ptr() as *const __m256i;
        let s2_ptr = sub2.as_ptr() as *const __m256i;

        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_loadu_si256(d_ptr.add(i));
            let v = _mm256_add_epi16(v, _mm256_loadu_si256(a1_ptr.add(i)));
            let v = _mm256_add_epi16(v, _mm256_loadu_si256(a2_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s1_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s2_ptr.add(i)));
            _mm256_storeu_si256(d_ptr.add(i), v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_delta_avx2_11(
    vals: &mut [i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr() as *mut __m256i;
        let a_ptr = add.as_ptr() as *const __m256i;
        let s_ptr = sub.as_ptr() as *const __m256i;

        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_loadu_si256(d_ptr.add(i));
            let v = _mm256_add_epi16(v, _mm256_loadu_si256(a_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_loadu_si256(s_ptr.add(i)));
            _mm256_storeu_si256(d_ptr.add(i), v);
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
unsafe fn vec_delta_avx512_11(
    vals: &mut [i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr() as *mut __m512i;
        let a_ptr = add.as_ptr() as *const __m512i;
        let s_ptr = sub.as_ptr() as *const __m512i;

        for i in 0..(HIDDEN_SIZE / 32) {
            let v = _mm512_loadu_si512(d_ptr.add(i));
            let v = _mm512_add_epi16(v, _mm512_loadu_si512(a_ptr.add(i)));
            let v = _mm512_sub_epi16(v, _mm512_loadu_si512(s_ptr.add(i)));
            _mm512_storeu_si512(d_ptr.add(i), v);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_delta_avx512(
    vals: &mut [i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr() as *mut __m512i;
        let a1_ptr = add1.as_ptr() as *const __m512i;
        let a2_ptr = add2.as_ptr() as *const __m512i;
        let s1_ptr = sub1.as_ptr() as *const __m512i;
        let s2_ptr = sub2.as_ptr() as *const __m512i;

        for i in 0..(HIDDEN_SIZE / 32) {
            let v = _mm512_loadu_si512(d_ptr.add(i));
            let v = _mm512_add_epi16(v, _mm512_loadu_si512(a1_ptr.add(i)));
            let v = _mm512_add_epi16(v, _mm512_loadu_si512(a2_ptr.add(i)));
            let v = _mm512_sub_epi16(v, _mm512_loadu_si512(s1_ptr.add(i)));
            let v = _mm512_sub_epi16(v, _mm512_loadu_si512(s2_ptr.add(i)));
            _mm512_storeu_si512(d_ptr.add(i), v);
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
        // Sixteen i32 lanes. Each lane only ever sums a few dozen products of
        // at most QA * QA * 128, so i32 cannot overflow here.
        let mut sum = _mm512_setzero_si512();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m512i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m512i;

            for chunk in 0..(HIDDEN_SIZE / 32) {
                let v = _mm512_loadu_si512(v_ptr.add(chunk));
                let x = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
                let w = _mm512_loadu_si512(w_ptr.add(chunk));

                // x is in 0..=QA and |w| * QA fits in an i16 (checked when the
                // network is loaded), so this 16-bit product is exact.
                let xw = _mm512_mullo_epi16(x, w);
                // madd sums adjacent pairs into i32 lanes: x * (x * w) = x^2 * w.
                sum = _mm512_add_epi32(sum, _mm512_madd_epi16(x, xw));
            }
        };

        evaluate_side(&acc.vals[us], 0);
        evaluate_side(&acc.vals[them], HIDDEN_SIZE);

        let mut output = i64::from(_mm512_reduce_add_epi32(sum));
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
unsafe fn vec_delta_neon_11(
    vals: &mut [i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub: &[i16; HIDDEN_SIZE],
) {
    use std::arch::aarch64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr();
        let a_ptr = add.as_ptr();
        let s_ptr = sub.as_ptr();

        for i in (0..HIDDEN_SIZE).step_by(8) {
            let mut v = vld1q_s16(d_ptr.add(i));
            v = vaddq_s16(v, vld1q_s16(a_ptr.add(i)));
            v = vsubq_s16(v, vld1q_s16(s_ptr.add(i)));
            vst1q_s16(d_ptr.add(i), v);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_delta_neon(
    vals: &mut [i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    use std::arch::aarch64::*;
    unsafe {
        let d_ptr = vals.as_mut_ptr();
        let a1_ptr = add1.as_ptr();
        let a2_ptr = add2.as_ptr();
        let s1_ptr = sub1.as_ptr();
        let s2_ptr = sub2.as_ptr();

        for i in (0..HIDDEN_SIZE).step_by(8) {
            let mut v = vld1q_s16(d_ptr.add(i));
            v = vaddq_s16(v, vld1q_s16(a1_ptr.add(i)));
            v = vaddq_s16(v, vld1q_s16(a2_ptr.add(i)));
            v = vsubq_s16(v, vld1q_s16(s1_ptr.add(i)));
            v = vsubq_s16(v, vld1q_s16(s2_ptr.add(i)));
            vst1q_s16(d_ptr.add(i), v);
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
