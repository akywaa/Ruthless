use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, PieceType, Square};

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
    /// Present when a Heimdall-format network is loaded. The legacy network
    /// keeps using the single linear output layer above.
    pub heimdall: Option<Box<HeimdallNet>>,
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
        for (new_bucket, &source) in LEGACY_BUCKET_SOURCE.iter().enumerate() {
            if source == bucket {
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

/// Allocates the (large, effectively permanent) network storage on a 64-byte
/// boundary, which is what the SIMD kernels load from.
fn allocate_network() -> &'static mut Network {
    // The network is read at runtime: the embedded file holds a 10-bucket net
    // that is expanded here, while a 16-bucket network with threat rows would
    // not fit the binary comfortably. The block is never freed.
    let layout = std::alloc::Layout::from_size_align(std::mem::size_of::<Network>(), 64)
        .expect("invalid network layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Network;
    assert!(!ptr.is_null(), "out of memory allocating network");
    unsafe { &mut *ptr }
}

fn init_network() -> &'static Network {
    let net = allocate_network();

    if !load_external_network(net) {
        load_legacy_network(NETWORK_BYTES, net);
    }

    #[cfg(target_arch = "x86_64")]
    {
        // The AVX2 output layer multiplies clipped activations (0..=QA)
        // by output weights in 16-bit lanes, which is only exact while
        // |weight| * QA fits into an i16. Reject a net that would
        // silently produce wrong evaluations instead.
        if has_avx2() && net.heimdall.is_none() {
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

/// Which network architecture the engine is running.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NetFormat {
    /// The original embedded network: 10 buckets expanded to 16, linear output.
    Legacy,
    /// 16hm piece-square inputs plus threat inputs feeding the multilayer head.
    Heimdall,
}

pub const HEIMDALL_NET_BYTES: usize = FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS * HIDDEN_SIZE * 2
    + TOTAL_THREATS * HIDDEN_SIZE
    + HIDDEN_SIZE * 2
    + HIDDEN_SIZE * L1_SIZE * NUM_OUTPUT_BUCKETS
    + NUM_OUTPUT_BUCKETS * L1_SIZE * 4
    + L2_IN * NUM_OUTPUT_BUCKETS * L2_SIZE * 4
    + NUM_OUTPUT_BUCKETS * L2_SIZE * 4
    + L2_SIZE * NUM_OUTPUT_BUCKETS * 4
    + NUM_OUTPUT_BUCKETS * 4;

/// Recognises a network file from its exact size. The two layouts cannot be
/// confused: they differ by 35 MB.
pub fn detect_format(len: usize) -> Option<NetFormat> {
    if len == HEIMDALL_NET_BYTES {
        Some(NetFormat::Heimdall)
    } else if (LEGACY_NET_BYTES..HEIMDALL_NET_BYTES).contains(&len) {
        Some(NetFormat::Legacy)
    } else {
        None
    }
}

/// Cached answer to "is the active network a Heimdall one", read once per
/// evaluation by the scaling in `eval`.
#[inline(always)]
pub fn uses_heimdall() -> bool {
    network().heimdall.is_some()
}

struct NetBytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> NetBytes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.pos + len;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| format!("network truncated at byte {}", self.pos))?;
        self.pos = end;
        Ok(slice)
    }

    fn i32(&mut self) -> Result<i32, String> {
        let raw = self.take(4)?;
        Ok(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn i16_row(&mut self, dst: &mut [i16; HIDDEN_SIZE]) -> Result<(), String> {
        let raw = self.take(HIDDEN_SIZE * 2)?;
        read_row(raw, dst);
        Ok(())
    }
}

/// Loads a Heimdall-format network: the same disk layout Bullet writes, in the
/// order `piece-square weights, threat rows, feature bias, L1, L2, L3`.
///
/// The layer-1 weights are stored input-major on disk (`input, bucket, output`)
/// and transposed here into contiguous per-neuron rows, which is what every
/// kernel in this file expects.
fn load_heimdall_bytes(bytes: &[u8], net: &mut Network) -> Result<(), String> {
    if bytes.len() != HEIMDALL_NET_BYTES {
        return Err(format!(
            "expected {HEIMDALL_NET_BYTES} bytes for the 16hm multilayer layout, got {}",
            bytes.len()
        ));
    }

    let mut reader = NetBytes::new(bytes);

    for feature in 0..FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS {
        reader.i16_row(&mut net.feature_weights[feature].vals)?;
    }

    let raw = reader.take(TOTAL_THREATS * HIDDEN_SIZE)?;
    // Signed bytes and raw bytes have the same layout and every bit pattern is
    // a valid i8, so this reinterpretation is sound.
    let signed = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
    let mut rows = vec![0i8; TOTAL_THREATS * HIDDEN_SIZE];
    rows.copy_from_slice(signed);

    reader.i16_row(&mut net.feature_bias.vals)?;

    let mut head: Box<HeimdallHead> = Box::new(unsafe { std::mem::zeroed() });

    for input in 0..HIDDEN_SIZE {
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            let chunk = reader.take(L1_SIZE)?;
            for (output, &byte) in chunk.iter().enumerate() {
                head.l1_weights[bucket][output][input] = byte as i8;
            }
        }
    }
    for bucket in 0..NUM_OUTPUT_BUCKETS {
        for output in 0..L1_SIZE {
            head.l1_bias[bucket][output] = reader.i32()?;
        }
    }
    for input in 0..L2_IN {
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            for output in 0..L2_SIZE {
                head.l2_weights[bucket][input][output] = reader.i32()?;
            }
        }
    }
    for bucket in 0..NUM_OUTPUT_BUCKETS {
        for output in 0..L2_SIZE {
            head.l2_bias[bucket][output] = reader.i32()?;
        }
    }
    for input in 0..L2_SIZE {
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            head.l3_weights[bucket][input][0] = reader.i32()?;
        }
    }
    for bucket in 0..NUM_OUTPUT_BUCKETS {
        head.l3_bias[bucket][0] = reader.i32()?;
    }

    debug_assert_eq!(reader.pos, HEIMDALL_NET_BYTES, "unread bytes in the network file");

    net.heimdall = Some(Box::new(HeimdallNet {
        head: *head,
        threat_weights: Some(Box::new(ThreatWeights { rows })),
    }));
    Ok(())
}

fn external_net_path() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("RUTHLESS_NET") {
        return Some(path.into());
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    for name in ["heimdall.bin", "resources/heimdall.bin"] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Tries a network file next to the binary (or at `RUTHLESS_NET`). Anything that
/// does not match a known layout is reported and ignored, so the engine keeps
/// running on the embedded network.
fn load_external_network(net: &mut Network) -> bool {
    let Some(path) = external_net_path() else {
        return false;
    };

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("info string cannot read network {}: {error}", path.display());
            return false;
        }
    };

    match detect_format(bytes.len()) {
        Some(NetFormat::Heimdall) => match load_heimdall_bytes(&bytes, net) {
            Ok(()) => {
                eprintln!(
                    "info string loaded heimdall network {} ({} MiB)",
                    path.display(),
                    bytes.len() / (1024 * 1024)
                );
                true
            }
            Err(error) => {
                eprintln!("info string rejected network {}: {error}", path.display());
                false
            }
        },
        Some(NetFormat::Legacy) => {
            load_legacy_network(&bytes, net);
            true
        }
        None => {
            eprintln!(
                "info string network {} has an unsupported size ({} bytes)",
                path.display(),
                bytes.len()
            );
            false
        }
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
    let us = board.side_to_move as usize;
    let them = us ^ 1;
    if let Some(heimdall) = net.heimdall.as_deref() {
        return forward_heimdall(
            &board.accumulator,
            &board.threat,
            us,
            them,
            bucket,
            &heimdall.head,
        );
    }
    evaluate_acc(&board.accumulator, us, them, bucket, net)
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

// ---------------------------------------------------------------------------
// Heimdall multilayer output head
// ---------------------------------------------------------------------------
//
// The head replaces the linear `1024 -> 1` layer of the legacy network with
//
//     pairwise(512) -> 32 -> dual activation(64) -> 16 -> 1,  x8 buckets,
//
// which is the multilayer layout Bullet trains for Heimdall-format networks.
// Quantisation follows that format exactly: the first layer consumes one
// unsigned byte per pair of hidden lanes, `>> 8` requantises into the
// Q = 64 space used by the remaining layers, and the final dot product is
// dequantised once in 64-bit arithmetic.

/// Hidden lanes of one perspective are paired as `(i, i + PAIR_COUNT)`.
pub const PAIR_COUNT: usize = HIDDEN_SIZE / 2;

/// Width of the first head layer, per output bucket. Both perspectives' pairwise
/// activations feed it together, so its input is `2 * PAIR_COUNT = HIDDEN_SIZE`.
pub const L1_SIZE: usize = 32;
/// The first layer is activated twice (CReLU and SCReLU), hence twice the width.
pub const L2_IN: usize = 2 * L1_SIZE;
pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 1;

pub const QUANT_BITS: u32 = 6;
pub const QUANT: i32 = 1 << QUANT_BITS;
pub const QUANT_CUBED: i32 = QUANT * QUANT * QUANT;
pub const QUANT_FOURTH: i64 = (QUANT as i64) * (QUANT as i64) * (QUANT as i64) * (QUANT as i64);
pub const FT_SCALE_BITS: u32 = 7;
/// Bin scale of a Heimdall network's output, already in centipawns.
pub const EVAL_SCALE: i32 = 282;
/// `16 + QUANT_BITS - FT_SCALE_BITS - 2 * FT_QUANT_BITS - L1_QUANT_BITS`, with
/// `FT_QUANT_BITS = 8` and `L1_QUANT_BITS = 7`; the total is -8, so the first
/// layer is requantised with a right shift by this amount.
pub const L1_SHIFT: u32 = 8;

/// Output head plus the threat feature rows. Rows are 512 signed bytes per
/// threat, which is what the first head layer consumes directly.
#[repr(C, align(64))]
pub struct HeimdallNet {
    pub head: HeimdallHead,
    pub threat_weights: Option<Box<ThreatWeights>>,
}

pub struct ThreatWeights {
    /// `TOTAL_THREATS * HIDDEN_SIZE` rows of signed bytes.
    pub rows: Vec<i8>,
}

#[cfg(test)]
impl ThreatWeights {
    /// Synthetic weights for tests: the layout and arithmetic are what matter
    /// here, not whether the numbers came out of a trainer.
    pub fn random(seed: u64) -> Self {
        let mut state = seed | 1;
        let mut rows = vec![0i8; TOTAL_THREATS * HIDDEN_SIZE];
        for value in rows.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = (state >> 32) as i8;
        }
        Self { rows }
    }
}

#[repr(C, align(64))]
pub struct HeimdallHead {
    /// `[bucket][output][input]`: one neuron's weights are contiguous, so the
    /// dot product walks the row in order.
    pub l1_weights: [[[i8; HIDDEN_SIZE]; L1_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l1_bias: [[i32; L1_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l2_weights: [[[i32; L2_SIZE]; L2_IN]; NUM_OUTPUT_BUCKETS],
    pub l2_bias: [[i32; L2_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l3_weights: [[[i32; L3_SIZE]; L2_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l3_bias: [[i32; L3_SIZE]; NUM_OUTPUT_BUCKETS],
}

/// Sum of the active threat rows, per perspective, without bias. Kept apart
/// from the piece-square accumulator so a threat-only change never touches the
/// 768-input feature rows.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThreatAccumulator {
    pub vals: [[i16; HIDDEN_SIZE]; 2],
}

impl ThreatAccumulator {
    pub fn new() -> Self {
        Self {
            vals: [[0; HIDDEN_SIZE]; 2],
        }
    }
}

/// Pairwise reduction of one perspective: `(x[i] * x[i + PAIR_COUNT]) >> 16`
/// after scaling the first factor up by [`FT_SCALE_BITS`], leaving one byte per
/// pair. The second factor is only clipped from above because the first is
/// non-negative, which makes a negative product impossible to exceed the low
/// clamp.
fn pairwise_side_scalar(
    psq: &[i16; HIDDEN_SIZE],
    threats: &[i16; HIDDEN_SIZE],
    out: &mut [u8; PAIR_COUNT],
) {
    for i in 0..PAIR_COUNT {
        let a = i32::from(psq[i].wrapping_add(threats[i])).clamp(0, i32::from(QA));
        let b = i32::from(psq[i + PAIR_COUNT].wrapping_add(threats[i + PAIR_COUNT]))
            .min(i32::from(QA));
        out[i] = (((a << FT_SCALE_BITS) * b) >> 16).clamp(0, 255) as u8;
    }
}

/// Reference first head layer, including requantisation and the dual activation.
fn l1_forward_scalar(
    weights: &[[i8; HIDDEN_SIZE]; L1_SIZE],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    out: &mut [i32; L2_IN],
) {
    for j in 0..L1_SIZE {
        let mut sum = 0i64;
        for i in 0..HIDDEN_SIZE {
            sum += i64::from(input[i]) * i64::from(weights[j][i]);
        }
        let output = ((sum as i32).wrapping_add(bias[j])) >> L1_SHIFT;

        out[j] = output.clamp(0, QUANT) << QUANT_BITS;
        // Squaring is done wide and then capped: real networks keep the
        // requantised value small, but the cap is what the format specifies.
        out[L1_SIZE + j] = (i64::from(output) * i64::from(output)).min(i64::from(QUANT * QUANT)) as i32;
    }
}

fn l2_forward_scalar(
    weights: &[[i32; L2_SIZE]; L2_IN],
    bias: &[i32; L2_SIZE],
    input: &[i32; L2_IN],
    out: &mut [i32; L2_SIZE],
) {
    *out = *bias;
    for i in 0..L2_IN {
        for j in 0..L2_SIZE {
            // i64 accumulation truncated to 32 bits is exactly wrapping i32
            // arithmetic, without relying on overflow being defined.
            out[j] = (i64::from(out[j]) + i64::from(input[i]) * i64::from(weights[i][j])) as i32;
        }
    }
}

fn l3_forward_scalar(
    weights: &[[i32; L3_SIZE]; L2_SIZE],
    bias: &[i32; L3_SIZE],
    input: &[i32; L2_SIZE],
) -> i32 {
    let mut result = i64::from(bias[0]);
    for i in 0..L2_SIZE {
        let activated = input[i].clamp(0, QUANT_CUBED);
        result += i64::from(activated) * i64::from(weights[i][0]);
    }
    (result * i64::from(EVAL_SCALE) / QUANT_FOURTH) as i32
}

/// Scalar reference for the whole head: the oracle the vectorised pass is
/// tested against. Production always goes through [`forward_heimdall`], whose
/// per-layer kernels fall back to the scalar versions on their own.
#[cfg(test)]
fn forward_scalar(
    psq: &Accumulator,
    threats: &ThreatAccumulator,
    us: usize,
    them: usize,
    bucket: usize,
    head: &HeimdallHead,
) -> i32 {
    let mut input = [0u8; HIDDEN_SIZE];
    pairwise_side_scalar(&psq.vals[us], &threats.vals[us], (&mut input[..PAIR_COUNT]).try_into().unwrap());
    pairwise_side_scalar(&psq.vals[them], &threats.vals[them], (&mut input[PAIR_COUNT..]).try_into().unwrap());

    let mut l1 = [0i32; L2_IN];
    l1_forward_scalar(&head.l1_weights[bucket], &head.l1_bias[bucket], &input, &mut l1);

    let mut l2 = [0i32; L2_SIZE];
    l2_forward_scalar(&head.l2_weights[bucket], &head.l2_bias[bucket], &l1, &mut l2);

    l3_forward_scalar(&head.l3_weights[bucket], &head.l3_bias[bucket], &l2)
}

#[inline(always)]
fn pairwise_side(psq: &[i16; HIDDEN_SIZE], threats: &[i16; HIDDEN_SIZE], out: &mut [u8; PAIR_COUNT]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { pairwise_side_avx2(psq, threats, out) };
            return;
        }
    }
    pairwise_side_scalar(psq, threats, out)
}

#[inline(always)]
fn l1_forward(
    weights: &[[i8; HIDDEN_SIZE]; L1_SIZE],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    out: &mut [i32; L2_IN],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        unsafe { l1_forward_avx512(weights, bias, input, out) };
        return;
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        if has_avx2() {
            unsafe { l1_forward_avx2(weights, bias, input, out) };
            return;
        }
        l1_forward_scalar(weights, bias, input, out);
    }

    #[cfg(not(target_arch = "x86_64"))]
    l1_forward_scalar(weights, bias, input, out)
}

#[inline(always)]
fn l2_forward(
    weights: &[[i32; L2_SIZE]; L2_IN],
    bias: &[i32; L2_SIZE],
    input: &[i32; L2_IN],
    out: &mut [i32; L2_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { l2_forward_avx2(weights, bias, input, out) };
            return;
        }
    }
    l2_forward_scalar(weights, bias, input, out)
}

/// Forward pass of a Heimdall network. `psq` holds the piece-square
/// accumulator, `threats` the active threat rows; both are summed in 16-bit
/// space before the pairwise activation.
fn forward_heimdall(
    psq: &Accumulator,
    threats: &ThreatAccumulator,
    us: usize,
    them: usize,
    bucket: usize,
    head: &HeimdallHead,
) -> i32 {
    let mut input = [0u8; HIDDEN_SIZE];
    pairwise_side(&psq.vals[us], &threats.vals[us], (&mut input[..PAIR_COUNT]).try_into().unwrap());
    pairwise_side(&psq.vals[them], &threats.vals[them], (&mut input[PAIR_COUNT..]).try_into().unwrap());

    let mut l1 = [0i32; L2_IN];
    l1_forward(&head.l1_weights[bucket], &head.l1_bias[bucket], &input, &mut l1);

    let mut l2 = [0i32; L2_SIZE];
    l2_forward(&head.l2_weights[bucket], &head.l2_bias[bucket], &l1, &mut l2);

    l3_forward_scalar(&head.l3_weights[bucket], &head.l3_bias[bucket], &l2)
}

// ---------------------------------------------------------------------------
// Threat inputs (60144 features)
// ---------------------------------------------------------------------------
//
// A threat feature encodes one attack relation: which piece of which colour
// attacks or defends which piece of which colour, from where and to where. The
// index depends on the king square of the perspective only through mirroring,
// exactly like the piece-square inputs, so a position and its horizontal mirror
// produce the same features.
//
// Layout (from the Heimdall/Bullet format):
//
//   for attacker colour, attacker type:            (12 groups)
//     for victim colour placed inside that group:  x PIECE_TARGET_COUNT
//       for attacker square:                       x span
//         for target square:                       x attacks below it
//
// which sums to exactly `TOTAL_THREATS`. Redundant relations are dropped: a
// pawn never reports queens or bishops (the other side of the relation already
// encodes it) and same-type relations are stored once, in one direction.

pub const TOTAL_THREATS: usize = 60144;

/// Upper bound of active threat features in a legal position. The collector
/// asserts instead of truncating, so a corrupt board cannot silently produce a
/// wrong evaluation.
pub const MAX_THREATS: usize = 256;

const THREAT_SENTINEL: u16 = u16::MAX;

/// For every attacker type, the position of each victim type inside its feature
/// group, or -1 when that relation is not indexed. Kings are never indexed.
#[rustfmt::skip]
const PIECE_TARGET_MAP: [[i8; 6]; 6] = [
    // pawn  -> pawn, knight, rook
    [0, 1, -1, 2, -1, -1],
    // knight -> pawn, knight, bishop, rook, queen
    [0, 1, 2, 3, 4, -1],
    // bishop -> pawn, knight, bishop, rook
    [0, 1, 2, 3, -1, -1],
    // rook -> pawn, knight, bishop, rook
    [0, 1, 2, 3, -1, -1],
    // queen -> pawn, knight, bishop, rook, queen
    [0, 1, 2, 3, 4, -1],
    // king -> nothing
    [-1, -1, -1, -1, -1, -1],
];

#[rustfmt::skip]
const PIECE_TARGET_COUNT: [u16; 6] = [3, 5, 4, 4, 5, 0];

struct ThreatTables {
    /// First index of an attacker square, per attacker colour and type.
    offsets: [[[u16; 64]; 6]; 2],
    /// Number of indices one attacker square owns, per colour and type.
    span: [[u16; 6]; 2],
    /// First index of an attacker colour and type group.
    global: [[u16; 6]; 2],
    /// Base index of a relation, selected by whether the attacker square is
    /// below the target square. `THREAT_SENTINEL` marks excluded relations.
    attack: [[[[[u16; 2]; 6]; 2]; 6]; 2],
    /// Number of attacked squares below a given target, per attacker square.
    piece_index: [[[[u8; 64]; 64]; 6]; 2],
}

static THREAT_TABLES: std::sync::OnceLock<ThreatTables> = std::sync::OnceLock::new();

#[inline(always)]
fn threat_tables() -> &'static ThreatTables {
    THREAT_TABLES.get_or_init(init_threat_tables)
}

/// Attacks of a piece in feature space: the board is a1-based, so a white pawn
/// attacks towards higher squares and a black pawn towards lower ones, which is
/// also how the threat tables are laid out.
#[inline(always)]
fn threat_attacks(piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    match piece.piece_type() {
        PieceType::Pawn => crate::attacks::pawn_attacks(piece.color(), sq),
        PieceType::Knight => crate::attacks::knight_attacks(sq),
        PieceType::Bishop => crate::attacks::bishop_attacks(sq, occ),
        PieceType::Rook => crate::attacks::rook_attacks(sq, occ),
        PieceType::Queen => crate::attacks::bishop_attacks(sq, occ) | crate::attacks::rook_attacks(sq, occ),
        PieceType::King => crate::attacks::king_attacks(sq),
        PieceType::None => Bitboard::EMPTY,
    }
}

/// Square mask that brings a position into the feature space of one perspective:
/// ranks are flipped for black and files are mirrored when the king sits on the
/// e-h half. `0x38` flips ranks and `7` mirrors files.
#[inline(always)]
pub fn perspective_mask(ksq: Square, view: Color) -> u8 {
    let rank_flip = if view == Color::Black { 56 } else { 0 };
    let file_flip = if ksq.file() >= 4 { 7 } else { 0 };
    rank_flip ^ file_flip
}

fn init_threat_tables() -> ThreatTables {
    // Every field is a plain integer array, so all-zero is a valid table.
    let mut tables: ThreatTables = unsafe { std::mem::zeroed() };

    let mut global = 0u32;
    for color in 0..2 {
        for (kind, &target_count) in PIECE_TARGET_COUNT.iter().enumerate() {
            let piece = Piece::new(Color::from_index(color), PieceType::from_index(kind));
            let mut offset = 0u16;
            for sq in 0..64 {
                let square = Square::new(sq as u8);
                tables.offsets[color][kind][sq] = offset;
                // Pawns never stand on the first or last rank, and an index
                // built from one would be meaningless.
                let back_rank = kind == 0 && (square.rank() == 0 || square.rank() == 7);
                if !back_rank {
                    offset += threat_attacks(piece, square, Bitboard::EMPTY).count() as u16;
                }
            }
            tables.span[color][kind] = offset;
            tables.global[color][kind] = global as u16;
            global += 2 * u32::from(target_count) * u32::from(offset);
        }
    }
    assert_eq!(
        global as usize, TOTAL_THREATS,
        "threat index tables must cover exactly the published feature count"
    );

    for color in 0..2 {
        for kind in 0..6 {
            let piece = Piece::new(Color::from_index(color), PieceType::from_index(kind));
            for from in 0..64 {
                let attacks = threat_attacks(piece, Square::new(from as u8), Bitboard::EMPTY);
                for to in 0..64 {
                    tables.piece_index[color][kind][from][to] = (0..to)
                        .filter(|&sq| attacks.contains(Square::new(sq as u8)))
                        .count() as u8;
                }
            }
        }
    }

    for acolor in 0..2 {
        for (akind, targets) in PIECE_TARGET_MAP.iter().enumerate() {
            for vcolor in 0..2 {
                for (vkind, &map) in targets.iter().enumerate() {
                    // Same-type relations are stored once: opposite colours
                    // encode the pair in both directions, same-colour pairs
                    // (defences) in one. Pawn defences are the exception and
                    // are kept from both sides.
                    let opposed = acolor != vcolor;
                    let semi_excluded = akind == vkind && (opposed || akind != 0);
                    for below in 0..2 {
                        if map < 0 || (below == 1 && semi_excluded) {
                            tables.attack[acolor][akind][vcolor][vkind][below] = THREAT_SENTINEL;
                        } else {
                            let color_base =
                                vcolor as u32 * u32::from(PIECE_TARGET_COUNT[akind]);
                            let feature = u32::from(tables.global[acolor][akind])
                                + (color_base + map as u32) * u32::from(tables.span[acolor][akind]);
                            tables.attack[acolor][akind][vcolor][vkind][below] = feature as u16;
                        }
                    }
                }
            }
        }
    }

    tables
}

/// Feature index of a single threat. `from` must be below `to` when the relation
/// is one of the same-type pairs that the format stores only once.
#[inline(always)]
fn threat_index(
    view: Color,
    ksq: Square,
    attacker: Piece,
    from: Square,
    victim: Piece,
    to: Square,
) -> Option<u16> {
    let tables = threat_tables();
    let mask = perspective_mask(ksq, view);
    let from = (from as u8 ^ mask) as usize;
    let to = (to as u8 ^ mask) as usize;

    let a = oriented_piece(attacker, view);
    let v = oriented_piece(victim, view);
    let (acolor, akind) = (a / 6, a % 6);
    let (vcolor, vkind) = (v / 6, v % 6);

    let base = tables.attack[acolor][akind][vcolor][vkind][usize::from(from < to)];
    if base == THREAT_SENTINEL {
        return None;
    }

    let index = base + tables.offsets[acolor][akind][from] + u16::from(tables.piece_index[acolor][akind][from][to]);
    debug_assert!((index as usize) < TOTAL_THREATS);
    Some(index)
}

/// Active threat features of one perspective, kept sorted so that two lists can
/// be diffed in one linear pass.
#[derive(Copy, Clone)]
pub struct ThreatList {
    indices: [u16; MAX_THREATS],
    count: usize,
}

impl ThreatList {
    pub fn new() -> Self {
        Self {
            indices: [0; MAX_THREATS],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.count]
    }

    #[inline(always)]
    fn push(&mut self, index: u16) {
        assert!(self.count < MAX_THREATS, "too many active threat features");
        self.indices[self.count] = index;
        self.count += 1;
    }
}

/// Collects every threat feature of one perspective from scratch. Moves use the
/// incremental path instead, but this is the definition the incremental state is
/// checked against.
pub fn collect_threats(board: &Board, view: Color, out: &mut ThreatList) {
    out.count = 0;
    let ksq = board.king_square(view);
    let mask = perspective_mask(ksq, view);

    let mut by_type = [Bitboard::EMPTY; 6];
    for (kind, slot) in by_type.iter_mut().enumerate() {
        for color in 0..2 {
            *slot |= board.pieces[Piece::new(Color::from_index(color), PieceType::from_index(kind))];
        }
    }

    // Union of the piece types each attacker type is allowed to threaten.
    let mut victims = [Bitboard::EMPTY; 6];
    for (akind, targets) in PIECE_TARGET_MAP.iter().enumerate() {
        for (vkind, &map) in targets.iter().enumerate() {
            if map >= 0 {
                victims[akind] |= by_type[vkind];
            }
        }
    }

    let kings = by_type[PieceType::King as usize];
    let mut attackers = board.occupied & !kings;

    while !attackers.is_empty() {
        let from = attackers.pop_lsb();
        let attacker = board.piece_on[from];
        let akind = attacker.piece_type() as usize;
        let from_label = (from as u8 ^ mask) as usize;

        // Same-type pairs are indexed once. Pawn defences are the exception:
        // a pawn does not attack backwards, so nothing is duplicated there.
        let mut same_type = by_type[akind];
        if akind == PieceType::Pawn as usize {
            same_type &= board.occupied_co[!attacker.color()];
        }

        let mut targets = threat_attacks(attacker, from, board.occupied) & !kings & victims[akind];
        while !targets.is_empty() {
            let to = targets.pop_lsb();
            if same_type.contains(to) && (to as u8 ^ mask) as usize > from_label {
                continue;
            }
            let victim = board.piece_on[to];
            if let Some(index) = threat_index(view, ksq, attacker, from, victim, to) {
                out.push(index);
            }
        }
    }

    out.indices[..out.count].sort_unstable();
}

/// Applies or removes the rows of a threat index list.
fn apply_threat_rows_scalar(acc: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16], add: bool) {
    for &index in list {
        let base = index as usize * HIDDEN_SIZE;
        let row = &weights.rows[base..base + HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            let w = i16::from(row[i]);
            acc[i] = if add {
                acc[i].wrapping_add(w)
            } else {
                acc[i].wrapping_sub(w)
            };
        }
    }
}

#[inline(always)]
fn apply_threat_rows(acc: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16], add: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { apply_threat_rows_avx2(acc, weights, list, add) };
            return;
        }
    }
    apply_threat_rows_scalar(acc, weights, list, add);
}

impl ThreatAccumulator {
    /// Rebuilds one perspective from scratch: zero, plus every row of the list.
    pub fn refresh_side(&mut self, side: usize, list: &ThreatList, weights: &ThreatWeights) {
        let mut acc = [0i16; HIDDEN_SIZE];
        apply_threat_rows(&mut acc, weights, list.as_slice(), true);
        self.vals[side] = acc;
    }

    /// Diffs two sorted feature lists and applies only the changed rows. Calling
    /// it with the arguments swapped undoes the update exactly: the accumulator
    /// is the sum of the rows of the list, and both directions add and subtract
    /// exactly the rows that entered or left it.
    pub fn apply_diff(&mut self, side: usize, weights: &ThreatWeights, old: &[u16], new: &[u16]) {
        let mut adds = [0u16; MAX_THREATS];
        let mut subs = [0u16; MAX_THREATS];
        let mut add_count = 0usize;
        let mut sub_count = 0usize;

        let (mut i, mut j) = (0usize, 0usize);
        while i < old.len() && j < new.len() {
            match old[i].cmp(&new[j]) {
                std::cmp::Ordering::Less => {
                    subs[sub_count] = old[i];
                    sub_count += 1;
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    adds[add_count] = new[j];
                    add_count += 1;
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < old.len() {
            subs[sub_count] = old[i];
            sub_count += 1;
            i += 1;
        }
        while j < new.len() {
            adds[add_count] = new[j];
            add_count += 1;
            j += 1;
        }

        let acc = &mut self.vals[side];
        if sub_count > 0 {
            apply_threat_rows(acc, weights, &subs[..sub_count], false);
        }
        if add_count > 0 {
            apply_threat_rows(acc, weights, &adds[..add_count], true);
        }
    }
}

/// Threat weights of the active network, if it is a Heimdall-format one. The
/// legacy net has no threat inputs, so this also serves as the gate that keeps
/// the whole threat path off it.
#[inline(always)]
pub fn threat_weights() -> Option<&'static ThreatWeights> {
    network()
        .heimdall
        .as_deref()
        .and_then(|net| net.threat_weights.as_deref())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_rows_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16], add: bool) {
    use std::arch::x86_64::*;
    unsafe {
        for &index in list {
            let row = weights.rows.as_ptr().add(index as usize * HIDDEN_SIZE);
            let mut i = 0;
            while i < HIDDEN_SIZE {
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(i) as *const __m128i));
                let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let r = if add {
                    _mm256_add_epi16(a, w)
                } else {
                    _mm256_sub_epi16(a, w)
                };
                _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 16;
            }
        }
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
            for (bucket, &source) in LEGACY_BUCKET_SOURCE.iter().enumerate() {
                for (other, &other_source) in LEGACY_BUCKET_SOURCE.iter().enumerate() {
                    if other_source == source {
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

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn centered(&mut self, span: i32) -> i32 {
            (self.next() % span as u64) as i32 - span / 2
        }
    }

    fn random_head(seed: u64) -> Box<HeimdallNet> {
        // Zeroed storage keeps the 160 KiB of layers off the test stack.
        let mut net: Box<HeimdallNet> = Box::new(HeimdallNet {
            head: unsafe { std::mem::zeroed() },
            threat_weights: None,
        });
        let mut rng = Rng(seed | 1);
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            let head = &mut net.head;
            for j in 0..L1_SIZE {
                for i in 0..HIDDEN_SIZE {
                    head.l1_weights[bucket][j][i] = (rng.next() & 0xFF) as u8 as i8;
                }
                head.l1_bias[bucket][j] = rng.centered(8192);
            }
            for i in 0..L2_IN {
                for j in 0..L2_SIZE {
                    head.l2_weights[bucket][i][j] = rng.centered(4096);
                }
            }
            for j in 0..L2_SIZE {
                head.l2_bias[bucket][j] = rng.centered(4096);
            }
            for i in 0..L2_SIZE {
                head.l3_weights[bucket][i][0] = rng.centered(4096);
            }
            head.l3_bias[bucket][0] = rng.centered(4096);
        }
        net
    }

    /// Accumulators shaped like a real position: lane values within a few
    /// hundred of zero plus a bias so the pairwise clamps are exercised.
    fn random_accumulators(seed: u64) -> (Accumulator, ThreatAccumulator) {
        let mut rng = Rng(seed | 1);
        let mut psq = Accumulator {
            vals: [[0i16; HIDDEN_SIZE]; 2],
        };
        let mut threat = ThreatAccumulator::new();
        for side in 0..2 {
            for i in 0..HIDDEN_SIZE {
                psq.vals[side][i] = rng.centered(1200) as i16;
                threat.vals[side][i] = rng.centered(600) as i16;
            }
        }
        (psq, threat)
    }

    #[test]
    fn pairwise_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            let mut rng = Rng(0x5EED);
            for _ in 0..64 {
                let (psq, threat) = random_accumulators(rng.next());
                for side in 0..2 {
                    let mut expected = [0u8; PAIR_COUNT];
                    pairwise_side_scalar(&psq.vals[side], &threat.vals[side], &mut expected);
                    let mut actual = [0u8; PAIR_COUNT];
                    unsafe { pairwise_side_avx2(&psq.vals[side], &threat.vals[side], &mut actual) };
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn l1_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            let net = random_head(0xABCD);
            let mut rng = Rng(0x1234);
            for _ in 0..16 {
                let (psq, threat) = random_accumulators(rng.next());
                let mut input = [0u8; HIDDEN_SIZE];
                pairwise_side_scalar(&psq.vals[0], &threat.vals[0], (&mut input[..PAIR_COUNT]).try_into().unwrap());
                pairwise_side_scalar(&psq.vals[1], &threat.vals[1], (&mut input[PAIR_COUNT..]).try_into().unwrap());
                for bucket in 0..NUM_OUTPUT_BUCKETS {
                    let mut expected = [0i32; L2_IN];
                    l1_forward_scalar(
                        &net.head.l1_weights[bucket],
                        &net.head.l1_bias[bucket],
                        &input,
                        &mut expected,
                    );
                    let mut actual = [0i32; L2_IN];
                    unsafe {
                        l1_forward_avx2(
                            &net.head.l1_weights[bucket],
                            &net.head.l1_bias[bucket],
                            &input,
                            &mut actual,
                        )
                    };
                    assert_eq!(actual, expected, "bucket {bucket}");
                }
            }
        }
    }

    #[test]
    fn l2_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            let net = random_head(0x9876);
            let mut rng = Rng(0x4321);
            for _ in 0..32 {
                let mut input = [0i32; L2_IN];
                for value in input.iter_mut() {
                    *value = rng.centered(1 << 20);
                }
                for bucket in 0..NUM_OUTPUT_BUCKETS {
                    let mut expected = [0i32; L2_SIZE];
                    l2_forward_scalar(
                        &net.head.l2_weights[bucket],
                        &net.head.l2_bias[bucket],
                        &input,
                        &mut expected,
                    );
                    let mut actual = [0i32; L2_SIZE];
                    unsafe {
                        l2_forward_avx2(
                            &net.head.l2_weights[bucket],
                            &net.head.l2_bias[bucket],
                            &input,
                            &mut actual,
                        )
                    };
                    assert_eq!(actual, expected, "bucket {bucket}");
                }
            }
        }
    }

    #[test]
    fn heimdall_head_simd_matches_scalar() {
        let net = random_head(0x00DD_BA11);
        let mut rng = Rng(0xFEED);
        for _ in 0..32 {
            let (psq, threat) = random_accumulators(rng.next());
            let us = (rng.next() & 1) as usize;
            let them = us ^ 1;
            for bucket in 0..NUM_OUTPUT_BUCKETS {
                let expected = forward_scalar(&psq, &threat, us, them, bucket, &net.head);
                let actual = forward_heimdall(&psq, &threat, us, them, bucket, &net.head);
                assert_eq!(actual, expected, "bucket {bucket}");
            }
        }
    }

    #[test]
    fn head_input_shapes_match_the_kernels() {
        assert_eq!(L2_IN, 2 * L1_SIZE);
        assert_eq!(L2_SIZE % 8, 0, "the AVX2 layer-2 kernel stores two vectors");
        assert_eq!(PAIR_COUNT % 32, 0, "the AVX2 pairwise kernel packs 32 pairs");
        assert_eq!(HIDDEN_SIZE % 32, 0);
        assert_eq!(L1_SHIFT, 8);
        assert_eq!(QUANT_FOURTH, 16_777_216);
    }

    #[test]
    fn network_storage_is_aligned_for_simd() {
        let net = network() as *const Network as usize;
        assert_eq!(net % 64, 0, "the network must be 64-byte aligned");
    }

    #[test]
    fn detect_format_recognises_both_layouts() {
        assert_eq!(detect_format(LEGACY_NET_BYTES), Some(NetFormat::Legacy));
        assert_eq!(detect_format(NETWORK_BYTES.len()), Some(NetFormat::Legacy));
        assert_eq!(detect_format(HEIMDALL_NET_BYTES), Some(NetFormat::Heimdall));
        assert_eq!(detect_format(HEIMDALL_NET_BYTES - 1), Some(NetFormat::Legacy));
        assert_eq!(detect_format(0), None);
        assert_eq!(detect_format(HEIMDALL_NET_BYTES + 1), None);
    }

    #[test]
    fn heimdall_loader_reads_the_disk_layout() {
        let mut bytes = vec![0u8; HEIMDALL_NET_BYTES];
        let ft_bytes = FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS * HIDDEN_SIZE * 2;
        let threat_bytes = TOTAL_THREATS * HIDDEN_SIZE;
        let bias_bytes = HIDDEN_SIZE * 2;
        let l1_bytes = HIDDEN_SIZE * L1_SIZE * NUM_OUTPUT_BUCKETS;
        let l1_bias_bytes = NUM_OUTPUT_BUCKETS * L1_SIZE * 4;
        let l2_bytes = L2_IN * NUM_OUTPUT_BUCKETS * L2_SIZE * 4;
        let l2_bias_bytes = NUM_OUTPUT_BUCKETS * L2_SIZE * 4;
        let l3_bytes = L2_SIZE * NUM_OUTPUT_BUCKETS * 4;

        let put_i16 = |bytes: &mut [u8], offset: usize, value: i16| {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        };
        let put_i32 = |bytes: &mut [u8], offset: usize, value: i32| {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        };

        // Last piece-square feature, last lane.
        let last_feature = FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS - 1;
        put_i16(&mut bytes, last_feature * HIDDEN_SIZE * 2 + 511 * 2, -1234);
        // Last threat row, last lane: a signed byte, so 0x80 is -128.
        bytes[ft_bytes + (TOTAL_THREATS - 1) * HIDDEN_SIZE + 511] = 0x80;
        put_i16(&mut bytes, ft_bytes + threat_bytes + 511 * 2, 777);
        // Layer 1 is stored input-major: (input, bucket, output).
        let l1_base = ft_bytes + threat_bytes + bias_bytes;
        bytes[l1_base + ((511 * NUM_OUTPUT_BUCKETS + 7) * L1_SIZE + 31)] = 0xF9;
        let l1_bias_base = l1_base + l1_bytes;
        put_i32(&mut bytes, l1_bias_base + (7 * L1_SIZE + 31) * 4, 123_456);
        let l2_base = l1_bias_base + l1_bias_bytes;
        put_i32(&mut bytes, l2_base + ((63 * NUM_OUTPUT_BUCKETS + 7) * L2_SIZE + 15) * 4, -654_321);
        let l2_bias_base = l2_base + l2_bytes;
        put_i32(&mut bytes, l2_bias_base + (7 * L2_SIZE + 15) * 4, 4242);
        let l3_base = l2_bias_base + l2_bias_bytes;
        put_i32(&mut bytes, l3_base + (15 * NUM_OUTPUT_BUCKETS + 7) * 4, 99);
        let l3_bias_base = l3_base + l3_bytes;
        put_i32(&mut bytes, l3_bias_base + 7 * 4, -99);

        let net = allocate_network();
        load_heimdall_bytes(&bytes, net).expect("the synthetic network must load");

        assert_eq!((net as *mut Network as usize) % 64, 0);
        assert_eq!(net.feature_weights[last_feature].vals[511], -1234);
        assert_eq!(net.feature_bias.vals[511], 777);

        let heimdall = net.heimdall.as_ref().expect("head must be present");
        assert_eq!(heimdall.head.l1_weights[7][31][511], -7);
        assert_eq!(heimdall.head.l1_bias[7][31], 123_456);
        assert_eq!(heimdall.head.l2_weights[7][63][15], -654_321);
        assert_eq!(heimdall.head.l2_bias[7][15], 4242);
        assert_eq!(heimdall.head.l3_weights[7][15][0], 99);
        assert_eq!(heimdall.head.l3_bias[7][0], -99);

        let rows = &heimdall.threat_weights.as_ref().unwrap().rows;
        assert_eq!(rows[(TOTAL_THREATS - 1) * HIDDEN_SIZE + 511], -128);
        assert_eq!(rows.len(), TOTAL_THREATS * HIDDEN_SIZE);
    }

    #[test]
    fn heimdall_loader_rejects_other_sizes() {
        let net = allocate_network();
        let short = vec![0u8; HEIMDALL_NET_BYTES - 8];
        assert!(load_heimdall_bytes(&short, net).is_err());
    }

    const THREAT_TEST_FENS: [&str; 7] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1N2/PPPN1PPP/R1BQKB1R w KQ - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "2r1k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    ];

    /// Mirrors the piece placement of a FEN left to right. Castling rights and
    /// the en passant square are dropped: the threat inputs only depend on the
    /// pieces and the kings.
    fn mirror_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().unwrap();
        let side = parts.next().unwrap_or("w");
        let ranks: Vec<String> = placement
            .split('/')
            .map(|rank| rank.chars().rev().collect())
            .collect();
        format!("{} {side} - - 0 1", ranks.join("/"))
    }

    #[test]
    fn threat_index_space_is_exactly_60144_features() {
        let tables = threat_tables();
        let mut total = 0usize;
        for color in 0..2 {
            for (kind, &target_count) in PIECE_TARGET_COUNT.iter().enumerate() {
                let span = tables.span[color][kind] as usize;
                total += 2 * target_count as usize * span;
                assert_eq!(tables.offsets[color][kind][0], 0);
                assert!(
                    tables.offsets[color][kind].windows(2).all(|w| w[0] <= w[1]),
                    "attacker square offsets must not decrease"
                );
                for below in 0..2 {
                    for vcolor in 0..2 {
                        for vkind in 0..6 {
                            let base = tables.attack[color][kind][vcolor][vkind][below];
                            assert!(base == THREAT_SENTINEL || (base as usize) < TOTAL_THREATS);
                        }
                    }
                }
            }
        }
        assert_eq!(total, TOTAL_THREATS, "the tables must cover the whole input space");
    }

    #[test]
    fn threat_features_track_a_single_pawn_attack() {
        // White pawn e4 attacks the black knight on d5, and nothing else on the
        // board attacks or defends another piece.
        let board = Board::from_fen("7k/8/8/3n4/4P3/8/8/K7 w - - 0 1").unwrap();
        let mut list = ThreatList::new();

        collect_threats(&board, Color::White, &mut list);
        assert_eq!(list.as_slice().len(), 1, "white should see its pawn threat");
        let white_index = list.as_slice()[0];

        collect_threats(&board, Color::Black, &mut list);
        assert_eq!(list.as_slice().len(), 1, "black should see the same relation");
        assert_ne!(white_index, list.as_slice()[0]);
    }

    #[test]
    fn threat_features_are_mirror_invariant() {
        for fen in THREAT_TEST_FENS {
            let board = Board::from_fen(fen).unwrap();
            let mirrored = Board::from_fen(&mirror_fen(fen)).unwrap();
            for view in [Color::White, Color::Black] {
                let mut plain = ThreatList::new();
                let mut flipped = ThreatList::new();
                collect_threats(&board, view, &mut plain);
                collect_threats(&mirrored, view, &mut flipped);
                assert_eq!(plain.as_slice(), flipped.as_slice(), "{fen} for {view:?}");
            }
        }
    }

    #[test]
    fn threat_lists_are_deterministic_and_unique() {
        let mut board = Board::default();
        let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
        for ply in 0..200 {
            for view in [Color::White, Color::Black] {
                let mut first = ThreatList::new();
                let mut second = ThreatList::new();
                collect_threats(&board, view, &mut first);
                collect_threats(&board, view, &mut second);
                assert_eq!(first.as_slice(), second.as_slice(), "ply {ply}");
                assert!(
                    first.as_slice().iter().all(|&index| (index as usize) < TOTAL_THREATS),
                    "threat index out of range at ply {ply}"
                );
                for pair in first.as_slice().windows(2) {
                    assert!(pair[0] < pair[1], "duplicate threat index at ply {ply}");
                }
            }

            let moves = generate_legal_moves(&board);
            if moves.count == 0 {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            board.make_move(moves.get((rng as usize) % moves.count));
            board.reset_accumulator_stack();
        }
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
#[inline(always)]
unsafe fn hsum_epi32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let hi = _mm256_extracti128_si256::<1>(v);
        let lo = _mm256_castsi256_si128(v);
        let mut sum = _mm_add_epi32(lo, hi);
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        _mm_cvtsi128_si32(sum)
    }
}

/// Pairwise reduction, 32 pairs per iteration. `mulhi` is the same trick the
/// scalar reference spells out as `(a << FT_SCALE_BITS) * b >> 16`: the high
/// half of the 32-bit product is exactly the arithmetic shift of the product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_side_avx2(
    psq: &[i16; HIDDEN_SIZE],
    threats: &[i16; HIDDEN_SIZE],
    out: &mut [u8; PAIR_COUNT],
) {
    use std::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA);
        let max = _mm256_set1_epi16(255);

        let pair = |offset: usize| {
            let lo = _mm256_loadu_si256(psq.as_ptr().add(offset) as *const __m256i);
            let lo = _mm256_add_epi16(lo, _mm256_loadu_si256(threats.as_ptr().add(offset) as *const __m256i));
            let hi = _mm256_loadu_si256(psq.as_ptr().add(offset + PAIR_COUNT) as *const __m256i);
            let hi = _mm256_add_epi16(
                hi,
                _mm256_loadu_si256(threats.as_ptr().add(offset + PAIR_COUNT) as *const __m256i),
            );
            // The first factor is clipped on both sides, the second only from
            // above: a non-negative factor cannot make the product too small.
            let a = _mm256_min_epi16(_mm256_max_epi16(lo, zero), qa);
            let b = _mm256_min_epi16(hi, qa);
            let product = _mm256_mulhi_epi16(_mm256_slli_epi16::<{ FT_SCALE_BITS as i32 }>(a), b);
            _mm256_min_epi16(_mm256_max_epi16(product, zero), max)
        };

        let mut i = 0;
        while i < PAIR_COUNT {
            let p0 = pair(i);
            let p1 = pair(i + 16);
            // packus works inside 128-bit lanes, so the two halves have to be
            // rejoined before the bytes go out in order.
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(p0, p1));
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, packed);
            i += 32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_forward_avx2(
    weights: &[[i8; HIDDEN_SIZE]; L1_SIZE],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    out: &mut [i32; L2_IN],
) {
    use std::arch::x86_64::*;
    unsafe {
        for j in 0..L1_SIZE {
            let mut acc = _mm256_setzero_si256();
            let row = weights[j].as_ptr();
            let mut i = 0;
            // 255 * 127 * 2 fits an i32 lane with room to spare, so widening
            // both operands to 16 bits keeps every product exact.
            while i < HIDDEN_SIZE {
                let x = _mm256_cvtepu8_epi16(_mm_loadu_si128(input.as_ptr().add(i) as *const __m128i));
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(i) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, w));
                i += 16;
            }

            let output = hsum_epi32_avx2(acc).wrapping_add(bias[j]) >> L1_SHIFT;
            out[j] = output.clamp(0, QUANT) << QUANT_BITS;
            out[L1_SIZE + j] =
                (i64::from(output) * i64::from(output)).min(i64::from(QUANT * QUANT)) as i32;
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn l1_forward_avx512(
    weights: &[[i8; HIDDEN_SIZE]; L1_SIZE],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    out: &mut [i32; L2_IN],
) {
    use std::arch::x86_64::*;
    unsafe {
        for j in 0..L1_SIZE {
            let mut acc = _mm512_setzero_si512();
            let row = weights[j].as_ptr();
            let mut i = 0;
            while i < HIDDEN_SIZE {
                let x = _mm512_cvtepu8_epi16(_mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i));
                let w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(row.add(i) as *const __m256i));
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x, w));
                i += 32;
            }

            let output = _mm512_reduce_add_epi32(acc).wrapping_add(bias[j]) >> L1_SHIFT;
            out[j] = output.clamp(0, QUANT) << QUANT_BITS;
            out[L1_SIZE + j] =
                (i64::from(output) * i64::from(output)).min(i64::from(QUANT * QUANT)) as i32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_forward_avx2(
    weights: &[[i32; L2_SIZE]; L2_IN],
    bias: &[i32; L2_SIZE],
    input: &[i32; L2_IN],
    out: &mut [i32; L2_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        // Wrapping 32-bit sums are exactly the low half of the scalar's 64-bit
        // accumulation, so both paths agree bit for bit.
        let mut acc0 = _mm256_loadu_si256(bias.as_ptr() as *const __m256i);
        let mut acc1 = _mm256_loadu_si256(bias.as_ptr().add(8) as *const __m256i);
        for i in 0..L2_IN {
            let x = _mm256_set1_epi32(input[i]);
            let row = weights[i].as_ptr() as *const __m256i;
            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(x, _mm256_loadu_si256(row)));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(x, _mm256_loadu_si256(row.add(1))));
        }
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc0);
        _mm256_storeu_si256(out.as_mut_ptr().add(8) as *mut __m256i, acc1);
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
