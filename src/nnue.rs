use crate::alloc::{HugeBox, HugeBuffer};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, PIECE_NB, Piece, PieceType, Square};
use std::mem::MaybeUninit;

pub const HIDDEN_SIZE: usize = 512;
pub const NUM_INPUT_BUCKETS: usize = 16;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const FEATURES_PER_BUCKET: usize = 768;
pub const SCALE: i32 = 400;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

// 16hm king input buckets (a-d files, e-h mirrored)
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

// old 10-bucket layout for embedded net
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

// map 16 buckets back to legacy 10 buckets
const LEGACY_BUCKET_SOURCE: [usize; NUM_INPUT_BUCKETS] = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];

pub const LEGACY_NUM_INPUT_BUCKETS: usize = 10;

// size of the embedded 10-bucket network: weights, bias and output layer
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
    // multilayer head when using multilayer net
    pub multilayer: Option<Box<MultilayerNet>>,
}

static NETWORK_BYTES: &[u8] = include_bytes!("../resources/ruthless.bin");

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

#[repr(C, align(64))]
struct ZeroWeights([i16; HIDDEN_SIZE]);

static ZERO_WEIGHTS: ZeroWeights = ZeroWeights([0; HIDDEN_SIZE]);

static NETWORK_PTR: AtomicPtr<Network> = AtomicPtr::new(std::ptr::null_mut());
static NETWORK_INIT: OnceLock<&'static Network> = OnceLock::new();

#[inline(always)]
pub fn network() -> &'static Network {
    let ptr = NETWORK_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        return unsafe { &*ptr };
    }
    init_network_cold()
}

#[cold]
#[inline(never)]
fn init_network_cold() -> &'static Network {
    let net = NETWORK_INIT.get_or_init(init_network);
    NETWORK_PTR.store(*net as *const Network as *mut Network, Ordering::Release);
    net
}

// Eagerly build every lazily-initialised table. Without this the first
// Board::from_fen pays for the 40+ MB network copy and the threat table
// construction, which shows up as a large one-off cost in the first bench
// position (and as a hiccup on the first real search).
pub fn warmup() {
    let _ = network();
    let _ = threat_tables();
    let _ = threat_weights();
}

// copies one 512-lane weight row out of the little-endian network image
#[inline(always)]
fn read_row(src: &[u8], dst: &mut [i16; HIDDEN_SIZE]) {
    debug_assert!(src.len() >= HIDDEN_SIZE * 2);
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr() as *mut u8, HIDDEN_SIZE * 2);
    }
    #[cfg(not(target_endian = "little"))]
    for (d, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
        *d = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
}

// expands the embedded 10-bucket network into the 16-bucket feature space
fn load_legacy_network(bytes: &[u8], net: &mut Network) {
    assert!(bytes.len() >= LEGACY_NET_BYTES, "embedded network is truncated");

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

// 64-byte aligned static network alloc
fn allocate_network() -> &'static mut Network {
    // embedded 10-bucket net gets expanded here
    let buffer = HugeBox::<Network>::zeroed();
    let ptr = buffer.leak();
    unsafe { &mut *ptr }
}

fn init_network() -> &'static Network {
    let net = allocate_network();

    if !load_external_network(net) {
        match detect_format(NETWORK_BYTES.len()) {
            Some(NetFormat::Multilayer) => {
                load_multilayer_bytes(NETWORK_BYTES, net).expect("failed to load embedded multilayer network");
            }
            Some(NetFormat::Legacy) => {
                load_legacy_network(NETWORK_BYTES, net);
            }
            None => {
                panic!("Embedded network has unknown size: {} bytes", NETWORK_BYTES.len());
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // make sure weights * QA don't overflow i16 in avx2
        if has_avx2() && net.multilayer.is_none() {
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

// which network architecture the engine is running
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NetFormat {
    // the original embedded network: 10 buckets expanded to 16, linear output
    Legacy,
    // 16hm piece-square inputs plus threat inputs feeding the multilayer head
    Multilayer,
}

pub const MULTILAYER_NET_BYTES: usize = FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS * HIDDEN_SIZE * 2
    + TOTAL_THREATS * HIDDEN_SIZE
    + HIDDEN_SIZE * 2
    + HIDDEN_SIZE * L1_SIZE * NUM_OUTPUT_BUCKETS
    + NUM_OUTPUT_BUCKETS * L1_SIZE * 4
    + L2_IN * NUM_OUTPUT_BUCKETS * L2_SIZE * 4
    + NUM_OUTPUT_BUCKETS * L2_SIZE * 4
    + L2_SIZE * NUM_OUTPUT_BUCKETS * 4
    + NUM_OUTPUT_BUCKETS * 4;

// detect net layout by file size
pub fn detect_format(len: usize) -> Option<NetFormat> {
    if len == MULTILAYER_NET_BYTES {
        Some(NetFormat::Multilayer)
    } else if (LEGACY_NET_BYTES..MULTILAYER_NET_BYTES).contains(&len) {
        Some(NetFormat::Legacy)
    } else {
        None
    }
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
        let slice = self.data.get(self.pos..end).ok_or_else(|| format!("network truncated at byte {}", self.pos))?;
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

// load multilayer net and transpose l1 weights
fn load_multilayer_bytes(bytes: &[u8], net: &mut Network) -> Result<(), String> {
    if bytes.len() != MULTILAYER_NET_BYTES {
        return Err(format!(
            "expected {MULTILAYER_NET_BYTES} bytes for the 16hm multilayer layout, got {}",
            bytes.len()
        ));
    }

    let mut reader = NetBytes::new(bytes);

    for feature in 0..FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS {
        reader.i16_row(&mut net.feature_weights[feature].vals)?;
    }

    let raw = reader.take(TOTAL_THREATS * HIDDEN_SIZE)?;
    // safe to cast bytes to i8
    let signed = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
    let mut rows = HugeBuffer::<i8>::zeroed(TOTAL_THREATS * HIDDEN_SIZE);
    rows.copy_from_slice(signed);

    reader.i16_row(&mut net.feature_bias.vals)?;

    let mut head: Box<MultilayerHead> = Box::new(unsafe { std::mem::zeroed() });

    for input in 0..HIDDEN_SIZE {
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            let chunk = reader.take(L1_SIZE)?;
            for (output, &byte) in chunk.iter().enumerate() {
                head.l1_weights[bucket][output][input] = byte as i8;
            }
        }
    }
    rebuild_l1_transposed(&mut head);
    for bucket in 0..NUM_OUTPUT_BUCKETS {
        for output in 0..L1_SIZE {
            head.l1_bias[bucket][output] = reader.i32()?;
        }
    }
    for input in 0..L2_IN {
        for bucket in 0..NUM_OUTPUT_BUCKETS {
            for output in 0..L2_SIZE {
                let value = reader.i32()?;
                if !(-128..=127).contains(&value) {
                    return Err(format!("l2 weight out of i8 range: {value}"));
                }
                head.l2_weights[bucket][output][input] = value as i16;
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

    debug_assert_eq!(reader.pos, MULTILAYER_NET_BYTES, "unread bytes in the network file");

    net.multilayer = Some(Box::new(MultilayerNet {
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
    for name in ["ruthless.bin", "resources/ruthless.bin"] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// try loading external net, fallback to embedded
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
        Some(NetFormat::Multilayer) => match load_multilayer_bytes(&bytes, net) {
            Ok(()) => {
                eprintln!("info string loaded multilayer network {} ({} MiB)", path.display(), bytes.len() / (1024 * 1024));
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
            eprintln!("info string network {} has an unsupported size ({} bytes)", path.display(), bytes.len());
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
    #[allow(dead_code)]
    pub fn refresh_side_occ(&mut self, piece_on: &[Piece; 64], mut occ: Bitboard, ksq: Square, color: Color) {
        let net = network();
        let side = color as usize;
        self.vals[side] = net.feature_bias.vals;

        while !occ.is_empty() {
            let sq = occ.pop_lsb();
            let piece = piece_on[sq];
            let idx = feature_index_side(piece, sq, ksq, color);
            add_feature_weights(&mut self.vals[side], &net.feature_weights[idx].vals);
        }
    }

    #[inline(always)]
    pub fn apply_delta_from_side(&mut self, src: &Accumulator, side: usize, delta: &FeatureDelta) {
        let net = network();
        // the second perspective is applied right after this one, so warm its
        // weight rows while the first one is being computed
        #[cfg(target_arch = "x86_64")]
        if side == 0 && has_avx2() {
            unsafe {
                prefetch_psq_list(net, &delta.adds[1][..delta.add_count[1] as usize]);
                prefetch_psq_list(net, &delta.subs[1][..delta.sub_count[1] as usize]);
            }
        }
        let adds = &delta.adds[side];
        let subs = &delta.subs[side];
        let add_count = delta.add_count[side];
        let sub_count = delta.sub_count[side];
        if add_count == 0 && sub_count == 0 {
            self.vals[side] = src.vals[side];
            return;
        }
        if add_count == 1 && sub_count == 1 {
            let add = &net.feature_weights[adds[0] as usize].vals;
            let sub = &net.feature_weights[subs[0] as usize].vals;
            apply_delta_weights_11_from(&mut self.vals[side], &src.vals[side], add, sub);
            return;
        }
        if add_count == 1 && sub_count == 2 {
            let add = &net.feature_weights[adds[0] as usize].vals;
            let sub1 = &net.feature_weights[subs[0] as usize].vals;
            let sub2 = &net.feature_weights[subs[1] as usize].vals;
            apply_delta_weights_12_from(&mut self.vals[side], &src.vals[side], add, sub1, sub2);
            return;
        }
        let add1 = feature_weights_or_zero(net, adds[0]);
        let add2 = feature_weights_or_zero(net, adds[1]);
        let sub1 = feature_weights_or_zero(net, subs[0]);
        let sub2 = feature_weights_or_zero(net, subs[1]);
        apply_delta_weights_from(&mut self.vals[side], &src.vals[side], add1, add2, sub1, sub2);
    }
}

// cached runtime avx2 check
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

        // 0 = not checked yet, 1 = unsupported, 2 = supported
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

// AVX-512 VNNI support, cached after the first runtime check
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
fn has_avx512_vnni() -> bool {
    #[cfg(target_feature = "avx512vnni")] { return true; }
    use std::sync::atomic::{AtomicU8, Ordering};

    static VNNI: AtomicU8 = AtomicU8::new(0);

    match VNNI.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let supported = std::arch::is_x86_feature_detected!("avx512vnni");
            VNNI.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
            supported
        }
    }
}

// AVX-VNNI (256-bit) support, cached after the first runtime check
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx_vnni() -> bool {
    #[cfg(target_feature = "avxvnni")] { return true; }
    use std::sync::atomic::{AtomicU8, Ordering};

    static VNNI: AtomicU8 = AtomicU8::new(0);

    match VNNI.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let supported = std::arch::is_x86_feature_detected!("avxvnni");
            VNNI.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
            supported
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn add_feature_weights(vals: &mut [i16; HIDDEN_SIZE], weights: &[i16; HIDDEN_SIZE]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        unsafe { vec_add_avx512(vals, weights) };
        return;
    }

    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx512f", target_feature = "avx512bw"))))]
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
fn apply_delta_weights_from(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { vec_delta_avx2_from(dst, src, add1, add2, sub1, sub2) };
            return;
        }
    }
    for i in 0..HIDDEN_SIZE {
        dst[i] = src[i].wrapping_add(add1[i]).wrapping_add(add2[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
    }
}

#[inline(always)]
fn apply_delta_weights_11_from(dst: &mut [i16; HIDDEN_SIZE], src: &[i16; HIDDEN_SIZE], add: &[i16; HIDDEN_SIZE], sub: &[i16; HIDDEN_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { vec_delta_avx2_11_from(dst, src, add, sub) };
            return;
        }
    }
    for i in 0..HIDDEN_SIZE {
        dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_delta_avx2_from(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    add1: &[i16; HIDDEN_SIZE],
    add2: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = dst.as_mut_ptr() as *mut __m256i;
        let s_ptr = src.as_ptr() as *const __m256i;
        let a1_ptr = add1.as_ptr() as *const __m256i;
        let a2_ptr = add2.as_ptr() as *const __m256i;
        let s1_ptr = sub1.as_ptr() as *const __m256i;
        let s2_ptr = sub2.as_ptr() as *const __m256i;
        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_load_si256(s_ptr.add(i));
            let v = _mm256_add_epi16(v, _mm256_load_si256(a1_ptr.add(i)));
            let v = _mm256_add_epi16(v, _mm256_load_si256(a2_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_load_si256(s1_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_load_si256(s2_ptr.add(i)));
            _mm256_store_si256(d_ptr.add(i), v);
        }
    }
}

#[inline(always)]
fn apply_delta_weights_12_from(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { vec_delta_avx2_12_from(dst, src, add, sub1, sub2) };
            return;
        }
    }
    for i in 0..HIDDEN_SIZE {
        dst[i] = src[i].wrapping_add(add[i]).wrapping_sub(sub1[i]).wrapping_sub(sub2[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_delta_avx2_12_from(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    add: &[i16; HIDDEN_SIZE],
    sub1: &[i16; HIDDEN_SIZE],
    sub2: &[i16; HIDDEN_SIZE],
) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = dst.as_mut_ptr() as *mut __m256i;
        let s_ptr = src.as_ptr() as *const __m256i;
        let a_ptr = add.as_ptr() as *const __m256i;
        let s1_ptr = sub1.as_ptr() as *const __m256i;
        let s2_ptr = sub2.as_ptr() as *const __m256i;
        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_load_si256(s_ptr.add(i));
            let v = _mm256_add_epi16(v, _mm256_load_si256(a_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_load_si256(s1_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_load_si256(s2_ptr.add(i)));
            _mm256_store_si256(d_ptr.add(i), v);
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_delta_avx2_11_from(dst: &mut [i16; HIDDEN_SIZE], src: &[i16; HIDDEN_SIZE], add: &[i16; HIDDEN_SIZE], sub: &[i16; HIDDEN_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        let d_ptr = dst.as_mut_ptr() as *mut __m256i;
        let s_ptr = src.as_ptr() as *const __m256i;
        let a_ptr = add.as_ptr() as *const __m256i;
        let sub_ptr = sub.as_ptr() as *const __m256i;
        for i in 0..(HIDDEN_SIZE / 16) {
            let v = _mm256_load_si256(s_ptr.add(i));
            let v = _mm256_add_epi16(v, _mm256_load_si256(a_ptr.add(i)));
            let v = _mm256_sub_epi16(v, _mm256_load_si256(sub_ptr.add(i)));
            _mm256_store_si256(d_ptr.add(i), v);
        }
    }
}

#[inline(always)]
pub fn king_bucket(sq: Square) -> usize {
    if !sq.is_valid() {
        return 0; // fallback for invalid squares (e.g., missing king in test positions)
    }
    let file = sq.file();
    let rank = sq.rank();
    let mirrored_file = if file > 3 { 7 - file } else { file };
    BUCKET_LAYOUT[(rank * 4 + mirrored_file) as usize]
}

// legacy 10-bucket lookup for tests
#[cfg(test)]
pub fn legacy_king_bucket(sq: Square) -> usize {
    if !sq.is_valid() {
        return 0;
    }
    let file = sq.file();
    let rank = sq.rank();
    let mirrored_file = if file > 3 { 7 - file } else { file };
    LEGACY_BUCKET_LAYOUT[(rank * 4 + mirrored_file) as usize]
}

// need full refresh when king changes bucket or crosses mirror
#[allow(dead_code)]
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
    if pc_color == view { pt } else { pt + 6 }
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
pub fn king_bucket_flip(ksq: Square, view: Color) -> (usize, u8) {
    let oriented_ksq = if view == Color::White { ksq } else { Square::new((ksq as u8) ^ 56) };
    let flip = if oriented_ksq.file() > 3 { 7 } else { 0 };
    (king_bucket(oriented_ksq), flip)
}
// king-dependent feature bases, cached per accumulator frame.
// feature index = base[view] + oriented_piece * 64 + (square ^ mask[view])
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FeatureBases {
    // king bucket * 768
    pub base: [usize; 2],
    // vertical flip for the black view (56) xor the horizontal mirror (0 or 7)
    pub mask: [u8; 2],
}

impl FeatureBases {
    #[inline(always)]
    pub fn from_kings(w_ksq: Square, b_ksq: Square) -> Self {
        let (w_bucket, w_flip) = king_bucket_flip(w_ksq, Color::White);
        let (b_bucket, b_flip) = king_bucket_flip(b_ksq, Color::Black);
        Self {
            base: [w_bucket * 768, b_bucket * 768],
            mask: [w_flip, 56 ^ b_flip],
        }
    }

    // set one perspective from its king square
    #[inline(always)]
    pub fn set_king(&mut self, ksq: Square, view: usize) {
        let color = Color::from_index(view);
        let (bucket, flip) = king_bucket_flip(ksq, color);
        self.base[view] = bucket * 768;
        self.mask[view] = if color == Color::White { flip } else { 56 ^ flip };
    }

    #[inline(always)]
    pub fn bucket(&self, view: usize) -> usize {
        self.base[view] / 768
    }

    #[inline(always)]
    pub fn mirrored(&self, view: usize) -> bool {
        self.mask[view] & 7 != 0
    }

    #[inline(always)]
    pub fn index(&self, piece: Piece, sq: Square, view: usize) -> usize {
        self.base[view]
            + oriented_piece(piece, Color::from_index(view)) * 64
            + ((sq as u8 ^ self.mask[view]) as usize)
    }
}



#[inline(always)]
fn apply_psq_lists(vals: &mut [i16; HIDDEN_SIZE], dst: &mut [i16; HIDDEN_SIZE], net: &'static Network, adds: &[u16], subs: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { apply_psq_lists_avx2(vals, dst, net, adds, subs) };
            return;
        }
    }
    for &idx in adds {
        let w = &net.feature_weights[idx as usize].vals;
        for i in 0..HIDDEN_SIZE {
            vals[i] = vals[i].wrapping_add(w[i]);
        }
    }
    for &idx in subs {
        let w = &net.feature_weights[idx as usize].vals;
        for i in 0..HIDDEN_SIZE {
            vals[i] = vals[i].wrapping_sub(w[i]);
        }
    }
    *dst = *vals;
}

// prefetch one chunk of a feature row: `lanes` i16 starting at `base`
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_psq_chunk(net: &'static Network, index: u16, base: usize, lines: usize) {
    use std::arch::x86_64::*;
    unsafe {
        let chunk = (net.feature_weights.as_ptr().add(index as usize) as *const i8).add(base * 2);
        let mut line = 0;
        while line < lines {
            _mm_prefetch(chunk.add(line * 64), _MM_HINT_T0);
            line += 1;
        }
    }
}

// 192 hidden lanes (12 YMM accumulators)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_psq_chunk_12(
    vals: &mut [i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    net: &'static Network,
    adds: &[u16],
    subs: &[u16],
    base: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let s = vals.as_ptr().add(base) as *const __m256i;
        let mut a0 = _mm256_load_si256(s);
        let mut a1 = _mm256_load_si256(s.add(1));
        let mut a2 = _mm256_load_si256(s.add(2));
        let mut a3 = _mm256_load_si256(s.add(3));
        let mut a4 = _mm256_load_si256(s.add(4));
        let mut a5 = _mm256_load_si256(s.add(5));
        let mut a6 = _mm256_load_si256(s.add(6));
        let mut a7 = _mm256_load_si256(s.add(7));
        let mut a8 = _mm256_load_si256(s.add(8));
        let mut a9 = _mm256_load_si256(s.add(9));
        let mut a10 = _mm256_load_si256(s.add(10));
        let mut a11 = _mm256_load_si256(s.add(11));

        // one prefetch per row, a few rows ahead (see apply_threat_chunk_12)
        const PREFETCH_AHEAD: usize = 4;
        let rows = net.feature_weights.as_ptr() as *const i16;
        for (n, &idx) in subs.iter().enumerate() {
            if let Some(&ahead) = subs.get(n + PREFETCH_AHEAD) {
                _mm_prefetch(rows.add(ahead as usize * HIDDEN_SIZE + base) as *const i8, _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + base) as *const __m256i;
            a0 = _mm256_sub_epi16(a0, _mm256_load_si256(r));
            a1 = _mm256_sub_epi16(a1, _mm256_load_si256(r.add(1)));
            a2 = _mm256_sub_epi16(a2, _mm256_load_si256(r.add(2)));
            a3 = _mm256_sub_epi16(a3, _mm256_load_si256(r.add(3)));
            a4 = _mm256_sub_epi16(a4, _mm256_load_si256(r.add(4)));
            a5 = _mm256_sub_epi16(a5, _mm256_load_si256(r.add(5)));
            a6 = _mm256_sub_epi16(a6, _mm256_load_si256(r.add(6)));
            a7 = _mm256_sub_epi16(a7, _mm256_load_si256(r.add(7)));
            a8 = _mm256_sub_epi16(a8, _mm256_load_si256(r.add(8)));
            a9 = _mm256_sub_epi16(a9, _mm256_load_si256(r.add(9)));
            a10 = _mm256_sub_epi16(a10, _mm256_load_si256(r.add(10)));
            a11 = _mm256_sub_epi16(a11, _mm256_load_si256(r.add(11)));
        }
        for (n, &idx) in adds.iter().enumerate() {
            if let Some(&ahead) = adds.get(n + PREFETCH_AHEAD) {
                _mm_prefetch(rows.add(ahead as usize * HIDDEN_SIZE + base) as *const i8, _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + base) as *const __m256i;
            a0 = _mm256_add_epi16(a0, _mm256_load_si256(r));
            a1 = _mm256_add_epi16(a1, _mm256_load_si256(r.add(1)));
            a2 = _mm256_add_epi16(a2, _mm256_load_si256(r.add(2)));
            a3 = _mm256_add_epi16(a3, _mm256_load_si256(r.add(3)));
            a4 = _mm256_add_epi16(a4, _mm256_load_si256(r.add(4)));
            a5 = _mm256_add_epi16(a5, _mm256_load_si256(r.add(5)));
            a6 = _mm256_add_epi16(a6, _mm256_load_si256(r.add(6)));
            a7 = _mm256_add_epi16(a7, _mm256_load_si256(r.add(7)));
            a8 = _mm256_add_epi16(a8, _mm256_load_si256(r.add(8)));
            a9 = _mm256_add_epi16(a9, _mm256_load_si256(r.add(9)));
            a10 = _mm256_add_epi16(a10, _mm256_load_si256(r.add(10)));
            a11 = _mm256_add_epi16(a11, _mm256_load_si256(r.add(11)));
        }

        let v = vals.as_mut_ptr().add(base) as *mut __m256i;
        let d = dst.as_mut_ptr().add(base) as *mut __m256i;
        _mm256_store_si256(v, a0);
        _mm256_store_si256(v.add(1), a1);
        _mm256_store_si256(v.add(2), a2);
        _mm256_store_si256(v.add(3), a3);
        _mm256_store_si256(v.add(4), a4);
        _mm256_store_si256(v.add(5), a5);
        _mm256_store_si256(v.add(6), a6);
        _mm256_store_si256(v.add(7), a7);
        _mm256_store_si256(v.add(8), a8);
        _mm256_store_si256(v.add(9), a9);
        _mm256_store_si256(v.add(10), a10);
        _mm256_store_si256(v.add(11), a11);
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
        _mm256_store_si256(d.add(8), a8);
        _mm256_store_si256(d.add(9), a9);
        _mm256_store_si256(d.add(10), a10);
        _mm256_store_si256(d.add(11), a11);
    }
}

// 128 hidden lanes (8 YMM accumulators)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_psq_chunk_8(
    vals: &mut [i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    net: &'static Network,
    adds: &[u16],
    subs: &[u16],
    base: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let s = vals.as_ptr().add(base) as *const __m256i;
        let mut a0 = _mm256_load_si256(s);
        let mut a1 = _mm256_load_si256(s.add(1));
        let mut a2 = _mm256_load_si256(s.add(2));
        let mut a3 = _mm256_load_si256(s.add(3));
        let mut a4 = _mm256_load_si256(s.add(4));
        let mut a5 = _mm256_load_si256(s.add(5));
        let mut a6 = _mm256_load_si256(s.add(6));
        let mut a7 = _mm256_load_si256(s.add(7));

        let rows = net.feature_weights.as_ptr() as *const i16;
        for &idx in subs {
            let r = rows.add(idx as usize * HIDDEN_SIZE + base) as *const __m256i;
            a0 = _mm256_sub_epi16(a0, _mm256_load_si256(r));
            a1 = _mm256_sub_epi16(a1, _mm256_load_si256(r.add(1)));
            a2 = _mm256_sub_epi16(a2, _mm256_load_si256(r.add(2)));
            a3 = _mm256_sub_epi16(a3, _mm256_load_si256(r.add(3)));
            a4 = _mm256_sub_epi16(a4, _mm256_load_si256(r.add(4)));
            a5 = _mm256_sub_epi16(a5, _mm256_load_si256(r.add(5)));
            a6 = _mm256_sub_epi16(a6, _mm256_load_si256(r.add(6)));
            a7 = _mm256_sub_epi16(a7, _mm256_load_si256(r.add(7)));
        }
        for &idx in adds {
            let r = rows.add(idx as usize * HIDDEN_SIZE + base) as *const __m256i;
            a0 = _mm256_add_epi16(a0, _mm256_load_si256(r));
            a1 = _mm256_add_epi16(a1, _mm256_load_si256(r.add(1)));
            a2 = _mm256_add_epi16(a2, _mm256_load_si256(r.add(2)));
            a3 = _mm256_add_epi16(a3, _mm256_load_si256(r.add(3)));
            a4 = _mm256_add_epi16(a4, _mm256_load_si256(r.add(4)));
            a5 = _mm256_add_epi16(a5, _mm256_load_si256(r.add(5)));
            a6 = _mm256_add_epi16(a6, _mm256_load_si256(r.add(6)));
            a7 = _mm256_add_epi16(a7, _mm256_load_si256(r.add(7)));
        }

        let v = vals.as_mut_ptr().add(base) as *mut __m256i;
        let d = dst.as_mut_ptr().add(base) as *mut __m256i;
        _mm256_store_si256(v, a0);
        _mm256_store_si256(v.add(1), a1);
        _mm256_store_si256(v.add(2), a2);
        _mm256_store_si256(v.add(3), a3);
        _mm256_store_si256(v.add(4), a4);
        _mm256_store_si256(v.add(5), a5);
        _mm256_store_si256(v.add(6), a6);
        _mm256_store_si256(v.add(7), a7);
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_psq_lists_avx2(vals: &mut [i16; HIDDEN_SIZE], dst: &mut [i16; HIDDEN_SIZE], net: &'static Network, adds: &[u16], subs: &[u16]) {
    unsafe {
        // warm only the first chunk (128 bytes) of the first 1-2 rows
        // hardware prefetcher will pull the rest
        const WARM_ROWS: usize = 2;
        let warm = WARM_ROWS.min(subs.len());
        for &idx in &subs[..warm] {
            prefetch_psq_chunk(net, idx, 0, 2);
        }
        let warm = WARM_ROWS.min(adds.len());
        for &idx in &adds[..warm] {
            prefetch_psq_chunk(net, idx, 0, 2);
        }

        // 192 + 192 + 128 lanes cover the 512-lane row
        apply_psq_chunk_12(vals, dst, net, adds, subs, 0);
        apply_psq_chunk_12(vals, dst, net, adds, subs, 192);
        apply_psq_chunk_8(vals, dst, net, adds, subs, 384);
    }
}

// prefetch the first chunk of a small feature list (at most two entries per list)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn prefetch_psq_list(net: &'static Network, list: &[u16]) {
    unsafe {
        for &idx in list {
            prefetch_psq_chunk(net, idx, 0, 2);
        }
    }
}

// FinnyTable for incremental accumulator updates
#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct FinnyEntry {
    pub vals: [i16; HIDDEN_SIZE],
    pub pieces: [Bitboard; PIECE_NB],
    pub initialized: bool,
}

pub type FinnyTable = [[[FinnyEntry; 2]; NUM_INPUT_BUCKETS]; 2];

#[inline(always)]
pub fn refresh_side_with_finny(
    entry: &mut FinnyEntry,
    board_pieces: &[Bitboard; PIECE_NB],
    bases: &FeatureBases,
    view: usize,
    dst: &mut [i16; HIDDEN_SIZE],
) {
    let net = network();
    if !entry.initialized {
        entry.vals = net.feature_bias.vals;
        entry.pieces = [Bitboard::EMPTY; PIECE_NB];
        entry.initialized = true;
    }

    let mut adds = [0u16; 32];
    let mut subs = [0u16; 32];
    let mut add_count = 0usize;
    let mut sub_count = 0usize;

    // the king-dependent part of the index is fixed for the whole refresh
    let mask = bases.mask[view];
    for p in 0..PIECE_NB {
        let cur = board_pieces[p];
        let prev = entry.pieces[p];
        if cur == prev {
            continue;
        }
        let piece = Piece::from_index(p);
        // first lane of this piece's 64-square block
        let piece_base = bases.base[view] + oriented_piece(piece, Color::from_index(view)) * 64;
        let mut added = cur & !prev;
        while !added.is_empty() {
            let sq = added.pop_lsb();
            adds[add_count] = (piece_base + ((sq as u8 ^ mask) as usize)) as u16;
            add_count += 1;
        }
        let mut removed = prev & !cur;
        while !removed.is_empty() {
            let sq = removed.pop_lsb();
            subs[sub_count] = (piece_base + ((sq as u8 ^ mask) as usize)) as u16;
            sub_count += 1;
        }
        entry.pieces[p] = cur;
    }

    if add_count > 0 || sub_count > 0 {
        apply_psq_lists(&mut entry.vals, dst, net, &adds[..add_count], &subs[..sub_count]);
    } else {
        *dst = entry.vals;
    }
}

// empty slot marker
const NO_FEATURE: u16 = u16::MAX;

// feature delta per move (u16 to keep struct small)
#[derive(Copy, Clone)]
pub struct FeatureDelta {
    adds: [[u16; 2]; 2],
    subs: [[u16; 2]; 2],
    add_count: [u8; 2],
    sub_count: [u8; 2],
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
    pub fn add(&mut self, piece: Piece, sq: Square, bases: &FeatureBases, skip: Option<Color>) {
        if skip != Some(Color::White) {
            let view = Color::White as usize;
            self.adds[view][self.add_count[view] as usize] = bases.index(piece, sq, view) as u16;
            self.add_count[view] += 1;
        }
        if skip != Some(Color::Black) {
            let view = Color::Black as usize;
            self.adds[view][self.add_count[view] as usize] = bases.index(piece, sq, view) as u16;
            self.add_count[view] += 1;
        }
    }

    #[inline(always)]
    pub fn sub(&mut self, piece: Piece, sq: Square, bases: &FeatureBases, skip: Option<Color>) {
        if skip != Some(Color::White) {
            let view = Color::White as usize;
            self.subs[view][self.sub_count[view] as usize] = bases.index(piece, sq, view) as u16;
            self.sub_count[view] += 1;
        }
        if skip != Some(Color::Black) {
            let view = Color::Black as usize;
            self.subs[view][self.sub_count[view] as usize] = bases.index(piece, sq, view) as u16;
            self.sub_count[view] += 1;
        }
    }
}

// weight vector of a delta slot, or the shared zero vector when the slot is
// unused.
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
    let y = i32::from(x).max(0).min(i32::from(QA));
    y * y
}

// scalar fallback and reference for simd tests
fn evaluate_scalar(acc: &Accumulator, us: usize, them: usize, bucket: usize, net: &'static Network) -> i32 {
    let mut output = 0i64;

    // first weight half pairs with the side to move, the second half with the
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
pub fn evaluate(board: &mut Board) -> i32 {
    board.ensure_threats_ready();
    board.ensure_accumulator_ready();
    let net = network();
    let bucket = output_bucket(board);
    let us = board.side_to_move as usize;
    let them = us ^ 1;
    if let Some(ml) = net.multilayer.as_deref() {
        let threat_us = board.threat_side_immut(us);
        let threat_them = board.threat_side_immut(them);
        return forward_multilayer(board.accumulator_immut(), threat_us, threat_them, us, them, bucket, &ml.head);
    }
    evaluate_acc(board.accumulator_immut(), us, them, bucket, net)
}

#[inline(always)]
pub fn evaluate_scaled(board: &mut Board, mul: i32, div: i32) -> i32 {
    board.ensure_threats_ready();
    board.ensure_accumulator_ready();
    let net = network();
    let bucket = output_bucket(board);
    let us = board.side_to_move as usize;
    let them = us ^ 1;
    if let Some(ml) = net.multilayer.as_deref() {
        let threat_us = board.threat_side_immut(us);
        let threat_them = board.threat_side_immut(them);
        let raw = forward_multilayer_raw(board.accumulator_immut(), threat_us, threat_them, us, them, bucket, &ml.head);
        return ((i128::from(raw) * i128::from(EVAL_SCALE) * i128::from(mul))
            / (i128::from(QUANT_FOURTH) * i128::from(div))) as i32;
    }
    evaluate_acc(board.accumulator_immut(), us, them, bucket, net) * mul / div
}

#[inline(always)]
fn evaluate_acc(acc: &Accumulator, us: usize, them: usize, bucket: usize, net: &'static Network) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        return unsafe { evaluate_avx512(acc, us, them, bucket, net) };
    }

    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx512f", target_feature = "avx512bw"))))]
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

// multilayer head: pairwise(512) -> 32 -> 16 -> 1

// hidden lanes of one perspective are paired as (i, i + PAIR_COUNT)
pub const PAIR_COUNT: usize = HIDDEN_SIZE / 2;
// l1 width per bucket (takes both sides' pairwise outputs)
pub const L1_SIZE: usize = 32;
// the AVX2 kernel consumes the L1 input in blocks of 4 bytes (4 pairwise activations)
pub const L1_INPUT_BLOCKS: usize = HIDDEN_SIZE / 4;
// and produces it in blocks of 8 outputs
pub const L1_OUTPUT_BLOCKS: usize = L1_SIZE / 8;
// [input_block][output_block] weight blocks of 32 bytes (8 outputs x 4 inputs)
pub const L1_TRANSPOSED_BLOCKS: usize = L1_INPUT_BLOCKS * L1_OUTPUT_BLOCKS;
// lane order produced by packus_epi16 on two 16-lane halves (AVX2 lane split);
// weights are stored in this order so pairwise_side does not need a permute
const PAIRWISE_PACK_ORDER: [usize; 32] = {
    let mut order = [0usize; 32];
    let mut i = 0;
    while i < 8 {
        order[i] = i;
        order[8 + i] = 16 + i;
        order[16 + i] = 8 + i;
        order[24 + i] = 24 + i;
        i += 1;
    }
    order
};

// first layer activated twice (CReLU + SCReLU), so double width
pub const L2_IN: usize = 2 * L1_SIZE;
pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 1;

pub const QUANT_BITS: u32 = 6;
pub const QUANT: i32 = 1 << QUANT_BITS;
pub const QUANT_CUBED: i32 = QUANT * QUANT * QUANT;
pub const QUANT_FOURTH: i64 = (QUANT as i64) * (QUANT as i64) * (QUANT as i64) * (QUANT as i64);
pub const FT_SCALE_BITS: u32 = 7;
pub const EVAL_SCALE: i32 = 400;
// requant shift for l1
pub const L1_SHIFT: u32 = 8;

// multilayer head + threat weights (512 i8 per threat)
#[repr(C, align(64))]
pub struct MultilayerNet {
    pub head: MultilayerHead,
    pub threat_weights: Option<Box<ThreatWeights>>,
}

pub struct ThreatWeights {
    // TOTAL_THREATS * HIDDEN_SIZE rows of signed bytes
    pub rows: HugeBuffer<i8>,
}

#[cfg(test)]
impl ThreatWeights {
    // random threat weights for tests
    pub fn random(seed: u64) -> Self {
        let mut state = seed | 1;
        let mut rows = HugeBuffer::<i8>::zeroed(TOTAL_THREATS * HIDDEN_SIZE);
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
pub struct MultilayerHead {
    // [bucket][output][input]
    pub l1_weights: [[[i8; HIDDEN_SIZE]; L1_SIZE]; NUM_OUTPUT_BUCKETS],
    // AVX2 copy of the first layer: [bucket][input_block * L1_OUTPUT_BLOCKS + output_block][8 * 4]
    // laid out for maddubs: 32 bytes = 8 outputs, 4 consecutive inputs each
    pub l1_weights_t: [[[i8; 32]; L1_TRANSPOSED_BLOCKS]; NUM_OUTPUT_BUCKETS],
    pub l1_bias: [[i32; L1_SIZE]; NUM_OUTPUT_BUCKETS],
    // [bucket][output][input]
    pub l2_weights: [[[i16; L2_IN]; L2_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l2_bias: [[i32; L2_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l3_weights: [[[i32; L3_SIZE]; L2_SIZE]; NUM_OUTPUT_BUCKETS],
    pub l3_bias: [[i32; L3_SIZE]; NUM_OUTPUT_BUCKETS],
}

// build the AVX2-friendly first-layer copy from the logical weights.
// the input index is also reordered to the pairwise_side output order, which
// lets that kernel store packed bytes directly (no vpermq).
fn rebuild_l1_transposed(head: &mut MultilayerHead) {
    for bucket in 0..NUM_OUTPUT_BUCKETS {
        for input_block in 0..L1_INPUT_BLOCKS {
            let chunk = input_block / 8;
            let pos = (input_block % 8) * 4;
            for output_block in 0..L1_OUTPUT_BLOCKS {
                let mut block = [0i8; 32];
                for jj in 0..8 {
                    let output = output_block * 8 + jj;
                    for ii in 0..4 {
                        let input = chunk * 32 + PAIRWISE_PACK_ORDER[pos + ii];
                        block[jj * 4 + ii] = head.l1_weights[bucket][output][input];
                    }
                }
                head.l1_weights_t[bucket][input_block * L1_OUTPUT_BLOCKS + output_block] = block;
            }
        }
    }
}

// threat accumulator
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThreatAccumulator {
    pub vals: [[i16; HIDDEN_SIZE]; 2],
}

impl ThreatAccumulator {
    pub fn new() -> Self {
        Self { vals: [[0; HIDDEN_SIZE]; 2] }
    }
}

// pairwise activation: (a * b) >> 16
fn pairwise_side_scalar(psq: &[i16; HIDDEN_SIZE], threats: &[i16; HIDDEN_SIZE], out: &mut [u8; PAIR_COUNT]) -> u64 {
    let mut nnz = 0u64;
    for i in 0..PAIR_COUNT {
        let a = i32::from(psq[i].wrapping_add(threats[i])).max(0).min(i32::from(QA));
        let b = i32::from(psq[i + PAIR_COUNT].wrapping_add(threats[i + PAIR_COUNT])).min(i32::from(QA));
        out[i] = (((a << FT_SCALE_BITS) * b) >> 16).max(0).min(255) as u8;
        if out[i] != 0 {
            nnz |= 1u64 << (i / 4);
        }
    }
    nnz
}

// reference first head layer, including requantization and the dual activation
fn l1_forward_scalar(weights: &[[i8; HIDDEN_SIZE]; L1_SIZE], bias: &[i32; L1_SIZE], input: &[u8; HIDDEN_SIZE], out: &mut [i16; L2_IN]) {
    for j in 0..L1_SIZE {
        let mut sum = 0i64;
        for i in 0..HIDDEN_SIZE {
            sum += i64::from(input[i]) * i64::from(weights[j][i]);
        }
        let output = ((sum as i32).wrapping_add(bias[j])) >> L1_SHIFT;

        // clamp before squaring for crelu/screlu
        let clamped = output.max(0).min(QUANT);
        out[j] = (clamped << QUANT_BITS) as i16;
        out[L1_SIZE + j] = (clamped * clamped) as i16;
    }
}

fn l2_forward_scalar(weights: &[[i16; L2_IN]; L2_SIZE], bias: &[i32; L2_SIZE], input: &[i16; L2_IN], out: &mut [i32; L2_SIZE]) {
    for j in 0..L2_SIZE {
        let mut sum = i64::from(bias[j]);
        for i in 0..L2_IN {
            sum += i64::from(input[i]) * i64::from(weights[j][i]);
        }
        out[j] = sum as i32;
    }
}

#[inline(always)]
fn l3_forward_raw(weights: &[[i32; L3_SIZE]; L2_SIZE], bias: &[i32; L3_SIZE], input: &[i32; L2_SIZE]) -> i64 {
    let mut result = i64::from(bias[0]);
    for i in 0..L2_SIZE {
        let activated = input[i].max(0).min(QUANT_CUBED);
        result += i64::from(activated) * i64::from(weights[i][0]);
    }
    result
}

fn l3_forward_scalar(weights: &[[i32; L3_SIZE]; L2_SIZE], bias: &[i32; L3_SIZE], input: &[i32; L2_SIZE]) -> i32 {
    let result = l3_forward_raw(weights, bias, input);
    (result * i64::from(EVAL_SCALE) / QUANT_FOURTH) as i32
}

// scalar reference for multilayer head
#[cfg(test)]
fn forward_scalar(psq: &Accumulator, threats: &ThreatAccumulator, us: usize, them: usize, bucket: usize, head: &MultilayerHead) -> i32 {
    let mut input = [0u8; HIDDEN_SIZE];
    pairwise_side_scalar(&psq.vals[us], &threats.vals[us], (&mut input[..PAIR_COUNT]).try_into().unwrap());
    pairwise_side_scalar(&psq.vals[them], &threats.vals[them], (&mut input[PAIR_COUNT..]).try_into().unwrap());

    let mut l1 = [0i16; L2_IN];
    l1_forward_scalar(&head.l1_weights[bucket], &head.l1_bias[bucket], &input, &mut l1);

    let mut l2 = [0i32; L2_SIZE];
    l2_forward_scalar(&head.l2_weights[bucket], &head.l2_bias[bucket], &l1, &mut l2);

    l3_forward_scalar(&head.l3_weights[bucket], &head.l3_bias[bucket], &l2)
}

// the first layer can either consume pairwise bytes in logical order
// (scalar / AVX-512 kernels) or in the packus lane order (AVX2 kernel)
#[allow(dead_code)]
#[inline(always)]
fn l1_packed_layout() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    {
        return has_avx2();
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    )))]
    {
        false
    }
}

#[inline(always)]
fn pairwise_side(psq: &[i16; HIDDEN_SIZE], threats: &[i16; HIDDEN_SIZE], out: &mut [u8; PAIR_COUNT]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            if l1_packed_layout() {
                return unsafe { pairwise_side_avx2::<true>(psq, threats, out) };
            }
            return unsafe { pairwise_side_avx2::<false>(psq, threats, out) };
        }
    }
    pairwise_side_scalar(psq, threats, out)
}

#[inline(always)]
fn l1_forward(
    weights: &[[i8; HIDDEN_SIZE]; L1_SIZE],
    transposed: &[[i8; 32]; L1_TRANSPOSED_BLOCKS],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    nnz: [u64; 2],
    out: &mut [i16; L2_IN],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        if has_avx512_vnni() {
            unsafe { l1_forward_avx512_vnni(weights, bias, input, out) };
        } else {
            unsafe { l1_forward_avx512(weights, bias, input, out) };
        }
        return;
    }

    #[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx512f", target_feature = "avx512bw"))))]
    {
        if has_avx2() {
            if has_avx_vnni() {
                unsafe { l1_forward_avx2_vnni(transposed, bias, input, nnz, out) };
            } else {
                unsafe { l1_forward_avx2(transposed, bias, input, nnz, out) };
            }
            return;
        }
        l1_forward_scalar(weights, bias, input, out);
    }

    #[cfg(not(target_arch = "x86_64"))]
    l1_forward_scalar(weights, bias, input, out)
}

#[inline(always)]
fn l2_forward(weights: &[[i16; L2_IN]; L2_SIZE], bias: &[i32; L2_SIZE], input: &[i16; L2_IN], out: &mut [i32; L2_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { l2_forward_avx2(weights, bias, input, out) };
            return;
        }
    }
    l2_forward_scalar(weights, bias, input, out)
}

// multilayer forward pass
fn forward_multilayer_raw(
    psq: &Accumulator,
    threat_us: &[i16; HIDDEN_SIZE],
    threat_them: &[i16; HIDDEN_SIZE],
    us: usize,
    them: usize,
    bucket: usize,
    head: &MultilayerHead,
) -> i64 {
    let mut input = [0u8; HIDDEN_SIZE];
    let nnz_us = pairwise_side(&psq.vals[us], threat_us, (&mut input[..PAIR_COUNT]).try_into().unwrap());
    let nnz_them = pairwise_side(&psq.vals[them], threat_them, (&mut input[PAIR_COUNT..]).try_into().unwrap());

    let mut l1 = [0i16; L2_IN];
    l1_forward(&head.l1_weights[bucket], &head.l1_weights_t[bucket], &head.l1_bias[bucket], &input, [nnz_us, nnz_them], &mut l1);

    let mut l2 = [0i32; L2_SIZE];
    l2_forward(&head.l2_weights[bucket], &head.l2_bias[bucket], &l1, &mut l2);

    l3_forward_raw(&head.l3_weights[bucket], &head.l3_bias[bucket], &l2)
}

fn forward_multilayer(
    psq: &Accumulator,
    threat_us: &[i16; HIDDEN_SIZE],
    threat_them: &[i16; HIDDEN_SIZE],
    us: usize,
    them: usize,
    bucket: usize,
    head: &MultilayerHead,
) -> i32 {
    let raw = forward_multilayer_raw(psq, threat_us, threat_them, us, them, bucket, head);
    (raw * i64::from(EVAL_SCALE) / QUANT_FOURTH) as i32
}

// 60144 threat features (piece attacks/defenses, mirror invariant)

pub const TOTAL_THREATS: usize = 60144;

// max active threat features per position
pub const MAX_THREATS: usize = 256;

const THREAT_SENTINEL: u16 = u16::MAX;

// victim slot index per attacker type (-1 if ignored)
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
    // first index of an attacker square, per attacker color and type
    offsets: [[[u16; 64]; 6]; 2],
    // number of indices one attacker square owns, per color and type
    span: [[u16; 6]; 2],
    // first index of an attacker color and type group
    global: [[u16; 6]; 2],
    // base threat index (THREAT_SENTINEL if excluded)
    attack: [[[[[u16; 2]; 6]; 2]; 6]; 2],
    // precomputed empty-board attacks for popcnt indexing, 6 KB vs 48 KB
    pseudo_attacks: [[[Bitboard; 64]; 6]; 2],

    // flat tables for fast threat_index_fast: [side][attacker_piece][victim_piece]
    // side: 0=white, 1=black; piece index 0..11 (Piece enum, excluding None)
    attack_by_piece: [[[[u16; 2]; 12]; 12]; 2],
    offsets_by_piece: [[[u16; 64]; 12]; 2],
    pseudo_by_piece: [[[Bitboard; 64]; 12]; 2],
}

static THREAT_TABLES_PTR: AtomicPtr<ThreatTables> = AtomicPtr::new(std::ptr::null_mut());
static THREAT_TABLES_INIT: OnceLock<ThreatTables> = OnceLock::new();

#[inline(always)]
fn threat_tables() -> &'static ThreatTables {
    let ptr = THREAT_TABLES_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        return unsafe { &*ptr };
    }
    init_threat_tables_cold()
}

#[cold]
#[inline(never)]
fn init_threat_tables_cold() -> &'static ThreatTables {
    let tables = THREAT_TABLES_INIT.get_or_init(init_threat_tables);
    THREAT_TABLES_PTR.store(tables as *const ThreatTables as *mut ThreatTables, Ordering::Release);
    tables
}

// piece attacks in feature space
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

#[inline(always)]
fn threat_attacks_with(tables: &crate::attacks::SliderTables, piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    match piece.piece_type() {
        PieceType::Pawn => crate::attacks::pawn_attacks(piece.color(), sq),
        PieceType::Knight => crate::attacks::knight_attacks(sq),
        PieceType::Bishop => tables.bishop_attacks(sq, occ),
        PieceType::Rook => tables.rook_attacks(sq, occ),
        PieceType::Queen => tables.bishop_attacks(sq, occ) | tables.rook_attacks(sq, occ),
        PieceType::King => crate::attacks::king_attacks(sq),
        PieceType::None => Bitboard::EMPTY,
    }
}

// xor mask for rank flip / file mirror
#[inline(always)]
pub fn perspective_mask(ksq: Square, view: Color) -> u8 {
    let rank_flip = if view == Color::Black { 56 } else { 0 };
    let file_flip = if ksq.file() >= 4 { 7 } else { 0 };
    rank_flip ^ file_flip
}

fn init_threat_tables() -> ThreatTables {
    // all fields are plain int arrays, zero is a valid state
    let mut tables: ThreatTables = unsafe { std::mem::zeroed() };

    let mut global = 0u32;
    for color in 0..2 {
        for (kind, &target_count) in PIECE_TARGET_COUNT.iter().enumerate() {
            let piece = Piece::new(Color::from_index(color), PieceType::from_index(kind));
            let mut offset = 0u16;
            for sq in 0..64 {
                let square = Square::new(sq as u8);
                tables.offsets[color][kind][sq] = offset;
                // skip back-rank squares for pawns
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
                let square = Square::new(from as u8);
                tables.pseudo_attacks[color][kind][from] = threat_attacks(piece, square, Bitboard::EMPTY);
            }
        }
    }

    for acolor in 0..2 {
        for (akind, targets) in PIECE_TARGET_MAP.iter().enumerate() {
            for vcolor in 0..2 {
                for (vkind, &map) in targets.iter().enumerate() {
                    // store same-type pairs once (except pawn defenses)
                    let opposed = acolor != vcolor;
                    let semi_excluded = akind == vkind && (opposed || akind != 0);
                    for below in 0..2 {
                        if map < 0 || (below == 1 && semi_excluded) {
                            tables.attack[acolor][akind][vcolor][vkind][below] = THREAT_SENTINEL;
                        } else {
                            let color_base = vcolor as u32 * u32::from(PIECE_TARGET_COUNT[akind]);
                            let feature = u32::from(tables.global[acolor][akind]) + (color_base + map as u32) * u32::from(tables.span[acolor][akind]);
                            tables.attack[acolor][akind][vcolor][vkind][below] = feature as u16;
                        }
                    }
                }
            }
        }
}

// Populate flat tables for fast threat_index_fast
    for side in 0..2 {
        let view = Color::from_index(side);
        for ap in 0..12 {
            let oa = oriented_piece(Piece::from_index(ap), view);
            let (acolor, akind) = (oa / 6, oa % 6);
            tables.offsets_by_piece[side][ap] = tables.offsets[acolor][akind];
            tables.pseudo_by_piece[side][ap] = tables.pseudo_attacks[acolor][akind];
            for vp in 0..12 {
                let ov = oriented_piece(Piece::from_index(vp), view);
                let (vcolor, vkind) = (ov / 6, ov % 6);
                tables.attack_by_piece[side][ap][vp] = tables.attack[acolor][akind][vcolor][vkind];
            }
        }
    }

    tables
}

// compute threat feature index
#[inline(always)]
#[allow(dead_code)]
fn threat_index(tables: &'static ThreatTables, mask: u8, attacker: usize, from: Square, victim: usize, to: Square) -> Option<u16> {
    const PIECE_COLOR: [usize; 12] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    const PIECE_KIND: [usize; 12] = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5];
    let from = (from as u8 ^ mask) as usize;
    let to = (to as u8 ^ mask) as usize;

    let (acolor, akind) = (PIECE_COLOR[attacker], PIECE_KIND[attacker]);
    let (vcolor, vkind) = (PIECE_COLOR[victim], PIECE_KIND[victim]);

    let base = tables.attack[acolor][akind][vcolor][vkind][usize::from(from < to)];
    if base == THREAT_SENTINEL {
        return None;
    }

    let attacks = tables.pseudo_attacks[acolor][akind][from];
    let below_mask = (1u64 << to) - 1;
    let index = base + tables.offsets[acolor][akind][from] + u16::from((attacks.0 & below_mask).count_ones() as u16);
    debug_assert!((index as usize) < TOTAL_THREATS);
    Some(index)
}

// fast threat index using flat tables
#[inline(always)]
fn threat_index_fast(
    tables: &'static ThreatTables,
    side: usize,
    mask: u8,
    attacker: Piece,
    from: Square,
    victim: Piece,
    to: Square,
) -> Option<u16> {
    let a = attacker as usize;
    let v = victim as usize;
    debug_assert!(a < 12 && v < 12);
    let from_lbl = (from as u8 ^ mask) as usize;
    let to_lbl = (to as u8 ^ mask) as usize;
    let base = unsafe {
        *tables.attack_by_piece.get_unchecked(side).get_unchecked(a).get_unchecked(v).get_unchecked(usize::from(from_lbl < to_lbl))
    };
    if base == THREAT_SENTINEL {
        return None;
    }
    let attacks = unsafe { *tables.pseudo_by_piece.get_unchecked(side).get_unchecked(a).get_unchecked(from_lbl) };
    let offset = unsafe { *tables.offsets_by_piece.get_unchecked(side).get_unchecked(a).get_unchecked(from_lbl) };
    let below_mask = (1u64 << to_lbl) - 1;
    Some(base + offset + (attacks.0 & below_mask).count_ones() as u16)
}

// sorted active threat indices (uninit buffer to skip zeroing)
#[derive(Copy, Clone)]
pub struct ThreatList {
    indices: [MaybeUninit<u16>; MAX_THREATS],
    count: usize,
}

impl std::fmt::Debug for ThreatList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl ThreatList {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            indices: [MaybeUninit::uninit(); MAX_THREATS],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn init_one(slot: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            let p = slot.as_mut_ptr();
            std::ptr::addr_of_mut!((*p).count).write(0);
            slot.assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn init_pair(slot: &mut MaybeUninit<[Self; 2]>) -> &mut [Self; 2] {
        unsafe {
            let p = slot.as_mut_ptr() as *mut Self;
            std::ptr::addr_of_mut!((*p).count).write(0);
            std::ptr::addr_of_mut!((*p.add(1)).count).write(0);
            slot.assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.indices.as_ptr().cast::<u16>(), self.count) }
    }

    #[inline(always)]
    fn push(&mut self, index: u16) {
        assert!(self.count < MAX_THREATS, "too many active threat features");
        self.indices[self.count].write(index);
        self.count += 1;
    }

    // orders the indices so two lists can be compared in one linear pass
    #[inline(always)]
    fn sort(&mut self) {
        unsafe {
            std::slice::from_raw_parts_mut(self.indices.as_mut_ptr().cast::<u16>(), self.count).sort_unstable();
        }
    }

}

// cached board info for threat scan
struct ThreatScan {
    tables: &'static ThreatTables,
    slider_tables: &'static crate::attacks::SliderTables,
    mask: u8,
    kings: Bitboard,
    by_type: [Bitboard; 6],
    victims: [Bitboard; 6],
}

impl ThreatScan {
    fn new(board: &Board, view: Color) -> Self {
        let ksq = board.king_square(view);
        let mask = perspective_mask(ksq, view);

        let mut by_type = [Bitboard::EMPTY; 6];
        for (kind, slot) in by_type.iter_mut().enumerate() {
            for color in 0..2 {
                *slot |= board.pieces[Piece::new(Color::from_index(color), PieceType::from_index(kind))];
            }
        }

        // union of the piece types each attacker type is allowed to threaten
        let mut victims = [Bitboard::EMPTY; 6];
        for (akind, targets) in PIECE_TARGET_MAP.iter().enumerate() {
            for (vkind, &map) in targets.iter().enumerate() {
                if map >= 0 {
                    victims[akind] |= by_type[vkind];
                }
            }
        }

        Self {
            tables: threat_tables(),
            slider_tables: crate::attacks::slider_tables(),
            mask,
            kings: by_type[PieceType::King as usize],
            by_type,
            victims,
        }
    }
}

// appends every relation of the piece standing on from. Indices stay
// unsorted; the caller sorts once after the last attacker.
#[inline(always)]
fn emit_attacker_threats(board: &Board, scan: &ThreatScan, view: Color, from: Square, out: &mut ThreatList) {
    let attacker = board.piece_on[from];
    let akind = attacker.piece_type() as usize;
    let from_label = (from as u8 ^ scan.mask) as usize;
    // let attacker_idx = oriented_piece(attacker, view); // no longer needed

    // index same-type pairs once (except pawn defenses)
    let mut same_type = scan.by_type[akind];
    if akind == PieceType::Pawn as usize {
        same_type &= board.occupied_co[!attacker.color()];
    }

    let mut targets = threat_attacks_with(scan.slider_tables, attacker, from, board.occupied) & !scan.kings & scan.victims[akind];
    while !targets.is_empty() {
        let to = targets.pop_lsb();
        if same_type.contains(to) && (to as u8 ^ scan.mask) as usize > from_label {
            continue;
        }
        let victim = board.piece_on[to];
        if let Some(index) = threat_index_fast(scan.tables, view as usize, scan.mask, attacker, from, victim, to) {
            out.push(index);
        }
    }
}

// collect all threats for one side from scratch
pub fn collect_threats(board: &Board, view: Color, out: &mut ThreatList) {
    out.count = 0;
    let scan = ThreatScan::new(board, view);
    let mut attackers = board.occupied & !scan.kings;
    while !attackers.is_empty() {
        let from = attackers.pop_lsb();
        emit_attacker_threats(board, &scan, view, from, out);
    }
    out.sort();
}

pub(crate) struct SharedThreatScan {
    tables: &'static ThreatTables,
    slider_tables: &'static crate::attacks::SliderTables,
    pub(crate) masks: [u8; 2],
    pub(crate) kings: Bitboard,
    pub(crate) by_type: [Bitboard; 6],
    pub(crate) victims: [Bitboard; 6],
}

impl SharedThreatScan {
    pub(crate) fn new(board: &Board) -> Self {
        let masks = [
            perspective_mask(board.king_square(Color::White), Color::White),
            perspective_mask(board.king_square(Color::Black), Color::Black),
        ];
        let mut scan = Self {
            tables: threat_tables(),
            slider_tables: crate::attacks::slider_tables(),
            masks,
            kings: Bitboard::EMPTY,
            by_type: [Bitboard::EMPTY; 6],
            victims: [Bitboard::EMPTY; 6],
        };
        scan.refresh_pieces(board);
        scan
    }

    #[inline(always)]
    pub(crate) fn refresh_pieces(&mut self, board: &Board) {
        for kind in 0..6 {
            let w = board.pieces[kind];
            let b = board.pieces[kind + 6];
            self.by_type[kind] = w | b;
        }
        self.kings = self.by_type[PieceType::King as usize];
        // PIECE_TARGET_MAP in closed form: pawns/knight/rook/queen/bishop
        // victims are a few fixed unions, so skip the 6x6 map walk.
        let b = &self.by_type;
        let non_king = board.occupied & !self.kings;
        self.victims[PieceType::Pawn as usize] = b[0] | b[1] | b[3];
        self.victims[PieceType::Knight as usize] = non_king;
        self.victims[PieceType::Bishop as usize] = b[0] | b[1] | b[2] | b[3];
        self.victims[PieceType::Rook as usize] = b[0] | b[1] | b[2] | b[3];
        self.victims[PieceType::Queen as usize] = non_king;
        self.victims[PieceType::King as usize] = Bitboard::EMPTY;
    }
}

#[inline(always)]
fn emit_attacker_threats_both(board: &Board, scan: &SharedThreatScan, from: Square, need: [bool; 2], out: &mut [ThreatList; 2]) {
    let attacker = board.piece_on[from];
    let akind = attacker.piece_type() as usize;
    let mut same_type = scan.by_type[akind];
    if akind == PieceType::Pawn as usize {
        same_type &= board.occupied_co[!attacker.color()];
    }

    let mut targets = threat_attacks_with(scan.slider_tables, attacker, from, board.occupied) & !scan.kings & scan.victims[akind];
    while !targets.is_empty() {
        let to = targets.pop_lsb();
        let victim = board.piece_on[to];
        let is_same = same_type.contains(to);
        for side in 0..2 {
            if !need[side] {
                continue;
            }
            let mask = scan.masks[side];
            let from_label = (from as u8 ^ mask) as usize;
            if is_same && (to as u8 ^ mask) as usize > from_label {
                continue;
            }
            // let view = Color::from_index(side); // no longer needed
            if let Some(index) = threat_index_fast(scan.tables, side, mask, attacker, from, victim, to) {
                out[side].push(index);
            }
        }
    }
}

pub fn collect_threats_both_unsorted(board: &Board, need: [bool; 2], out: &mut [ThreatList; 2]) {
    if need[0] {
        out[0].count = 0;
    }
    if need[1] {
        out[1].count = 0;
    }
    let scan = SharedThreatScan::new(board);
    let mut attackers = board.occupied & !scan.kings;
    while !attackers.is_empty() {
        let from = attackers.pop_lsb();
        emit_attacker_threats_both(board, &scan, from, need, out);
    }
}

#[inline(always)]
fn push_threat_both(scan: &SharedThreatScan, attacker: Piece, from: Square, victim: Piece, to: Square, need: [bool; 2], out: &mut [ThreatList; 2]) {
    for side in 0..2 {
        if !need[side] {
            continue;
        }
        // let view = Color::from_index(side); // no longer needed
        if let Some(index) = threat_index_fast(scan.tables, side, scan.masks[side], attacker, from, victim, to) {
            out[side].push(index);
        }
    }
}

// Cached per-attacker invariants for the flat threat tables. `attacker` and
// `from` stay fixed while walking the victims of one square, so the two table
// rows (pseudo attacks and offsets) are loaded once per attacker instead of
// once per target square.
struct ThreatEmitter<'a> {
    need: [bool; 2],
    masks: [u8; 2],
    from_lbl: [usize; 2],
    attack: [&'a [[u16; 2]; 12]; 2],
    pseudo: [Bitboard; 2],
    offset: [u16; 2],
}

impl<'a> ThreatEmitter<'a> {
    #[inline(always)]
    fn new(scan: &'a SharedThreatScan, a: usize, need: [bool; 2], from_lbl_0: usize, from_lbl_1: usize) -> Self {
        Self {
            need,
            masks: scan.masks,
            from_lbl: [from_lbl_0, from_lbl_1],
            attack: [&scan.tables.attack_by_piece[0][a], &scan.tables.attack_by_piece[1][a]],
            pseudo: [scan.tables.pseudo_by_piece[0][a][from_lbl_0], scan.tables.pseudo_by_piece[1][a][from_lbl_1]],
            offset: [scan.tables.offsets_by_piece[0][a][from_lbl_0], scan.tables.offsets_by_piece[1][a][from_lbl_1]],
        }
    }

    #[inline(always)]
    fn push(&self, victim: Piece, to: Square, out: &mut [ThreatList; 2]) {
        let v = victim as usize;
        if v >= 12 {
            return; // Piece::None or invalid, no threat
        }
        if self.need[0] {
            let to_lbl = (to as u8 ^ self.masks[0]) as usize;
            let base = unsafe { *self.attack[0].get_unchecked(v).get_unchecked(usize::from(self.from_lbl[0] < to_lbl)) };
            if base != THREAT_SENTINEL {
                let below_mask = (1u64 << to_lbl) - 1;
                out[0].push(base + self.offset[0] + (self.pseudo[0].0 & below_mask).count_ones() as u16);
            }
        }
        if self.need[1] {
            let to_lbl = (to as u8 ^ self.masks[1]) as usize;
            let base = unsafe { *self.attack[1].get_unchecked(v).get_unchecked(usize::from(self.from_lbl[1] < to_lbl)) };
            if base != THREAT_SENTINEL {
                let below_mask = (1u64 << to_lbl) - 1;
                out[1].push(base + self.offset[1] + (self.pseudo[1].0 & below_mask).count_ones() as u16);
            }
        }
    }
}

pub fn append_changed_square_threats_both(board: &Board, scan: &SharedThreatScan, changed: Bitboard, need: [bool; 2], out: &mut [ThreatList; 2]) {
    let non_kings = board.occupied & !scan.kings;
    let mut sqs = changed & non_kings;

    while !sqs.is_empty() {
        let sq = sqs.pop_lsb();
        let piece = board.piece_on[sq];
        let pkind = piece.piece_type() as usize;

        let sq_lbl_0 = (sq as u8 ^ scan.masks[0]) as usize;
        let sq_lbl_1 = (sq as u8 ^ scan.masks[1]) as usize;

        let mut targets = threat_attacks_with(scan.slider_tables, piece, sq, board.occupied) & !scan.kings & scan.victims[pkind];
        if !targets.is_empty() {
            let emitter = ThreatEmitter::new(scan, piece as usize, need, sq_lbl_0, sq_lbl_1);
            while !targets.is_empty() {
                let to = targets.pop_lsb();
                emitter.push(board.piece_on[to], to, out);
            }
        }

        let mut attackers = attackers_to_square(&board.pieces, board.occupied, sq, scan.slider_tables, pkind) & !changed;
        while !attackers.is_empty() {
            let from = attackers.pop_lsb();
            let attacker = board.piece_on[from];
            let from_lbl_0 = (from as u8 ^ scan.masks[0]) as usize;
            let from_lbl_1 = (from as u8 ^ scan.masks[1]) as usize;
            let emitter = ThreatEmitter::new(scan, attacker as usize, need, from_lbl_0, from_lbl_1);
            emitter.push(piece, sq, out);
        }
    }
}

pub fn collect_xray_threat_diffs_both(
    board: &Board,
    scan: &SharedThreatScan,
    changed: Bitboard,
    occ_before: Bitboard,
    occ_after: Bitboard,
    need: [bool; 2],
    subs: &mut [ThreatList; 2],
    adds: &mut [ThreatList; 2],
) {
    let occ_delta = occ_before ^ occ_after;
    if occ_delta.is_empty() {
        return;
    }

    let non_changed = !changed;
    let non_changed_victims = non_changed & !scan.kings;
    let queens = board.pieces[Piece::WhiteQueen] | board.pieces[Piece::BlackQueen];

    let tables = scan.slider_tables;
    let diag_pieces = (board.pieces[Piece::WhiteBishop] | board.pieces[Piece::BlackBishop] | queens) & non_changed;
    if !diag_pieces.is_empty() {
        let mut diag_sliders = Bitboard::EMPTY;
        let mut d = occ_delta;
        while !d.is_empty() {
            let sq = d.pop_lsb();
            // only occ_before is needed: the first changed square on a slider's
            // ray is occupied in occ_before and sees the slider from there
            diag_sliders |= tables.bishop_attacks(sq, occ_before) & diag_pieces;
        }
        while !diag_sliders.is_empty() {
            let from = diag_sliders.pop_lsb();
            let attacker = board.piece_on[from];
            let akind = attacker.piece_type() as usize;
            let vic_mask = non_changed_victims & scan.victims[akind];

            let att_b = tables.bishop_attacks(from, occ_before) & vic_mask;
            let att_a = tables.bishop_attacks(from, occ_after) & vic_mask;

            // Precompute invariant values for both sides for this attacker/from
            let a = attacker as usize;
            let from_lbl_0 = (from as u8 ^ scan.masks[0]) as usize;
            let from_lbl_1 = (from as u8 ^ scan.masks[1]) as usize;
            let emitter = ThreatEmitter::new(scan, a, need, from_lbl_0, from_lbl_1);

            let mut lost = att_b & !att_a;
            while !lost.is_empty() {
                let to = lost.pop_lsb();
                emitter.push(board.piece_on[to], to, subs);
            }

            let mut gained = att_a & !att_b;
            while !gained.is_empty() {
                let to = gained.pop_lsb();
                emitter.push(board.piece_on[to], to, adds);
            }
        }
    }

    let orth_pieces = (board.pieces[Piece::WhiteRook] | board.pieces[Piece::BlackRook] | queens) & non_changed;
    if !orth_pieces.is_empty() {
        let mut orth_sliders = Bitboard::EMPTY;
        let mut d = occ_delta;
        while !d.is_empty() {
            let sq = d.pop_lsb();
            orth_sliders |= tables.rook_attacks(sq, occ_before) & orth_pieces;
        }
        while !orth_sliders.is_empty() {
            let from = orth_sliders.pop_lsb();
            let attacker = board.piece_on[from];
            let akind = attacker.piece_type() as usize;
            let vic_mask = non_changed_victims & scan.victims[akind];

            let att_b = tables.rook_attacks(from, occ_before) & vic_mask;
            let att_a = tables.rook_attacks(from, occ_after) & vic_mask;

            // Precompute invariant values for both sides for this attacker/from
            let a = attacker as usize;
            let from_lbl_0 = (from as u8 ^ scan.masks[0]) as usize;
            let from_lbl_1 = (from as u8 ^ scan.masks[1]) as usize;
            let emitter = ThreatEmitter::new(scan, a, need, from_lbl_0, from_lbl_1);

            let mut lost = att_b & !att_a;
            while !lost.is_empty() {
                let to = lost.pop_lsb();
                emitter.push(board.piece_on[to], to, subs);
            }

            let mut gained = att_a & !att_b;
            while !gained.is_empty() {
                let to = gained.pop_lsb();
                emitter.push(board.piece_on[to], to, adds);
            }
        }
    }
}

// every piece of either color that can attack sq under the given occupancy,
// restricted to the attacker types that may threaten a piece of victim_kind
#[inline(always)]
fn attackers_to_square(
    pieces: &[Bitboard; PIECE_NB],
    occ: Bitboard,
    sq: Square,
    tables: &crate::attacks::SliderTables,
    victim_kind: usize,
) -> Bitboard {
    use crate::attacks::{knight_attacks, pawn_attacks};

    let knights = pieces[Piece::WhiteKnight as usize] | pieces[Piece::BlackKnight as usize];
    let queens = pieces[Piece::WhiteQueen as usize] | pieces[Piece::BlackQueen as usize];
    let mut attackers = knight_attacks(sq) & knights;

    // a queen can only be threatened by knights and queens
    if victim_kind == PieceType::Queen as usize {
        return attackers | (tables.bishop_attacks(sq, occ) | tables.rook_attacks(sq, occ)) & queens;
    }

    let bishops = pieces[Piece::WhiteBishop as usize] | pieces[Piece::BlackBishop as usize];
    let rooks = pieces[Piece::WhiteRook as usize] | pieces[Piece::BlackRook as usize];
    attackers |= tables.bishop_attacks(sq, occ) & (bishops | queens);
    attackers |= tables.rook_attacks(sq, occ) & (rooks | queens);

    // pawns cannot threaten bishops
    if victim_kind != PieceType::Bishop as usize {
        // black pawn on sq attacks squares a white pawn attacks it from
        attackers |= pawn_attacks(Color::Black, sq) & pieces[Piece::WhitePawn as usize]
            | pawn_attacks(Color::White, sq) & pieces[Piece::BlackPawn as usize];
    }

    attackers
}

// applies or removes the rows of a threat index list
fn apply_threat_rows_scalar(acc: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16], add: bool) {
    for &index in list {
        let base = index as usize * HIDDEN_SIZE;
        let row = &weights.rows[base..base + HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            let w = i16::from(row[i]);
            acc[i] = if add { acc[i].wrapping_add(w) } else { acc[i].wrapping_sub(w) };
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
    // rebuilds one perspective from scratch: zero, plus every row of the list
pub fn refresh_side(&mut self, side: usize, list: &ThreatList, weights: &ThreatWeights) {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                unsafe { refresh_threat_side_avx2(&mut self.vals[side], weights, list.as_slice()) };
                return;
            }
        }
        self.vals[side].fill(0);
        apply_threat_rows_scalar(&mut self.vals[side], weights, list.as_slice(), true);
    }

    // applies a known set of removed and added rows directly
    #[inline(always)]
    pub fn apply_lists(&mut self, side: usize, weights: &ThreatWeights, subs: &[u16], adds: &[u16]) {
        let acc = &mut self.vals[side];
        if !subs.is_empty() {
            apply_threat_rows(acc, weights, subs, false);
        }
        if !adds.is_empty() {
            apply_threat_rows(acc, weights, adds, true);
        }
    }

    #[inline(always)]
    pub fn apply_lists_from(&mut self, src: &ThreatAccumulator, side: usize, weights: &ThreatWeights, subs: &[u16], adds: &[u16]) {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                unsafe {
                    apply_threat_delta_from_avx2(&mut self.vals[side], &src.vals[side], weights, subs, adds);
                }
                return;
            }
        }
        self.vals[side].copy_from_slice(&src.vals[side]);
        self.apply_lists(side, weights, subs, adds);
    }
}

// prefetch first chunk of a threat row (128-256 bytes = 2-4 cache lines)
// let hardware prefetcher pull the rest
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_threat_row(weights: &ThreatWeights, index: u16) {
    use std::arch::x86_64::*;
    unsafe {
        let row = weights.rows.as_ptr().add(index as usize * HIDDEN_SIZE) as *const i8;
        _mm_prefetch(row, _MM_HINT_T0);
        _mm_prefetch(row.add(64), _MM_HINT_T0);
        _mm_prefetch(row.add(128), _MM_HINT_T0);
        _mm_prefetch(row.add(256), _MM_HINT_T0);
    }
}

// prefetch one chunk (base..base + lines * 64) of a row
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_threat_chunk(weights: &ThreatWeights, index: u16, base: usize, lines: usize) {
    use std::arch::x86_64::*;
    unsafe {
        let chunk = weights.rows.as_ptr().add(index as usize * HIDDEN_SIZE + base) as *const i8;
        let mut line = 0;
        while line < lines {
            _mm_prefetch(chunk.add(line * 64), _MM_HINT_T0);
            line += 1;
        }
    }
}

// 192 hidden lanes (12 YMM accumulators)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_chunk_12(
    dst: &mut [i16; HIDDEN_SIZE],
    src: &[i16; HIDDEN_SIZE],
    weights: &ThreatWeights,
    subs: &[u16],
    adds: &[u16],
    base: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let s = src.as_ptr().add(base) as *const __m256i;
        let mut a0 = _mm256_load_si256(s);
        let mut a1 = _mm256_load_si256(s.add(1));
        let mut a2 = _mm256_load_si256(s.add(2));
        let mut a3 = _mm256_load_si256(s.add(3));
        let mut a4 = _mm256_load_si256(s.add(4));
        let mut a5 = _mm256_load_si256(s.add(5));
        let mut a6 = _mm256_load_si256(s.add(6));
        let mut a7 = _mm256_load_si256(s.add(7));
        let mut a8 = _mm256_load_si256(s.add(8));
        let mut a9 = _mm256_load_si256(s.add(9));
        let mut a10 = _mm256_load_si256(s.add(10));
        let mut a11 = _mm256_load_si256(s.add(11));

        // One prefetch per row, a few rows ahead. The old code issued one per
        // cache line of the whole row, which saturated the line fill buffers
        // and made _mm_prefetch itself a top profiler entry.
        const PREFETCH_AHEAD: usize = 4;
        let rows = weights.rows.as_ptr();
        for (n, &idx) in subs.iter().enumerate() {
            if let Some(&ahead) = subs.get(n + PREFETCH_AHEAD) {
                _mm_prefetch(rows.add(ahead as usize * HIDDEN_SIZE + base), _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + base);
            a0 = _mm256_sub_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_sub_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_sub_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_sub_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_sub_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_sub_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_sub_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_sub_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
            a8 = _mm256_sub_epi16(a8, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(128) as *const __m128i)));
            a9 = _mm256_sub_epi16(a9, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(144) as *const __m128i)));
            a10 = _mm256_sub_epi16(a10, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(160) as *const __m128i)));
            a11 = _mm256_sub_epi16(a11, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(176) as *const __m128i)));
        }
        for (n, &idx) in adds.iter().enumerate() {
            if let Some(&ahead) = adds.get(n + PREFETCH_AHEAD) {
                _mm_prefetch(rows.add(ahead as usize * HIDDEN_SIZE + base), _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + base);
            a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_add_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_add_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_add_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_add_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_add_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_add_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
            a8 = _mm256_add_epi16(a8, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(128) as *const __m128i)));
            a9 = _mm256_add_epi16(a9, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(144) as *const __m128i)));
            a10 = _mm256_add_epi16(a10, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(160) as *const __m128i)));
            a11 = _mm256_add_epi16(a11, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(176) as *const __m128i)));
        }

        let d = dst.as_mut_ptr().add(base) as *mut __m256i;
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
        _mm256_store_si256(d.add(8), a8);
        _mm256_store_si256(d.add(9), a9);
        _mm256_store_si256(d.add(10), a10);
        _mm256_store_si256(d.add(11), a11);
    }
}

// 128 hidden lanes (8 YMM accumulators)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_chunk_8(dst: &mut [i16; HIDDEN_SIZE], src: &[i16; HIDDEN_SIZE], weights: &ThreatWeights, subs: &[u16], adds: &[u16], base: usize) {
    use std::arch::x86_64::*;
    unsafe {
        let s = src.as_ptr().add(base) as *const __m256i;
        let mut a0 = _mm256_load_si256(s);
        let mut a1 = _mm256_load_si256(s.add(1));
        let mut a2 = _mm256_load_si256(s.add(2));
        let mut a3 = _mm256_load_si256(s.add(3));
        let mut a4 = _mm256_load_si256(s.add(4));
        let mut a5 = _mm256_load_si256(s.add(5));
        let mut a6 = _mm256_load_si256(s.add(6));
        let mut a7 = _mm256_load_si256(s.add(7));

        let rows = weights.rows.as_ptr();
        for &idx in subs {
            let r = rows.add(idx as usize * HIDDEN_SIZE + base);
            a0 = _mm256_sub_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_sub_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_sub_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_sub_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_sub_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_sub_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_sub_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_sub_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
        }
        for &idx in adds {
            let r = rows.add(idx as usize * HIDDEN_SIZE + base);
            a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_add_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_add_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_add_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_add_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_add_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_add_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
        }

        let d = dst.as_mut_ptr().add(base) as *mut __m256i;
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_delta_from_avx2(dst: &mut [i16; HIDDEN_SIZE], src: &[i16; HIDDEN_SIZE], weights: &ThreatWeights, subs: &[u16], adds: &[u16]) {
    unsafe {
        // warm only the first chunk (128 bytes) of the first 1-2 rows
        // hardware prefetcher will pull the rest
        const WARM_ROWS: usize = 2;
        let warm = WARM_ROWS.min(subs.len());
        for &idx in &subs[..warm] {
            prefetch_threat_chunk(weights, idx, 0, 2);
        }
        let warm = WARM_ROWS.min(adds.len());
        for &idx in &adds[..warm] {
            prefetch_threat_chunk(weights, idx, 0, 2);
        }

        // a short list is fully warmed above
        apply_threat_chunk_12(dst, src, weights, subs, adds, 0);
        apply_threat_chunk_12(dst, src, weights, subs, adds, 192);
        apply_threat_chunk_8(dst, src, weights, subs, adds, 384);
    }
}

#[allow(dead_code)]
pub fn prefetch_threat_rows(weights: &ThreatWeights, list: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        for &idx in list {
            let p = weights.rows.as_ptr().add(idx as usize * HIDDEN_SIZE);
            for line in (0..HIDDEN_SIZE).step_by(64) {
                _mm_prefetch(p.add(line) as *const i8, _MM_HINT_T0);
            }
        }
    }
}

// get threat weights if using multilayer net
#[allow(dead_code)]
static THREAT_WEIGHTS_PTR: AtomicPtr<Option<&'static ThreatWeights>> = AtomicPtr::new(std::ptr::null_mut());
static THREAT_WEIGHTS_INIT: OnceLock<Option<&'static ThreatWeights>> = OnceLock::new();

#[allow(dead_code)]
#[inline(always)]
pub fn threat_weights() -> Option<&'static ThreatWeights> {
    let ptr = THREAT_WEIGHTS_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        return unsafe { *ptr };
    }
    init_threat_weights_cold()
}

#[cold]
#[inline(never)]
fn init_threat_weights_cold() -> Option<&'static ThreatWeights> {
    let weights = THREAT_WEIGHTS_INIT.get_or_init(|| network().multilayer.as_deref().and_then(|net| net.threat_weights.as_deref()));
    THREAT_WEIGHTS_PTR.store(weights as *const Option<&'static ThreatWeights> as *mut Option<&'static ThreatWeights>, Ordering::Release);
    *weights
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_threat_rows_avx2(acc: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16], add: bool) {
    use std::arch::x86_64::*;
    // prefetch threat rows ahead to hide cache misses
    const PREFETCH_AHEAD: usize = 4;
    unsafe {
        for (n, &index) in list.iter().enumerate() {
            if let Some(&ahead) = list.get(n + PREFETCH_AHEAD) {
                let ptr = weights.rows.as_ptr().add(ahead as usize * HIDDEN_SIZE);
                for line in (0..HIDDEN_SIZE).step_by(64) {
                    _mm_prefetch(ptr.add(line) as *const i8, _MM_HINT_T0);
                }
            }
            let row = weights.rows.as_ptr().add(index as usize * HIDDEN_SIZE);
            let mut i = 0;
            while i < HIDDEN_SIZE {
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(i) as *const __m128i));
                let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                let r = if add { _mm256_add_epi16(a, w) } else { _mm256_sub_epi16(a, w) };
                _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn refresh_threat_side_avx2(dst: &mut [i16; HIDDEN_SIZE], weights: &ThreatWeights, list: &[u16]) {
    use std::arch::x86_64::*;
    unsafe {
        // 192 lanes (12 YMM) + 192 lanes (12 YMM) + 128 lanes (8 YMM) = 512 lanes
        // This reduces from 4 passes to 3 passes over the threat list
        let rows = weights.rows.as_ptr();
        let len = list.len();

        // First 192 lanes (12 YMM = 12 * 16 = 192 i16 = 384 bytes)
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut a4 = _mm256_setzero_si256();
        let mut a5 = _mm256_setzero_si256();
        let mut a6 = _mm256_setzero_si256();
        let mut a7 = _mm256_setzero_si256();
        let mut a8 = _mm256_setzero_si256();
        let mut a9 = _mm256_setzero_si256();
        let mut a10 = _mm256_setzero_si256();
        let mut a11 = _mm256_setzero_si256();

        let main_end = len.saturating_sub(4);
        for (n, &idx) in list.iter().enumerate() {
            if n < main_end {
                let ahead = list[n + 4];
                let p = rows.add(ahead as usize * HIDDEN_SIZE) as *const i8;
                _mm_prefetch(p, _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE);
            a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_add_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_add_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_add_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_add_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_add_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_add_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
            a8 = _mm256_add_epi16(a8, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(128) as *const __m128i)));
            a9 = _mm256_add_epi16(a9, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(144) as *const __m128i)));
            a10 = _mm256_add_epi16(a10, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(160) as *const __m128i)));
            a11 = _mm256_add_epi16(a11, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(176) as *const __m128i)));
        }

        let d = dst.as_mut_ptr() as *mut __m256i;
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
        _mm256_store_si256(d.add(8), a8);
        _mm256_store_si256(d.add(9), a9);
        _mm256_store_si256(d.add(10), a10);
        _mm256_store_si256(d.add(11), a11);

        // Next 192 lanes (base = 192)
        a0 = _mm256_setzero_si256();
        a1 = _mm256_setzero_si256();
        a2 = _mm256_setzero_si256();
        a3 = _mm256_setzero_si256();
        a4 = _mm256_setzero_si256();
        a5 = _mm256_setzero_si256();
        a6 = _mm256_setzero_si256();
        a7 = _mm256_setzero_si256();
        a8 = _mm256_setzero_si256();
        a9 = _mm256_setzero_si256();
        a10 = _mm256_setzero_si256();
        a11 = _mm256_setzero_si256();

        for (n, &idx) in list.iter().enumerate() {
            if n < main_end {
                let ahead = list[n + 4];
                let p = rows.add(ahead as usize * HIDDEN_SIZE + 192) as *const i8;
                _mm_prefetch(p, _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + 192);
            a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_add_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_add_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_add_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_add_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_add_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_add_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
            a8 = _mm256_add_epi16(a8, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(128) as *const __m128i)));
            a9 = _mm256_add_epi16(a9, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(144) as *const __m128i)));
            a10 = _mm256_add_epi16(a10, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(160) as *const __m128i)));
            a11 = _mm256_add_epi16(a11, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(176) as *const __m128i)));
        }

        let d = dst.as_mut_ptr().add(192) as *mut __m256i;
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
        _mm256_store_si256(d.add(8), a8);
        _mm256_store_si256(d.add(9), a9);
        _mm256_store_si256(d.add(10), a10);
        _mm256_store_si256(d.add(11), a11);

        // Last 128 lanes (base = 384, 8 YMM)
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut a4 = _mm256_setzero_si256();
        let mut a5 = _mm256_setzero_si256();
        let mut a6 = _mm256_setzero_si256();
        let mut a7 = _mm256_setzero_si256();

        for (n, &idx) in list.iter().enumerate() {
            if n < main_end {
                let ahead = list[n + 4];
                let p = rows.add(ahead as usize * HIDDEN_SIZE + 384) as *const i8;
                _mm_prefetch(p, _MM_HINT_T0);
            }
            let r = rows.add(idx as usize * HIDDEN_SIZE + 384);
            a0 = _mm256_add_epi16(a0, _mm256_cvtepi8_epi16(_mm_load_si128(r as *const __m128i)));
            a1 = _mm256_add_epi16(a1, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(16) as *const __m128i)));
            a2 = _mm256_add_epi16(a2, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(32) as *const __m128i)));
            a3 = _mm256_add_epi16(a3, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(48) as *const __m128i)));
            a4 = _mm256_add_epi16(a4, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(64) as *const __m128i)));
            a5 = _mm256_add_epi16(a5, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(80) as *const __m128i)));
            a6 = _mm256_add_epi16(a6, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(96) as *const __m128i)));
            a7 = _mm256_add_epi16(a7, _mm256_cvtepi8_epi16(_mm_load_si128(r.add(112) as *const __m128i)));
        }

        let d = dst.as_mut_ptr().add(384) as *mut __m256i;
        _mm256_store_si256(d, a0);
        _mm256_store_si256(d.add(1), a1);
        _mm256_store_si256(d.add(2), a2);
        _mm256_store_si256(d.add(3), a3);
        _mm256_store_si256(d.add(4), a4);
        _mm256_store_si256(d.add(5), a5);
        _mm256_store_si256(d.add(6), a6);
        _mm256_store_si256(d.add(7), a7);
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
            // horizontal mirroring: both files of a mirrored pair share a bucket
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
        let mut bytes = vec![0u8; LEGACY_NET_BYTES];
        for feature in 0..FEATURES_PER_BUCKET * LEGACY_NUM_INPUT_BUCKETS {
            let base = feature * HIDDEN_SIZE * 2;
            for lane in 0..HIDDEN_SIZE {
                let value = (feature * 31 + lane) as i16;
                bytes[base + lane * 2..base + lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
        let net = allocate_network();
        load_legacy_network(&bytes, net);
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
        // e1 -> d1 keeps bucket 3 but crosses the mirror line
        assert!(needs_accumulator_refresh(Square::E1, Square::D1, Color::White));
        // a3 -> b3 stays inside bucket 8
        assert!(!needs_accumulator_refresh(Square::A3, Square::B3, Color::White));
        // a1 -> a3 crosses from bucket 0 to bucket 8
        assert!(needs_accumulator_refresh(Square::A1, Square::A3, Color::White));
        // the same boundaries from black's point of view
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

    fn random_head(seed: u64) -> Box<MultilayerNet> {
        // heap-alloc so tests don't blow the stack
        let mut net: Box<MultilayerNet> = Box::new(MultilayerNet {
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
            for j in 0..L2_SIZE {
                for i in 0..L2_IN {
                    head.l2_weights[bucket][j][i] = (rng.next() & 0xFF) as i8 as i16;
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
        rebuild_l1_transposed(&mut net.head);
        net
    }

    // random accumulators for tests
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
            let check = |packed_layout: bool, psq: &[i16; HIDDEN_SIZE], threat: &[i16; HIDDEN_SIZE], logical: &[u8; PAIR_COUNT]| {
                let mut expected = *logical;
                if packed_layout {
                    for chunk in (0..PAIR_COUNT).step_by(32) {
                        for p in 0..32 {
                            expected[chunk + p] = logical[chunk + PAIRWISE_PACK_ORDER[p]];
                        }
                    }
                }
                let mut actual = [0u8; PAIR_COUNT];
                let nnz = if packed_layout {
                    unsafe { pairwise_side_avx2::<true>(psq, threat, &mut actual) }
                } else {
                    unsafe { pairwise_side_avx2::<false>(psq, threat, &mut actual) }
                };
                assert_eq!(actual, expected, "packed layout: {packed_layout}");

                let mut expected_nnz = 0u64;
                for block in (0..PAIR_COUNT).step_by(4) {
                    if expected[block..block + 4].iter().any(|&value| value != 0) {
                        expected_nnz |= 1u64 << (block / 4);
                    }
                }
                assert_eq!(nnz, expected_nnz, "packed layout: {packed_layout}");
            };

            let mut rng = Rng(0x5EED);
            for _ in 0..64 {
                let (psq, threat) = random_accumulators(rng.next());
                for side in 0..2 {
                    let mut logical = [0u8; PAIR_COUNT];
                    pairwise_side_scalar(&psq.vals[side], &threat.vals[side], &mut logical);
                    check(true, &psq.vals[side], &threat.vals[side], &logical);
                    check(false, &psq.vals[side], &threat.vals[side], &logical);
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
                // the AVX2 first layer consumes the packus lane order
                let mut packed_input = [0u8; HIDDEN_SIZE];
                for chunk in (0..HIDDEN_SIZE).step_by(32) {
                    for p in 0..32 {
                        packed_input[chunk + p] = input[chunk + PAIRWISE_PACK_ORDER[p]];
                    }
                }
                for bucket in 0..NUM_OUTPUT_BUCKETS {
                    let mut expected = [0i16; L2_IN];
                    l1_forward_scalar(&net.head.l1_weights[bucket], &net.head.l1_bias[bucket], &input, &mut expected);
                    let mut actual = [0i16; L2_IN];
                    unsafe {
                        l1_forward_avx2(
                            &net.head.l1_weights_t[bucket],
                            &net.head.l1_bias[bucket],
                            &packed_input,
                            [u64::MAX; 2],
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
                let mut input = [0i16; L2_IN];
                for value in input.iter_mut() {
                    *value = rng.centered(1 << 13) as i16;
                }
                for bucket in 0..NUM_OUTPUT_BUCKETS {
                    let mut expected = [0i32; L2_SIZE];
                    l2_forward_scalar(&net.head.l2_weights[bucket], &net.head.l2_bias[bucket], &input, &mut expected);
                    let mut actual = [0i32; L2_SIZE];
                    unsafe { l2_forward_avx2(&net.head.l2_weights[bucket], &net.head.l2_bias[bucket], &input, &mut actual) };
                    assert_eq!(actual, expected, "bucket {bucket}");
                }
            }
        }
    }

    #[test]
    fn multilayer_head_simd_matches_scalar() {
        let net = random_head(0x00DD_BA11);
        let mut rng = Rng(0xFEED);
        for _ in 0..32 {
            let (psq, threat) = random_accumulators(rng.next());
            let us = (rng.next() & 1) as usize;
            let them = us ^ 1;
            for bucket in 0..NUM_OUTPUT_BUCKETS {
                let expected = forward_scalar(&psq, &threat, us, them, bucket, &net.head);
                let actual = forward_multilayer(&psq, &threat.vals[us], &threat.vals[them], us, them, bucket, &net.head);
                assert_eq!(actual, expected, "bucket {bucket}");
            }
        }
    }

    #[test]
    fn head_input_shapes_match_the_kernels() {
        assert_eq!(L2_IN, 2 * L1_SIZE);
        assert_eq!(L2_SIZE % 8, 0, "the AVX2 layer-2 kernel reduces four accumulators");
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
    fn detect_format_recognizes_both_layouts() {
        assert_eq!(detect_format(LEGACY_NET_BYTES), Some(NetFormat::Legacy));
        assert_eq!(detect_format(NETWORK_BYTES.len()), Some(NetFormat::Multilayer));
        assert_eq!(detect_format(MULTILAYER_NET_BYTES), Some(NetFormat::Multilayer));
        assert_eq!(detect_format(MULTILAYER_NET_BYTES - 1), Some(NetFormat::Legacy));
        assert_eq!(detect_format(0), None);
        assert_eq!(detect_format(MULTILAYER_NET_BYTES + 1), None);
    }

    #[test]
    fn multilayer_loader_reads_the_disk_layout() {
        let mut bytes = vec![0u8; MULTILAYER_NET_BYTES];
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

        // last piece-square feature, last lane
        let last_feature = FEATURES_PER_BUCKET * NUM_INPUT_BUCKETS - 1;
        put_i16(&mut bytes, last_feature * HIDDEN_SIZE * 2 + 511 * 2, -1234);
        // last threat row, last lane: a signed byte, so 0x80 is -128
        bytes[ft_bytes + (TOTAL_THREATS - 1) * HIDDEN_SIZE + 511] = 0x80;
        put_i16(&mut bytes, ft_bytes + threat_bytes + 511 * 2, 777);
        // layer 1 is stored input-major: (input, bucket, output)
        let l1_base = ft_bytes + threat_bytes + bias_bytes;
        bytes[l1_base + ((511 * NUM_OUTPUT_BUCKETS + 7) * L1_SIZE + 31)] = 0xF9;
        let l1_bias_base = l1_base + l1_bytes;
        put_i32(&mut bytes, l1_bias_base + (7 * L1_SIZE + 31) * 4, 123_456);
        let l2_base = l1_bias_base + l1_bias_bytes;
        put_i32(&mut bytes, l2_base + ((63 * NUM_OUTPUT_BUCKETS + 7) * L2_SIZE + 15) * 4, -123);
        let l2_bias_base = l2_base + l2_bytes;
        put_i32(&mut bytes, l2_bias_base + (7 * L2_SIZE + 15) * 4, 4242);
        let l3_base = l2_bias_base + l2_bias_bytes;
        put_i32(&mut bytes, l3_base + (15 * NUM_OUTPUT_BUCKETS + 7) * 4, 99);
        let l3_bias_base = l3_base + l3_bytes;
        put_i32(&mut bytes, l3_bias_base + 7 * 4, -99);

        let net = allocate_network();
        load_multilayer_bytes(&bytes, net).expect("the synthetic network must load");

        assert_eq!((net as *mut Network as usize) % 64, 0);
        assert_eq!(net.feature_weights[last_feature].vals[511], -1234);
        assert_eq!(net.feature_bias.vals[511], 777);

        let multilayer = net.multilayer.as_ref().expect("head must be present");
        assert_eq!(multilayer.head.l1_weights[7][31][511], -7);
        assert_eq!(multilayer.head.l1_bias[7][31], 123_456);
        assert_eq!(multilayer.head.l2_weights[7][15][63], -123);
        assert_eq!(multilayer.head.l2_bias[7][15], 4242);
        assert_eq!(multilayer.head.l3_weights[7][15][0], 99);
        assert_eq!(multilayer.head.l3_bias[7][0], -99);

        let rows = &multilayer.threat_weights.as_ref().unwrap().rows;
        assert_eq!(rows[(TOTAL_THREATS - 1) * HIDDEN_SIZE + 511], -128);
        assert_eq!(rows.len(), TOTAL_THREATS * HIDDEN_SIZE);
    }

    #[test]
    fn multilayer_loader_rejects_other_sizes() {
        let net = allocate_network();
        let short = vec![0u8; MULTILAYER_NET_BYTES - 8];
        assert!(load_multilayer_bytes(&short, net).is_err());
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

    // mirror fen horizontally
    fn mirror_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().unwrap();
        let side = parts.next().unwrap_or("w");
        let ranks: Vec<String> = placement.split('/').map(|rank| rank.chars().rev().collect()).collect();
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
        // white pawn e4 attacks the black knight on d5, and nothing else on the
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

    fn eval_ordering(board: &mut Board, swap: bool) -> i32 {
        let net = network();
        let us = board.side_to_move as usize;
        let them = (!board.side_to_move) as usize;
        let bucket = output_bucket(board);
        let (a, b) = if swap { (them, us) } else { (us, them) };
        let acc = board.accumulator_immut();
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
            ("K+R w", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
            ("startpos w", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("midgame white +R", "r1bqk2r/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"),
        ];
        for (name, fen) in cases {
            let mut board = Board::from_fen(fen).unwrap();
            println!(
                "{}: normal={} swapped={}",
                name,
                eval_ordering(&mut board, false),
                eval_ordering(&mut board, true)
            );
        }

        let net = network();
        let mut k = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut kq = Board::from_fen("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let changed = k.accumulator_immut().vals[0]
            .iter()
            .zip(kq.accumulator_immut().vals[0].iter())
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
        println!(
            "output_weights: maxabs={} count>100={} total={}",
            ow_max,
            ow_gt100,
            NUM_OUTPUT_BUCKETS * 2 * HIDDEN_SIZE
        );
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
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
                let acc = board.accumulator_immut();
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
                let acc = board.accumulator_immut();
                // scalar eval in same perspective order
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
            // reset stack so random walk doesn't overflow
            board.reset_accumulator_stack();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
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

// pairwise activation using mulhi; returns a bitmask of nonzero 4-byte blocks.
// PACKED selects the packus lane order (AVX2 first layer) or logical order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_side_avx2<const PACKED: bool>(psq: &[i16; HIDDEN_SIZE], threats: &[i16; HIDDEN_SIZE], out: &mut [u8; PAIR_COUNT]) -> u64 {
    use std::arch::x86_64::*;
    unsafe {
        // both arrays come from 64-byte aligned accumulators and every load
        // below is at a multiple of 32 bytes
        assert_eq!(psq.as_ptr() as usize % 32, 0, "pairwise input must be 32-byte aligned");
        assert_eq!(threats.as_ptr() as usize % 32, 0, "pairwise input must be 32-byte aligned");
        let psq_ptr = psq.as_ptr();
        let threat_ptr = threats.as_ptr();

        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA);

        let pair = |offset: usize| {
            let lo = _mm256_load_si256(psq_ptr.add(offset) as *const __m256i);
            let lo = _mm256_add_epi16(lo, _mm256_load_si256(threat_ptr.add(offset) as *const __m256i));
            let hi = _mm256_load_si256(psq_ptr.add(offset + PAIR_COUNT) as *const __m256i);
            let hi = _mm256_add_epi16(hi, _mm256_load_si256(threat_ptr.add(offset + PAIR_COUNT) as *const __m256i));
            // clamp a to [0, QA], b only needs min(QA)
            let a = _mm256_min_epi16(_mm256_max_epi16(lo, zero), qa);
            let b = _mm256_min_epi16(hi, qa);
            _mm256_mulhi_epi16(_mm256_slli_epi16::<{ FT_SCALE_BITS as i32 }>(a), b)
        };

        let mut nnz = 0u64;
        let mut i = 0;
        while i < PAIR_COUNT {
            let p0 = pair(i);
            let p1 = pair(i + 16);
            // PACKED keeps the packus lane order for the AVX2 first layer
            let packed = _mm256_packus_epi16(p0, p1);
            let packed = if PACKED {
                packed
            } else {
                _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
            };
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, packed);
            // 8 dwords per iteration: set bits mark non-all-zero blocks
            let zero_dwords = _mm256_castsi256_ps(_mm256_cmpeq_epi32(packed, _mm256_setzero_si256()));
            let bits = !(_mm256_movemask_ps(zero_dwords) as u64) & 0xFF;
            nnz |= bits << (i / 4);
            i += 32;
        }
        nnz
    }
}

// Unpacks one byte of the nonzero-block mask into its set bit positions.
// All 8 slots are written so the fill loop stays branchless; callers keep
// slack at the end of the index buffer.
const fn build_nnz_table() -> [[u8; 8]; 256] {
    let mut table = [[0u8; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut n = 0usize;
        let mut bit = 0usize;
        while bit < 8 {
            if byte & (1 << bit) != 0 {
                table[byte][n] = bit as u8;
                n += 1;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
}

static NNZ_TABLE: [[u8; 8]; 256] = build_nnz_table();

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_forward_avx2(
    weights: &[[i8; 32]; L1_TRANSPOSED_BLOCKS],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    nnz: [u64; 2],
    out: &mut [i16; L2_IN],
) {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        // 32 outputs, 8 lanes each: no horizontal add at the end.
        // Two blocks are processed per iteration into two independent
        // accumulator sets, which halves the loop trip count and breaks the
        // dependency chain on acc0..3.
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut acc4 = _mm256_setzero_si256();
        let mut acc5 = _mm256_setzero_si256();
        let mut acc6 = _mm256_setzero_si256();
        let mut acc7 = _mm256_setzero_si256();

        // Expand the mask into a dense list of block indices first: 16
        // branchless table lookups instead of 128 dependent tzcnt steps.
        let mut idx_buf = [0u16; 128 + 8];
        let mut count = 0usize;
        let mut base = 0u16;
        for chunk in [nnz[0].to_le_bytes(), nnz[1].to_le_bytes()] {
            for &byte in chunk.iter() {
                let entry = &NNZ_TABLE[byte as usize];
                for k in 0..8 {
                    idx_buf[count + k] = base + entry[k] as u16;
                }
                count += byte.count_ones() as usize;
                base += 8;
            }
        }

        macro_rules! block4 {
            ($index:expr, $a0:ident, $a1:ident, $a2:ident, $a3:ident) => {{
                let idx = $index as usize;
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(input.as_ptr().add(idx * 4) as *const i32));
                let w = weights.as_ptr().add(idx * L1_OUTPUT_BLOCKS) as *const __m256i;
                $a0 = _mm256_add_epi32($a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256(w)), ones));
                $a1 = _mm256_add_epi32($a1, _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256(w.add(1))), ones));
                $a2 = _mm256_add_epi32($a2, _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256(w.add(2))), ones));
                $a3 = _mm256_add_epi32($a3, _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256(w.add(3))), ones));
            }};
        }

        let mut i = 0usize;
        while i + 1 < count {
            block4!(idx_buf[i], acc0, acc1, acc2, acc3);
            block4!(idx_buf[i + 1], acc4, acc5, acc6, acc7);
            i += 2;
        }
        if i < count {
            block4!(idx_buf[i], acc0, acc1, acc2, acc3);
        }
        acc0 = _mm256_add_epi32(acc0, acc4);
        acc1 = _mm256_add_epi32(acc1, acc5);
        acc2 = _mm256_add_epi32(acc2, acc6);
        acc3 = _mm256_add_epi32(acc3, acc7);

        let zero = _mm256_setzero_si256();
        let quant = _mm256_set1_epi32(QUANT);
        macro_rules! finish {
            ($acc:expr, $group:expr) => {{
                let biased = _mm256_add_epi32($acc, _mm256_loadu_si256(bias.as_ptr().add(($group) * 8) as *const __m256i));
                let shifted = _mm256_srai_epi32::<8>(biased);
                let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), quant);
                let crelu = _mm256_slli_epi32::<6>(clamped);
                let screlu = _mm256_mullo_epi32(clamped, clamped);
                let crelu16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(crelu, zero));
                let screlu16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(screlu, zero));
                _mm_storeu_si128(out.as_mut_ptr().add(($group) * 8) as *mut __m128i, _mm256_castsi256_si128(crelu16));
                _mm_storeu_si128(out.as_mut_ptr().add(L1_SIZE + ($group) * 8) as *mut __m128i, _mm256_castsi256_si128(screlu16));
            }};
        }
        finish!(acc0, 0);
        finish!(acc1, 1);
        finish!(acc2, 2);
        finish!(acc3, 3);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn l1_forward_avx2_vnni(
    weights: &[[i8; 32]; L1_TRANSPOSED_BLOCKS],
    bias: &[i32; L1_SIZE],
    input: &[u8; HIDDEN_SIZE],
    nnz: [u64; 2],
    out: &mut [i16; L2_IN],
) {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        macro_rules! block {
            ($index:expr) => {{
                let x = _mm256_set1_epi32(std::ptr::read_unaligned(input.as_ptr().add(($index) * 4) as *const i32));
                let w = weights.as_ptr().add(($index) * L1_OUTPUT_BLOCKS) as *const __m256i;
                acc0 = _mm256_dpwssd_epi32(acc0, _mm256_maddubs_epi16(x, _mm256_load_si256(w)), ones);
                acc1 = _mm256_dpwssd_epi32(acc1, _mm256_maddubs_epi16(x, _mm256_load_si256(w.add(1))), ones);
                acc2 = _mm256_dpwssd_epi32(acc2, _mm256_maddubs_epi16(x, _mm256_load_si256(w.add(2))), ones);
                acc3 = _mm256_dpwssd_epi32(acc3, _mm256_maddubs_epi16(x, _mm256_load_si256(w.add(3))), ones);
            }};
        }

        let mut bits = nnz[0];
        while bits != 0 {
            let index = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            block!(index);
        }
        let mut bits = nnz[1];
        while bits != 0 {
            let index = 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            block!(index);
        }

        let zero = _mm256_setzero_si256();
        let quant = _mm256_set1_epi32(QUANT);
        macro_rules! finish {
            ($acc:expr, $group:expr) => {{
                let biased = _mm256_add_epi32($acc, _mm256_loadu_si256(bias.as_ptr().add(($group) * 8) as *const __m256i));
                let shifted = _mm256_srai_epi32::<8>(biased);
                let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), quant);
                let crelu = _mm256_slli_epi32::<6>(clamped);
                let screlu = _mm256_mullo_epi32(clamped, clamped);
                let crelu16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(crelu, zero));
                let screlu16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(screlu, zero));
                _mm_storeu_si128(out.as_mut_ptr().add(($group) * 8) as *mut __m128i, _mm256_castsi256_si128(crelu16));
                _mm_storeu_si128(out.as_mut_ptr().add(L1_SIZE + ($group) * 8) as *mut __m128i, _mm256_castsi256_si128(screlu16));
            }};
        }
        finish!(acc0, 0);
        finish!(acc1, 1);
        finish!(acc2, 2);
        finish!(acc3, 3);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn l1_forward_avx512(weights: &[[i8; HIDDEN_SIZE]; L1_SIZE], bias: &[i32; L1_SIZE], input: &[u8; HIDDEN_SIZE], out: &mut [i16; L2_IN]) {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm512_set1_epi16(1);
        let mut j = 0;
        while j < L1_SIZE {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let mut a4 = _mm512_setzero_si512();
            let mut a5 = _mm512_setzero_si512();
            let mut a6 = _mm512_setzero_si512();
            let mut a7 = _mm512_setzero_si512();

            let r0 = weights[j].as_ptr() as *const __m512i;
            let r1 = weights[j + 1].as_ptr() as *const __m512i;
            let r2 = weights[j + 2].as_ptr() as *const __m512i;
            let r3 = weights[j + 3].as_ptr() as *const __m512i;
            let r4 = weights[j + 4].as_ptr() as *const __m512i;
            let r5 = weights[j + 5].as_ptr() as *const __m512i;
            let r6 = weights[j + 6].as_ptr() as *const __m512i;
            let r7 = weights[j + 7].as_ptr() as *const __m512i;

            let mut i = 0;
            while i < HIDDEN_SIZE {
                let x = _mm512_loadu_si512(input.as_ptr().add(i) as *const __m512i);
                let idx = i / 64;
                a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r0.add(idx))), ones));
                a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r1.add(idx))), ones));
                a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r2.add(idx))), ones));
                a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r3.add(idx))), ones));
                a4 = _mm512_add_epi32(a4, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r4.add(idx))), ones));
                a5 = _mm512_add_epi32(a5, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r5.add(idx))), ones));
                a6 = _mm512_add_epi32(a6, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r6.add(idx))), ones));
                a7 = _mm512_add_epi32(a7, _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(r7.add(idx))), ones));
                i += 64;
            }

            for (lane, acc) in [a0, a1, a2, a3, a4, a5, a6, a7].into_iter().enumerate() {
                let output = _mm512_reduce_add_epi32(acc).wrapping_add(bias[j + lane]) >> L1_SHIFT;
                let clamped = output.max(0).min(QUANT);
                out[j + lane] = (clamped << QUANT_BITS) as i16;
                out[L1_SIZE + j + lane] = (clamped * clamped) as i16;
            }
            j += 8;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn l1_forward_avx512_vnni(weights: &[[i8; HIDDEN_SIZE]; L1_SIZE], bias: &[i32; L1_SIZE], input: &[u8; HIDDEN_SIZE], out: &mut [i16; L2_IN]) {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm512_set1_epi16(1);
        let mut j = 0;
        while j < L1_SIZE {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            let mut a2 = _mm512_setzero_si512();
            let mut a3 = _mm512_setzero_si512();
            let mut a4 = _mm512_setzero_si512();
            let mut a5 = _mm512_setzero_si512();
            let mut a6 = _mm512_setzero_si512();
            let mut a7 = _mm512_setzero_si512();

            let r0 = weights[j].as_ptr() as *const __m512i;
            let r1 = weights[j + 1].as_ptr() as *const __m512i;
            let r2 = weights[j + 2].as_ptr() as *const __m512i;
            let r3 = weights[j + 3].as_ptr() as *const __m512i;
            let r4 = weights[j + 4].as_ptr() as *const __m512i;
            let r5 = weights[j + 5].as_ptr() as *const __m512i;
            let r6 = weights[j + 6].as_ptr() as *const __m512i;
            let r7 = weights[j + 7].as_ptr() as *const __m512i;

            let mut i = 0;
            while i < HIDDEN_SIZE {
                let x = _mm512_loadu_si512(input.as_ptr().add(i) as *const __m512i);
                let idx = i / 64;
                a0 = _mm512_dpwssd_epi32(a0, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r0.add(idx))), ones);
                a1 = _mm512_dpwssd_epi32(a1, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r1.add(idx))), ones);
                a2 = _mm512_dpwssd_epi32(a2, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r2.add(idx))), ones);
                a3 = _mm512_dpwssd_epi32(a3, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r3.add(idx))), ones);
                a4 = _mm512_dpwssd_epi32(a4, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r4.add(idx))), ones);
                a5 = _mm512_dpwssd_epi32(a5, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r5.add(idx))), ones);
                a6 = _mm512_dpwssd_epi32(a6, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r6.add(idx))), ones);
                a7 = _mm512_dpwssd_epi32(a7, _mm512_maddubs_epi16(x, _mm512_loadu_si512(r7.add(idx))), ones);
                i += 64;
            }

            for (lane, acc) in [a0, a1, a2, a3, a4, a5, a6, a7].into_iter().enumerate() {
                let output = _mm512_reduce_add_epi32(acc).wrapping_add(bias[j + lane]) >> L1_SHIFT;
                let clamped = output.max(0).min(QUANT);
                out[j + lane] = (clamped << QUANT_BITS) as i16;
                out[L1_SIZE + j + lane] = (clamped * clamped) as i16;
            }
            j += 8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_forward_avx2(weights: &[[i16; L2_IN]; L2_SIZE], bias: &[i32; L2_SIZE], input: &[i16; L2_IN], out: &mut [i32; L2_SIZE]) {
    use std::arch::x86_64::*;
    unsafe {
        // process 4 outputs at once
        let mut j = 0;
        while j < L2_SIZE {
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut a2 = _mm256_setzero_si256();
            let mut a3 = _mm256_setzero_si256();
            let r0 = weights[j].as_ptr();
            let r1 = weights[j + 1].as_ptr();
            let r2 = weights[j + 2].as_ptr();
            let r3 = weights[j + 3].as_ptr();
            let mut i = 0;
            while i < L2_IN {
                let x = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(x, _mm256_loadu_si256(r0.add(i) as *const __m256i)));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(x, _mm256_loadu_si256(r1.add(i) as *const __m256i)));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(x, _mm256_loadu_si256(r2.add(i) as *const __m256i)));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(x, _mm256_loadu_si256(r3.add(i) as *const __m256i)));
                i += 16;
            }
            let ab = _mm256_hadd_epi32(a0, a1);
            let cd = _mm256_hadd_epi32(a2, a3);
            let merged = _mm256_hadd_epi32(ab, cd);
            let sums = _mm_add_epi32(_mm256_castsi256_si128(merged), _mm256_extracti128_si256::<1>(merged));
            let bias4 = _mm_loadu_si128(bias.as_ptr().add(j) as *const __m128i);
            _mm_storeu_si128(out.as_mut_ptr().add(j) as *mut __m128i, _mm_add_epi32(sums, bias4));
            j += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_avx2(acc: &Accumulator, us: usize, them: usize, bucket: usize, net: &'static Network) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA);
        // 8 i32 accumulators (no overflow)
        let mut sum = _mm256_setzero_si256();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m256i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m256i;

            for chunk in 0..(HIDDEN_SIZE / 16) {
                let v = _mm256_load_si256(v_ptr.add(chunk));
                let x = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
                let w = _mm256_loadu_si256(w_ptr.add(chunk));

                // x * w fits in i16, then madd gives x^2 * w
                let xw = _mm256_mullo_epi16(x, w);
                // madd sums adjacent pairs into i32 lanes: x * (x * w) = x^2 * w
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn evaluate_avx512(acc: &Accumulator, us: usize, them: usize, bucket: usize, net: &'static Network) -> i32 {
    use std::arch::x86_64::*;

    unsafe {
        let zero = _mm512_setzero_si512();
        let qa = _mm512_set1_epi16(QA);
        // 16 i32 accumulators (no overflow)
        let mut sum = _mm512_setzero_si512();

        let mut evaluate_side = |vals: &[i16; HIDDEN_SIZE], offset: usize| {
            let v_ptr = vals.as_ptr() as *const __m512i;
            let w_ptr = net.output_weights[bucket][offset..].as_ptr() as *const __m512i;

            for chunk in 0..(HIDDEN_SIZE / 32) {
                let v = _mm512_loadu_si512(v_ptr.add(chunk));
                let x = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
                let w = _mm512_loadu_si512(w_ptr.add(chunk));

                // x * w fits in i16, then madd gives x^2 * w
                let xw = _mm512_mullo_epi16(x, w);
                // madd sums adjacent pairs into i32 lanes: x * (x * w) = x^2 * w
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
unsafe fn evaluate_neon(acc: &Accumulator, us: usize, them: usize, bucket: usize, net: &'static Network) -> i32 {
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
