use crate::types::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    LowerBound = 2,
    UpperBound = 3,
}

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub score: i32,
    pub raw_eval: i16,
    pub depth: u8,
    pub flag: TTFlag,
    pub best_move: Move,
}

pub const RAW_EVAL_NONE: i16 = i16::MIN;
const HAS_EVAL_MASK: u64 = 1 << 32;

#[repr(C, align(64))]
struct Cluster {
    entries: [[AtomicU64; 2]; 3],
    _pad: u64,
}

impl Cluster {
    fn empty() -> Self {
        Self {
            entries: std::array::from_fn(|_| [AtomicU64::new(0), AtomicU64::new(0)]),
            _pad: 0,
        }
    }
}

#[inline(always)]
fn pack_meta(key32: u32, score: i16, depth: u8, gen_bound: u8) -> u64 {
    (key32 as u64)
        | (((score as u16) as u64) << 32)
        | ((depth as u64) << 48)
        | ((gen_bound as u64) << 56)
}

#[inline(always)]
fn key32_of(meta: u64) -> u32 {
    (meta & 0xFFFF_FFFF) as u32
}

#[inline(always)]
fn score_of(meta: u64) -> i16 {
    ((meta >> 32) & 0xFFFF) as u16 as i16
}

#[inline(always)]
fn depth_of(meta: u64) -> u8 {
    ((meta >> 48) & 0xFF) as u8
}

#[inline(always)]
fn gen_bound_of(meta: u64) -> u8 {
    ((meta >> 56) & 0xFF) as u8
}

#[inline(always)]
fn bound(meta: u64) -> TTFlag {
    match gen_bound_of(meta) & 3 {
        1 => TTFlag::Exact,
        2 => TTFlag::LowerBound,
        3 => TTFlag::UpperBound,
        _ => TTFlag::None,
    }
}

#[inline(always)]
fn age(meta: u64) -> u8 {
    gen_bound_of(meta) >> 2
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    mask: usize,
    generation: AtomicU8,
}

unsafe impl Sync for TranspositionTable {}
unsafe impl Send for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let cluster_count = (size_bytes / std::mem::size_of::<Cluster>()).next_power_of_two();
        let mask = cluster_count - 1;

        let clusters = (0..cluster_count).map(|_| Cluster::empty()).collect();

        Self {
            clusters,
            mask,
            generation: AtomicU8::new(0),
        }
    }

    #[inline(always)]
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = (key as usize) & self.mask;
        let key32 = (key >> 32) as u32;
        let cluster = &self.clusters[idx];

        for entry in &cluster.entries {
            let meta = entry[0].load(Ordering::Relaxed);
            if key32_of(meta) == key32 && bound(meta) != TTFlag::None {
                let data1 = entry[1].load(Ordering::Relaxed);
                let mv = data1 as u16;
                let raw_eval = if data1 & HAS_EVAL_MASK != 0 {
                    ((data1 >> 16) & 0xFFFF) as u16 as i16
                } else {
                    RAW_EVAL_NONE
                };
                return Some(TTEntry {
                    score: score_of(meta) as i32,
                    raw_eval,
                    depth: depth_of(meta),
                    flag: bound(meta),
                    best_move: Move(mv),
                });
            }
        }

        None
    }

    #[inline(always)]
    pub fn store(
        &self,
        key: u64,
        score: i32,
        depth: u8,
        flag: TTFlag,
        best_move: Move,
        raw_eval: i16,
    ) {
        let idx = (key as usize) & self.mask;
        let key32 = (key >> 32) as u32;
        let cluster = &self.clusters[idx];
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x3F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            let meta = entry[0].load(Ordering::Relaxed);
            if key32_of(meta) == key32 || bound(meta) == TTFlag::None {
                replace_idx = i;
                break;
            }

            let entry_age = (64 + curr_gen - age(meta)) & 0x3F;
            let priority = (depth_of(meta) as i32) - (entry_age as i32 * 8);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        let target = &cluster.entries[replace_idx];

        let keep_move = best_move == Move::NULL && key32_of(target[0].load(Ordering::Relaxed)) == key32;
        let entry_score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let gen_bound = (curr_gen << 2) | (flag as u8);

        let mv = if keep_move {
            let prev = target[1].load(Ordering::Relaxed);
            (prev & 0xFFFF) | ((raw_eval as u16 as u64) << 16) | HAS_EVAL_MASK
        } else {
            (best_move.0 as u64) | ((raw_eval as u16 as u64) << 16) | HAS_EVAL_MASK
        };

        let mut word = target[0].load(Ordering::Relaxed);
        loop {
            match target[0].compare_exchange_weak(
                word,
                pack_meta(key32, entry_score, depth, gen_bound),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
        }
        target[1].store(mv, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for entry in &cluster.entries {
                entry[0].store(0, Ordering::Relaxed);
                entry[1].store(0, Ordering::Relaxed);
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }
}
