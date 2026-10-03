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
fn pack_data(score: i16, raw_eval: i16, depth: u8, gen_bound: u8, mv: u16) -> u64 {
    (mv as u64)
        | ((gen_bound as u64) << 16)
        | ((depth as u64) << 24)
        | (((score as u16) as u64) << 32)
        | (((raw_eval as u16) as u64) << 48)
}

#[inline(always)]
fn bound_of(gen_bound: u8) -> TTFlag {
    match gen_bound & 3 {
        1 => TTFlag::Exact,
        2 => TTFlag::LowerBound,
        3 => TTFlag::UpperBound,
        _ => TTFlag::None,
    }
}

#[inline(always)]
fn age_of(data: u64) -> u8 {
    (((data >> 16) & 0xFF) >> 2) as u8
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
        let cluster = &self.clusters[idx];

        for entry in &cluster.entries {
            let stored_key = entry[0].load(Ordering::Relaxed);
            if stored_key == key {
                let data = entry[1].load(Ordering::Relaxed);
                let gen_bound = ((data >> 16) & 0xFF) as u8;
                let flag = bound_of(gen_bound);
                if flag != TTFlag::None {
                    return Some(TTEntry {
                        best_move: Move((data & 0xFFFF) as u16),
                        depth: ((data >> 24) & 0xFF) as u8,
                        score: (((data >> 32) & 0xFFFF) as u16 as i16) as i32,
                        raw_eval: ((data >> 48) & 0xFFFF) as u16 as i16,
                        flag,
                    });
                }
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
        let cluster = &self.clusters[idx];
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x3F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            let stored_key = entry[0].load(Ordering::Relaxed);
            let data = entry[1].load(Ordering::Relaxed);
            if stored_key == key || bound_of(((data >> 16) & 0xFF) as u8) == TTFlag::None {
                replace_idx = i;
                break;
            }

            let entry_age = (64 + curr_gen - age_of(data)) & 0x3F;
            let depth = ((data >> 24) & 0xFF) as i32;
            let priority = depth - (entry_age as i32 * 8);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        let target = &cluster.entries[replace_idx];

        let keep_move = best_move == Move::NULL
            && target[0].load(Ordering::Relaxed) == key;
        let entry_score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let gen_bound = (curr_gen << 2) | (flag as u8);

        let mv = if keep_move {
            (target[1].load(Ordering::Relaxed) & 0xFFFF) as u16
        } else {
            best_move.0
        };

        let mut word = target[0].load(Ordering::Relaxed);
        loop {
            match target[0].compare_exchange_weak(
                word,
                key,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
        }
        target[1].store(pack_data(entry_score, raw_eval, depth, gen_bound, mv), Ordering::Relaxed);
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
