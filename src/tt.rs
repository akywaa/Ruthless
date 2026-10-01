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
    pub key: u64,
    pub score: i32,
    pub depth: u8,
    pub flag: TTFlag,
    pub best_move: Move,
}

#[repr(C, align(32))]
struct Cluster {
    entries: [AtomicU64; 3],
    _pad: u64,
}

impl Cluster {
    fn empty() -> Self {
        Self {
            entries: [AtomicU64::new(0); 3],
            _pad: 0,
        }
    }
}

#[inline(always)]
fn pack(best_move: u16, key16: u16, score: i16, depth: u8, gen_bound: u8) -> u64 {
    (best_move as u64)
        | ((key16 as u64) << 16)
        | (((score as u16) as u64) << 32)
        | ((depth as u64) << 48)
        | ((gen_bound as u64) << 56)
}

#[inline(always)]
fn key16_of(word: u64) -> u16 {
    ((word >> 16) & 0xFFFF) as u16
}

#[inline(always)]
fn score_of(word: u64) -> i16 {
    ((word >> 32) & 0xFFFF) as u16 as i16
}

#[inline(always)]
fn depth_of(word: u64) -> u8 {
    ((word >> 48) & 0xFF) as u8
}

#[inline(always)]
fn gen_bound_of(word: u64) -> u8 {
    ((word >> 56) & 0xFF) as u8
}

#[inline(always)]
fn best_move_of(word: u64) -> Move {
    Move(word as u16)
}

#[inline(always)]
fn bound(word: u64) -> TTFlag {
    match gen_bound_of(word) & 3 {
        1 => TTFlag::Exact,
        2 => TTFlag::LowerBound,
        3 => TTFlag::UpperBound,
        _ => TTFlag::None,
    }
}

#[inline(always)]
fn age(word: u64) -> u8 {
    gen_bound_of(word) >> 2
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
        let key16 = (key >> 48) as u16;
        let cluster = &self.clusters[idx];

        for entry in &cluster.entries {
            let word = entry.load(Ordering::Relaxed);
            if key16_of(word) == key16 && bound(word) != TTFlag::None {
                return Some(TTEntry {
                    key,
                    score: score_of(word) as i32,
                    depth: depth_of(word),
                    flag: bound(word),
                    best_move: best_move_of(word),
                });
            }
        }

        None
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, depth: u8, flag: TTFlag, best_move: Move) {
        let idx = (key as usize) & self.mask;
        let key16 = (key >> 48) as u16;
        let cluster = &self.clusters[idx];
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x3F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            let word = entry.load(Ordering::Relaxed);
            if key16_of(word) == key16 || bound(word) == TTFlag::None {
                replace_idx = i;
                break;
            }

            let entry_age = (64 + curr_gen - age(word)) & 0x3F;
            let priority = (depth_of(word) as i32) - (entry_age as i32 * 8);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        let target = &cluster.entries[replace_idx];

        // Keep existing best_move when storing a short/depth<=0 search for same key
        let keep_move = best_move == Move::NULL && key16_of(target.load(Ordering::Relaxed)) == key16;
        let entry_score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let gen_bound = (curr_gen << 2) | (flag as u8);

        if keep_move {
            // Reload current word to preserve best_move, then store atomically
            let mut word = target.load(Ordering::Relaxed);
            while target
                .compare_exchange_weak(
                    word,
                    pack(best_move_of(word).0, key16, entry_score, depth, gen_bound),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                word = target.load(Ordering::Relaxed);
            }
        } else {
            target.store(pack(best_move.0, key16, entry_score, depth, gen_bound), Ordering::Relaxed);
        }
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for entry in &cluster.entries {
                entry.store(0, Ordering::Relaxed);
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }
}
