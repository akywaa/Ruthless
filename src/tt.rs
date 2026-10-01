use crate::types::Move;
use std::sync::atomic::{AtomicU8, Ordering};

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

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct ClusterEntry {
    key16: u16,
    score: i16,
    best_move: Move,
    depth: u8,
    gen_bound: u8,
}

impl ClusterEntry {
    #[inline(always)]
    fn bound(&self) -> TTFlag {
        match self.gen_bound & 3 {
            1 => TTFlag::Exact,
            2 => TTFlag::LowerBound,
            3 => TTFlag::UpperBound,
            _ => TTFlag::None,
        }
    }

    #[inline(always)]
    fn age(&self) -> u8 {
        self.gen_bound >> 2
    }
}

#[repr(C, align(32))]
struct Cluster {
    entries: [ClusterEntry; 3],
    _pad: u16,
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

        let mut clusters = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            clusters.push(Cluster {
                entries: [ClusterEntry::default(); 3],
                _pad: 0,
            });
        }

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
            if entry.key16 == key16 && entry.bound() != TTFlag::None {
                return Some(TTEntry {
                    key,
                    score: entry.score as i32,
                    depth: entry.depth,
                    flag: entry.bound(),
                    best_move: entry.best_move,
                });
            }
        }

        None
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, depth: u8, flag: TTFlag, best_move: Move) {
        let idx = (key as usize) & self.mask;
        let key16 = (key >> 48) as u16;
        let cluster_ptr = self.clusters.as_ptr() as *mut Cluster;
        let cluster = unsafe { &mut *cluster_ptr.add(idx) };
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x3F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;

        for (i, entry) in cluster.entries.iter_mut().enumerate() {
            if entry.key16 == key16 || entry.bound() == TTFlag::None {
                replace_idx = i;
                break;
            }

            let entry_age = (64 + curr_gen - entry.age()) & 0x3F;
            let priority = (entry.depth as i32) - (entry_age as i32 * 8);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        let target = &mut cluster.entries[replace_idx];

        if best_move != Move::NULL || target.key16 != key16 {
            target.best_move = best_move;
        }

        target.key16 = key16;
        target.score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        target.depth = depth;
        target.gen_bound = (curr_gen << 2) | (flag as u8);
    }

    pub fn clear(&self) {
        let cluster_ptr = self.clusters.as_ptr() as *mut Cluster;
        for i in 0..self.clusters.len() {
            unsafe {
                *cluster_ptr.add(i) = Cluster {
                    entries: [ClusterEntry::default(); 3],
                    _pad: 0,
                };
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }
}