use crate::types::Move;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u64,
    pub score: i32,
    pub depth: u8,
    pub flag: TTFlag,
    pub best_move: Move,
}

struct RawEntry {
    key: AtomicU64,
    data: AtomicU64,
}

pub struct TranspositionTable {
    entries: Vec<RawEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let count = (mb * 1024 * 1024) / std::mem::size_of::<RawEntry>();
        let size = count.next_power_of_two();
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(RawEntry {
                key: AtomicU64::new(0),
                data: AtomicU64::new(0),
            });
        }

        Self {
            entries,
            mask: size - 1,
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = (key as usize) & self.mask;
        let entry = &self.entries[idx];
        if entry.key.load(Ordering::Relaxed) == key {
            let data = entry.data.load(Ordering::Relaxed);
            let score = (data as u32) as i32;
            let depth = ((data >> 32) & 0xFF) as u8;
            let flag = match ((data >> 40) & 0xFF) as u8 {
                1 => TTFlag::LowerBound,
                2 => TTFlag::UpperBound,
                _ => TTFlag::Exact,
            };
            let best_move = Move((data >> 48) as u16);
            Some(TTEntry {
                key,
                score,
                depth,
                flag,
                best_move,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, depth: u8, flag: TTFlag, best_move: Move) {
        let idx = (key as usize) & self.mask;
        let entry = &self.entries[idx];
        let entry_key = entry.key.load(Ordering::Relaxed);
        let entry_data = entry.data.load(Ordering::Relaxed);
        let entry_depth = ((entry_data >> 32) & 0xFF) as u8;

        if entry_key != key || depth >= entry_depth {
            let data = (score as u32 as u64)
                | ((depth as u64) << 32)
                | ((flag as u64) << 40)
                | ((best_move.0 as u64) << 48);

            entry.key.store(key, Ordering::Relaxed);
            entry.data.store(data, Ordering::Relaxed);
        }
    }

    pub fn clear(&self) {
        for entry in &self.entries {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }
}