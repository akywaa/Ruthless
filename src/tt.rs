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
    entries: [CacheEntry; 4],
}

struct CacheEntry {
    key: AtomicU64,
    data: AtomicU64,
}

/// Allocates zeroed clusters in one go. Zero is a valid state for every field
/// (an all-zero key/data pair simply never matches), and allocating with
/// `alloc_zeroed` lets the OS hand back pre-zeroed pages instead of touching
/// every byte in user space, which matters for multi-gigabyte hash sizes.
fn alloc_zeroed_clusters(count: usize) -> Box<[Cluster]> {
    let layout = std::alloc::Layout::array::<Cluster>(count).expect("TT size overflow");
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout) as *mut Cluster;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, count))
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
    clusters: Box<[Cluster]>,
    len: usize,
    generation: AtomicU8,
}

unsafe impl Sync for TranspositionTable {}
unsafe impl Send for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb.max(1).saturating_mul(1024 * 1024);
        let cluster_count = (size_bytes / std::mem::size_of::<Cluster>()).max(1);

        Self {
            clusters: alloc_zeroed_clusters(cluster_count),
            len: cluster_count,
            generation: AtomicU8::new(0),
        }
    }

    /// Multiply-shift indexing: maps the full 64-bit key onto `0..len` and
    /// honours exactly the requested size instead of rounding down to a power
    /// of two (Hash 24000 no longer silently becomes 16384).
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.len as u128) >> 64) as usize
    }

    #[inline(always)]
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let idx = self.index(key);
        let cluster = &self.clusters[idx];

        for entry in &cluster.entries {
            // Load data first, then the xored key. If a concurrent store
            // races between the two loads, the xor check simply fails
            // instead of returning garbage.
            let data = entry.data.load(Ordering::Relaxed);
            let stored_key = entry.key.load(Ordering::Relaxed);

            if stored_key ^ data == key {
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
        let idx = self.index(key);
        let cluster = &self.clusters[idx];
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x3F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;
        let mut found_match = false;
        let mut old_data = 0; // cached entry data, avoids a second load

        for i in 0..4 {
            let data = cluster.entries[i].data.load(Ordering::Relaxed);
            let stored_key = cluster.entries[i].key.load(Ordering::Relaxed);

            // A race with a concurrent store makes this xor check fail.
            if stored_key ^ data == key {
                replace_idx = i;
                found_match = true;
                old_data = data;
                break;
            }

            let gen_bound = ((data >> 16) & 0xFF) as u8;

            if bound_of(gen_bound) == TTFlag::None {
                replace_idx = i;
                lowest_score = i32::MIN;
                continue;
            }

            let entry_age = (64 + curr_gen - age_of(data)) & 0x3F;
            let entry_depth = ((data >> 24) & 0xFF) as i32;
            let priority = entry_depth - (entry_age as i32 * 8);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        if found_match {
            let old_depth = ((old_data >> 24) & 0xFF) as i32;
            let old_age_diff = (64 + curr_gen - age_of(old_data)) & 0x3F;
            if flag != TTFlag::Exact && old_age_diff == 0 && (depth as i32) + 4 <= old_depth {
                if best_move != Move::NULL {
                    let new_mv = best_move.0 as u64;
                    if new_mv != (old_data & 0xFFFF) {
                        let updated = (old_data & !0xFFFF) | new_mv;
                        let entry = &cluster.entries[replace_idx];
                        entry.data.store(updated, Ordering::Relaxed);
                        entry.key.store(key ^ updated, Ordering::Relaxed);
                    }
                }
                return;
            }
        }

        let target_entry = &cluster.entries[replace_idx];

        let mv = if best_move == Move::NULL && found_match {
            (old_data & 0xFFFF) as u16
        } else {
            best_move.0
        };

        let entry_score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let gen_bound = (curr_gen << 2) | (flag as u8);
        let new_data = pack_data(entry_score, raw_eval, depth, gen_bound, mv);

        // Write plain data first, then the xored key so a racing reader
        // either sees a consistent pair or fails the xor check.
        target_entry.data.store(new_data, Ordering::Relaxed);
        target_entry.key.store(key ^ new_data, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for entry in &cluster.entries {
                entry.key.store(0, Ordering::Relaxed);
                entry.data.store(0, Ordering::Relaxed);
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_keeps_deeper_entry() {
        let tt = TranspositionTable::new(1);
        tt.new_search();
        tt.store(42, 100, 8, TTFlag::Exact, Move(1), 0);
        tt.store(42, 50, 0, TTFlag::LowerBound, Move::NULL, 0);

        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.depth, 8);
        assert!(entry.best_move == Move(1));
    }
}
