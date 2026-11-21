use crate::alloc::HugeBuffer;
use crate::types::Move;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

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
    pub is_pv: bool,
}

pub const RAW_EVAL_NONE: i16 = i16::MIN;
pub const DEFAULT_HASH_MB: usize = 32;

#[repr(C, align(64))]
struct Cluster {
    entries: [CacheEntry; 4],
}

struct CacheEntry {
    key: AtomicU64,
    data: AtomicU64,
}

// zeroed tt clusters
fn alloc_zeroed_clusters(count: usize) -> HugeBuffer<Cluster> {
    HugeBuffer::zeroed(count)
}

#[inline(always)]
fn pack_data(score: i16, raw_eval: i16, depth: u8, gen_bound: u8, mv: u16) -> u64 {
    (mv as u64) | ((gen_bound as u64) << 16) | ((depth as u64) << 24) | (((score as u16) as u64) << 32) | (((raw_eval as u16) as u64) << 48)
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
fn is_pv_of(gen_bound: u8) -> bool {
    (gen_bound & 4) != 0
}

#[inline(always)]
fn age_of(data: u64) -> u8 {
    (((data >> 16) & 0xFF) >> 3) as u8
}

pub struct TranspositionTable {
    clusters: HugeBuffer<Cluster>,
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

    // fastrange indexing for arbitrary hash sizes
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        ((key as u128 * self.len as u128) >> 64) as usize
    }

    // prefetch tt cluster
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let idx = self.index(key);
            let ptr = self.clusters.as_ptr().add(idx) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = key;
        }
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
            let data = entry.data.load(Ordering::Relaxed);
            let stored_key = entry.key.load(Ordering::Relaxed);

            if stored_key ^ data == key {
                let gen_bound = ((data >> 16) & 0xFF) as u8;
                return Some(TTEntry {
                    best_move: Move((data & 0xFFFF) as u16),
                    depth: ((data >> 24) & 0xFF) as u8,
                    score: (((data >> 32) & 0xFFFF) as u16 as i16) as i32,
                    raw_eval: ((data >> 48) & 0xFFFF) as u16 as i16,
                    flag: bound_of(gen_bound),
                    is_pv: is_pv_of(gen_bound),
                });
            }
        }

        None
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, depth: u8, flag: TTFlag, best_move: Move, raw_eval: i16, is_pv: bool) {
        let idx = self.index(key);
        let cluster = &self.clusters[idx];
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x1F;

        let mut replace_idx = 0;
        let mut lowest_score = i32::MAX;
        let mut found_match = false;
        let mut old_data = 0; // cached entry data, avoids a second load

        for i in 0..4 {
            let data = cluster.entries[i].data.load(Ordering::Relaxed);
            let stored_key = cluster.entries[i].key.load(Ordering::Relaxed);

            // a race with a concurrent store makes this xor check fail
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

            let entry_age = (32 + curr_gen - age_of(data)) & 0x1F;
            let entry_depth = ((data >> 24) & 0xFF) as i32;
            let is_pv_entry = is_pv_of(gen_bound);
            let priority = entry_depth - (entry_age as i32 * 8) + (is_pv_entry as i32 * 4);

            if priority < lowest_score {
                lowest_score = priority;
                replace_idx = i;
            }
        }

        if found_match {
            let old_depth = ((old_data >> 24) & 0xFF) as i32;
            let old_age_diff = (32 + curr_gen - age_of(old_data)) & 0x1F;
            if flag != TTFlag::Exact && old_age_diff == 0 && (depth as i32) + 4 <= old_depth {
                // keep deeper entry, only fill missing best move
                if best_move != Move::NULL && (old_data & 0xFFFF) == 0 {
                    let updated = (old_data & !0xFFFF) | best_move.0 as u64;
                    let entry = &cluster.entries[replace_idx];
                    entry.data.store(updated, Ordering::Relaxed);
                    entry.key.store(key ^ updated, Ordering::Relaxed);
                }
                return;
            }
        }

        if found_match && depth == 0 && ((old_data >> 24) & 0xFF) > 0 {
            return;
        }

        let target_entry = &cluster.entries[replace_idx];

        let mv = if best_move == Move::NULL && found_match {
            (old_data & 0xFFFF) as u16
        } else {
            best_move.0
        };

        // keep pv flag sticky
        let old_is_pv = found_match && is_pv_of(((old_data >> 16) & 0xFF) as u8);
        let store_is_pv = is_pv || old_is_pv;

        let raw_eval = if raw_eval == crate::tt::RAW_EVAL_NONE && found_match {
            ((old_data >> 48) & 0xFFFF) as u16 as i16
        } else {
            raw_eval
        };

        let entry_score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let gen_bound = ((curr_gen & 0x1F) << 3) | ((store_is_pv as u8) << 2) | (flag as u8);
        let new_data = pack_data(entry_score, raw_eval, depth, gen_bound, mv);

        // lockless xor store
        target_entry.data.store(new_data, Ordering::Relaxed);
        target_entry.key.store(key ^ new_data, Ordering::Relaxed);
    }

    // store static eval without overwriting real entry
    #[inline(always)]
    pub fn store_eval(&self, key: u64, raw_eval: i16) {
        let idx = self.index(key);
        let cluster = &self.clusters[idx];

        // one pass: remember the first empty slot while looking for a match
        let mut empty: Option<usize> = None;
        for (i, entry) in cluster.entries.iter().enumerate() {
            let data = entry.data.load(Ordering::Relaxed);
            if entry.key.load(Ordering::Relaxed) ^ data == key {
                let new = (data & !(0xFFFFu64 << 48)) | (((raw_eval as u16) as u64) << 48);
                entry.data.store(new, Ordering::Relaxed);
                entry.key.store(key ^ new, Ordering::Relaxed);
                return;
            }
            if empty.is_none() && bound_of(((data >> 16) & 0xFF) as u8) == TTFlag::None {
                empty = Some(i);
            }
        }

        let Some(i) = empty else {
            return;
        };
        let curr_gen = self.generation.load(Ordering::Relaxed) & 0x1F;
        let new = pack_data(0, raw_eval, 0, (curr_gen & 0x1F) << 3, 0);
        let slot = &cluster.entries[i];
        slot.data.store(new, Ordering::Relaxed);
        slot.key.store(key ^ new, Ordering::Relaxed);
    }

    // uci hashfull permille
    pub fn hashfull(&self) -> usize {
        let sample = self.len.min(1000);
        let mut used = 0usize;

        for cluster in self.clusters.as_slice().iter().take(sample) {
            for entry in &cluster.entries {
                let data = entry.data.load(Ordering::Relaxed);
                if bound_of(((data >> 16) & 0xFF) as u8) != TTFlag::None {
                    used += 1;
                }
            }
        }

        used / 4
    }

    pub fn clear(&self) {
        for cluster in self.clusters.as_slice() {
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
        tt.store(42, 100, 8, TTFlag::Exact, Move(1), 0, true);
        tt.store(42, 50, 0, TTFlag::LowerBound, Move::NULL, 0, false);

        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.depth, 8);
        assert!(entry.best_move == Move(1));
        // shallow non-pv store shouldn't clear pv flag
        assert!(entry.is_pv);
    }

    #[test]
    fn shallow_store_does_not_replace_deeper_move() {
        let tt = TranspositionTable::new(1);
        tt.new_search();
        tt.store(42, 100, 8, TTFlag::Exact, Move(1), 0, true);
        tt.store(42, -50, 1, TTFlag::UpperBound, Move(2), 0, false);

        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.depth, 8);
        assert_eq!(entry.score, 100);
        assert!(entry.best_move == Move(1));
    }

    #[test]
    fn shallow_store_fills_missing_deeper_move() {
        let tt = TranspositionTable::new(1);
        tt.new_search();
        tt.store(42, 100, 8, TTFlag::UpperBound, Move::NULL, 0, false);
        tt.store(42, -50, 1, TTFlag::UpperBound, Move(7), 0, false);

        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.depth, 8);
        assert!(entry.best_move == Move(7));
    }
}
