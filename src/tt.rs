use crate::types::Move;

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

pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let count = (mb * 1024 * 1024) / std::mem::size_of::<TTEntry>();
        let size = count.next_power_of_two();
        let empty = TTEntry {
            key: 0,
            score: 0,
            depth: 0,
            flag: TTFlag::Exact,
            best_move: Move::NULL,
        };

        Self {
            entries: vec![empty; size],
            mask: size - 1,
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let entry = &self.entries[(key as usize) & self.mask];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn store(&mut self, key: u64, score: i32, depth: u8, flag: TTFlag, best_move: Move) {
        let idx = (key as usize) & self.mask;
        let entry = &mut self.entries[idx];
        if entry.key != key || depth >= entry.depth {
            *entry = TTEntry {
                key,
                score,
                depth,
                flag,
                best_move,
            };
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            entry.key = 0;
            entry.score = 0;
            entry.depth = 0;
            entry.best_move = Move::NULL;
        }
    }
}
