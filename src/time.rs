use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::Board;
use crate::types::Color;

// move overhead in ms
pub static MOVE_OVERHEAD_MS: AtomicU64 = AtomicU64::new(15);

#[inline(always)]
pub fn move_overhead() -> u128 {
    MOVE_OVERHEAD_MS.load(Ordering::Relaxed) as u128
}

pub fn set_move_overhead(ms: u64) {
    MOVE_OVERHEAD_MS.store(ms, Ordering::Relaxed);
}

#[derive(Clone, Default, Debug)]
pub struct TimeControl {
    pub wtime: Option<u128>,
    pub btime: Option<u128>,
    pub winc: u128,
    pub binc: u128,
    pub movestogo: Option<u128>,
    pub movetime: Option<u128>,
    // go nodes: stop after this many nodes instead of by the clock
    pub nodes: Option<u64>,
}

pub struct TimeLimits {
    pub soft_limit: Option<u128>,
    pub hard_limit: Option<u128>,
    // true for fixed movetime
    pub fixed: bool,
    // node limit from go nodes
    pub node_limit: Option<u64>,
}

pub struct TimeManager;

impl TimeManager {
    pub fn allocate(board: &Board, tc: &TimeControl) -> TimeLimits {
        let overhead = move_overhead();

        if let Some(nodes) = tc.nodes {
            return TimeLimits {
                soft_limit: None,
                hard_limit: None,
                fixed: false,
                node_limit: Some(nodes.max(1)),
            };
        }

        if let Some(mt) = tc.movetime {
            let t = mt.saturating_sub(overhead).max(5);
            return TimeLimits {
                soft_limit: Some(t),
                hard_limit: Some(t),
                fixed: true,
                node_limit: None,
            };
        }

        let (my_time, my_inc) = match board.side_to_move {
            Color::White => (tc.wtime, tc.winc),
            Color::Black => (tc.btime, tc.binc),
        };

        if let Some(time) = my_time {
            let usable_time = time.saturating_sub(overhead) as f64;
            let inc = my_inc as f64;

            // cyclic time control (e.g. 40 moves in X minutes)
            if let Some(moves) = tc.movestogo {
                let moves = moves.clamp(2, 50) as f64;
                let base = (usable_time / moves) + 0.8 * inc;

                let soft = (base as u128).min((usable_time * 0.4) as u128).max(5);
                let hard = ((base * 2.5) as u128).min((usable_time * 0.8) as u128).max(soft);

                return TimeLimits {
                    soft_limit: Some(soft),
                    hard_limit: Some(hard),
                    fixed: false,
                    node_limit: None,
                };
            }

            // estimate remaining moves for fischer clock
            let move_num = board.fullmove_number as f64;
            let remaining_moves = 20.0 + 35.0 * f64::exp(-0.035 * move_num);

            // use full increment on fischer clock
            let base_time = (usable_time / remaining_moves) + inc;

            // don't allocate less than a fraction of our increment if we have one
            let min_time = (inc * 0.8).max(10.0);

            // cap soft limit to prevent blowing too much bank on a single complicated position
            let max_soft = usable_time * 0.35;

            let soft = base_time.max(min_time).min(max_soft).max(5.0) as u128;

            // hard limit is significantly larger to allow search.rs to extend time for PV node drops
            let hard = (base_time * 3.5).min(usable_time * 0.8).max(soft as f64) as u128;

            TimeLimits {
                soft_limit: Some(soft),
                hard_limit: Some(hard),
                fixed: false,
                node_limit: None,
            }
        } else {
            TimeLimits {
                soft_limit: None,
                hard_limit: None,
                fixed: false,
                node_limit: None,
            }
        }
    }
}
