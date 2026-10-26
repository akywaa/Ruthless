use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::numa::bind_thread;
use crate::search::{Searcher, MATE_SCORE};
use crate::time::TimeLimits;
use crate::tt::TranspositionTable;
use crate::types::Move;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone)]
pub struct SearchParams {
    pub board: Board,
    pub depth: u8,
    pub soft_time: Option<u128>,
    pub hard_time: Option<u128>,
    pub stop: Arc<AtomicBool>,
    pub clear_histories: bool,
}

#[derive(Clone)]
struct SearchResult {
    best_move: Move,
    score: i32,
    depth: u8,
    pv: Vec<Move>,
}

#[derive(Clone)]
pub enum Command {
    Idle,
    Search(SearchParams),
    Exit,
}

pub struct PoolState {
    pub command: Command,
    pub generation: u64,
}

pub struct ThreadPool {
    pub tt: Arc<TranspositionTable>,
    pub main_searcher: Searcher,
    pub shared_nodes: Arc<AtomicU64>,
    shared_state: Arc<(Mutex<PoolState>, Condvar)>,
    handles: Vec<thread::JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
    active_helpers: Arc<AtomicU64>,
    num_threads: usize,
    clear_pending: bool,
    results: Arc<Mutex<Vec<Option<SearchResult>>>>,
}

impl ThreadPool {
    pub fn new(mb: usize, num_threads: usize) -> Self {
        let tt = Arc::new(TranspositionTable::new(mb));
        let stop_signal = Arc::new(AtomicBool::new(false));
        let shared_nodes = Arc::new(AtomicU64::new(0));
        let active_helpers = Arc::new(AtomicU64::new(0));

        let main_searcher = Searcher::new(
            Arc::clone(&tt),
            Arc::clone(&stop_signal),
            0,
            num_threads,
            Arc::clone(&shared_nodes),
        );

        let mut pool = Self {
            tt,
            main_searcher,
            shared_nodes,
            shared_state: Arc::new((
                Mutex::new(PoolState {
                    command: Command::Idle,
                    generation: 0,
                }),
                Condvar::new(),
            )),
            handles: Vec::new(),
            stop_signal,
            active_helpers,
            num_threads: 1, // Resized correctly below
            clear_pending: false,
            results: Arc::new(Mutex::new(vec![None])),
        };

        pool.set_threads(num_threads);
        pool
    }

    pub fn set_hash(&mut self, mb: usize) {
        let old_threads = self.num_threads;
        self.set_threads(1); // Kill helpers to safely drop old TT

        self.tt = Arc::new(TranspositionTable::new(mb));
        self.main_searcher.tt = Arc::clone(&self.tt);

        self.set_threads(old_threads); // Respawn and bind to new TT
    }

    pub fn set_threads(&mut self, new_num: usize) {
        let target_num = new_num.max(1);
        if self.num_threads == target_num && !self.handles.is_empty() {
            return;
        }

        // Send Exit signal to stop existing threads gracefully
        {
            let (lock, cvar) = &*self.shared_state;
            let mut state = lock.lock().unwrap();
            state.command = Command::Exit;
            state.generation += 1;
            cvar.notify_all();
        }

        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }

        self.num_threads = target_num;
        self.main_searcher.num_threads = target_num;

        let results = Arc::new(Mutex::new(vec![None; target_num]));
        self.results = Arc::clone(&results);

        // Reset to Idle so freshly spawned helpers wait for a real command
        // instead of inheriting the stale Exit above.
        {
            let (lock, _cvar) = &*self.shared_state;
            lock.lock().unwrap().command = Command::Idle;
        }

        // Spawn new helpers
        for id in 1..target_num {
            let state_clone = Arc::clone(&self.shared_state);
            let tt_clone = Arc::clone(&self.tt);
            let stop_clone = Arc::clone(&self.stop_signal);
            let nodes_clone = Arc::clone(&self.shared_nodes);
            let active_clone = Arc::clone(&self.active_helpers);
            let results_clone = Arc::clone(&results);
            let total_threads = target_num;

            let handle = thread::spawn(move || {
                bind_thread(id);
                let mut helper = Searcher::new(
                    tt_clone,
                    stop_clone,
                    id,
                    total_threads,
                    nodes_clone,
                );

                let mut last_gen = 0;
                loop {
                    let mut state = state_clone.0.lock().unwrap();
                    // Wake up only if generation changed and we are no longer idle
                    while state.generation == last_gen || matches!(state.command, Command::Idle) {
                        if matches!(state.command, Command::Exit) {
                            return;
                        }
                        state = state_clone.1.wait(state).unwrap();
                    }
                    last_gen = state.generation;

                    let command = state.command.clone();
                    drop(state);

                    match command {
                        Command::Search(params) => {
                            if params.clear_histories {
                                helper.clear_histories();
                            }
                            let mut board_clone = params.board;
                            helper.stop = Arc::clone(&params.stop);
                            active_clone.fetch_add(1, Ordering::Relaxed);
                            helper.search_helper(&mut board_clone, params.depth, params.soft_time);

                            if helper.completed_depth > 0 && helper.root_best_move != Move::NULL {
                                let mut pv = helper.extract_pv(&mut board_clone, helper.completed_depth);
                                if pv.first() != Some(&helper.root_best_move) {
                                    pv.clear();
                                    pv.push(helper.root_best_move);
                                }
                                results_clone.lock().unwrap()[id] = Some(SearchResult {
                                    best_move: helper.root_best_move,
                                    score: helper.root_score,
                                    depth: helper.completed_depth,
                                    pv,
                                });
                            }

                            active_clone.fetch_sub(1, Ordering::Release);
                        }
                        Command::Exit => return,
                        Command::Idle => unreachable!(),
                    }
                }
            });

            self.handles.push(handle);
        }
    }

    pub fn clear(&mut self) {
        self.main_searcher.clear();
        self.clear_pending = true;
    }

    pub fn new_search_stop(&mut self) -> Arc<AtomicBool> {
        let stop = Arc::new(AtomicBool::new(false));
        self.stop_signal = Arc::clone(&stop);
        self.main_searcher.stop = Arc::clone(&stop);
        stop
    }

    pub fn start_search(&mut self, board: &mut Board, depth: u8, limits: TimeLimits) -> Move {
        // Wait for helpers from the previous search to fully stop. Otherwise they
        // would keep searching the stale position and pollute the TT with moves
        // from the previous ply.
        while self.active_helpers.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }

        {
            let mut results = self.results.lock().unwrap();
            for slot in results.iter_mut() {
                *slot = None;
            }
        }

        bind_thread(0);
        self.shared_nodes.store(0, Ordering::Relaxed);
        let search_start = Instant::now();

        let clear_histories = self.clear_pending;
        self.clear_pending = false;

        let params = SearchParams {
            board: board.clone(),
            depth,
            soft_time: limits.soft_limit,
            hard_time: limits.hard_limit,
            stop: Arc::clone(&self.stop_signal),
            clear_histories,
        };

        // Wake up sleeping helpers instantly
        if self.num_threads > 1 {
            let (lock, cvar) = &*self.shared_state;
            let mut state = lock.lock().unwrap();
            state.command = Command::Search(params);
            state.generation += 1;
            cvar.notify_all();
        }

        self.main_searcher.fixed_time = limits.fixed;
        self.main_searcher.node_limit = limits.node_limit;
        let best_move =
            self.main_searcher.search(board, depth, limits.soft_limit, limits.hard_limit);

        self.stop_signal.store(true, Ordering::Relaxed);

        // Single-threaded searches need none of the helper bookkeeping.
        if self.num_threads == 1 {
            return best_move;
        }

        // Helpers publish their last completed iteration before leaving the
        // search loop, so wait for them before reading the result slots.
        while self.active_helpers.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }

        // Put helpers back to sleep
        if self.num_threads > 1 {
            let (lock, cvar) = &*self.shared_state;
            let mut state = lock.lock().unwrap();
            state.command = Command::Idle;
            state.generation += 1;
            cvar.notify_all();
        }

        let main_depth = self.main_searcher.completed_depth;
        let mut main_pv = self.main_searcher.extract_pv(board, main_depth);
        if main_pv.first() != Some(&best_move) {
            main_pv.clear();
            if best_move != Move::NULL {
                main_pv.push(best_move);
            }
        }
        let main_result = SearchResult {
            best_move,
            score: self.main_searcher.root_score,
            depth: main_depth,
            pv: main_pv,
        };

        let legal_moves = generate_legal_moves(board);
        let helpers: Vec<SearchResult> = {
            let results = self.results.lock().unwrap();
            results
                .iter()
                .skip(1)
                .flatten()
                .filter(|result| legal_moves.as_slice().contains(&result.best_move))
                .cloned()
                .collect()
        };

        let selected = Self::choose_best(main_result, helpers);
        if selected.best_move != best_move || selected.depth > main_depth {
            let total_nodes = self.shared_nodes.load(Ordering::Relaxed)
                + (self.main_searcher.nodes & 1023);
            let elapsed = search_start.elapsed().as_millis().max(1);
            let nps = (total_nodes as u128 * 1000) / elapsed;
            print_search_result(&selected, total_nodes, nps, elapsed);
        }

        selected.best_move
    }

    fn choose_best(main: SearchResult, helpers: Vec<SearchResult>) -> SearchResult {
        let candidates: Vec<SearchResult> = std::iter::once(main.clone())
            .chain(helpers)
            .filter(|result| result.best_move != Move::NULL && result.depth > 0)
            .collect();

        if candidates.is_empty() {
            return main;
        }

        let mate_threshold = MATE_SCORE - 100;
        if let Some(best) = candidates
            .iter()
            .filter(|result| result.score.abs() > mate_threshold)
            .max_by_key(|result| result.score)
        {
            return best.clone();
        }

        // Stockfish-style vote: scores are shifted so the worst result has a
        // small positive weight, then every thread votes for its best move.
        let min_score = candidates.iter().map(|result| result.score).min().unwrap();
        let mut votes: Vec<(Move, i64)> = Vec::with_capacity(candidates.len());

        for result in &candidates {
            let weight = i64::from(result.score - min_score) + 14;
            if let Some((_, total)) = votes.iter_mut().find(|(m, _)| *m == result.best_move) {
                *total += weight;
            } else {
                votes.push((result.best_move, weight));
            }
        }

        let max_votes = votes.iter().map(|(_, total)| *total).max().unwrap();
        candidates
            .into_iter()
            .filter(|result| {
                votes
                    .iter()
                    .any(|(m, total)| *m == result.best_move && *total == max_votes)
            })
            .max_by_key(|result| (result.depth, result.pv.len()))
            .unwrap_or(main)
    }
}

fn print_search_result(result: &SearchResult, nodes: u64, nps: u128, time: u128) {
    let pv = result
        .pv
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    if result.score.abs() > MATE_SCORE - 100 {
        let mate_dist = (MATE_SCORE - result.score.abs() + 1) / 2;
        let sign = if result.score > 0 { 1 } else { -1 };
        println!(
            "info depth {} score mate {} nodes {} nps {} time {} pv {}",
            result.depth,
            mate_dist * sign,
            nodes,
            nps,
            time,
            pv
        );
    } else {
        println!(
            "info depth {} score cp {} nodes {} nps {} time {} pv {}",
            result.depth, result.score, nodes, nps, time, pv
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(best_move: u16, score: i32, depth: u8) -> SearchResult {
        SearchResult {
            best_move: Move(best_move),
            score,
            depth,
            pv: vec![Move(best_move)],
        }
    }

    #[test]
    fn voting_prefers_the_higher_scored_move() {
        let main = result(1, 20, 10);
        let helpers = vec![result(2, 100, 11), result(3, 0, 9)];
        assert!(ThreadPool::choose_best(main, helpers).best_move == Move(2));
    }

    #[test]
    fn mate_result_wins_over_regular_eval() {
        let main = result(1, 100, 10);
        let helpers = vec![result(2, MATE_SCORE - 3, 8)];
        assert!(ThreadPool::choose_best(main, helpers).best_move == Move(2));
    }

    #[test]
    fn same_move_prefers_the_deeper_result() {
        let main = result(1, 50, 10);
        let helpers = vec![result(1, 40, 12)];
        let picked = ThreadPool::choose_best(main, helpers);
        assert!(picked.best_move == Move(1));
        assert_eq!(picked.depth, 12);
    }

    #[test]
    fn no_completed_results_returns_main() {
        let main = result(1, 0, 0);
        assert!(ThreadPool::choose_best(main, Vec::new()).best_move == Move(1));
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Release the lock before joining: the workers need it to observe Exit.
        {
            let (lock, cvar) = &*self.shared_state;
            let mut state = lock.lock().unwrap();
            state.command = Command::Exit;
            state.generation += 1;
            cvar.notify_all();
        }

        while let Some(handle) = self.handles.pop() {
            let _ = handle.join();
        }
    }
}
