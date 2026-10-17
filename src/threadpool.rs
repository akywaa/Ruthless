use crate::board::Board;
use crate::numa::bind_thread;
use crate::search::Searcher;
use crate::time::TimeLimits;
use crate::tt::TranspositionTable;
use crate::types::Move;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

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
                            active_clone.fetch_sub(1, Ordering::Relaxed);
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
        while self.active_helpers.load(Ordering::Relaxed) != 0 {
            std::thread::yield_now();
        }

        bind_thread(0);
        self.shared_nodes.store(0, Ordering::Relaxed);

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

        let best_move =
            self.main_searcher.search(board, depth, limits.soft_limit, limits.hard_limit);

        self.stop_signal.store(true, Ordering::Relaxed);

        // Put helpers back to sleep
        if self.num_threads > 1 {
            let (lock, cvar) = &*self.shared_state;
            let mut state = lock.lock().unwrap();
            state.command = Command::Idle;
            state.generation += 1;
            cvar.notify_all();
        }

        best_move
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
