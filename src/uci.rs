use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::threadpool::ThreadPool;
use crate::time::{TimeControl, TimeLimits, TimeManager};
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

pub fn uci_loop() {
    let mut board = Board::default();
    let pool = Arc::new(Mutex::new(ThreadPool::new(32, 1)));
    let mut stop = pool.lock().unwrap().new_search_stop();
    let mut pending_stop: Option<Arc<AtomicBool>> = None;
    let stdin = io::stdin();

    let (go_tx, go_rx) = mpsc::channel::<(Board, u8, TimeControl, bool, Arc<AtomicBool>)>();
    let search_pool = Arc::clone(&pool);

    let _search_thread = thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            while let Ok((mut search_board, depth, tc, wait_for_stop, thread_release)) = go_rx.recv()
            {
                let limits = TimeManager::allocate(&search_board, &tc);
                let best_move = search_pool
                    .lock()
                    .unwrap()
                    .start_search(&mut search_board, depth, limits);

                while wait_for_stop && !thread_release.load(Ordering::Relaxed) {
                    thread::sleep(std::time::Duration::from_millis(1));
                }

                println!("bestmove {}", best_move);
                let _ = io::stdout().flush();
            }
        })
        .unwrap();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name Ruthless 0.2");
                println!("id author Ruthless Team");
                println!("option name Hash type spin default 32 min 1 max 1048576");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name RfpBase type spin default 80 min 10 max 200");
                println!("option name RfpImproving type spin default 20 min 0 max 100");
                println!("option name FutilityBase type spin default 70 min 10 max 200");
                println!("option name FutilityMargin type spin default 75 min 10 max 200");
                println!("option name NmpBase type spin default 3 min 1 max 6");
                println!("option name NmpEvalDiv type spin default 200 min 50 max 500");
                println!("option name CorrWeight type spin default 153 min 20 max 400");
                println!("option name CorrLimit type spin default 13 min 2 max 60");
                println!("option name LmrBase type spin default 0 min -100 max 200");
                println!("option name LmrDivisor type spin default 240 min 100 max 400");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name NumaBind type check default false");
                println!("option name Move Overhead type spin default 25 min 0 max 5000");
                println!("uciok");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            "setoption" => {
                halt(&stop, &mut pending_stop);
                handle_setoption(&tokens[1..], &mut pool.lock().unwrap());
            }
            "eval" => {
                println!("nnue eval: {} cp", crate::eval::evaluate(&board));
            }
            "isready" => {
                println!("readyok");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            "ucinewgame" => {
                halt(&stop, &mut pending_stop);
                pool.lock().unwrap().clear();
                board = Board::default();
            }
            "position" => {
                handle_position(&mut board, &tokens[1..]);
            }
            "go" => {
                halt(&stop, &mut pending_stop);
                stop = pool.lock().unwrap().new_search_stop();

                let (depth, tc, wait_for_stop) = parse_go(&tokens[1..]);
                let release = Arc::new(AtomicBool::new(false));
                if wait_for_stop {
                    pending_stop = Some(Arc::clone(&release));
                }

                let _ = go_tx.send((board.clone(), depth, tc, wait_for_stop, release));
            }
            "stop" => {
                halt(&stop, &mut pending_stop);
            }
            "bench" => {
                halt(&stop, &mut pending_stop);
                run_bench(&mut pool.lock().unwrap());
            }
            "quit" => {
                halt(&stop, &mut pending_stop);
                break;
            }
            _ => {}
        }
    }

    halt(&stop, &mut pending_stop);
}

fn halt(stop: &Arc<AtomicBool>, pending_stop: &mut Option<Arc<AtomicBool>>) {
    if let Some(flag) = pending_stop.take() {
        flag.store(true, Ordering::Relaxed);
    }
    stop.store(true, Ordering::Relaxed);
}

fn handle_setoption(tokens: &[&str], pool: &mut ThreadPool) {
    let mut name = String::new();
    let mut value = String::new();
    let mut is_name = false;
    let mut is_value = false;

    for &t in tokens {
        if t == "name" {
            is_name = true;
            is_value = false;
        } else if t == "value" {
            is_name = false;
            is_value = true;
        } else if is_name {
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(t);
        } else if is_value {
            if !value.is_empty() {
                value.push(' ');
            }
            value.push_str(t);
        }
    }

    match name.to_lowercase().as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                pool.set_hash(mb.clamp(1, 1048576));
            }
        }
        "threads" => {
            if let Ok(t) = value.parse::<usize>() {
                pool.set_threads(t.clamp(1, 256));
            }
        }
        "rfpbase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::RFP_BASE.store(v, Ordering::Relaxed); }
        }
        "rfpimproving" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::RFP_IMPROVING.store(v, Ordering::Relaxed); }
        }
        "futilitybase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::FUTILITY_BASE.store(v, Ordering::Relaxed); }
        }
        "futilitymargin" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::FUTILITY_MARGIN.store(v, Ordering::Relaxed); }
        }
        "nmpbase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::NMP_BASE.store(v, Ordering::Relaxed); }
        }
        "nmpevaldiv" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::NMP_EVAL_DIV.store(v, Ordering::Relaxed); }
        }
        "corrweight" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::CORR_WEIGHT.store(v, Ordering::Relaxed); }
        }
        "corrlimit" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::CORR_LIMIT.store(v, Ordering::Relaxed); }
        }
        "lmrbase" => {
            if let Ok(v) = value.parse::<i32>() {
                crate::search::LMR_BASE.store(v, Ordering::Relaxed);
                crate::search::rebuild_lmr_table();
            }
        }
        "lmrdivisor" => {
            if let Ok(v) = value.parse::<i32>() {
                crate::search::LMR_DIVISOR.store(v, Ordering::Relaxed);
                crate::search::rebuild_lmr_table();
            }
        }
        "syzygypath" => {
            crate::syzygy::init(&value);
        }
        "numabind" => {
            crate::numa::set_bind_enabled(value.eq_ignore_ascii_case("true"));
        }
        "move overhead" | "moveoverhead" => {
            if let Ok(v) = value.parse::<u64>() {
                crate::time::set_move_overhead(v.clamp(0, 5000));
            }
        }
        _ => {}
    }
}

fn handle_position(board: &mut Board, tokens: &[&str]) {
    if tokens.is_empty() {
        return;
    }

    let mut move_start = 0;
    if tokens[0] == "startpos" {
        *board = Board::default();
        move_start = 1;
    } else if tokens[0] == "fen" {
        let mut fen_parts = Vec::new();
        for &t in &tokens[1..] {
            if t == "moves" {
                break;
            }
            fen_parts.push(t);
        }
        let fen = fen_parts.join(" ");
        if let Ok(b) = Board::from_fen(&fen) {
            *board = b;
        }
        move_start = 1 + fen_parts.len();
    }

    if move_start < tokens.len() && tokens[move_start] == "moves" {
        for &m_str in &tokens[move_start + 1..] {
            let moves = generate_legal_moves(board);
            let m_lower = m_str.to_ascii_lowercase();
            for &m in moves.as_slice() {
                if m.to_string() == m_lower {
                    board.make_move(m);
                    board.reset_accumulator_stack();
                    break;
                }
            }
        }
    }
}

fn parse_go(tokens: &[&str]) -> (u8, TimeControl, bool) {
    let mut depth: u8 = 64;
    let mut tc = TimeControl::default();
    let mut wait_for_stop = false;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "infinite" | "ponder" => {
                wait_for_stop = true;
            }
            "depth" => {
                if let Some(val) = tokens.get(i + 1).and_then(|s| s.parse::<u8>().ok()) {
                    // Beyond MAX_PLY there is nothing to search and depth + 1 in
                    // the in-check extension would overflow.
                    depth = val.min(crate::search::MAX_PLY as u8);
                }
                i += 1;
            }
            "movetime" => {
                if let Some(val) = tokens.get(i + 1).and_then(|s| s.parse().ok()) {
                    tc.movetime = Some(val);
                }
                i += 1;
            }
            "wtime" => {
                tc.wtime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "btime" => {
                tc.btime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "winc" => {
                tc.winc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 1;
            }
            "binc" => {
                tc.binc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 1;
            }
            "movestogo" => {
                tc.movestogo = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    (depth, tc, wait_for_stop)
}

pub fn run_bench_cli() {
    let mut pool = ThreadPool::new(32, 1);
    run_bench(&mut pool);
}

fn run_bench(pool: &mut ThreadPool) {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/8/1k1NK3/r7/2R2P1P/3n2P1/8/8 b - - 0 59",
        "r1r3k1/1bqnbp1N/ppn1p1p1/4P1B1/8/2N5/PPB1QPPP/R3R1K1 w - - 3 9",
        "2kr1b1r/pp1qpp2/1np4p/3n1Pp1/3PN2B/2N2Q2/1PP3PP/2KRR3 w - g6 0 12",
        "6k1/8/pB2p1p1/3p3p/7q/PB6/Q7/2R3K1 w - - 0 37",
        "8/8/1p2k2P/1P6/P3B3/4B1K1/1b3P2/4n3 w - - 7 77",
        "7r/p1r1p2p/2k5/1p2n1N1/1Pp5/2R1P1P1/P4P1P/3R1K2 w - - 2 31",
        "5rk1/1p2r1p1/p4np1/P2p4/2pNbP2/2P1PB2/6PP/3RR1K1 w - - 2 23",
        "8/3k4/3b3p/R1p1p2P/p1Pr4/P2P1r2/2K1R3/4B3 b - - 4 75",
    ];

    let mut total_nodes = 0;
    let start = std::time::Instant::now();

    for fen in fens {
        if let Ok(mut board) = Board::from_fen(fen) {
            pool.new_search_stop();
            pool.start_search(
                &mut board,
                12,
                TimeLimits {
                    soft_limit: None,
                    hard_limit: None,
                    fixed: false,
                },
            );
            total_nodes += pool.shared_nodes.load(Ordering::Relaxed)
                + (pool.main_searcher.nodes & 1023);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let nps = (total_nodes as f64 / elapsed) as u64;

    println!();
    println!("Bench: {total_nodes} nodes {nps} nps");
}
