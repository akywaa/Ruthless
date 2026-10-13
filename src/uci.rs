use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::Searcher;
use crate::time::{TimeControl, TimeManager};
use crate::tt::TranspositionTable;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub fn uci_loop() {
    let mut board = Board::default();
    let mut tt_size_mb = 32;
    let mut num_threads = 1;
    let mut tt = Arc::new(TranspositionTable::new(tt_size_mb));
    let stop_signal = Arc::new(AtomicBool::new(false));
    let shared_nodes = Arc::new(AtomicU64::new(0));
    let soft_stop_votes = Arc::new(AtomicUsize::new(0));

    let mut searcher = Searcher::new(
        Arc::clone(&tt),
        Arc::clone(&stop_signal),
        0,
        num_threads,
        Arc::clone(&shared_nodes),
        Arc::clone(&soft_stop_votes),
    );
    let mut helpers: Vec<Searcher> = Vec::new();
    let stdin = io::stdin();

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
                println!("uciok");
            }
            "setoption" => {
                handle_setoption(&tokens[1..], &mut tt_size_mb, &mut num_threads, &mut tt);
                searcher.tt = Arc::clone(&tt);
                searcher.num_threads = num_threads;

                // Keep helper searchers persistent across searches
                helpers.clear();
                for id in 1..num_threads {
                    helpers.push(Searcher::new(
                        Arc::clone(&tt),
                        Arc::clone(&stop_signal),
                        id,
                        num_threads,
                        Arc::clone(&shared_nodes),
                        Arc::clone(&soft_stop_votes),
                    ));
                }
            }
            "eval" => {
                println!("nnue eval: {} cp", crate::eval::evaluate(&board));
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                searcher.clear();
                for helper in &mut helpers {
                    helper.clear();
                }
                board = Board::default();
            }
            "position" => {
                handle_position(&mut board, &tokens[1..]);
            }
            "go" => {
                handle_go(&mut board, &mut searcher, &mut helpers, &tokens[1..]);
            }
            "bench" => {
                run_bench(&mut searcher);
            }
            "quit" => break,
            _ => {}
        }
    }
}

fn handle_setoption(
    tokens: &[&str],
    tt_size_mb: &mut usize,
    num_threads: &mut usize,
    tt: &mut Arc<TranspositionTable>,
) {
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
            value.push_str(t);
        }
    }

    match name.to_lowercase().as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                *tt_size_mb = mb.clamp(1, 1048576);
                *tt = Arc::new(TranspositionTable::new(*tt_size_mb));
            }
        }
        "threads" => {
            if let Ok(t) = value.parse::<usize>() {
                *num_threads = t.clamp(1, 256);
            }
        }
        "rfpbase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::RFP_BASE.store(v, std::sync::atomic::Ordering::Relaxed); }
        }
        "rfpimproving" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::RFP_IMPROVING.store(v, std::sync::atomic::Ordering::Relaxed); }
        }
        "futilitybase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::FUTILITY_BASE.store(v, std::sync::atomic::Ordering::Relaxed); }
        }
        "futilitymargin" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::FUTILITY_MARGIN.store(v, std::sync::atomic::Ordering::Relaxed); }
        }
        "nmpbase" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::NMP_BASE.store(v, std::sync::atomic::Ordering::Relaxed); }
        }
        "nmpevaldiv" => {
            if let Ok(v) = value.parse::<i32>() { crate::search::NMP_EVAL_DIV.store(v, std::sync::atomic::Ordering::Relaxed); }
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

fn handle_go(
    board: &mut Board,
    main_searcher: &mut Searcher,
    helpers: &mut Vec<Searcher>,
    tokens: &[&str],
) {
    let mut depth: u8 = 64;
    let mut tc = TimeControl::default();

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if let Some(val) = tokens.get(i + 1).and_then(|s| s.parse().ok()) {
                    depth = val;
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

    let limits = TimeManager::allocate(board, &tc);

    main_searcher.soft_stop_votes.store(0, Ordering::Relaxed);
    main_searcher.shared_nodes.store(0, Ordering::Relaxed);
    main_searcher.stop.store(false, Ordering::Relaxed);

    let stop_signal = Arc::clone(&main_searcher.stop);

    let best_move = if helpers.is_empty() {
        let m = main_searcher.search(board, depth, limits.soft_limit, limits.hard_limit);
        stop_signal.store(true, Ordering::Relaxed);
        m
    } else {
        std::thread::scope(|s| {
            for helper in helpers.iter_mut() {
                let mut helper_board = board.clone();
                let soft = limits.soft_limit;
                s.spawn(move || {
                    helper.search_helper(&mut helper_board, depth, soft);
                });
            }

            let m = main_searcher.search(board, depth, limits.soft_limit, limits.hard_limit);
            stop_signal.store(true, Ordering::Relaxed);
            m
        })
    };

    println!("bestmove {}", best_move);
}

pub fn run_bench_cli() {
    let tt = Arc::new(TranspositionTable::new(32));
    let stop_signal = Arc::new(AtomicBool::new(false));
    let shared_nodes = Arc::new(AtomicU64::new(0));
    let soft_stop_votes = Arc::new(AtomicUsize::new(0));

    let mut searcher = Searcher::new(
        Arc::clone(&tt),
        Arc::clone(&stop_signal),
        0,
        1,
        Arc::clone(&shared_nodes),
        Arc::clone(&soft_stop_votes),
    );

    run_bench(&mut searcher);
}

fn run_bench(searcher: &mut Searcher) {
    // A diverse set of positions to evaluate engine performance and train the PGO
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
            searcher.shared_nodes.store(0, Ordering::Relaxed);
            searcher.search(&mut board, 12, None, None);
            total_nodes += searcher.shared_nodes.load(Ordering::Relaxed) + (searcher.nodes & 1023);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let nps = (total_nodes as f64 / elapsed) as u64;

    println!();
    println!("Bench: {total_nodes} nodes {nps} nps");
}
