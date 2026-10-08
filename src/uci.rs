use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::Searcher;
use crate::tt::TranspositionTable;
use crate::types::Color;
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
    let mut movetime: Option<u128> = None;
    let mut wtime: Option<u128> = None;
    let mut btime: Option<u128> = None;
    let mut winc: u128 = 0;
    let mut binc: u128 = 0;
    let mut movestogo: Option<u128> = None;

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
                    movetime = Some(val);
                }
                i += 1;
            }
            "wtime" => {
                wtime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "btime" => {
                btime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "winc" => {
                winc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 1;
            }
            "binc" => {
                binc = tokens.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 1;
            }
            "movestogo" => {
                movestogo = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let overhead_ms = 10;

    let (soft_time, hard_time) = if let Some(mt) = movetime {
        let t = mt.saturating_sub(overhead_ms).max(5);
        (Some(t), Some(t))
    } else {
        let (my_time, my_inc) = if board.side_to_move == Color::White {
            (wtime, winc)
        } else {
            (btime, binc)
        };

        if let Some(time) = my_time {
            let usable_time = time.saturating_sub(overhead_ms);

            if let Some(moves) = movestogo {
                let moves = (moves.clamp(1, 50) as f64).min(20.0);
                let base = (usable_time as f64 / moves) + 0.8 * my_inc as f64;
                let soft = (base as u128).min(usable_time * 4 / 10).max(5);
                let hard = ((base * 2.5) as u128).min(usable_time * 80 / 100).max(soft);
                (Some(soft), Some(hard))
            } else {
                let base_time = usable_time / 20 + (my_inc * 4) / 5;
                let min_time = (my_inc * 8 / 10).max(10);
                let max_soft = usable_time * 35 / 100;
                let soft = base_time.max(min_time).min(max_soft).max(5);
                let hard = ((base_time * 25) / 10).min(usable_time * 80 / 100).max(soft);
                (Some(soft), Some(hard))
            }
        } else {
            (None, None)
        }
    };

    main_searcher.soft_stop_votes.store(0, Ordering::Relaxed);
    main_searcher.shared_nodes.store(0, Ordering::Relaxed);
    main_searcher.stop.store(false, Ordering::Relaxed);

    let stop_signal = Arc::clone(&main_searcher.stop);

    let best_move = if helpers.is_empty() {
        let m = main_searcher.search(board, depth, soft_time, hard_time);
        stop_signal.store(true, Ordering::Relaxed);
        m
    } else {
        std::thread::scope(|s| {
            for helper in helpers.iter_mut() {
                let mut helper_board = board.clone();
                s.spawn(move || {
                    helper.search_helper(&mut helper_board, depth, soft_time);
                });
            }

            let m = main_searcher.search(board, depth, soft_time, hard_time);
            stop_signal.store(true, Ordering::Relaxed);
            m
        })
    };

    println!("bestmove {}", best_move);
}
