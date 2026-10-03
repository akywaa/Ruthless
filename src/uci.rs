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
                println!("uciok");
            }
            "setoption" => {
                handle_setoption(&tokens[1..], &mut tt_size_mb, &mut num_threads, &mut tt);
                searcher.tt = Arc::clone(&tt);
                searcher.num_threads = num_threads;
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                searcher.clear();
                board = Board::default();
            }
            "position" => {
                handle_position(&mut board, &tokens[1..]);
            }
            "go" => {
                handle_go(&mut board, &mut searcher, num_threads, &tokens[1..]);
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
            for &m in moves.as_slice() {
                if m.to_string() == m_str {
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
    threads: usize,
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

    let overhead_ms = 20;

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
                // Allocate more time per move similar to top engines
                let moves = (moves.clamp(1, 50) as f64).min(25.0);
                let base = (usable_time as f64 / moves) + 0.8 * my_inc as f64;
                let soft = (base as u128).clamp(10, usable_time * 4 / 10);
                let hard = ((base * 3.5) as u128).min(usable_time * 80 / 100).max(soft);
                (Some(soft), Some(hard))
            } else {
                let base_time = usable_time / 25 + (my_inc * 3) / 4;
                let soft = base_time.clamp(10, usable_time / 3);
                let hard = (base_time * 3).min(usable_time * 80 / 100).max(soft);
                (Some(soft), Some(hard))
            }
        } else {
            (None, None)
        }
    };

    main_searcher.num_threads = threads;
    main_searcher.soft_stop_votes.store(0, Ordering::Relaxed);
    main_searcher.shared_nodes.store(0, Ordering::Relaxed);
    main_searcher.stop.store(false, Ordering::Relaxed);

    let stop_signal = Arc::clone(&main_searcher.stop);
    let tt = Arc::clone(&main_searcher.tt);
    let shared_nodes = Arc::clone(&main_searcher.shared_nodes);
    let soft_stop_votes = Arc::clone(&main_searcher.soft_stop_votes);

    let best_move = if threads > 1 {
        std::thread::scope(|s| {
            for id in 1..threads {
                let mut helper_searcher = Searcher::new(
                    Arc::clone(&tt),
                    Arc::clone(&stop_signal),
                    id,
                    threads,
                    Arc::clone(&shared_nodes),
                    Arc::clone(&soft_stop_votes),
                );
                let mut helper_board = board.clone();
                s.spawn(move || {
                    helper_searcher.search_helper(&mut helper_board, depth, soft_time);
                });
            }

            let m = main_searcher.search(board, depth, soft_time, hard_time);
            stop_signal.store(true, Ordering::Relaxed);
            m
        })
    } else {
        let m = main_searcher.search(board, depth, soft_time, hard_time);
        stop_signal.store(true, Ordering::Relaxed);
        m
    };

    println!("bestmove {}", best_move);
}
