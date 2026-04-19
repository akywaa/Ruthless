use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::Searcher;
use crate::types::Color;
use std::io::{self, BufRead};

pub fn uci_loop() {
    let mut board = Board::default();
    let mut searcher = Searcher::new(32);
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
                println!("id name Ruthless 0.1");
                println!("id author Ruthless Team");
                println!("uciok");
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
                handle_go(&mut board, &mut searcher, &tokens[1..]);
            }
            "quit" => break,
            _ => {}
        }
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

fn handle_go(board: &mut Board, searcher: &mut Searcher, tokens: &[&str]) {
    let mut depth: u8 = 64;
    let mut movetime: Option<u128> = None;
    let mut wtime: Option<u128> = None;
    let mut btime: Option<u128> = None;
    let mut winc: u128 = 0;
    let mut binc: u128 = 0;

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
            _ => {}
        }
        i += 1;
    }

    let time_budget = if let Some(mt) = movetime {
        Some(mt)
    } else {
        let (my_time, my_inc) = if board.side_to_move == Color::White {
            (wtime, winc)
        } else {
            (btime, binc)
        };

        my_time.map(|t| (t / 25 + my_inc / 2).max(10))
    };

    let best_move = searcher.search(board, depth, time_budget);
    println!("bestmove {}", best_move);
}
