mod attacks;
mod bitboard;
mod board;
mod movegen;
mod types;

use board::Board;
use movegen::generate_legal_moves;
use std::time::Instant;

pub fn perft(board: &mut Board, depth: usize) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_legal_moves(board);
    if depth == 1 {
        return moves.count as u64;
    }

    let mut nodes = 0;
    for &m in moves.as_slice() {
        let undo = board.make_move(m);
        nodes += perft(board, depth - 1);
        board.undo_move(m, undo);
    }
    nodes
}

fn main() {
    let mut board = Board::default();
    println!("Ruthless Chess Engine");

    for depth in 1..=5 {
        let start = Instant::now();
        let nodes = perft(&mut board, depth);
        let elapsed = start.elapsed();
        let nps = if elapsed.as_secs_f64() > 0.0 {
            (nodes as f64 / elapsed.as_secs_f64()) as u64
        } else {
            0
        };
        println!(
            "Depth {}: {:>10} nodes | {:>8.2?} | {:>10} nps",
            depth, nodes, elapsed, nps
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perft_startpos() {
        let mut board = Board::default();
        assert_eq!(perft(&mut board, 1), 20);
        assert_eq!(perft(&mut board, 2), 400);
        assert_eq!(perft(&mut board, 3), 8902);
        assert_eq!(perft(&mut board, 4), 197281);
    }

    #[test]
    fn test_perft_kiwipete() {
        let mut board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&mut board, 1), 48);
        assert_eq!(perft(&mut board, 2), 2039);
        assert_eq!(perft(&mut board, 3), 97862);
    }
}