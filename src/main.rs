mod bitboard;
mod board;
mod types;

use board::{Board, STARTING_FEN};
use types::Color;

fn main() {
    let board = Board::default();
    println!("Board initialized from default FEN: {}", STARTING_FEN);
    println!("White Pawns: {:064b}", board.pieces[types::Piece::WhitePawn].0);
    println!("Side to move: {:?}", board.side_to_move);
    println!("White King at: {:?}", board.king_square(Color::White));
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::Square;

    #[test]
    fn test_starting_fen_parsing() {
        let board = Board::from_fen(STARTING_FEN).expect("Valid starting FEN");
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.king_square(Color::White), Square::E1);
        assert_eq!(board.king_square(Color::Black), Square::E8);
        assert_eq!(board.occupied.count(), 32);
    }
}