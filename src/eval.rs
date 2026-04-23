use crate::board::Board;
use crate::nnue;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    nnue::evaluate(&board.accumulator, board.side_to_move)
}
