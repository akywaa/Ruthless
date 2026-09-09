use crate::board::Board;
use crate::nnue;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

const NORM_NUM: i32 = 9;
const NORM_DEN: i32 = 20;

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let eval = nnue::evaluate(board);
    let material = board.non_pawn_material();
    let scale = (700 + material / 20).min(1024);
    eval * scale * NORM_NUM / (1024 * NORM_DEN)
}
