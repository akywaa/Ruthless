use crate::board::Board;
use crate::nnue;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// this pair is what keeps the engine playing well. the raw net output is way
// hotter than the piece values and the search margins assume, so we squeeze it
// back down here. pulled it out once and strength fell off a cliff, so leave it
const NORM_NUM: i32 = 9;
const NORM_DEN: i32 = 20;

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let eval = nnue::evaluate(board);
    let material = board.non_pawn_material();
    let scale = (700 + material / 20).min(1024);

    if nnue::uses_heimdall() {
        // A Heimdall network is trained against its own 282 scale and already
        // outputs centipawns, so squeezing it through the legacy normalisation
        // would make it play at 45% of the evaluation it was trained for.
        eval * scale / 1024
    } else {
        eval * scale * NORM_NUM / (1024 * NORM_DEN)
    }
}