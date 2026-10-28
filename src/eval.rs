use crate::board::Board;
use crate::nnue;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// this pair is what keeps the engine playing well. the raw net output is way
// hotter than the piece values and the search margins assume, so we squeeze it
// back down here. pulled it out once and strength fell off a cliff, so leave it
const NORM_NUM: i32 = 9;
const NORM_DEN: i32 = 20;

/// Progress bonus for the side that has to mate a bare king.
///
/// The net is flat in these positions: every king and rook move of a won
/// K+R vs K comes back with the same score, so the search has no reason to
/// drive the enemy king anywhere and a won endgame is shuffled into a
/// fifty-move draw. Rewarding the distance of the bare king from the centre
/// and the approach of our own king supplies the missing gradient. It is
/// symmetric, so the defending side is steered towards the centre as well.
#[inline(always)]
fn mop_up(board: &Board) -> i32 {
    let us = board.side_to_move;
    let them = !us;

    let our_count = board.occupied_co[us as usize].count();
    let their_count = board.occupied_co[them as usize].count();

    // Only when the weaker side has nothing but a king and we own material.
    let (strong, weak, sign) = if their_count == 1 && our_count > 1 {
        (us, them, 1)
    } else if our_count == 1 && their_count > 1 {
        (them, us, -1)
    } else {
        return 0;
    };

    let weak_king = board.king_square(weak);
    let strong_king = board.king_square(strong);
    if !weak_king.is_valid() || !strong_king.is_valid() {
        return 0;
    }

    // 0 in a corner, 6 in the four centre squares.
    let edge = weak_king.file().min(7 - weak_king.file()) as i32
        + weak_king.rank().min(7 - weak_king.rank()) as i32;
    // Chebyshev distance between the kings: 1 when adjacent, 7 furthest.
    let king_dist = (strong_king.file() as i32 - weak_king.file() as i32)
        .abs()
        .max((strong_king.rank() as i32 - weak_king.rank() as i32).abs());

    sign * (20 * (6 - edge) + 6 * (7 - king_dist))
}

#[inline(always)]
pub fn evaluate(board: &Board) -> i32 {
    let eval = nnue::evaluate(board);
    let material = board.non_pawn_material();
    let scale = (700 + material / 20).min(1024);
    eval * scale * NORM_NUM / (1024 * NORM_DEN) + mop_up(board)
}