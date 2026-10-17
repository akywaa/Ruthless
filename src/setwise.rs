use crate::attacks::{bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType, Square};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Expand every destination bit into a move, walking the bitboard bit by bit.
#[inline(always)]
pub fn extract_moves(
    from: Square,
    attacks: Bitboard,
    list: &mut MoveList,
    pt: PieceType,
    move_type: MoveType,
) {
    let mut attacks = attacks;
    while !attacks.is_empty() {
        list.push(Move::new(from, attacks.pop_lsb(), pt, move_type));
    }
}

/// Slider lookup index. Prefers BMI2 `pext` (compressed-occupancy index) and
/// falls back to the classic magic multiply-shift when BMI2 is unavailable.
/// The attacker table is built with the matching layout, see `attacks.rs`.
#[inline(always)]
pub fn pext_slider_idx(occ: u64, mask: u64, offset: usize, shift: u8, magic: u64) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        let _ = (shift, magic);
        offset + _pext_u64(occ, mask) as usize
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        offset + (((occ & mask).wrapping_mul(magic)) >> shift) as usize
    }
}

/// Compressed occupancy index used to build the PEXT slider table.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn compressed_index(occ: u64, mask: u64) -> usize {
    unsafe { _pext_u64(occ, mask) as usize }
}

#[inline(always)]
pub fn generate_knight_moves(board: &Board, us: Color, target: Bitboard, list: &mut MoveList) {
    let mut knights = board.pieces[Piece::new(us, PieceType::Knight)].0;

    // Setwise AVX-512 parallel move generation for knights: gather up to eight
    // attack bitboards at once with one vpgatherqq.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq"
    ))]
    unsafe {
        let table_ptr = crate::attacks::attacks().knight_attacks.as_ptr() as *const i64;
        while knights != 0 {
            let mut from_sqs = [0i32; 8];
            let mut k = 0;
            while knights != 0 && k < 8 {
                from_sqs[k] = knights.trailing_zeros() as i32;
                knights &= knights - 1;
                k += 1;
            }

            let v_from = _mm256_loadu_si256(from_sqs.as_ptr() as *const __m256i);
            let v_attacks = _mm512_i32gather_epi64::<8>(v_from, table_ptr);
            let v_target = _mm512_set1_epi64(target.0 as i64);
            let v_valid = _mm512_and_si512(v_attacks, v_target);

            let mut valid_arr = [0u64; 8];
            _mm512_storeu_si512(valid_arr.as_mut_ptr() as *mut _, v_valid);

            for i in 0..k {
                extract_moves(
                    Square::new(from_sqs[i] as u8),
                    Bitboard(valid_arr[i]),
                    list,
                    PieceType::None,
                    MoveType::Normal,
                );
            }
        }
        return;
    }

    while knights != 0 {
        let from = Square::new(knights.trailing_zeros() as u8);
        knights &= knights - 1;
        let att = crate::attacks::knight_attacks(from) & target;
        extract_moves(from, att, list, PieceType::None, MoveType::Normal);
    }
}

#[inline(always)]
pub fn generate_bishop_moves(
    board: &Board,
    us: Color,
    target: Bitboard,
    occ: Bitboard,
    list: &mut MoveList,
) {
    let mut bishops = board.pieces[Piece::new(us, PieceType::Bishop)].0;
    while bishops != 0 {
        let from = Square::new(bishops.trailing_zeros() as u8);
        bishops &= bishops - 1;
        let att = bishop_attacks(from, occ) & target;
        extract_moves(from, att, list, PieceType::None, MoveType::Normal);
    }
}

#[inline(always)]
pub fn generate_rook_moves(
    board: &Board,
    us: Color,
    target: Bitboard,
    occ: Bitboard,
    list: &mut MoveList,
) {
    let mut rooks = board.pieces[Piece::new(us, PieceType::Rook)].0;
    while rooks != 0 {
        let from = Square::new(rooks.trailing_zeros() as u8);
        rooks &= rooks - 1;
        let att = rook_attacks(from, occ) & target;
        extract_moves(from, att, list, PieceType::None, MoveType::Normal);
    }
}

#[inline(always)]
pub fn generate_queen_moves(
    board: &Board,
    us: Color,
    target: Bitboard,
    occ: Bitboard,
    list: &mut MoveList,
) {
    let mut queens = board.pieces[Piece::new(us, PieceType::Queen)].0;
    while queens != 0 {
        let from = Square::new(queens.trailing_zeros() as u8);
        queens &= queens - 1;
        let att = (bishop_attacks(from, occ) | rook_attacks(from, occ)) & target;
        extract_moves(from, att, list, PieceType::None, MoveType::Normal);
    }
}
