use crate::attacks::{bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType, Square};
use std::sync::atomic::{AtomicBool, Ordering};

// Slider indexing style, picked once at startup. PEXT needs BMI2, but on AMD
// Zen 1/2 (family 0x17) BMI2 is emulated in microcode and is much slower than
// a magic multiply-shift, so those CPUs keep using the magic tables.
static USE_PEXT: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn use_pext() -> bool {
    USE_PEXT.load(Ordering::Relaxed)
}

/// Chooses between the PEXT and the magic slider tables for this machine.
pub fn detect_slider_style() {
    #[cfg(target_arch = "x86_64")]
    {
        let fast = std::arch::is_x86_feature_detected!("bmi2") && !is_slow_pext_amd();
        USE_PEXT.store(fast, Ordering::Relaxed);
    }
}

#[cfg(target_arch = "x86_64")]
fn is_slow_pext_amd() -> bool {
    // CPUID leaves are available on every x86_64 CPU.
    let leaf1 = std::arch::x86_64::__cpuid(1);
    let is_amd = leaf1.ebx == 0x6874_7541
        && leaf1.edx == 0x6972_6D65
        && leaf1.ecx == 0x444D_4163;
    if !is_amd {
        return false;
    }

    let base_family = (leaf1.eax >> 8) & 0xF;
    let ext_family = (leaf1.eax >> 20) & 0xFF;
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };

    // 0x17 covers Zen 1, Zen+ and Zen 2.
    family == 0x17
}

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

/// Compressed-occupancy index for the PEXT slider table. The table is built in
/// this exact order, so the result is the offset-relative slot.
#[inline(always)]
pub fn pext_index(occ: u64, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // Safety: only called when runtime detection found BMI2 and a CPU where
        // PEXT is not microcoded.
        unsafe { pext_u64(occ, mask) as usize }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        unreachable!("PEXT slider tables are x86_64 only")
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn pext_u64(occ: u64, mask: u64) -> u64 {
    std::arch::x86_64::_pext_u64(occ, mask)
}

/// Classic magic multiply-shift index for the magic slider table.
#[inline(always)]
pub fn magic_index(occ: u64, mask: u64, shift: u8, magic: u64) -> usize {
    (((occ & mask).wrapping_mul(magic)) >> shift) as usize
}

#[inline(always)]
pub fn generate_knight_moves(board: &Board, us: Color, target: Bitboard, list: &mut MoveList) {
    let mut knights = board.pieces[Piece::new(us, PieceType::Knight)];
    while !knights.is_empty() {
        let from = knights.pop_lsb();
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
