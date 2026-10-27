use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square};

/// Static exchange evaluation: does the exchange on `to` win at least
/// `threshold` for the side to move?
///
/// `threshold` is compared against the value of the victim minus everything the
/// exchange costs us, so `see(board, m, 0)` asks whether the capture is not
/// losing material.
pub fn see(board: &Board, m: Move, threshold: i32) -> bool {
    if m == Move::NULL {
        return 0 >= threshold;
    }

    let from = m.from();
    let to = m.to();
    let move_type = m.move_type();

    if move_type == MoveType::Castling {
        return 0 >= threshold;
    }

    // Invalid move for SEE if there is no piece on the origin square
    if board.piece_on[from] == Piece::None {
        return 0 >= threshold;
    }

    let mut gain = [0i32; 32];
    let mut d = 0;

    let target_pt = if move_type == MoveType::EnPassant {
        PieceType::Pawn
    } else if board.piece_on[to] == Piece::None {
        PieceType::None
    } else {
        board.piece_on[to].piece_type()
    };

    gain[0] = if target_pt != PieceType::None {
        PIECE_VALUES[target_pt as usize]
    } else {
        0
    };

    if move_type == MoveType::Promotion {
        gain[0] += PIECE_VALUES[m.promo_type() as usize] - PIECE_VALUES[PieceType::Pawn as usize];
    }

    let mut pt = if move_type == MoveType::Promotion {
        m.promo_type()
    } else {
        board.piece_on[from].piece_type()
    };

    // Early exits before the attacker chain is even looked up. Both follow from
    // how the resolution below behaves: it only ever subtracts from `gain[0]`,
    // so a victim that does not cover the threshold can never reach it, and if
    // the victim still covers our attacker plus the threshold the exchange is
    // already won even when that attacker is lost - recapturing is the best the
    // opponent can do, and every deeper exchange only makes that capture worth
    // less, which can only raise our result.
    let swap = gain[0] - threshold;
    if swap < 0 {
        return false;
    }
    if PIECE_VALUES[pt as usize] <= swap {
        return true;
    }

    let mut occ = board.occupied;
    occ.clear(from);
    if move_type == MoveType::EnPassant {
        let cap_sq = Square::from_coords(to.file(), from.rank());
        occ.clear(cap_sq);
    }

    let bishops = board.pieces[Piece::WhiteBishop]
        | board.pieces[Piece::BlackBishop]
        | board.pieces[Piece::WhiteQueen]
        | board.pieces[Piece::BlackQueen];
    let rooks = board.pieces[Piece::WhiteRook]
        | board.pieces[Piece::BlackRook]
        | board.pieces[Piece::WhiteQueen]
        | board.pieces[Piece::BlackQueen];

    let mut attackers = all_attackers(board, to, occ);
    let mut side = !board.side_to_move;

    loop {
        let (next_pt, next_sq) = least_valuable_attacker(board, attackers & occ, side);
        if next_pt == PieceType::None {
            break;
        }

        d += 1;
        gain[d] = PIECE_VALUES[pt as usize];
        pt = next_pt;
        occ.clear(next_sq);

        attackers |= bishop_attacks(to, occ) & bishops;
        attackers |= rook_attacks(to, occ) & rooks;

        let opp_attackers = attackers & occ & board.occupied_co[!side];
        if pt == PieceType::King && !opp_attackers.is_empty() {
            d -= 1;
            break;
        }

        side = !side;
    }

    while d > 0 {
        gain[d - 1] = gain[d - 1].saturating_sub(gain[d].max(0));
        d -= 1;
    }

    gain[0] >= threshold
}

fn all_attackers(board: &Board, sq: Square, occ: Bitboard) -> Bitboard {
    let p_attacks = (pawn_attacks(Color::Black, sq) & board.pieces[Piece::WhitePawn])
        | (pawn_attacks(Color::White, sq) & board.pieces[Piece::BlackPawn]);
    let n_attacks = knight_attacks(sq) & (board.pieces[Piece::WhiteKnight] | board.pieces[Piece::BlackKnight]);
    let k_attacks = king_attacks(sq) & (board.pieces[Piece::WhiteKing] | board.pieces[Piece::BlackKing]);
    let b_attacks = bishop_attacks(sq, occ)
        & (board.pieces[Piece::WhiteBishop]
            | board.pieces[Piece::BlackBishop]
            | board.pieces[Piece::WhiteQueen]
            | board.pieces[Piece::BlackQueen]);
    let r_attacks = rook_attacks(sq, occ)
        & (board.pieces[Piece::WhiteRook]
            | board.pieces[Piece::BlackRook]
            | board.pieces[Piece::WhiteQueen]
            | board.pieces[Piece::BlackQueen]);

    p_attacks | n_attacks | k_attacks | b_attacks | r_attacks
}

fn least_valuable_attacker(board: &Board, attackers: Bitboard, side: Color) -> (PieceType, Square) {
    let side_attackers = attackers & board.occupied_co[side];
    if side_attackers.is_empty() {
        return (PieceType::None, Square::None);
    }

    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        let piece = Piece::new(side, pt);
        let subset = side_attackers & board.pieces[piece];
        if !subset.is_empty() {
            return (pt, subset.lsb());
        }
    }

    (PieceType::None, Square::None)
}

