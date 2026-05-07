use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square};

pub fn see(board: &Board, m: Move, threshold: i32) -> bool {
    see_value(board, m) >= threshold
}

pub fn see_value(board: &Board, m: Move) -> i32 {
    let from = m.from();
    let to = m.to();
    let move_type = m.move_type();

    if move_type == MoveType::Castling {
        return 0;
    }

    let mut gain = [0i32; 32];
    let mut d = 0;

    let target_pt = if move_type == MoveType::EnPassant {
        PieceType::Pawn
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

        if next_pt == PieceType::King {
            let opp_attackers = attackers & occ & board.occupied_co[!side];
            if !opp_attackers.is_empty() {
                break;
            }
        }

        d += 1;
        gain[d] = PIECE_VALUES[pt as usize];
        pt = next_pt;
        occ.clear(next_sq);

        if pt == PieceType::Pawn || pt == PieceType::Bishop || pt == PieceType::Queen {
            attackers |= bishop_attacks(to, occ) & bishops;
        }
        if pt == PieceType::Rook || pt == PieceType::Queen {
            attackers |= rook_attacks(to, occ) & rooks;
        }

        side = !side;
    }

    while d > 0 {
        gain[d - 1] = gain[d - 1].saturating_sub(gain[d].max(0));
        d -= 1;
    }

    gain[0]
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
