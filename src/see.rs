use crate::attacks::{king_attacks, knight_attacks, pawn_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square};

// static exchange eval >= threshold
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

    // invalid move for SEE if there is no piece on the origin square
    if board.piece_on[from] == Piece::None {
        return 0 >= threshold;
    }

    let target_pt = if move_type == MoveType::EnPassant {
        PieceType::Pawn
    } else if board.piece_on[to] == Piece::None {
        PieceType::None
    } else {
        board.piece_on[to].piece_type()
    };

    let mut swap = if target_pt != PieceType::None { PIECE_VALUES[target_pt as usize] } else { 0 };
    if move_type == MoveType::Promotion {
        swap += PIECE_VALUES[m.promo_type() as usize] - PIECE_VALUES[PieceType::Pawn as usize];
    }

    swap -= threshold;
    if swap < 0 {
        return false;
    }

    let pt = if move_type == MoveType::Promotion {
        m.promo_type()
    } else {
        board.piece_on[from].piece_type()
    };

    swap = PIECE_VALUES[pt as usize] - swap;
    if swap <= 0 {
        return true;
    }

    let mut occ = board.occupied;
    occ.clear(from);
    if move_type == MoveType::EnPassant {
        let cap_sq = Square::from_coords(to.file(), from.rank());
        occ.clear(cap_sq);
    }

    let bishops = board.pieces[Piece::WhiteBishop] | board.pieces[Piece::BlackBishop] | board.pieces[Piece::WhiteQueen] | board.pieces[Piece::BlackQueen];
    let rooks = board.pieces[Piece::WhiteRook] | board.pieces[Piece::BlackRook] | board.pieces[Piece::WhiteQueen] | board.pieces[Piece::BlackQueen];

    let tables = crate::attacks::slider_tables();
    let mut attackers = all_attackers(board, to, occ, tables);
    let mut side = !board.side_to_move;
    let mut res = 1;

    loop {
        let (next_pt, next_sq) = least_valuable_attacker(board, attackers & occ, side);
        if next_pt == PieceType::None {
            break;
        }

        occ.clear(next_sq);
        if matches!(next_pt, PieceType::Pawn | PieceType::Bishop | PieceType::Queen) {
            attackers |= tables.bishop_attacks(to, occ) & bishops;
        }
        if matches!(next_pt, PieceType::Rook | PieceType::Queen) {
            attackers |= tables.rook_attacks(to, occ) & rooks;
        }

        if next_pt == PieceType::King && !(attackers & occ & board.occupied_co[!side]).is_empty() {
            break;
        }

        res ^= 1;
        swap = PIECE_VALUES[next_pt as usize] - swap;
        if swap < res {
            break;
        }

        side = !side;
    }

    res != 0
}

fn all_attackers(board: &Board, sq: Square, occ: Bitboard, tables: &crate::attacks::SliderTables) -> Bitboard {
    let p_attacks = (pawn_attacks(Color::Black, sq) & board.pieces[Piece::WhitePawn]) | (pawn_attacks(Color::White, sq) & board.pieces[Piece::BlackPawn]);
    let n_attacks = knight_attacks(sq) & (board.pieces[Piece::WhiteKnight] | board.pieces[Piece::BlackKnight]);
    let k_attacks = king_attacks(sq) & (board.pieces[Piece::WhiteKing] | board.pieces[Piece::BlackKing]);
    let b_attacks = tables.bishop_attacks(sq, occ)
        & (board.pieces[Piece::WhiteBishop] | board.pieces[Piece::BlackBishop] | board.pieces[Piece::WhiteQueen] | board.pieces[Piece::BlackQueen]);
    let r_attacks = tables.rook_attacks(sq, occ)
        & (board.pieces[Piece::WhiteRook] | board.pieces[Piece::BlackRook] | board.pieces[Piece::WhiteQueen] | board.pieces[Piece::BlackQueen]);

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
