use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType, Square};

pub fn generate_legal_moves(board: &mut Board) -> MoveList {
    let mut list = MoveList::new();
    let mut pseudo_list = MoveList::new();
    generate_pseudo_moves(board, &mut pseudo_list);

    for &m in pseudo_list.as_slice() {
        let us = board.side_to_move;
        let undo = board.make_move(m);
        let ksq = board.king_square(us);
        if !board.is_square_attacked(ksq, board.side_to_move) {
            list.push(m);
        }
        board.undo_move(m, undo);
    }

    list
}

fn generate_pseudo_moves(board: &Board, list: &mut MoveList) {
    let us = board.side_to_move;
    let them = !us;
    let our_occ = board.occupied_co[us];
    let their_occ = board.occupied_co[them];
    let all_occ = board.occupied;

    generate_pawn_moves(board, us, their_occ, all_occ, list);

    let mut knights = board.pieces[Piece::new(us, PieceType::Knight)];
    while !knights.is_empty() {
        let from = knights.pop_lsb();
        let mut attacks = knight_attacks(from) & !our_occ;
        while !attacks.is_empty() {
            let to = attacks.pop_lsb();
            list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
        }
    }

    let mut bishops = board.pieces[Piece::new(us, PieceType::Bishop)];
    while !bishops.is_empty() {
        let from = bishops.pop_lsb();
        let mut attacks = bishop_attacks(from, all_occ) & !our_occ;
        while !attacks.is_empty() {
            let to = attacks.pop_lsb();
            list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
        }
    }

    let mut rooks = board.pieces[Piece::new(us, PieceType::Rook)];
    while !rooks.is_empty() {
        let from = rooks.pop_lsb();
        let mut attacks = rook_attacks(from, all_occ) & !our_occ;
        while !attacks.is_empty() {
            let to = attacks.pop_lsb();
            list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
        }
    }

    let mut queens = board.pieces[Piece::new(us, PieceType::Queen)];
    while !queens.is_empty() {
        let from = queens.pop_lsb();
        let mut attacks = (bishop_attacks(from, all_occ) | rook_attacks(from, all_occ)) & !our_occ;
        while !attacks.is_empty() {
            let to = attacks.pop_lsb();
            list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
        }
    }

    let king_sq = board.king_square(us);
    let mut king_moves = king_attacks(king_sq) & !our_occ;
    while !king_moves.is_empty() {
        let to = king_moves.pop_lsb();
        list.push(Move::new(king_sq, to, PieceType::None, MoveType::Normal));
    }

    generate_castling_moves(board, us, all_occ, list);
}

fn generate_pawn_moves(
    board: &Board,
    us: Color,
    their_occ: Bitboard,
    all_occ: Bitboard,
    list: &mut MoveList,
) {
    let pawns = board.pieces[Piece::new(us, PieceType::Pawn)];
    let (push_offset, start_rank, promo_rank): (i8, u8, u8) = match us {
        Color::White => (8, 1, 7),
        Color::Black => (-8, 6, 0),
    };

    let mut p = pawns;
    while !p.is_empty() {
        let from = p.pop_lsb();
        let to_idx = (from as i8 + push_offset) as u8;
        let to = Square::new(to_idx);

        if !all_occ.contains(to) {
            if to.rank() == promo_rank {
                for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
                    list.push(Move::new(from, to, pt, MoveType::Promotion));
                }
            } else {
                list.push(Move::new(from, to, PieceType::None, MoveType::Normal));

                if from.rank() == start_rank {
                    let double_to = Square::new((from as i8 + push_offset * 2) as u8);
                    if !all_occ.contains(double_to) {
                        list.push(Move::new(from, double_to, PieceType::None, MoveType::Normal));
                    }
                }
            }
        }

        let mut attacks = pawn_attacks(us, from) & their_occ;
        while !attacks.is_empty() {
            let cap_to = attacks.pop_lsb();
            if cap_to.rank() == promo_rank {
                for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
                    list.push(Move::new(from, cap_to, pt, MoveType::Promotion));
                }
            } else {
                list.push(Move::new(from, cap_to, PieceType::None, MoveType::Normal));
            }
        }

        if board.ep_square.is_valid() {
            let ep_attacks = pawn_attacks(us, from) & Bitboard::from_square(board.ep_square);
            if !ep_attacks.is_empty() {
                list.push(Move::new(from, board.ep_square, PieceType::None, MoveType::EnPassant));
            }
        }
    }
}

fn generate_castling_moves(board: &Board, us: Color, all_occ: Bitboard, list: &mut MoveList) {
    if board.is_square_attacked(board.king_square(us), !us) {
        return;
    }

    match us {
        Color::White => {
            if (board.castling_rights & Board::CASTLE_WK) != 0
                && (all_occ.0 & ((1u64 << Square::F1 as u8) | (1u64 << Square::G1 as u8))) == 0
                && !board.is_square_attacked(Square::F1, Color::Black)
                && !board.is_square_attacked(Square::G1, Color::Black)
            {
                list.push(Move::new(Square::E1, Square::G1, PieceType::None, MoveType::Castling));
            }
            if (board.castling_rights & Board::CASTLE_WQ) != 0
                && (all_occ.0 & ((1u64 << Square::B1 as u8) | (1u64 << Square::C1 as u8) | (1u64 << Square::D1 as u8))) == 0
                && !board.is_square_attacked(Square::D1, Color::Black)
                && !board.is_square_attacked(Square::C1, Color::Black)
            {
                list.push(Move::new(Square::E1, Square::C1, PieceType::None, MoveType::Castling));
            }
        }
        Color::Black => {
            if (board.castling_rights & Board::CASTLE_BK) != 0
                && (all_occ.0 & ((1u64 << Square::F8 as u8) | (1u64 << Square::G8 as u8))) == 0
                && !board.is_square_attacked(Square::F8, Color::White)
                && !board.is_square_attacked(Square::G8, Color::White)
            {
                list.push(Move::new(Square::E8, Square::G8, PieceType::None, MoveType::Castling));
            }
            if (board.castling_rights & Board::CASTLE_BQ) != 0
                && (all_occ.0 & ((1u64 << Square::B8 as u8) | (1u64 << Square::C8 as u8) | (1u64 << Square::D8 as u8))) == 0
                && !board.is_square_attacked(Square::D8, Color::White)
                && !board.is_square_attacked(Square::C8, Color::White)
            {
                list.push(Move::new(Square::E8, Square::C8, PieceType::None, MoveType::Castling));
            }
        }
    }
}

pub fn generate_noisy_moves(board: &mut Board) -> MoveList {
    let mut list = MoveList::new();
    let mut pseudo_list = MoveList::new();
    generate_pseudo_moves(board, &mut pseudo_list);

    for &m in pseudo_list.as_slice() {
        let is_noisy = board.piece_on[m.to()] != Piece::None
            || m.move_type() == MoveType::EnPassant
            || m.move_type() == MoveType::Promotion;

        if !is_noisy {
            continue;
        }

        let us = board.side_to_move;
        let undo = board.make_move(m);
        let ksq = board.king_square(us);
        if !board.is_square_attacked(ksq, board.side_to_move) {
            list.push(m);
        }
        board.undo_move(m, undo);
    }

    list
}