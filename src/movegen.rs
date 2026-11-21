use crate::attacks::{king_attacks, pawn_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::setwise::{extract_moves, generate_bishop_moves, generate_knight_moves, generate_queen_moves, generate_rook_moves};
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType, Square};

pub fn generate_legal_moves(board: &Board) -> MoveList {
    let mut list = MoveList::new();
    let mut pseudo = MoveList::new();
    generate_noisy_pseudo(board, &mut pseudo);
    generate_quiet_pseudo(board, &mut pseudo);

    let checkers = board.checkers();
    let pinned = board.pinned_pieces(board.side_to_move);
    for &m in pseudo.as_slice() {
        if board.is_legal_with(m, checkers, pinned) {
            list.push(m);
        }
    }

    list
}

pub fn generate_noisy_pseudo(board: &Board, list: &mut MoveList) {
    let us = board.side_to_move;
    let them = !us;
    let their_occ = board.occupied_co[them];
    let all_occ = board.occupied;
    let empty = !board.occupied;

    generate_pawn_captures_pseudo(board, us, their_occ, list);
    generate_pawn_promotions_pseudo(board, us, empty, list);

    generate_knight_moves(board, us, their_occ, list);
    generate_bishop_moves(board, us, their_occ, all_occ, list);
    generate_rook_moves(board, us, their_occ, all_occ, list);
    generate_queen_moves(board, us, their_occ, all_occ, list);

    let king_sq = board.king_square(us);
    let king_moves = king_attacks(king_sq) & their_occ;
    extract_moves(king_sq, king_moves, list, PieceType::None, MoveType::Normal);
}

pub fn generate_quiet_pseudo(board: &Board, list: &mut MoveList) {
    let us = board.side_to_move;
    let empty = !board.occupied;
    let all_occ = board.occupied;

    generate_pawn_pushes_pseudo(board, us, empty, list);

    generate_knight_moves(board, us, empty, list);
    generate_bishop_moves(board, us, empty, all_occ, list);
    generate_rook_moves(board, us, empty, all_occ, list);
    generate_queen_moves(board, us, empty, all_occ, list);

    let king_sq = board.king_square(us);
    let king_moves = king_attacks(king_sq) & empty;
    extract_moves(king_sq, king_moves, list, PieceType::None, MoveType::Normal);

    generate_castling_moves(board, us, all_occ, list);
}

const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;
const RANK_1: u64 = 0xFF;
const RANK_8: u64 = 0xFF00000000000000;
const RANK_2: u64 = 0x000000000000FF00;
const RANK_7: u64 = 0x00FF000000000000;

fn generate_pawn_captures_pseudo(board: &Board, us: Color, their_occ: Bitboard, list: &mut MoveList) {
    let pawns = board.pieces[Piece::new(us, PieceType::Pawn)].0;

    if board.ep_square.is_valid() {
        let ep = board.ep_square;
        let mut ep_pawns = Bitboard(pawn_attacks(!us, ep).0 & pawns);
        while !ep_pawns.is_empty() {
            let from = ep_pawns.pop_lsb();
            list.push(Move::new(from, ep, PieceType::None, MoveType::EnPassant));
        }
    }

    // ep handled per color below

    match us {
        Color::White => {
            for shift in [7u8, 9u8] {
                let file_mask = if shift == 7 { !FILE_A } else { !FILE_H };
                let captures = Bitboard(((pawns & file_mask) << shift) & their_occ.0);
                // promo targets are the highest ranks, so they come last in
                // ascending square order
                let mut normal = Bitboard(captures.0 & !RANK_8);
                while !normal.is_empty() {
                    let to = normal.pop_lsb();
                    let from = Square::new(to as u8 - shift);
                    list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
                }
                let mut promo = Bitboard(captures.0 & RANK_8);
                while !promo.is_empty() {
                    let to = promo.pop_lsb();
                    let from = Square::new(to as u8 - shift);
                    for pt in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                        list.push(Move::new(from, to, pt, MoveType::Promotion));
                    }
                }
            }
        }
        Color::Black => {
            for shift in [7u8, 9u8] {
                let file_mask = if shift == 7 { !FILE_H } else { !FILE_A };
                let captures = Bitboard(((pawns & file_mask) >> shift) & their_occ.0);
                // promo targets are the lowest ranks, so they come first in
                // ascending square order
                let mut promo = Bitboard(captures.0 & RANK_1);
                while !promo.is_empty() {
                    let to = promo.pop_lsb();
                    let from = Square::new(to as u8 + shift);
                    for pt in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                        list.push(Move::new(from, to, pt, MoveType::Promotion));
                    }
                }
                let mut normal = Bitboard(captures.0 & !RANK_1);
                while !normal.is_empty() {
                    let to = normal.pop_lsb();
                    let from = Square::new(to as u8 + shift);
                    list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
                }
            }
        }
    }
}

fn generate_pawn_pushes_pseudo(board: &Board, us: Color, empty: Bitboard, list: &mut MoveList) {
    let pawns = board.pieces[Piece::new(us, PieceType::Pawn)].0;
    let start_rank_mask: u64 = match us { Color::White => RANK_2, Color::Black => RANK_7 };

    match us {
        Color::White => {
            // single pushes (normal, non-promoting)
            let single = Bitboard((pawns << 8) & empty.0);
            let mut normal = Bitboard(single.0 & !RANK_8);
            while !normal.is_empty() {
                let to = normal.pop_lsb();
                let from = Square::new(to as u8 - 8);
                list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
            }
            // double pushes
            let double = Bitboard(((pawns & start_rank_mask) << 16) & empty.0 & (empty.0 << 8));
            let mut bb = double;
            while !bb.is_empty() {
                let to = bb.pop_lsb();
                let from = Square::new(to as u8 - 16);
                list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
            }
        }
        Color::Black => {
            let single = Bitboard((pawns >> 8) & empty.0);
            let mut normal = Bitboard(single.0 & !RANK_1);
            while !normal.is_empty() {
                let to = normal.pop_lsb();
                let from = Square::new(to as u8 + 8);
                list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
            }
            let double = Bitboard(((pawns & start_rank_mask) >> 16) & empty.0 & (empty.0 >> 8));
            let mut bb = double;
            while !bb.is_empty() {
                let to = bb.pop_lsb();
                let from = Square::new(to as u8 + 16);
                list.push(Move::new(from, to, PieceType::None, MoveType::Normal));
            }
        }
    }
}

fn generate_pawn_promotions_pseudo(board: &Board, us: Color, empty: Bitboard, list: &mut MoveList) {
    let pawns = board.pieces[Piece::new(us, PieceType::Pawn)].0;
    match us {
        Color::White => {
            let single = Bitboard((pawns << 8) & empty.0 & RANK_8);
            let mut bb = single;
            while !bb.is_empty() {
                let to = bb.pop_lsb();
                let from = Square::new(to as u8 - 8);
                for pt in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                    list.push(Move::new(from, to, pt, MoveType::Promotion));
                }
            }
        }
        Color::Black => {
            let single = Bitboard((pawns >> 8) & empty.0 & RANK_1);
            let mut bb = single;
            while !bb.is_empty() {
                let to = bb.pop_lsb();
                let from = Square::new(to as u8 + 8);
                for pt in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                    list.push(Move::new(from, to, pt, MoveType::Promotion));
                }
            }
        }
    }
}

fn generate_castling_moves(board: &Board, us: Color, all_occ: Bitboard, list: &mut MoveList) {
    let rights_mask = if us == Color::White {
        Board::CASTLE_WK | Board::CASTLE_WQ
    } else {
        Board::CASTLE_BK | Board::CASTLE_BQ
    };
    if (board.castling_rights & rights_mask) == 0 {
        return;
    }
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
