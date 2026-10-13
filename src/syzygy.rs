use crate::board::Board;

#[cfg(feature = "syzygy")]
use crate::types::{Color, Piece};
#[cfg(feature = "syzygy")]
use std::ffi::CString;

#[cfg(feature = "syzygy")]
#[link(name = "fathom", kind = "static")]
unsafe extern "C" {
    pub static mut TB_LARGEST: u32;
    pub fn tb_init(path: *const std::ffi::c_char) -> bool;
    pub fn tb_probe_wdl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        rule50: u32,
        castling: u32,
        ep: u32,
        turn: u8,
    ) -> u32;
}

pub fn init(path: &str) {
    #[cfg(feature = "syzygy")]
    {
        if path.is_empty() {
            return;
        }
        if let Ok(c_path) = CString::new(path) {
            unsafe {
                tb_init(c_path.as_ptr());
            }
        }
    }

    #[cfg(not(feature = "syzygy"))]
    let _ = path;
}

/// Number of pieces covered by the largest loaded tablebase (0 if none).
#[cfg(feature = "syzygy")]
pub fn max_pieces() -> u32 {
    unsafe { TB_LARGEST }
}

pub fn probe_wdl(board: &Board) -> Option<i32> {
    #[cfg(feature = "syzygy")]
    {
        let max_p = max_pieces();
        if max_p == 0 || board.occupied.count() > max_p {
            return None;
        }

        let white = board.occupied_co[Color::White as usize].0;
        let black = board.occupied_co[Color::Black as usize].0;

        let kings = board.pieces[Piece::WhiteKing].0 | board.pieces[Piece::BlackKing].0;
        let queens = board.pieces[Piece::WhiteQueen].0 | board.pieces[Piece::BlackQueen].0;
        let rooks = board.pieces[Piece::WhiteRook].0 | board.pieces[Piece::BlackRook].0;
        let bishops = board.pieces[Piece::WhiteBishop].0 | board.pieces[Piece::BlackBishop].0;
        let knights = board.pieces[Piece::WhiteKnight].0 | board.pieces[Piece::BlackKnight].0;
        let pawns = board.pieces[Piece::WhitePawn].0 | board.pieces[Piece::BlackPawn].0;

        let ep = if board.ep_square.is_valid() {
            board.ep_square as u32
        } else {
            0
        };

        let turn = if board.side_to_move == Color::White {
            1
        } else {
            0
        };

        let wdl = unsafe {
            tb_probe_wdl(
                white,
                black,
                kings,
                queens,
                rooks,
                bishops,
                knights,
                pawns,
                board.halfmove_clock as u32,
                board.castling_rights as u32,
                ep,
                turn,
            )
        };

        if wdl == 0xFFFF_FFFF {
            return None; // TB_RESULT_FAILED
        }

        match wdl {
            0 => Some(-20000),    // Loss
            1 | 2 | 3 => Some(0), // Blessed loss / draw / cursed win
            4 => Some(20000),     // Win
            _ => None,
        }
    }

    #[cfg(not(feature = "syzygy"))]
    {
        let _ = board;
        None
    }
}
