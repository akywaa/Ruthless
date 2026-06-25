use crate::types::{Piece, Square, PIECE_NB, SQUARE_NB};
use std::sync::OnceLock;

pub struct ZobristKeys {
    pub pieces: [[u64; SQUARE_NB]; PIECE_NB],
    pub castling: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
    pub fiftymove_clock: [u64; 16],
}

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();

#[inline(always)]
pub fn zobrist() -> &'static ZobristKeys {
    ZOBRIST.get_or_init(ZobristKeys::init)
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl ZobristKeys {
    fn init() -> Self {
        let mut rng = XorShift64(0x18F2_48A7_C593_4B01);
        let mut pieces = [[0u64; SQUARE_NB]; PIECE_NB];
        for p in 0..PIECE_NB {
            for sq in 0..SQUARE_NB {
                pieces[p][sq] = rng.next();
            }
        }

        let mut castling = [0u64; 16];
        for item in &mut castling {
            *item = rng.next();
        }

        let mut ep = [0u64; 8];
        for item in &mut ep {
            *item = rng.next();
        }

        let side = rng.next();

        let mut fiftymove_clock = [0u64; 16];
        for item in &mut fiftymove_clock {
            *item = rng.next();
        }

        Self {
            pieces,
            castling,
            ep,
            side,
            fiftymove_clock,
        }
    }
}

#[inline(always)]
pub fn piece_key(piece: Piece, sq: Square) -> u64 {
    zobrist().pieces[piece as usize][sq as usize]
}

#[inline(always)]
pub fn castling_key(rights: u8) -> u64 {
    zobrist().castling[(rights & 0xF) as usize]
}

#[inline(always)]
pub fn ep_key(file: u8) -> u64 {
    zobrist().ep[(file & 7) as usize]
}

#[inline(always)]
pub fn fiftymove_key(bucket: u8) -> u64 {
    zobrist().fiftymove_clock[(bucket & 15) as usize]
}

#[inline(always)]
pub fn side_key() -> u64 {
    zobrist().side
}
