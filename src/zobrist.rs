use crate::types::{Piece, Square, PIECE_NB, SQUARE_NB};
use std::sync::OnceLock;

pub struct ZobristKeys {
    pub pieces: [[u64; SQUARE_NB]; PIECE_NB],
    pub castling: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
    pub fiftymove_clock: [u64; 16],
}

// The keys only depend on compile-time constants, so they are built during
// const evaluation instead of behind a OnceLock. That keeps the hot paths
// (`piece_key` is called several times per move) free of atomic loads.
static ZOBRIST: ZobristKeys = ZobristKeys::init();

#[inline(always)]
pub fn zobrist() -> &'static ZobristKeys {
    &ZOBRIST
}

struct XorShift64(u64);

impl XorShift64 {
    const fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl ZobristKeys {
    const fn init() -> Self {
        let mut rng = XorShift64(0x18F2_48A7_C593_4B01);
        let mut pieces = [[0u64; SQUARE_NB]; PIECE_NB];
        // Const evaluation cannot use `for` loops, so the tables are filled by
        // index with `while` loops. The key sequence is identical to the one the
        // previous runtime initializer produced.
        let mut p = 0;
        while p < PIECE_NB {
            let mut sq = 0;
            while sq < SQUARE_NB {
                pieces[p][sq] = rng.next();
                sq += 1;
            }
            p += 1;
        }

        let mut castling = [0u64; 16];
        let mut i = 0;
        while i < 16 {
            castling[i] = rng.next();
            i += 1;
        }

        let mut ep = [0u64; 8];
        let mut i = 0;
        while i < 8 {
            ep[i] = rng.next();
            i += 1;
        }

        let side = rng.next();

        let mut fiftymove_clock = [0u64; 16];
        let mut i = 0;
        while i < 16 {
            fiftymove_clock[i] = rng.next();
            i += 1;
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

pub struct CuckooTable {
    pub keys: [u64; 8192],
    pub sq_a: [Square; 8192],
    pub sq_b: [Square; 8192],
}

static CUCKOO: OnceLock<CuckooTable> = OnceLock::new();

#[inline(always)]
pub fn cuckoo() -> &'static CuckooTable {
    CUCKOO.get_or_init(init_cuckoo)
}

#[inline(always)]
pub fn h1(h: u64) -> usize {
    ((h >> 32) & 0x1FFF) as usize
}

#[inline(always)]
pub fn h2(h: u64) -> usize {
    ((h >> 48) & 0x1FFF) as usize
}

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, queen_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::types::PieceType;

fn init_cuckoo() -> CuckooTable {
    let mut keys = [0u64; 8192];
    let mut sq_a = [Square::None; 8192];
    let mut sq_b = [Square::None; 8192];

    for piece in [
        Piece::WhiteKnight, Piece::WhiteBishop, Piece::WhiteRook, Piece::WhiteQueen, Piece::WhiteKing,
        Piece::BlackKnight, Piece::BlackBishop, Piece::BlackRook, Piece::BlackQueen, Piece::BlackKing,
    ] {
        for a in 0..64 {
            let sq1 = Square::new(a as u8);
            let att = match piece.piece_type() {
                PieceType::Knight => knight_attacks(sq1),
                PieceType::Bishop => bishop_attacks(sq1, Bitboard::EMPTY),
                PieceType::Rook => rook_attacks(sq1, Bitboard::EMPTY),
                PieceType::Queen => queen_attacks(sq1, Bitboard::EMPTY),
                PieceType::King => king_attacks(sq1),
                _ => Bitboard::EMPTY,
            };

            for b in (a + 1)..64 {
                let sq2 = Square::new(b as u8);
                if !att.contains(sq2) {
                    continue;
                }

                let mut key = piece_key(piece, sq1) ^ piece_key(piece, sq2) ^ side_key();
                let mut p_a = sq1;
                let mut p_b = sq2;
                let mut idx = h1(key);

                loop {
                    std::mem::swap(&mut keys[idx], &mut key);
                    std::mem::swap(&mut sq_a[idx], &mut p_a);
                    std::mem::swap(&mut sq_b[idx], &mut p_b);

                    if p_a == Square::None && p_b == Square::None {
                        break;
                    }

                    idx = if idx == h1(key) { h2(key) } else { h1(key) };
                }
            }
        }
    }

    CuckooTable { keys, sq_a, sq_b }
}
