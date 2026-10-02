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

#[allow(dead_code)]
pub struct CuckooTable {
    pub keys: [u64; 8192],
    pub sq_a: [Square; 8192],
    pub sq_b: [Square; 8192],
}

#[allow(dead_code)]
static CUCKOO: OnceLock<CuckooTable> = OnceLock::new();

#[inline(always)]
#[allow(dead_code)]
pub fn cuckoo() -> &'static CuckooTable {
    CUCKOO.get_or_init(init_cuckoo)
}

#[inline(always)]
#[allow(dead_code)]
pub fn h1(h: u64) -> usize {
    ((h >> 32) & 0x1FFF) as usize
}

#[inline(always)]
#[allow(dead_code)]
pub fn h2(h: u64) -> usize {
    ((h >> 48) & 0x1FFF) as usize
}

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, queen_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::types::PieceType;

#[allow(dead_code)]
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
