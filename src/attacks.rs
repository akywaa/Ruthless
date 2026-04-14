use crate::bitboard::Bitboard;
use crate::types::{Color, Square, SQUARE_NB};
use std::sync::OnceLock;

pub struct Magic {
    pub mask: u64,
    pub magic: u64,
    pub shift: u8,
    pub offset: usize,
}

pub struct AttackTables {
    pub pawn_attacks: [[Bitboard; SQUARE_NB]; 2],
    pub knight_attacks: [Bitboard; SQUARE_NB],
    pub king_attacks: [Bitboard; SQUARE_NB],
    pub bishop_magics: [Magic; SQUARE_NB],
    pub rook_magics: [Magic; SQUARE_NB],
    pub slider_table: Vec<Bitboard>,
}

static ATTACKS: OnceLock<AttackTables> = OnceLock::new();

#[inline(always)]
pub fn attacks() -> &'static AttackTables {
    ATTACKS.get_or_init(AttackTables::init)
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    attacks().pawn_attacks[color as usize][sq as usize]
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    attacks().knight_attacks[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    attacks().king_attacks[sq as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let m = &attacks().bishop_magics[sq as usize];
    let idx = m.offset + (((occ.0 & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize;
    attacks().slider_table[idx]
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let m = &attacks().rook_magics[sq as usize];
    let idx = m.offset + (((occ.0 & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize;
    attacks().slider_table[idx]
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

struct Prng(u64);
impl Prng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

impl AttackTables {
    fn init() -> Self {
        let mut pawn_attacks = [[Bitboard::EMPTY; SQUARE_NB]; 2];
        let mut knight_attacks = [Bitboard::EMPTY; SQUARE_NB];
        let mut king_attacks = [Bitboard::EMPTY; SQUARE_NB];

        for sq in 0..64 {
            let bb = 1u64 << sq;
            let file = sq % 8;

            let w_left = if file > 0 { bb << 7 } else { 0 };
            let w_right = if file < 7 { bb << 9 } else { 0 };
            pawn_attacks[Color::White as usize][sq] = Bitboard(w_left | w_right);

            let b_left = if file > 0 { bb >> 9 } else { 0 };
            let b_right = if file < 7 { bb >> 7 } else { 0 };
            pawn_attacks[Color::Black as usize][sq] = Bitboard(b_left | b_right);

            let mut k = 0u64;
            let offsets: [i8; 8] = [-17, -15, -10, -6, 6, 10, 15, 17];
            for off in offsets {
                let to = sq as i8 + off;
                if (0..64).contains(&to) && ((sq as i8 % 8) - (to % 8)).abs() <= 2 {
                    k |= 1u64 << to;
                }
            }
            knight_attacks[sq] = Bitboard(k);

            let mut kg = 0u64;
            let king_offsets: [i8; 8] = [-9, -8, -7, -1, 1, 7, 8, 9];
            for off in king_offsets {
                let to = sq as i8 + off;
                if (0..64).contains(&to) && ((sq as i8 % 8) - (to % 8)).abs() <= 1 {
                    kg |= 1u64 << to;
                }
            }
            king_attacks[sq] = Bitboard(kg);
        }

        let mut slider_table = Vec::with_capacity(107648);
        let mut prng = Prng(1070372);

        let (bishop_magics, _) = Self::init_slider(true, &mut slider_table, &mut prng);
        let (rook_magics, _) = Self::init_slider(false, &mut slider_table, &mut prng);

        Self {
            pawn_attacks,
            knight_attacks,
            king_attacks,
            bishop_magics,
            rook_magics,
            slider_table,
        }
    }

    fn init_slider(
        is_bishop: bool,
        table: &mut Vec<Bitboard>,
        prng: &mut Prng,
    ) -> ([Magic; SQUARE_NB], usize) {
        let mut magics = [const { Magic { mask: 0, magic: 0, shift: 0, offset: 0 } }; SQUARE_NB];

        for sq in 0..64 {
            let mask = if is_bishop {
                Self::bishop_mask(sq)
            } else {
                Self::rook_mask(sq)
            };
            let bits = mask.count_ones() as u8;
            let size = 1usize << bits;
            let shift = 64 - bits;
            let offset = table.len();

            let mut occupancies = vec![0u64; size];
            let mut attacks = vec![Bitboard::EMPTY; size];

            for i in 0..size {
                occupancies[i] = Self::index_to_occupancy(i, bits, mask);
                attacks[i] = if is_bishop {
                    Self::bishop_rays(sq, Bitboard(occupancies[i]))
                } else {
                    Self::rook_rays(sq, Bitboard(occupancies[i]))
                };
            }

            table.resize(offset + size, Bitboard::EMPTY);

            loop {
                let magic = prng.sparse();
                if ((mask.wrapping_mul(magic)) >> 56).count_ones() < 6 {
                    continue;
                }

                table[offset..offset + size].fill(Bitboard::EMPTY);
                let mut fail = false;

                for i in 0..size {
                    let idx = offset + (((occupancies[i].wrapping_mul(magic)) >> shift) as usize);
                    if table[idx] == Bitboard::EMPTY {
                        table[idx] = attacks[i];
                    } else if table[idx] != attacks[i] {
                        fail = true;
                        break;
                    }
                }

                if !fail {
                    magics[sq] = Magic { mask, magic, shift, offset };
                    break;
                }
            }
        }

        (magics, table.len())
    }

    fn index_to_occupancy(index: usize, bits: u8, mut mask: u64) -> u64 {
        let mut occ = 0u64;
        for i in 0..bits {
            let lsb = mask.trailing_zeros();
            mask &= mask - 1;
            if (index & (1 << i)) != 0 {
                occ |= 1u64 << lsb;
            }
        }
        occ
    }

    fn bishop_mask(sq: usize) -> u64 {
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        let mut mask = 0u64;
        for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let (mut cf, mut cr) = (f + df, r + dr);
            while cf > 0 && cf < 7 && cr > 0 && cr < 7 {
                mask |= 1u64 << (cr * 8 + cf);
                cf += df;
                cr += dr;
            }
        }
        mask
    }

    fn rook_mask(sq: usize) -> u64 {
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        let mut mask = 0u64;
        for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (mut cf, mut cr) = (f + df, r + dr);
            while (df != 0 && cf > 0 && cf < 7) || (dr != 0 && cr > 0 && cr < 7) {
                mask |= 1u64 << (cr * 8 + cf);
                cf += df;
                cr += dr;
            }
        }
        mask
    }

    fn bishop_rays(sq: usize, occ: Bitboard) -> Bitboard {
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        let mut bb = 0u64;
        for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let (mut cf, mut cr) = (f + df, r + dr);
            while (0..8).contains(&cf) && (0..8).contains(&cr) {
                let bit = 1u64 << (cr * 8 + cf);
                bb |= bit;
                if (occ.0 & bit) != 0 {
                    break;
                }
                cf += df;
                cr += dr;
            }
        }
        Bitboard(bb)
    }

    fn rook_rays(sq: usize, occ: Bitboard) -> Bitboard {
        let f = (sq % 8) as i8;
        let r = (sq / 8) as i8;
        let mut bb = 0u64;
        for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (mut cf, mut cr) = (f + df, r + dr);
            while (0..8).contains(&cf) && (0..8).contains(&cr) {
                let bit = 1u64 << (cr * 8 + cf);
                bb |= bit;
                if (occ.0 & bit) != 0 {
                    break;
                }
                cf += df;
                cr += dr;
            }
        }
        Bitboard(bb)
    }
}