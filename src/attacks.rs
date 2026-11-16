use crate::bitboard::Bitboard;
use crate::types::{Color, SQUARE_NB, Square};
use std::sync::OnceLock;

pub struct Magic {
    pub mask: u64,
    pub magic: u64,
    pub shift: u8,
    // store both magic and pext tables, chosen at startup
    pub magic_offset: usize,
    pub pext_offset: usize,
}

pub struct SliderTables {
    pub bishop_magics: [Magic; SQUARE_NB],
    pub rook_magics: [Magic; SQUARE_NB],
    pub slider_magic: Vec<Bitboard>,
    pub slider_pext: Vec<Bitboard>,
    // decided once at init to skip atomic load
    pub use_pext: bool,
}

// const static tables for non-slider attacks (computed at compile time)
pub const PAWN_ATTACKS: [[Bitboard; SQUARE_NB]; 2] = init_pawn_attacks();
pub const KNIGHT_ATTACKS: [Bitboard; SQUARE_NB] = init_knight_attacks();
pub const KING_ATTACKS: [Bitboard; SQUARE_NB] = init_king_attacks();
pub const BETWEEN: [[Bitboard; SQUARE_NB]; SQUARE_NB] = init_between();

const fn init_pawn_attacks() -> [[Bitboard; SQUARE_NB]; 2] {
    let mut table = [[Bitboard::EMPTY; SQUARE_NB]; 2];
    let mut sq = 0;
    while sq < 64 {
        let bb = 1u64 << sq;
        let file = sq % 8;

        let w_left = if file > 0 { bb << 7 } else { 0 };
        let w_right = if file < 7 { bb << 9 } else { 0 };
        table[Color::White as usize][sq] = Bitboard(w_left | w_right);

        let b_left = if file > 0 { bb >> 9 } else { 0 };
        let b_right = if file < 7 { bb >> 7 } else { 0 };
        table[Color::Black as usize][sq] = Bitboard(b_left | b_right);

        sq += 1;
    }
    table
}

const fn init_knight_attacks() -> [Bitboard; SQUARE_NB] {
    let mut table = [Bitboard::EMPTY; SQUARE_NB];
    let mut sq = 0;
    while sq < 64 {
        let mut k = 0u64;
        let offsets: [i8; 8] = [-17, -15, -10, -6, 6, 10, 15, 17];
        let mut i = 0;
        while i < 8 {
            let off = offsets[i];
            let to = sq as i8 + off;
            if to >= 0 && to < 64 && ((sq as i8 % 8) - (to % 8)).abs() <= 2 {
                k |= 1u64 << to;
            }
            i += 1;
        }
        table[sq] = Bitboard(k);
        sq += 1;
    }
    table
}

const fn init_king_attacks() -> [Bitboard; SQUARE_NB] {
    let mut table = [Bitboard::EMPTY; SQUARE_NB];
    let mut sq = 0;
    while sq < 64 {
        let mut kg = 0u64;
        let king_offsets: [i8; 8] = [-9, -8, -7, -1, 1, 7, 8, 9];
        let mut i = 0;
        while i < 8 {
            let off = king_offsets[i];
            let to = sq as i8 + off;
            if to >= 0 && to < 64 && ((sq as i8 % 8) - (to % 8)).abs() <= 1 {
                kg |= 1u64 << to;
            }
            i += 1;
        }
        table[sq] = Bitboard(kg);
        sq += 1;
    }
    table
}

const fn init_between() -> [[Bitboard; SQUARE_NB]; SQUARE_NB] {
    let mut table = [[Bitboard::EMPTY; SQUARE_NB]; SQUARE_NB];
    let mut sq1 = 0;
    while sq1 < 64 {
        let f1 = (sq1 % 8) as i8;
        let r1 = (sq1 / 8) as i8;
        let mut sq2 = 0;
        while sq2 < 64 {
            if sq1 != sq2 {
                let f2 = (sq2 % 8) as i8;
                let r2 = (sq2 / 8) as i8;
                let df = (f2 - f1).signum();
                let dr = (r2 - r1).signum();

                if f1 == f2 || r1 == r2 || (f1 - f2).abs() == (r1 - r2).abs() {
                    let mut bb = 0u64;
                    let mut cf = f1 + df;
                    let mut cr = r1 + dr;
                    while cf != f2 || cr != r2 {
                        bb |= 1u64 << (cr * 8 + cf);
                        cf += df;
                        cr += dr;
                    }
                    table[sq1][sq2] = Bitboard(bb);
                }
            }
            sq2 += 1;
        }
        sq1 += 1;
    }
    table
}

// still behind OnceLock because the tables need runtime allocation for slider magics
static SLIDER_TABLES: OnceLock<SliderTables> = OnceLock::new();

#[inline(always)]
pub fn slider_tables() -> &'static SliderTables {
    if let Some(t) = SLIDER_TABLES.get() {
        return t;
    }
    SLIDER_TABLES.get_or_init(SliderTables::init)
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color as usize][sq as usize]
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq as usize]
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let tables = slider_tables();
    let m = &tables.bishop_magics[sq as usize];
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    if tables.use_pext {
        return tables.slider_pext[m.pext_offset + crate::setwise::pext_index(occ.0, m.mask)];
    }
    tables.slider_magic[m.magic_offset + crate::setwise::magic_index(occ.0, m.mask, m.shift, m.magic)]
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let tables = slider_tables();
    let m = &tables.rook_magics[sq as usize];
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    if tables.use_pext {
        return tables.slider_pext[m.pext_offset + crate::setwise::pext_index(occ.0, m.mask)];
    }
    tables.slider_magic[m.magic_offset + crate::setwise::magic_index(occ.0, m.mask, m.shift, m.magic)]
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

#[inline(always)]
pub fn between(sq1: Square, sq2: Square) -> Bitboard {
    if !sq1.is_valid() || !sq2.is_valid() {
        return Bitboard::EMPTY;
    }
    BETWEEN[sq1 as usize][sq2 as usize]
}

impl SliderTables {
    fn init() -> Self {
        let use_pext = crate::setwise::detect_slider_style();

        let mut slider_magic = Vec::with_capacity(107648);
        let mut slider_pext = Vec::with_capacity(107648);
        let mut prng = Prng(1070372);

        let bishop_magics = Self::init_slider(true, &mut slider_magic, &mut slider_pext, &mut prng);
        let rook_magics = Self::init_slider(false, &mut slider_magic, &mut slider_pext, &mut prng);

        Self {
            bishop_magics,
            rook_magics,
            slider_magic,
            slider_pext,
            use_pext,
        }
    }

    fn init_slider(is_bishop: bool, magic_table: &mut Vec<Bitboard>, pext_table: &mut Vec<Bitboard>, prng: &mut Prng) -> [Magic; SQUARE_NB] {
        let mut magics = [const {
            Magic {
                mask: 0,
                magic: 0,
                shift: 0,
                magic_offset: 0,
                pext_offset: 0,
            }
        }; SQUARE_NB];

        for sq in 0..64 {
            let mask = if is_bishop { Self::bishop_mask(sq) } else { Self::rook_mask(sq) };
            let bits = mask.count_ones() as u8;
            let size = 1usize << bits;
            let shift = 64 - bits;
            let magic_offset = magic_table.len();
            let pext_offset = pext_table.len();

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

            // pext layout maps index i directly to attacks[i]
            pext_table.resize(pext_offset + size, Bitboard::EMPTY);
            pext_table[pext_offset..pext_offset + size].copy_from_slice(&attacks);

            magic_table.resize(magic_offset + size, Bitboard::EMPTY);

            loop {
                let magic = prng.sparse();
                if ((mask.wrapping_mul(magic)) >> 56).count_ones() < 6 {
                    continue;
                }

                magic_table[magic_offset..magic_offset + size].fill(Bitboard::EMPTY);
                let mut fail = false;

                for i in 0..size {
                    let idx = magic_offset + (((occupancies[i].wrapping_mul(magic)) >> shift) as usize);
                    if magic_table[idx] == Bitboard::EMPTY {
                        magic_table[idx] = attacks[i];
                    } else if magic_table[idx] != attacks[i] {
                        fail = true;
                        break;
                    }
                }

                if !fail {
                    magics[sq] = Magic {
                        mask,
                        magic,
                        shift,
                        magic_offset,
                        pext_offset,
                    };
                    break;
                }
            }
        }

        magics
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

#[inline(always)]
pub fn aligned(sq1: Square, sq2: Square, sq3: Square) -> bool {
    if !sq1.is_valid() || !sq2.is_valid() || !sq3.is_valid() {
        return false;
    }
    let line = BETWEEN[sq1 as usize][sq3 as usize];
    line.contains(sq2) || sq2 == sq3
}
