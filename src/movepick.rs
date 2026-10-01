use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::movegen::{generate_noisy_pseudo, generate_quiet_pseudo};
use crate::see::see;
use crate::types::{Color, Move, MoveList, MoveType, Piece, PieceType};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Stage {
    TTMove,
    GenerateNoisy,
    GoodNoisy,
    Killers,
    CounterMove,
    GenerateQuiets,
    Quiets,
    BadNoisy,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    killer_idx: usize,
    counter_move: Move,
    moves: MoveList,
    scores: [i32; 256],
    cur_idx: usize,
    bad_noisy: MoveList,
    qsearch: bool,
}

impl MovePicker {
    pub fn new(tt_move: Move, killers: [Move; 2], counter_move: Move) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers,
            killer_idx: 0,
            counter_move,
            moves: MoveList::new(),
            scores: [0; 256],
            cur_idx: 0,
            bad_noisy: MoveList::new(),
            qsearch: false,
        }
    }

    pub fn new_qsearch(tt_move: Move) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers: [Move::NULL; 2],
            killer_idx: 0,
            counter_move: Move::NULL,
            moves: MoveList::new(),
            scores: [0; 256],
            cur_idx: 0,
            bad_noisy: MoveList::new(),
            qsearch: true,
        }
    }

    pub fn next(
        &mut self,
        board: &mut Board,
        history: &[[[i32; 64]; 64]; 2],
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &[[[[[[i32; 64]; 64]; 12]; 2]; 2]; 4],
        noisy_history: &[[[[i32; 2]; 6]; 64]; 12],
        ply: u8,
        played_pieces: &[Piece; 64],
        played_moves: &[Move; 64],
        prev_in_check: &[bool; 64],
        prev_is_capture: &[bool; 64],
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateNoisy;
                    let m = self.tt_move;
                    if self.is_pseudo_legal_any(board, m) && self.is_legal(board, m) {
                        return Some(m);
                    }
                }
                Stage::GenerateNoisy => {
                    self.moves.count = 0;
                    generate_noisy_pseudo(board, &mut self.moves);
                    self.score_noisy(board, noisy_history);
                    self.cur_idx = 0;
                    self.stage = Stage::GoodNoisy;
                }
                Stage::GoodNoisy => {
                    while self.cur_idx < self.moves.count {
                        let m = self.pick_best();
                        if m == self.tt_move {
                            continue;
                        }

                        if see(board, m, 0) {
                            if self.is_legal(board, m) {
                                return Some(m);
                            }
                        } else {
                            self.bad_noisy.push(m);
                        }
                    }

                    if self.qsearch {
                        self.stage = Stage::Done;
                    } else {
                        self.stage = Stage::Killers;
                    }
                }
                Stage::Killers => {
                    while self.killer_idx < 2 {
                        let k = self.killers[self.killer_idx];
                        self.killer_idx += 1;
                        if k != Move::NULL
                            && k != self.tt_move
                            && !(self.killer_idx == 2 && k == self.killers[0])
                            && self.is_pseudo_legal(board, k)
                            && self.is_legal(board, k)
                        {
                            return Some(k);
                        }
                    }
                    self.stage = Stage::CounterMove;
                }
                Stage::CounterMove => {
                    self.stage = Stage::GenerateQuiets;
                    let cm = self.counter_move;
                    if cm != Move::NULL
                        && cm != self.tt_move
                        && cm != self.killers[0]
                        && cm != self.killers[1]
                        && self.is_pseudo_legal(board, cm)
                        && self.is_legal(board, cm)
                    {
                        return Some(cm);
                    }
                }
                Stage::GenerateQuiets => {
                    self.moves.count = 0;
                    generate_quiet_pseudo(board, &mut self.moves);
                    self.score_quiets(
                        board,
                        history,
                        pawn_history,
                        conthist,
                        ply,
                        played_pieces,
                        played_moves,
                        prev_in_check,
                        prev_is_capture,
                    );
                    self.cur_idx = 0;
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    while self.cur_idx < self.moves.count {
                        let m = self.pick_best();
                        if m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move {
                            continue;
                        }
                        if self.is_legal(board, m) {
                            return Some(m);
                        }
                    }
                    self.cur_idx = 0;
                    self.stage = Stage::BadNoisy;
                }
                Stage::BadNoisy => {
                    while self.cur_idx < self.bad_noisy.count {
                        let m = self.bad_noisy.moves[self.cur_idx];
                        self.cur_idx += 1;
                        if m != self.tt_move && self.is_legal(board, m) {
                            return Some(m);
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    fn pick_best(&mut self) -> Move {
        let mut best_score = self.scores[self.cur_idx];
        let mut best_idx = self.cur_idx;

        for i in (self.cur_idx + 1)..self.moves.count {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }

        self.scores.swap(self.cur_idx, best_idx);
        self.moves.moves.swap(self.cur_idx, best_idx);

        let m = self.moves.moves[self.cur_idx];
        self.cur_idx += 1;
        m
    }

    fn score_noisy(&mut self, board: &Board, noisy_history: &[[[[i32; 2]; 6]; 64]; 12]) {
        let threats = board.opponent_threats();

        for i in 0..self.moves.count {
            let m = self.moves.moves[i];
            let attacker = board.piece_on[m.from()];
            let captured = board.piece_on[m.to()];
            let victim_pt = match m.move_type() {
                MoveType::EnPassant => PieceType::Pawn,
                MoveType::Promotion => m.promo_type(),
                _ => captured.piece_type(),
            };

            let to_threatened = threats.contains(m.to()) as usize;
            let mvv_lva = PIECE_VALUES[victim_pt as usize] * 12 - PIECE_VALUES[attacker.piece_type() as usize];
            let hist = noisy_history[attacker as usize][m.to() as usize][victim_pt as usize][to_threatened];
            self.scores[i] = mvv_lva + hist;
        }
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        history: &[[[i32; 64]; 64]; 2],
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &[[[[[[i32; 64]; 64]; 12]; 2]; 2]; 4],
        ply: u8,
        played_pieces: &[Piece; 64],
        played_moves: &[Move; 64],
        prev_in_check: &[bool; 64],
        prev_is_capture: &[bool; 64],
    ) {
        let us = board.side_to_move;
        let them = !us;
        let threats = board.opponent_threats();
        let pawn_threats = board.opponent_pawn_threats();
        let p_idx = (board.pawn_hash as usize) & 511;
        let offsets = [1usize, 2, 4, 6];
        let ply_idx = ply as usize;

        // Wall pawns: protect the friendly king
        let my_ksq = board.king_square(us);
        let my_pawns = board.pieces[Piece::new(us, PieceType::Pawn)];
        let king_wall_pawns = king_attacks(my_ksq) & my_pawns;

        // Opponent attackable targets
        let their_occ = board.occupied_co[them];
        let occ = board.occupied;

        let escape_bonus = [0, 2000, 2200, 3000, 4500, 0];

        for i in 0..self.moves.count {
            let m = self.moves.moves[i];
            let from = m.from();
            let to = m.to();
            let piece = board.piece_on[from];
            let pt = piece.piece_type();

            let mut score = history[us as usize][from as usize][to as usize]
                + pawn_history[p_idx][piece as usize][to as usize];

            for (layer, &offset) in offsets.iter().enumerate() {
                if ply_idx >= offset {
                    let prev_idx = ply_idx - offset;
                    let prev_piece = played_pieces[prev_idx];
                    let prev_move = played_moves[prev_idx];
                    if prev_piece != Piece::None && prev_move != Move::NULL {
                        let chk = prev_in_check[prev_idx] as usize;
                        let cap = prev_is_capture[prev_idx] as usize;
                        score += conthist[layer][chk][cap][prev_piece as usize][prev_move.to() as usize][to as usize];
                    }
                }
            }

            // Tactical adjustments
            let from_threatened = threats.contains(from);
            let to_threatened = threats.contains(to);

            if from_threatened {
                score += escape_bonus[pt as usize];
            }
            if to_threatened {
                score -= escape_bonus[pt as usize] / 2;
            }

            if pawn_threats.contains(to) && pt != PieceType::Pawn {
                score -= 3000;
            }

            // Give a bonus if the move attacks an undefended enemy piece
            let attacks_from_to = match pt {
                PieceType::Knight => knight_attacks(to),
                PieceType::Bishop => bishop_attacks(to, occ),
                PieceType::Rook => rook_attacks(to, occ),
                PieceType::Queen => bishop_attacks(to, occ) | rook_attacks(to, occ),
                _ => Bitboard::EMPTY,
            };
            let attacks_enemy = attacks_from_to & their_occ & !threats;
            if !attacks_enemy.is_empty() {
                score += 1500;
            }

            // Discourage moving king shield pawns
            if pt == PieceType::Pawn && king_wall_pawns.contains(from) {
                score -= 1200;
            }

            self.scores[i] = score;
        }
    }

    fn is_pseudo_legal(&self, board: &Board, m: Move) -> bool {
        let from = m.from();
        let to = m.to();
        let pc = board.piece_on[from];
        if pc == Piece::None || pc.color() != board.side_to_move {
            return false;
        }
        if board.piece_on[to] != Piece::None || m.move_type() == MoveType::EnPassant {
            return false;
        }
        let mut quiets = MoveList::new();
        generate_quiet_pseudo(board, &mut quiets);
        quiets.as_slice().contains(&m)
    }

    fn is_legal(&self, board: &mut Board, m: Move) -> bool {
        let us = board.side_to_move;
        let undo = board.make_move(m);
        let ksq = board.king_square(us);
        let legal = !board.is_square_attacked(ksq, board.side_to_move);
        board.undo_move(m, undo);
        legal
    }

    fn is_pseudo_legal_any(&self, board: &Board, m: Move) -> bool {
        if m == Move::NULL {
            return false;
        }
        let from = m.from();
        let pc = board.piece_on[from];
        if pc == Piece::None || pc.color() != board.side_to_move {
            return false;
        }
        let mut list = MoveList::new();
        generate_noisy_pseudo(board, &mut list);
        generate_quiet_pseudo(board, &mut list);
        list.as_slice().contains(&m)
    }
}
