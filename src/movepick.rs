use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::movegen::{generate_noisy_pseudo, generate_quiet_pseudo};
use crate::see::see;
use crate::search::MAX_PLY;
use crate::types::{Move, MoveList, MoveType, Piece, PieceType};

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
        board: &Board,
        history: &[[[i32; 64]; 64]; 2],
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &[[[[[[i32; 64]; 64]; 12]; 2]; 2]; 4],
        noisy_history: &[[[[i32; 2]; 6]; 64]; 12],
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
        prev_in_check: &[bool; MAX_PLY],
        prev_is_capture: &[bool; MAX_PLY],
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateNoisy;
                    let m = self.tt_move;
                    let is_noisy = board.piece_on[m.to()] != Piece::None
                        || m.move_type() == MoveType::Promotion
                        || m.move_type() == MoveType::EnPassant;

                    if (!self.qsearch || is_noisy)
                        && self.is_pseudo_legal_any(board, m)
                        && board.is_legal(m)
                    {
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
                            if board.is_legal(m) {
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
                            && board.is_legal(k)
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
                        && board.is_legal(cm)
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
                        if board.is_legal(m) {
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
                        if m != self.tt_move && board.is_legal(m) {
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

            let attacker_idx = attacker as usize;
            let victim_idx = victim_pt as usize;

            // Guard against broken moves (Piece::None attacker/victim) that
            // would index out of bounds in PIECE_VALUES and noisy_history.
            if attacker_idx < 12 && victim_idx < 6 {
                let mvv_lva = PIECE_VALUES[victim_idx] * 12 - PIECE_VALUES[attacker.piece_type() as usize];
                let hist = noisy_history[attacker_idx][m.to() as usize][victim_idx][to_threatened];
                self.scores[i] = mvv_lva + hist;
            } else {
                self.scores[i] = 0;
            }
        }
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        history: &[[[i32; 64]; 64]; 2],
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &[[[[[[i32; 64]; 64]; 12]; 2]; 2]; 4],
        ply: u8,
        played_pieces: &[Piece; MAX_PLY],
        played_moves: &[Move; MAX_PLY],
        prev_in_check: &[bool; MAX_PLY],
        prev_is_capture: &[bool; MAX_PLY],
    ) {
        let us = board.side_to_move;
        let pawn_threats = board.opponent_pawn_threats();
        let p_idx = (board.pawn_hash as usize) & 511;
        let offsets = [1usize, 2, 4, 6];
        let ply_idx = ply as usize;

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

            if pawn_threats.contains(to) && pt != PieceType::Pawn {
                score -= 3000;
            }

            self.scores[i] = score;
        }
    }

    #[inline(always)]
    fn is_pseudo_legal(&self, board: &Board, m: Move) -> bool {
        m.move_type() != MoveType::EnPassant
            && board.piece_on[m.to()] == Piece::None
            && board.is_pseudo_legal(m)
    }

    #[inline(always)]
    fn is_pseudo_legal_any(&self, board: &Board, m: Move) -> bool {
        board.is_pseudo_legal(m)
    }
}
