use crate::board::Board;
use crate::eval::PIECE_VALUES;
use crate::movegen::{generate_noisy_pseudo, generate_quiet_pseudo};
use crate::see::see;
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
    counter_move: Move,
    moves: MoveList,
    scores: [i32; 256],
    cur_idx: usize,
    bad_noisy: MoveList,
    qsearch: bool,
}

impl MovePicker {
    pub fn new(
        tt_move: Move,
        killers: [Move; 2],
        counter_move: Move,
    ) -> Self {
        Self {
            stage: if tt_move != Move::NULL { Stage::TTMove } else { Stage::GenerateNoisy },
            tt_move,
            killers,
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
        conthist: &[[[[i32; 64]; 64]; 12]; 4],
        noisy_history: &[[[i32; 6]; 64]; 12],
        ply: u8,
        played_pieces: &[Piece; 64],
        played_moves: &[Move; 64],
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateNoisy;
                    let m = self.tt_move;
                    if self.is_legal(board, m) {
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
                    self.stage = Stage::CounterMove;
                    for &k in &self.killers {
                        if k != Move::NULL && k != self.tt_move && self.is_pseudo_legal(board, k) && self.is_legal(board, k) {
                            return Some(k);
                        }
                    }
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
                    self.score_quiets(board, history, pawn_history, conthist, ply, played_pieces, played_moves);
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

    fn score_noisy(&mut self, board: &Board, noisy_history: &[[[i32; 6]; 64]; 12]) {
        for i in 0..self.moves.count {
            let m = self.moves.moves[i];
            let attacker = board.piece_on[m.from()];
            let captured = board.piece_on[m.to()];
            let victim_pt = if m.move_type() == MoveType::EnPassant {
                PieceType::Pawn
            } else {
                captured.piece_type()
            };

            let mvv_lva = PIECE_VALUES[victim_pt as usize] * 10 - PIECE_VALUES[attacker.piece_type() as usize];
            let hist = noisy_history[attacker as usize][m.to() as usize][victim_pt as usize];
            self.scores[i] = mvv_lva + hist;
        }
    }

    fn score_quiets(
        &mut self,
        board: &Board,
        history: &[[[i32; 64]; 64]; 2],
        pawn_history: &[[[i32; 64]; 12]; 512],
        conthist: &[[[[i32; 64]; 64]; 12]; 4],
        ply: u8,
        played_pieces: &[Piece; 64],
        played_moves: &[Move; 64],
    ) {
        let us = board.side_to_move as usize;
        let p_idx = (board.pawn_hash as usize) & 511;
        let offsets = [1usize, 2, 4, 6];
        let ply_idx = ply as usize;

        for i in 0..self.moves.count {
            let m = self.moves.moves[i];
            let piece = board.piece_on[m.from()] as usize;

            let mut score = history[us][m.from() as usize][m.to() as usize]
                + pawn_history[p_idx][piece][m.to() as usize];

            for (layer, &offset) in offsets.iter().enumerate() {
                if ply_idx >= offset {
                    let prev_piece = played_pieces[ply_idx - offset];
                    let prev_move = played_moves[ply_idx - offset];
                    if prev_piece != Piece::None && prev_move != Move::NULL {
                        score += conthist[layer][prev_piece as usize][prev_move.to() as usize][m.to() as usize];
                    }
                }
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
}