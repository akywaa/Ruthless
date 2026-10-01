mod attacks;
mod bitboard;
mod board;
mod eval;
mod movegen;
mod search;
mod tt;
mod types;
mod uci;
mod zobrist;

fn main() {
    uci::uci_loop();
}
