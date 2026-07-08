mod attacks;
mod bitboard;
mod board;
mod eval;
mod movegen;
mod movepick;
mod nnue;
mod search;
mod see;
mod tt;
mod types;
mod uci;
mod zobrist;

fn main() {
    std::panic::set_hook(Box::new(|info| {
        let _ = std::fs::write("ruthless_panic.txt", format!("{info}"));
    }));

    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(uci::uci_loop)
        .unwrap()
        .join()
        .unwrap();
}
