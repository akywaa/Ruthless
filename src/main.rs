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
        let bt = std::backtrace::Backtrace::force_capture();
        let mut path = std::env::current_exe().unwrap_or_default();
        path.set_file_name(format!("ruthless_panic_{}.txt", std::process::id()));
        let _ = std::fs::write(path, format!("{info}\n{bt}"));
    }));

    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(uci::uci_loop)
        .unwrap()
        .join()
        .unwrap();
}
