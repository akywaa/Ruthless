# Ruthless

A UCI chess engine written in Rust.

The design is heavily inspired by [Reckless](https://github.com/codedeliveryservice/Reckless), which I read through a lot while writing mine.

## What's inside

- Bitboards with magic bitboard move generation
- Alpha-beta search (PVS, null move pruning, LMR)
- Quiescence search
- Transposition table, Zobrist hashing
- NNUE eval, the net is `ruthless.bin`
- UCI protocol

The net was trained on [linrock's bullet-training-data (S1)](https://huggingface.co/datasets/linrock/bullet-training-data/tree/main/S1).

## Building

You need a recent Rust toolchain (edition 2024, so up to date stable).

```
cargo build --release
```

Binary ends up in `target/release/`. On Windows it's `Ruthless.exe`.

## Running

It only speaks UCI, so you'll want a GUI like Cute Chess, Nibbler or Banksia. You can also poke it by hand in the terminal:

```
./target/release/Ruthless
uci
isready
position startpos moves e2e4 e7e5
go movetime 1000
```

## License

MIT