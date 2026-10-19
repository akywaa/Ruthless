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

The net was trained on [vondele's master-binpacks_relabel](https://huggingface.co/datasets/vondele/master-binpacks_relabel/tree/main).

## Building

You need a recent Rust toolchain (edition 2024, so up to date stable).

```
cargo build --release
```

Binary ends up in `target/release/`. On Windows it's `Ruthless.exe`.

The default build targets `x86-64-v2` (SSE4.2 + POPCNT), so the binary runs on
any x86-64 machine. `target-cpu=native` is intentionally not the default: such
binaries fail with `illegal instruction` on hosts that lack one of the build
machine's ISA extensions.

For releases, build one binary per microarchitecture level and ship them
separately:

```
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release  # baseline
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release  # AVX2, BMI2, FMA
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release  # AVX-512
```

`make release-v2` / `release-v3` / `release-v4` do the same on Unix shells.

Slider move generation picks PEXT or magic bitboards at startup: PEXT is used
when BMI2 is available *and* the CPU is not an AMD Zen 1/2 (family 0x17), where
PEXT is microcoded and slower than a magic multiply-shift.

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