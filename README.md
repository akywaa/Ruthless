# Ruthless

A UCI chess engine written in Rust. It uses an NNUE evaluation and a fairly standard alpha-beta search.

## Features

- Bitboards, magic or PEXT slider attacks (picked at startup)
- NNUE: 10 king-bucketed inputs, 512 hidden units, 8 output buckets, SCReLU. AVX2, AVX-512 and NEON paths
- PVS with null move pruning, LMR, futility pruning, razoring, ProbCut, singular extensions
- Quiescence search, SEE, check extensions
- Quiet, noisy, continuation and correction histories
- Shared transposition table, multithreaded search
- Syzygy WDL probing (optional, through Fathom)

## Building

You need a recent stable Rust toolchain.

    cargo build --release

The binary ends up in `target/release/`. By default it is built with
`target-cpu=native`, so it may not run on older CPUs. For something you want
to share, pick an x86-64 level explicitly:

    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

Use `x86-64-v2` for the baseline or `x86-64-v4` for AVX-512. `make release-v2`,
`release-v3` and `release-v4` do the same thing. The NNUE kernels pick AVX2 at
runtime, so a v2 build still uses the vector paths on AVX2 hardware.

For the last few percent, build with profile-guided optimization:

    cargo pgo instrument
    cargo pgo run -- bench
    cargo pgo optimize

(`make pgo` on Unix, `pgo.bat` on Windows.) This needs `llvm-tools`
(`rustup component add llvm-tools`) and `cargo-pgo` (`cargo install cargo-pgo`).
The optimized binary is copied to `target/release/` and kept as
`Ruthless-pgo` next to it, so a later plain `cargo build --release` cannot
overwrite the only copy.

To enable Syzygy support, put [Fathom](https://github.com/jdart1/Fathom) in
`deps/Fathom` before building. Without it the engine builds fine, just
without tablebases.

## Running

Ruthless only speaks UCI, so use a GUI such as Cute Chess, Nibbler or Banksia.
You can also talk to it directly:

    ./target/release/Ruthless
    uci
    position startpos moves e2e4 e7e5
    go movetime 1000

`Ruthless bench` runs a fixed-depth benchmark.

Options: `Hash`, `Threads`, `SyzygyPath`, `NumaBind`, plus a handful of search
parameters (`RfpBase`, `NmpBase`, `LmrDivisor`, ...) that I use for tuning.

## License

MIT

<sub>net trained on [master-binpacks_relabel](https://huggingface.co/datasets/vondele/master-binpacks_relabel/tree/main) (maybe switch to my own datagen later).</sub>