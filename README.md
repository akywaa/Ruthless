 <div align="center">

# Ruthless

*A UCI chess engine written in Rust.*

</div>

## Features

NNUE evaluation with AVX2/AVX-VNNI/AVX-512 support, paired with PVS alpha-beta search, modern pruning and reductions, history heuristics, and multithreaded search. Optional Syzygy tablebase support.

## Building

Requires a recent stable Rust toolchain.

```bash
cargo build --release
```

By default, the build targets the native CPU. For portable binaries, choose an x86-64 level explicitly:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

Use `x86-64-v2` for a wider baseline or `x86-64-v4` for AVX-512. The equivalent Make targets are `release-v2`, `release-v3`, and `release-v4`.

For profile-guided optimization:

```bash
cargo pgo instrument
cargo pgo run -- bench
cargo pgo optimize
```

Requires `llvm-tools` and `cargo-pgo`. The optimized binary is saved as `Ruthless-pgo` in `target/release/`.

To enable Syzygy support, place [Fathom](https://github.com/jdart1/Fathom) in `deps/Fathom` before building.

## Running

Ruthless speaks UCI and can be used with Cute Chess, Nibbler, Banksia, or any other UCI-compatible GUI.

```bash
./target/release/Ruthless
uci
position startpos moves e2e4 e7e5
go movetime 1000
```

Run `Ruthless bench` for a fixed-depth benchmark.

Available options include `Hash`, `Threads`, `SyzygyPath`, `NumaBind`, `Move Overhead`, and several search tuning parameters.

## License

MIT

<sub>Network trained on [master-binpacks_relabel](https://huggingface.co/datasets/vondele/master-binpacks_relabel/tree/main). Inspired by Reckless.</sub>
