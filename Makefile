# Ruthless Makefile

.PHONY: all pgo clean

all:
	cargo build --release

# Note: PGO requires llvm-tools and cargo-pgo.
# Install them with:
#   rustup component add llvm-tools
#   cargo install cargo-pgo
pgo:
	@echo "1/3: Building instrumented engine..."
	cargo pgo instrument
	@echo "2/3: Running search benchmarks to generate profile data..."
	@printf "uci\nposition startpos\ngo depth 13\nposition fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo depth 13\nquit\n" | cargo pgo run
	@echo "3/3: Building optimized engine..."
	cargo pgo optimize
	@echo "PGO build complete! The binary is located in target/<architecture>/release/"

clean:
	cargo clean
