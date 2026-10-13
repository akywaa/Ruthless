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
	cargo pgo run -- bench
	@echo "3/3: Building optimized engine..."
	cargo pgo optimize
	@echo "PGO build complete! The binary is located in target/x86_64-unknown-linux-gnu/release/"

clean:
	cargo clean
