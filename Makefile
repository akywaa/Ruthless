# Ruthless Makefile

.PHONY: all pgo clean release-v2 release-v3 release-v4

all:
	cargo build --release

# One binary per x86-64 microarchitecture level so a release never crashes with
# "illegal instruction" on older CPUs.
release-v2:
	RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release

release-v3:
	RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release

release-v4:
	RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release

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
