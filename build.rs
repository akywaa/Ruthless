use std::path::Path;

fn main() {
    // Declare the cfg as expected so `#[cfg(feature = "syzygy")]` does not trip
    // the `unexpected_cfgs` lint when Cargo.toml declares no such feature.
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"syzygy\"))");

    let fathom_c = Path::new("deps/Fathom/tbprobe.c");

    if fathom_c.exists() {
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.c");
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.h");

        cc::Build::new()
            .file("deps/Fathom/tbprobe.c")
            .include("deps/Fathom")
            .compile("fathom");

        // Tell rustc to enable the "syzygy" feature flag for our code.
        println!("cargo:rustc-cfg=feature=\"syzygy\"");
    } else {
        println!(
            "cargo:warning=deps/Fathom not found. Building without Syzygy Tablebase support."
        );
    }
}
