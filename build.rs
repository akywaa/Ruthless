use std::path::Path;

fn main() {
    // declare syzygy cfg for check-cfg
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"syzygy\"))");

    let fathom_c = Path::new("deps/Fathom/tbprobe.c");

    if fathom_c.exists() {
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.c");
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.h");

        cc::Build::new().file("deps/Fathom/tbprobe.c").include("deps/Fathom").compile("fathom");

        // enable syzygy feature flag
        println!("cargo:rustc-cfg=feature=\"syzygy\"");
    } else {
        println!("cargo:warning=deps/Fathom not found. Building without Syzygy Tablebase support.");
    }
}
