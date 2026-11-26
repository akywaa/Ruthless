use std::path::{Path, PathBuf};

fn main() {
    // declare syzygy cfg for check-cfg
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"syzygy\"))");

    let fathom_c = Path::new("deps/Fathom/tbprobe.c");

    if fathom_c.exists() {
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.c");
        println!("cargo:rerun-if-changed=deps/Fathom/tbprobe.h");

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let wrapper_c = out_dir.join("tb_wrapper.c");
        std::fs::write(
            &wrapper_c,
            r#"#include <stdint.h>
#include <stdbool.h>
#include "tbprobe.h"

unsigned tb_probe_wdl_wrapper(
    uint64_t white,
    uint64_t black,
    uint64_t kings,
    uint64_t queens,
    uint64_t rooks,
    uint64_t bishops,
    uint64_t knights,
    uint64_t pawns,
    unsigned rule50,
    unsigned castling,
    unsigned ep,
    bool turn)
{
    return tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns, rule50, castling, ep, turn);
}
"#,
        )
        .expect("failed to write tb_wrapper.c");

        let mut build = cc::Build::new();
        build
            .file("deps/Fathom/tbprobe.c")
            .file(&wrapper_c)
            .include("deps/Fathom");

        if build.get_compiler().is_like_msvc() {
            build.flag("/std:c11").flag("/experimental:c11atomics");
        } else {
            build.std("c11");
        }

        build.compile("fathom");

        // enable syzygy feature flag
        println!("cargo:rustc-cfg=feature=\"syzygy\"");
    } else {
        println!("cargo:warning=deps/Fathom not found. Building without Syzygy Tablebase support.");
    }
}