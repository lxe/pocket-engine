use std::{env, path::PathBuf, process::Command};

fn main() {
    for f in [
        "qwen35.cu",
        "quant.cuh",
        "fp8.cuh",
        "fp4.cuh",
        "ggml-common.h",
        "hip_compat.h",
    ] {
        println!("cargo:rerun-if-changed=src/kernels/{f}");
    }
    for name in [
        "CUDA_ROOT",
        "CUDA_COMPUTE_CAP",
        "NVCC",
        "ROCM_PATH",
        "ROCM_ARCH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("qwen35.fatbin");
    if env::var_os("CARGO_FEATURE_ROCM").is_some() {
        assert!(
            env::var_os("CARGO_FEATURE_CUDA").is_none(),
            "build with `cuda` or `rocm`, not both"
        );
        rocm(&out);
        return;
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        std::fs::write(&out, []).expect("write empty kernel image");
        return;
    }
    let nvcc = env::var_os("NVCC").map(PathBuf::from).unwrap_or_else(|| {
        env::var_os("CUDA_ROOT")
            .map(|root| PathBuf::from(root).join("bin/nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"))
    });
    // One architecture per build: 90a for H100, 120a for RTX 50-series. The
    // image carries SASS for it plus PTX the driver can JIT on newer devices.
    let arch = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "90a".into());
    let base = arch.strip_suffix(['a', 'f']).unwrap_or(&arch);
    assert!(
        (2..=3).contains(&base.len()) && base.chars().all(|c| c.is_ascii_digit()),
        "invalid CUDA_COMPUTE_CAP"
    );
    let status = Command::new(nvcc)
        .args([
            "--fatbin",
            "-std=c++17",
            "-O3",
            "--prec-div=true",
            "--prec-sqrt=true",
        ])
        .arg(format!(
            "-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]"
        ))
        .args(["src/kernels/qwen35.cu", "-o"])
        .arg(&out)
        .status()
        .expect("nvcc is required to build the CUDA feature");
    assert!(status.success(), "kernel compilation failed");
}

/// The same kernel sources as a HIP code object for one AMD architecture
/// (gfx942 for MI300X and MI325X), with hip_compat.h standing in for the CUDA
/// headers; links HIP and hipBLAS.
fn rocm(out: &std::path::Path) {
    let root = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-search=native={root}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/lib");
    let arch = env::var("ROCM_ARCH").unwrap_or_else(|_| "gfx942".into());
    let status = Command::new(format!("{root}/bin/hipcc"))
        .args(["--genco", "-std=c++17", "-O3", "-ffp-contract=fast"])
        .arg(format!("--offload-arch={arch}"))
        .args([
            "-include",
            "src/kernels/hip_compat.h",
            "-x",
            "hip",
            "src/kernels/qwen35.cu",
            "-o",
        ])
        .arg(out)
        .status()
        .expect("hipcc is required to build the rocm feature");
    assert!(status.success(), "kernel compilation failed");
}
