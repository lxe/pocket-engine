use std::{env, path::PathBuf, process::Command};

fn main() {
    for f in [
        "qwen35.cu",
        "quant.cuh",
        "fp8.cuh",
        "fp4.cuh",
        "lora.cuh",
        "ggml-common.h",
    ] {
        println!("cargo:rerun-if-changed=src/kernels/{f}");
    }
    for name in ["CUDA_ROOT", "CUDA_COMPUTE_CAP", "NVCC"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("qwen35.fatbin");
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
