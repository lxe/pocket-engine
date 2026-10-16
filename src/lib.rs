//! Qwen3.5 inference on hand-written CUDA kernels: a fused forward pass over
//! flat token batches, CUDA-graph decode steps and continuous batching behind
//! an OpenAI-compatible server.

pub mod engine;
pub mod gdnbench;
pub mod gguf;
pub mod gpu;
#[cfg(feature = "rocm")]
mod hip;
pub mod image;
pub mod linear;
pub mod lora;
pub mod model;
pub mod sched;
pub mod server;
pub mod source;
pub mod tools;
#[cfg(not(feature = "rocm"))]
pub mod tune;
pub mod tuning;
pub mod vision;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0} failed: {1}")]
    Cuda(&'static str, String),
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Request(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
}

pub type Result<T> = std::result::Result<T, Error>;
