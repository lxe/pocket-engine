//! Qwen3.5 inference on hand-written CUDA kernels: a fused forward pass over
//! flat token batches, CUDA-graph decode steps and continuous batching behind
//! an OpenAI-compatible server.

pub mod engine;
pub mod gpu;
pub mod model;
pub mod sched;
pub mod server;

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
