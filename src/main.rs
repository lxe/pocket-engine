use clap::{Args, Parser, Subcommand};
use pocket_engine::engine::{self, Engine, EngineConfig};
use pocket_engine::server::{self, ChatTokenizer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Qwen3.5 inference on hand-written CUDA kernels")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Model directory: config.json, safetensors, tokenizer.json, chat_template.jinja.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 256)]
    max_seqs: usize,
    #[arg(long, default_value_t = 4096)]
    max_batch_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    max_context: usize,
    #[arg(long, default_value_t = 0.9)]
    memory_fraction: f64,
    /// Run every step eagerly instead of replaying captured decode graphs.
    #[arg(long)]
    no_graphs: bool,
    /// Set up the GPU, then wait for this path to exist before reading the
    /// model (for example a file a downloader creates when it finishes).
    #[arg(long)]
    wait_for: Option<PathBuf>,
}

impl EngineArgs {
    fn config(&self) -> EngineConfig {
        EngineConfig {
            model: self.model.clone(),
            device: self.device,
            max_seqs: self.max_seqs,
            max_batch_tokens: self.max_batch_tokens,
            max_context: self.max_context,
            memory_fraction: self.memory_fraction,
            graphs: !self.no_graphs,
            wait_for: self.wait_for.clone(),
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve /v1/chat/completions.
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "127.0.0.1:8190")]
        bind: String,
        /// Model name reported by the API.
        #[arg(long, default_value = "qwen3.5")]
        name: String,
    },
    /// Teacher-forced scoring for accuracy checks. Reads JSON
    /// [{"prompt": [ids], "forced": [ids]}, ...] and writes, per pair and
    /// position, the model's pick and top log-probabilities.
    Score {
        #[command(flatten)]
        engine: EngineArgs,
        input: PathBuf,
        output: PathBuf,
    },
}

#[derive(serde::Deserialize)]
struct Pair {
    prompt: Vec<u32>,
    forced: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Serve {
            engine: args,
            bind,
            name,
        } => {
            let (tx, info, health) = engine::spawn(args.config())?;
            let tok = ChatTokenizer::load(&args.model)?;
            eprintln!(
                "{} on {}: {} KV blocks, up to {} sequences of {} tokens",
                name, info.device, info.kv_blocks, info.max_seqs, info.max_context
            );
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(server::serve(tok, tx, info, health, name, &bind))?;
        }
        Cmd::Score {
            engine: args,
            input,
            output,
        } => {
            let pairs: Vec<Pair> = serde_json::from_str(&std::fs::read_to_string(input)?)?;
            let mut e = Engine::new(args.config())?;
            let out = e.score(pairs.into_iter().map(|p| (p.prompt, p.forced)).collect())?;
            std::fs::write(output, serde_json::to_string(&out)?)?;
            eprintln!("{:?} {:?}", e.sched.counters, e.stats);
        }
    }
    Ok(())
}
