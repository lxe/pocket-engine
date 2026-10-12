use clap::{Args, Parser, Subcommand};
use pocket_engine::engine::{self, Engine, EngineConfig, Scoring};
use pocket_engine::image::Spec;
use pocket_engine::model::Config;
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
    /// Serve text only, without loading the vision encoder.
    #[arg(long)]
    no_vision: bool,
    /// Largest image after resizing, in pixels (each 32 x 32 pixels becomes
    /// one prompt token).
    #[arg(long, default_value_t = 4 << 20)]
    max_image_pixels: usize,
    /// Image embeddings that can wait at once for their prompt steps.
    #[arg(long, default_value_t = 16_384)]
    image_pool: usize,
}

impl EngineArgs {
    fn config(&self) -> EngineConfig {
        EngineConfig {
            device: self.device,
            max_seqs: self.max_seqs,
            max_batch_tokens: self.max_batch_tokens,
            max_context: self.max_context,
            memory_fraction: self.memory_fraction,
            graphs: !self.no_graphs,
            wait_for: self.wait_for.clone(),
            vision: !self.no_vision,
            max_image_pixels: self.max_image_pixels,
            image_pool: self.image_pool,
            ..EngineConfig::new(self.model.clone())
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
    /// [{"prompt": [ids] | "messages": [...], "forced": [ids]}, ...] (images
    /// in messages as data URLs) and writes, per item and position, the
    /// model's pick and top log-probabilities. Prompts built from messages
    /// go to OUTPUT.prompts.json.
    Score {
        #[command(flatten)]
        engine: EngineArgs,
        input: PathBuf,
        output: PathBuf,
    },
}

#[derive(serde::Deserialize)]
struct Pair {
    #[serde(default)]
    prompt: Vec<u32>,
    #[serde(default)]
    messages: Vec<serde_json::Value>,
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
            let spec = if info.vision {
                Some(Spec::load(&args.model, args.max_image_pixels)?)
            } else {
                None
            };
            eprintln!(
                "{} on {}: {} KV blocks, up to {} sequences of {} tokens",
                name, info.device, info.kv_blocks, info.max_seqs, info.max_context
            );
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(server::serve(tok, spec, tx, info, health, name, &bind))?;
        }
        Cmd::Score {
            engine: args,
            input,
            output,
        } => {
            let pairs: Vec<Pair> = serde_json::from_str(&std::fs::read_to_string(input)?)?;
            let tok = ChatTokenizer::load(&args.model)?;
            let spec = Spec::load(&args.model, args.max_image_pixels)?;
            let placeholder = Config::load(&args.model)?.image_token;
            let mut items = Vec::new();
            let mut prompts = Vec::new();
            for p in pairs {
                let item = if p.messages.is_empty() {
                    Scoring {
                        prompt: p.prompt,
                        forced: p.forced,
                        images: vec![],
                        rope: None,
                    }
                } else {
                    let (ids, urls) = tok.chat(&p.messages)?;
                    let encoded = urls
                        .iter()
                        .map(|u| {
                            server::data_url(u).unwrap_or_else(|| {
                                Err(pocket_engine::Error::Request(
                                    "score takes data: URLs".into(),
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let (prompt, images, rope) =
                        server::with_images(ids, &encoded, &spec, placeholder)?;
                    prompts.push(prompt.clone());
                    Scoring {
                        prompt,
                        forced: p.forced,
                        images,
                        rope,
                    }
                };
                items.push(item);
            }
            let mut e = Engine::new(args.config())?;
            let out = e.score(items)?;
            std::fs::write(&output, serde_json::to_string(&out)?)?;
            if !prompts.is_empty() {
                std::fs::write(
                    output.with_extension("prompts.json"),
                    serde_json::to_string(&prompts)?,
                )?;
            }
            eprintln!("{:?} {:?}", e.sched.counters, e.stats);
        }
    }
    Ok(())
}
