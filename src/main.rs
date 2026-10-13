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
    /// Quantize bf16 weights to FP8 at load (Hopper and Blackwell). FP8
    /// checkpoints load as FP8 without it.
    #[arg(long)]
    fp8: bool,
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
            fp8: self.fp8,
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
    /// Dequantizes rows of a GGUF tensor on the GPU and writes them as
    /// little-endian f32, for checking the quantized formats.
    Dequant {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 64)]
        rows: usize,
        output: PathBuf,
    },
    /// Times a quantized GGUF matrix against the same matrix in bf16 on cuBLAS,
    /// for several batch sizes, and checks that their outputs agree.
    BenchLinear {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, value_delimiter = ',', default_value = "1,4,8,16,32,64,128,512")]
        rows: Vec<usize>,
    },
    /// Quantizes a bf16 weight matrix to FP8 on the GPU (in the layout this
    /// device's cuBLASLt runs) and times cuBLASLt FP8 against bf16 cuBLAS.
    BenchFp8 {
        #[arg(long)]
        model: PathBuf,
        /// Text-model tensor, e.g. layers.0.mlp.gate_proj.weight.
        #[arg(long)]
        tensor: String,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "1,8,16,32,64,128,512,2048"
        )]
        rows: Vec<usize>,
    },
    /// Prints the token ids of a chat message, for checking tokenizers.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        text: String,
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
        Cmd::Dequant {
            model,
            tensor,
            rows,
            output,
        } => {
            use pocket_engine::gpu::{Arg, Gpu};
            let g = pocket_engine::gguf::Gguf::open(&model)?;
            let (info, _) = g.tensor(&tensor).ok_or("no such tensor")?;
            let (cols, n) = (info.dims[0], rows.min(info.numel() / info.dims[0]));
            let gpu = Gpu::new(0)?;
            let k = pocket_engine::linear::Quants::default().get(&gpu, info.ty)?;
            // The loader's own path: file to device, repacked on the GPU.
            let w = pocket_engine::source::upload_tensor(&gpu, &g, &tensor)?;
            let out = gpu.alloc(n * cols * 2)?;
            gpu.launch(
                k.dequant,
                (((n * cols / 8).div_ceil(256)) as u32, 1, 1),
                256,
                0,
                &[
                    Arg::P(w.ptr),
                    Arg::I(0),
                    Arg::I(n as i32),
                    Arg::I(cols as i32),
                    Arg::P(out.ptr),
                ],
            )?;
            let mut host = gpu.pinned(n * cols * 2)?;
            gpu.dtoh(&mut host, out.ptr, n * cols * 2)?;
            gpu.sync()?;
            let f: Vec<u8> = host
                .as_slice::<u16>()
                .iter()
                .flat_map(|&b| half::bf16::from_bits(b).to_f32().to_le_bytes())
                .collect();
            std::fs::write(output, f)?;
            eprintln!("{tensor}: {} rows x {cols}, {}", n, info.ty.name());
        }
        Cmd::BenchLinear {
            model,
            tensor,
            rows,
        } => {
            use pocket_engine::gpu::{Arg, Gpu};
            use pocket_engine::linear::{Kind, Linear, Part, Quants};
            use std::rc::Rc;
            let g = pocket_engine::gguf::Gguf::open(&model)?;
            let (info, data) = g.tensor(&tensor).ok_or("no such tensor")?;
            let (k, n) = (info.dims[0], info.dims[1]);
            let gpu = Gpu::new(0)?;
            let mut quants = Quants::default();
            let qk = quants.get(&gpu, info.ty)?;
            quants.prepare(&gpu, n * k)?;
            let w = Rc::new(pocket_engine::source::upload_tensor(&gpu, &g, &tensor)?);
            let wb = Rc::new(gpu.alloc(n * k * 2)?);
            gpu.launch(
                qk.dequant,
                (((n * k / 8).div_ceil(256)) as u32, 1, 1),
                256,
                0,
                &[
                    Arg::P(w.ptr),
                    Arg::I(0),
                    Arg::I(n as i32),
                    Arg::I(k as i32),
                    Arg::P(wb.ptr),
                ],
            )?;
            let quant = Linear {
                parts: vec![Part {
                    buf: w,
                    kind: Kind::Quant(info.ty),
                    scales: None,
                    rows: n,
                    off: 0,
                }],
                cols: k,
            };
            let dense = Linear {
                parts: vec![Part {
                    buf: wb,
                    kind: Kind::Bf16,
                    scales: None,
                    rows: n,
                    off: 0,
                }],
                cols: k,
            };
            let max = *rows.iter().max().unwrap_or(&1);
            let x: Vec<u8> = (0..max * k)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
                        .to_le_bytes()
                })
                .collect();
            let x = gpu.upload(&x)?;
            let (ya, yb) = (gpu.alloc(max * n * 4)?, gpu.alloc(max * n * 4)?);
            println!(
                "{tensor}: {n} x {k} {}, {:.1} MB",
                info.ty.name(),
                data.len() as f64 / 1e6
            );
            for &m in &rows {
                let time = |l: &Linear, y: u64| -> Result<f64, Box<dyn std::error::Error>> {
                    l.forward(
                        &gpu,
                        &quants,
                        pocket_engine::linear::Input::bf16(x.ptr),
                        m,
                        y,
                        n,
                        true,
                    )?;
                    gpu.sync()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..20 {
                        l.forward(
                            &gpu,
                            &quants,
                            pocket_engine::linear::Input::bf16(x.ptr),
                            m,
                            y,
                            n,
                            true,
                        )?;
                    }
                    gpu.sync()?;
                    Ok(t0.elapsed().as_secs_f64() / 20.0)
                };
                let (tq, tb) = (time(&quant, ya.ptr)?, time(&dense, yb.ptr)?);
                let mut ha = gpu.pinned(m * n * 4)?;
                let mut hb = gpu.pinned(m * n * 4)?;
                gpu.dtoh(&mut ha, ya.ptr, m * n * 4)?;
                gpu.dtoh(&mut hb, yb.ptr, m * n * 4)?;
                gpu.sync()?;
                let (a, b) = (ha.as_slice::<f32>(), hb.as_slice::<f32>());
                let scale = b.iter().fold(0f32, |s, v| s.max(v.abs()));
                let err =
                    a.iter().zip(b).fold(0f32, |s, (p, q)| s.max((p - q).abs())) / scale.max(1e-9);
                println!(
                    "rows {m:4}: quant {:8.1} us ({:6.0} GB/s of weights)  bf16 cuBLAS {:8.1} us  max rel diff {err:.1e}",
                    tq * 1e6,
                    data.len() as f64 / tq / 1e9,
                    tb * 1e6
                );
            }
        }
        Cmd::BenchFp8 {
            model,
            tensor,
            rows,
        } => {
            use pocket_engine::gpu::{Arg, Fp8Mode, Gpu};
            let src = pocket_engine::source::Source::open(&model)?;
            let dims = src.dims(&tensor).ok_or("no such tensor")?;
            let (n, k) = (dims[0], dims[1]);
            let gpu = Gpu::new(0)?;
            let mode = gpu.fp8_mode().ok_or("this GPU has no FP8 matmul")?;
            gpu.begin_uploads()?;
            let wb = src.matrix(&tensor, n, k)?.upload(&gpu)?;
            gpu.end_uploads()?;
            let wq = gpu.alloc(n * k)?;
            let ws = gpu.alloc(mode.weight_scale_bytes(n, k))?;
            match mode {
                Fp8Mode::Mx32 => gpu.launch(
                    gpu.kernel("fp8_weights_mx")?,
                    (((n * k / 32).div_ceil(256)).min(65_535) as u32, 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(wb.ptr),
                        Arg::I(n as i32),
                        Arg::I(k as i32),
                        Arg::P(wq.ptr),
                        Arg::P(ws.ptr),
                    ],
                )?,
                Fp8Mode::Block128 => gpu.launch(
                    gpu.kernel("fp8_weights_block")?,
                    (k.div_ceil(128) as u32, n.div_ceil(128) as u32, 1),
                    256,
                    0,
                    &[
                        Arg::P(wb.ptr),
                        Arg::I(n as i32),
                        Arg::I(k as i32),
                        Arg::P(wq.ptr),
                        Arg::P(ws.ptr),
                    ],
                )?,
            }
            let max = *rows.iter().max().unwrap_or(&1);
            let x: Vec<u8> = (0..max * k)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
                        .to_le_bytes()
                })
                .collect();
            let x = gpu.upload(&x)?;
            let xq = gpu.alloc(max * k)?;
            let xs = gpu.alloc(mode.act_scale_bytes(max.next_multiple_of(128), k))?;
            let (ya, yb) = (gpu.alloc(max * n * 4)?, gpu.alloc(max * n * 4)?);
            let act = gpu.kernel("fp8_activations")?;
            let mode_id = match mode {
                Fp8Mode::Block128 => 1,
                Fp8Mode::Mx32 => 2,
            };
            println!("{tensor}: {n} x {k}, {mode:?}");
            for &m in &rows {
                let quantize = || {
                    gpu.launch(
                        act,
                        (((m * k / 8).div_ceil(256)).min(65_535) as u32, 1, 1),
                        256,
                        0,
                        &[
                            Arg::P(x.ptr),
                            Arg::I(m as i32),
                            Arg::I(k as i32),
                            Arg::P(xq.ptr),
                            Arg::P(xs.ptr),
                            Arg::I(mode_id),
                            Arg::I(m as i32),
                        ],
                    )
                };
                let fp8 = || -> Result<(), Box<dyn std::error::Error>> {
                    quantize()?;
                    gpu.fp8_matmul(
                        mode, wq.ptr, ws.ptr, xq.ptr, xs.ptr, ya.ptr, n, k, m, n, true,
                    )?;
                    Ok(())
                };
                let dense = || gpu.gemm(x.ptr, wb.ptr, yb.ptr, m, k, n, true);
                let time = |f: &dyn Fn() -> Result<(), Box<dyn std::error::Error>>| -> Result<f64, Box<dyn std::error::Error>> {
                    f()?;
                    gpu.sync()?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..20 {
                        f()?;
                    }
                    gpu.sync()?;
                    Ok(t0.elapsed().as_secs_f64() / 20.0)
                };
                let tf = time(&fp8)?;
                let tb = time(&|| dense().map_err(Into::into))?;
                let mut ha = gpu.pinned(m * n * 4)?;
                let mut hb = gpu.pinned(m * n * 4)?;
                gpu.dtoh(&mut ha, ya.ptr, m * n * 4)?;
                gpu.dtoh(&mut hb, yb.ptr, m * n * 4)?;
                gpu.sync()?;
                let (a, b) = (ha.as_slice::<f32>(), hb.as_slice::<f32>());
                let norm = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                let err = a
                    .iter()
                    .zip(b)
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f32>()
                    .sqrt()
                    / norm.max(1e-9);
                println!(
                    "rows {m:4}: fp8 (with activation quantization) {:8.1} us  bf16 cuBLAS {:8.1} us  relative error {err:.2e}",
                    tf * 1e6,
                    tb * 1e6
                );
            }
        }
        Cmd::Tokenize { model, text } => {
            let tok = ChatTokenizer::load(&model)?;
            let (ids, _) = tok.chat(&[serde_json::json!({ "role": "user", "content": text })])?;
            println!("{}", serde_json::to_string(&ids)?);
        }
        Cmd::Score {
            engine: args,
            input,
            output,
        } => {
            let pairs: Vec<Pair> = serde_json::from_str(&std::fs::read_to_string(input)?)?;
            let tok = ChatTokenizer::load(&args.model)?;
            let spec = Spec::load(&args.model, args.max_image_pixels)?;
            let placeholder = Config::for_model(&args.model)?.image_token;
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
