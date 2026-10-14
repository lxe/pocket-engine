use clap::{Args, Parser, Subcommand};
use pocket_engine::engine::{self, Engine, EngineConfig, Scoring};
use pocket_engine::image::Spec;
use pocket_engine::model::Config;
use pocket_engine::server::{self, ChatTokenizer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Qwen3.5 and Qwen3.8 inference server")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Model directory or GGUF file.
    #[arg(long)]
    model: PathBuf,
    /// GPU index.
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Most requests at once (lowered to fit memory).
    #[arg(long, default_value_t = 256)]
    max_seqs: usize,
    /// Most tokens per step.
    #[arg(long, default_value_t = 4096)]
    max_batch_tokens: usize,
    /// Longest prompt plus output, in tokens.
    #[arg(long, default_value_t = 4096)]
    max_context: usize,
    /// Share of GPU memory to use.
    #[arg(long, default_value_t = 0.9)]
    memory_fraction: f64,
    /// Disable CUDA graphs.
    #[arg(long)]
    no_graphs: bool,
    /// Wait for this path to exist before loading, so GPU setup overlaps a
    /// download.
    #[arg(long)]
    wait_for: Option<PathBuf>,
    /// Text only.
    #[arg(long)]
    no_vision: bool,
    /// Largest image after resizing, in pixels.
    #[arg(long, default_value_t = 4 << 20)]
    max_image_pixels: usize,
    /// Image embeddings waiting at once.
    #[arg(long, default_value_t = 16_384)]
    image_pool: usize,
    /// Quantize a bf16 checkpoint to FP8.
    #[arg(long)]
    fp8: bool,
    /// Quantize a bf16 checkpoint to FP4 (RTX 50-series).
    #[arg(long)]
    fp4: bool,
    /// FP8 KV cache (about twice the tokens).
    #[arg(long)]
    kv_fp8: bool,
    /// f32 linear-attention state (twice the memory).
    #[arg(long)]
    state_f32: bool,
    /// Prompt prefixes to keep; 0 turns caching off [default: --max-seqs / 4].
    #[arg(long)]
    prefix_cache: Option<usize>,
    /// Draft N tokens a step with the multi-token-prediction layer (2 if N is
    /// omitted).
    #[arg(long, num_args = 0..=1, default_missing_value = "2")]
    mtp: Option<usize>,
    /// Draft N tokens a step by prompt lookup: what followed the last earlier
    /// occurrence of the latest few tokens (4 if N is omitted).
    #[arg(long, num_args = 0..=1, default_missing_value = "4")]
    ngram: Option<usize>,
    /// bf16 activations for GGUF weights, instead of int8 in batches.
    #[arg(long)]
    no_int8: bool,
    /// Serve a LoRA adapter as NAME (a PEFT directory or a llama.cpp GGUF
    /// adapter); repeat for more.
    #[arg(long, value_name = "NAME=PATH")]
    lora: Vec<String>,
    /// Adapters on the GPU at once; also allows loading adapters at runtime
    /// [default: 4 with --lora, else off].
    #[arg(long)]
    max_loras: Option<usize>,
    /// Largest adapter rank.
    #[arg(long, default_value_t = 16)]
    max_lora_rank: usize,
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
            fp4: self.fp4,
            kv_fp8: self.kv_fp8,
            state_bf16: !self.state_f32,
            prefix_cache: self.prefix_cache,
            mtp: self.mtp.unwrap_or(0),
            ngram: self.ngram.unwrap_or(0),
            int8: !self.no_int8,
            loras: self
                .lora
                .iter()
                .map(|s| match s.split_once('=') {
                    Some((n, p)) => (n.to_string(), p.into()),
                    None => {
                        let p = std::path::PathBuf::from(s);
                        let n = p
                            .file_stem()
                            .map_or(s.clone(), |f| f.to_string_lossy().into_owned());
                        (n, p)
                    }
                })
                .collect(),
            max_loras: self
                .max_loras
                .unwrap_or(if self.lora.is_empty() { 0 } else { 4 }),
            max_lora_rank: self.max_lora_rank,
            ..EngineConfig::new(self.model.clone())
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve the OpenAI-compatible API.
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1:8190")]
        bind: String,
        /// Model name in the API [default: the model's file or directory name].
        #[arg(long)]
        name: Option<String>,
    },
    /// Dump dequantized rows of a GGUF tensor.
    ///
    /// Dequantizes on the GPU and writes little-endian f32, for checking the
    /// quantized formats.
    Dequant {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 64)]
        rows: usize,
        output: PathBuf,
    },
    /// Time a quantized matrix against bf16.
    ///
    /// Times a GGUF matrix against the same matrix in bf16 on cuBLAS, for
    /// several batch sizes, and checks that their outputs agree.
    BenchLinear {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, value_delimiter = ',', default_value = "1,4,8,16,32,64,128,512")]
        rows: Vec<usize>,
    },
    /// Time the linear-attention kernels.
    ///
    /// `seqs` sequences of `tokens` tokens each, with 27B's head layout by
    /// default.
    BenchGdn {
        #[arg(long, default_value_t = 48)]
        heads: usize,
        #[arg(long, default_value_t = 16)]
        k_heads: usize,
        #[arg(long, default_value_t = 1)]
        seqs: usize,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        /// The chunked tensor-core kernels for prompts (gdn_prep, gdn_scan
        /// and the norm) instead of gdn_decode.
        #[arg(long)]
        chunk: bool,
        /// Compare gdn_decode's outputs and state with the chunked kernels' and
        /// gdn_recurrent's.
        #[arg(long)]
        check: bool,
        /// gdn_scan's blocks per head: 1, 2 or 4.
        #[arg(long, default_value_t = 1)]
        split: usize,
    },
    /// Time prompt attention.
    ///
    /// `seqs` prompts of `tokens` new tokens after `context` cached ones,
    /// `group` query heads per key and value head (9B: 4 kv heads, group 4;
    /// 27B: 4, 6).
    BenchAttn {
        #[arg(long, default_value_t = 4)]
        kv_heads: usize,
        #[arg(long, default_value_t = 4)]
        group: usize,
        #[arg(long, default_value_t = 1)]
        seqs: usize,
        #[arg(long, default_value_t = 4096)]
        tokens: usize,
        #[arg(long, default_value_t = 0)]
        context: usize,
        /// Compare with the paged decode kernel, token by token.
        #[arg(long)]
        check: bool,
        /// Query tiles per block and keys per tile, as "NT,BN" (default: the
        /// device's choice).
        #[arg(long)]
        shape: Option<String>,
        /// The warp-specialized kernel (wgmma on Hopper, tcgen05 on B200;
        /// groups of 4 or 6).
        #[arg(long)]
        wg: bool,
    },
    /// Time an FP8 matrix against bf16.
    ///
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
    /// Print the token ids of a chat message.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        text: String,
    },
    /// Score fixed continuations, for accuracy checks.
    ///
    /// Reads JSON [{"prompt": [ids] | "messages": [...], "forced": [ids]},
    /// ...] (images in messages as data URLs) and writes, per item and
    /// position, the model's pick and top log-probabilities. Prompts built
    /// from messages go to OUTPUT.prompts.json.
    Score {
        #[command(flatten)]
        engine: EngineArgs,
        /// Score with this adapter (loaded with --lora).
        #[arg(long)]
        adapter: Option<String>,
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
            let name = name.unwrap_or_else(|| {
                let p = &args.model;
                let stem = if p.is_dir() {
                    p.file_name()
                } else {
                    p.file_stem()
                };
                stem.map_or("qwen".into(), |s| s.to_string_lossy().into_owned())
            });
            let cfg = args.config();
            let lora_on = cfg.max_loras > 0;
            // The tokenizer and the structured-output tables (about a second)
            // load while the engine loads weights.
            let prep = {
                let model = args.model.clone();
                std::thread::spawn(move || -> pocket_engine::Result<_> {
                    let tok = ChatTokenizer::load(&model)?;
                    let grammars = server::grammars(&tok, &Config::for_model(&model)?)?;
                    Ok((tok, grammars))
                })
            };
            let (tx, info, health, adapters, engine_thread) = engine::spawn(cfg)?;
            let (tok, grammars) = prep.join().expect("tokenizer thread")?;
            let spec = if info.vision {
                Some(Spec::load(&args.model, args.max_image_pixels)?)
            } else {
                None
            };
            eprintln!(
                "{} on {}: {} KV blocks, up to {} sequences of {} tokens",
                name, info.device, info.kv_blocks, info.max_seqs, info.max_context
            );
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let served = rt.block_on(server::serve(
                tok,
                grammars,
                spec,
                tx,
                info,
                health,
                lora_on.then_some(adapters),
                name,
                &bind,
            ));
            // Dropping the server's tasks closes the engine's channel; let the
            // engine free the GPU before the process exits (on AMD, exiting
            // under it crashed in HIP's teardown).
            drop(rt);
            let _ = engine_thread.join();
            served?;
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
            // The tiled layout must decode to exactly the same bits.
            if let (Some(t), Some(tk)) = (
                pocket_engine::source::upload_tiled(&gpu, &g, &tensor)?,
                k.tiled.as_ref(),
            ) {
                let n16 = n / 16 * 16;
                let out2 = gpu.alloc(n16.max(16) * cols * 2)?;
                gpu.launch(
                    tk.dequant,
                    (((n16 * cols / 8).div_ceil(256)) as u32, 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(t.ptr),
                        Arg::I(0),
                        Arg::I(n16 as i32),
                        Arg::I(cols as i32),
                        Arg::P(out2.ptr),
                    ],
                )?;
                let mut h2 = gpu.pinned(n16.max(16) * cols * 2)?;
                gpu.dtoh(&mut h2, out2.ptr, n16 * cols * 2)?;
                gpu.sync()?;
                let same =
                    host.as_slice::<u16>()[..n16 * cols] == h2.as_slice::<u16>()[..n16 * cols];
                eprintln!(
                    "tiled layout: {}",
                    if same { "identical" } else { "DIFFERS" }
                );
                if !same {
                    let (a, b) = (host.as_slice::<u16>(), h2.as_slice::<u16>());
                    let i = (0..n16 * cols).find(|&i| a[i] != b[i]).unwrap_or(0);
                    let f = |v: u16| half::bf16::from_bits(v).to_f32();
                    return Err(format!(
                        "tiled dequantization differs from the plain decoder: row {}, column {}: {} vs {}",
                        i / cols,
                        i % cols,
                        f(a[i]),
                        f(b[i])
                    )
                    .into());
                }
            }
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
            let mut quants8 = Quants::default();
            quants8.int8_rows = 1;
            quants8.get(&gpu, info.ty)?;
            quants8.prepare(&gpu, n * k)?;
            let int8 = qk.tiled.as_ref().is_some_and(|t| t.mma8.is_some());
            // Weights stream from memory when serving, so each format cycles
            // through copies that together exceed the L2 cache several times.
            let copies = |bytes: usize| (512usize << 20).div_ceil(bytes).clamp(2, 32);
            let one = |buf: pocket_engine::gpu::Buf, kind: Kind| Linear {
                parts: vec![Part {
                    buf: Rc::new(buf),
                    kind,
                    scales: None,
                    rows: n,
                    off: 0,
                }],
                cols: k,
                int8: true,
                lora: None,
            };
            let mut quant = Vec::new();
            for _ in 0..copies(data.len()) {
                quant.push(one(
                    pocket_engine::source::upload_tensor(&gpu, &g, &tensor)?,
                    Kind::Quant(info.ty),
                ));
            }
            let mut tiled = Vec::new();
            if info.ty.tile_chunk().is_some() {
                for _ in 0..copies(data.len()) {
                    match pocket_engine::source::upload_tiled(&gpu, &g, &tensor)? {
                        Some(t) => tiled.push(one(t, Kind::Tiled(info.ty))),
                        None => break,
                    }
                }
            }
            let mut dense = Vec::new();
            for _ in 0..copies(n * k * 2) {
                let wb = gpu.alloc(n * k * 2)?;
                gpu.launch(
                    qk.dequant,
                    (((n * k / 8).div_ceil(256)) as u32, 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(quant[0].parts[0].buf.ptr),
                        Arg::I(0),
                        Arg::I(n as i32),
                        Arg::I(k as i32),
                        Arg::P(wb.ptr),
                    ],
                )?;
                dense.push(one(wb, Kind::Bf16));
            }
            let max = *rows.iter().max().unwrap_or(&1);
            let x: Vec<u8> = (0..max * k)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
                        .to_le_bytes()
                })
                .collect();
            let x = gpu.upload(&x)?;
            let (ya, yb, yt) = (
                gpu.alloc(max * n * 4)?,
                gpu.alloc(max * n * 4)?,
                gpu.alloc(max * n * 4)?,
            );
            println!(
                "{tensor}: {n} x {k} {}, {:.1} MB; weights cycle through copies (L2 cold)",
                info.ty.name(),
                data.len() as f64 / 1e6
            );
            for &m in &rows {
                let time_with = |quants: &Quants,
                                 ls: &[Linear],
                                 y: u64|
                 -> Result<f64, Box<dyn std::error::Error>> {
                    let run = |l: &Linear| {
                        l.forward(
                            &gpu,
                            quants,
                            pocket_engine::linear::Input::bf16(x.ptr),
                            m,
                            y,
                            n,
                            true,
                        )
                    };
                    for l in ls {
                        run(l)?;
                    }
                    gpu.sync()?;
                    let iters = 4 * ls.len();
                    let t0 = std::time::Instant::now();
                    for i in 0..iters {
                        run(&ls[i % ls.len()])?;
                    }
                    gpu.sync()?;
                    Ok(t0.elapsed().as_secs_f64() / iters as f64)
                };
                let time = |ls: &[Linear], y: u64| time_with(&quants, ls, y);
                let rel = |p: u64, q: u64| -> Result<f32, Box<dyn std::error::Error>> {
                    let (mut ha, mut hb) = (gpu.pinned(m * n * 4)?, gpu.pinned(m * n * 4)?);
                    gpu.dtoh(&mut ha, p, m * n * 4)?;
                    gpu.dtoh(&mut hb, q, m * n * 4)?;
                    gpu.sync()?;
                    let (a, b) = (ha.as_slice::<f32>(), hb.as_slice::<f32>());
                    let scale = b.iter().fold(0f32, |s, v| s.max(v.abs()));
                    Ok(a.iter().zip(b).fold(0f32, |s, (p, q)| s.max((p - q).abs()))
                        / scale.max(1e-9))
                };
                let (tq, tb) = (time(&quant, ya.ptr)?, time(&dense, yb.ptr)?);
                let err = rel(ya.ptr, yb.ptr)?;
                let tt = if tiled.is_empty() {
                    String::new()
                } else {
                    let t = time(&tiled, yt.ptr)?;
                    let mut s = format!(
                        "  tiled {:8.1} us ({:6.0} GB/s, diff {:.1e})",
                        t * 1e6,
                        data.len() as f64 / t / 1e9,
                        rel(yt.ptr, yb.ptr)?
                    );
                    if int8 {
                        let t = time_with(&quants8, &tiled, yt.ptr)?;
                        s += &format!(
                            "  int8 {:8.1} us (diff {:.1e})",
                            t * 1e6,
                            rel(yt.ptr, yb.ptr)?
                        );
                    }
                    s
                };
                println!(
                    "rows {m:4}: quant {:8.1} us ({:6.0} GB/s of weights)  bf16 cuBLAS {:8.1} us  max rel diff {err:.1e}{tt}",
                    tq * 1e6,
                    data.len() as f64 / tq / 1e9,
                    tb * 1e6
                );
            }
        }
        Cmd::BenchGdn {
            heads,
            k_heads,
            seqs,
            tokens,
            chunk,
            check,
            split,
        } => {
            use pocket_engine::gpu::{Arg, Gpu};
            let gpu = Gpu::new(0)?;
            let (hv, hk, d) = (heads, k_heads, 128usize);
            let c = 2 * hk * d + hv * d;
            let pw = c + hv * d + 2 * hv;
            let t = seqs * tokens;
            let rnd = |n: usize, scale: f32| -> Vec<u8> {
                (0..n)
                    .flat_map(|i| {
                        let x = ((i.wrapping_mul(2654435761) % 2001) as f32 / 1000.0 - 1.0) * scale;
                        half::bf16::from_f32(x).to_le_bytes()
                    })
                    .collect()
            };
            let f32s =
                |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let i32s =
                |v: Vec<i32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let conv = gpu.upload(&rnd(t * c, 1.0))?;
            let proj = gpu.upload(&rnd(t * pw, 1.0))?;
            let neg_a = gpu.upload(&f32s(vec![-0.01; hv]))?;
            let dt = gpu.upload(&f32s(vec![0.1; hv]))?;
            let nw = gpu.upload(&f32s(vec![1.0; d]))?;
            let state = gpu.alloc_zeroed(seqs * hv * d * d * 2)?;
            let cu = gpu.upload(&i32s((0..=seqs).map(|i| (i * tokens) as i32).collect()))?;
            let slots = gpu.upload(&i32s((0..seqs as i32).collect()))?;
            let fresh = gpu.upload(&i32s(vec![0; seqs]))?;
            let tok_seq = gpu.upload(&i32s((0..t).map(|i| (i / tokens) as i32).collect()))?;
            let out = gpu.alloc(t * hv * d * 2)?;
            let norm = gpu.kernel("gdn_norm")?;
            let prep = gpu.kernel("gdn_prep")?;
            gpu.set_smem(prep, pocket_engine::model::GDN_PREP_SMEM)?;
            let kscan = gpu.kernel(&format!("gdn_scan{split}"))?;
            gpu.set_smem(kscan, pocket_engine::model::gdn_scan_smem(split))?;
            let scratch = [
                gpu.alloc(t * hv * d * 2)?,
                gpu.alloc(t * hv * d * 2)?,
                gpu.alloc(t * hv * 4)?,
                gpu.alloc(t * hv * 4)?,
                gpu.alloc(t * hv * 32 * 2)?,
                gpu.alloc(t * hv * 4)?,
            ];
            // Kernels by name: "chunk" (prep, scan, norm), "decode" (gdn_decode)
            // or "recurrent" (gdn_recurrent). Phases: 1 the prep, 2 the scan or
            // recurrence, 4 the norm.
            // Where gdn_decode's buffers do not fit (AMD's 64 KB), "decode"
            // runs gdn_recurrent, as the engine does.
            let dshape = pocket_engine::model::gdn_decode_shape(true, gpu.smem_per_sm);
            let (nbuf, per_sm) = dshape.unwrap_or((0, 0));
            let kdecode = match dshape {
                Some(_) => {
                    let k = gpu.kernel("gdn_decode")?;
                    gpu.set_smem(k, nbuf * 2 * d * d)?;
                    Some(k)
                }
                None => None,
            };
            let launch = |name: &str, fresh_ptr: u64, phases: u8| -> pocket_engine::Result<()> {
                if name == "chunk" && phases & 1 != 0 {
                    gpu.launch(
                        prep,
                        ((t.div_ceil(32) + seqs) as u32, hv as u32, 1),
                        2 * d as u32,
                        pocket_engine::model::GDN_PREP_SMEM as u32,
                        &[
                            Arg::P(conv.ptr),
                            Arg::P(proj.ptr),
                            Arg::I(pw as i32),
                            Arg::I(c as i32),
                            Arg::I(hk as i32),
                            Arg::I(hv as i32),
                            Arg::P(neg_a.ptr),
                            Arg::P(dt.ptr),
                            Arg::P(cu.ptr),
                            Arg::I(seqs as i32),
                            Arg::P(0),
                            Arg::I(1),
                            Arg::I(0),
                            Arg::P(scratch[0].ptr),
                            Arg::P(scratch[1].ptr),
                            Arg::P(scratch[2].ptr),
                            Arg::P(scratch[3].ptr),
                            Arg::P(scratch[4].ptr),
                            Arg::P(scratch[5].ptr),
                        ],
                    )?;
                }
                if phases & 2 != 0 {
                    let (kern, smem, grid, extra) = match name {
                        "chunk" => (
                            kscan,
                            pocket_engine::model::gdn_scan_smem(split) as u32,
                            (seqs as u32, hv as u32, split as u32),
                            vec![
                                Arg::P(scratch[0].ptr),
                                Arg::P(scratch[1].ptr),
                                Arg::P(scratch[2].ptr),
                                Arg::P(scratch[3].ptr),
                                Arg::P(scratch[4].ptr),
                                Arg::P(scratch[5].ptr),
                                Arg::P(0),
                            ],
                        ),
                        "decode" if kdecode.is_some() => (
                            kdecode.expect("checked"),
                            (nbuf * 2 * d * d) as u32,
                            ((seqs * hv).min(gpu.sms * per_sm) as u32, 1, 1),
                            vec![Arg::I(seqs as i32), Arg::I(nbuf as i32)],
                        ),
                        _ => (
                            gpu.kernel("gdn_recurrent")?,
                            0,
                            (seqs as u32, hv as u32, 1),
                            vec![],
                        ),
                    };
                    let threads = if name == "decode" && kdecode.is_some() {
                        256
                    } else {
                        2 * d as u32
                    };
                    gpu.launch(
                        kern,
                        grid,
                        threads,
                        smem,
                        &[
                            Arg::P(conv.ptr),
                            Arg::P(proj.ptr),
                            Arg::I(pw as i32),
                            Arg::I(c as i32),
                            Arg::I(hk as i32),
                            Arg::I(hv as i32),
                            Arg::P(neg_a.ptr),
                            Arg::P(dt.ptr),
                            Arg::P(nw.ptr),
                            Arg::P(state.ptr),
                            Arg::I(1),
                            Arg::P(cu.ptr),
                            Arg::P(slots.ptr),
                            Arg::P(fresh_ptr),
                            Arg::P(out.ptr),
                            Arg::F(1e-6),
                            Arg::I(0),
                            Arg::P(0),
                            Arg::P(0),
                            Arg::I(0),
                            Arg::I(0),
                            Arg::P(0),
                            Arg::I(1),
                        ]
                        .into_iter()
                        .chain(extra)
                        .collect::<Vec<_>>(),
                    )?;
                }
                if name == "chunk" && phases & 4 != 0 {
                    gpu.launch(
                        norm,
                        (t as u32, 1, 1),
                        256,
                        0,
                        &[
                            Arg::P(proj.ptr),
                            Arg::I(pw as i32),
                            Arg::I(c as i32),
                            Arg::I(hv as i32),
                            Arg::P(nw.ptr),
                            Arg::P(out.ptr),
                            Arg::I(t as i32),
                            Arg::F(1e-6),
                            Arg::P(cu.ptr),
                            Arg::P(tok_seq.ptr),
                            Arg::P(0),
                            Arg::P(0),
                            Arg::I(0),
                            Arg::I(0),
                            Arg::P(0),
                            Arg::I(1),
                        ],
                    )?;
                }
                Ok(())
            };
            let which = if chunk { "chunk" } else { "decode" };
            if check {
                // Warm the state, then run both kernels from the same state.
                let fresh1 = gpu.upload(&i32s(vec![1; seqs]))?;
                launch("decode", fresh1.ptr, 7)?;
                let sbytes = seqs * hv * d * d * 2;
                let saved = gpu.alloc(sbytes)?;
                gpu.dtod(saved.ptr, state.ptr, sbytes)?;
                let fetch = |ptr: u64, bytes: usize| -> pocket_engine::Result<Vec<f32>> {
                    let mut h = gpu.pinned(bytes)?;
                    gpu.dtoh(&mut h, ptr, bytes)?;
                    gpu.sync()?;
                    Ok(h.as_slice::<u16>()
                        .iter()
                        .map(|&b| half::bf16::from_bits(b).to_f32())
                        .collect())
                };
                for (label, f, other) in [
                    ("fresh, against chunked", fresh1.ptr, "chunk"),
                    ("carried, against chunked", fresh.ptr, "chunk"),
                    ("carried, against gdn_recurrent", fresh.ptr, "recurrent"),
                ] {
                    gpu.dtod(state.ptr, saved.ptr, sbytes)?;
                    launch("decode", f, 7)?;
                    let (o1, s1) = (fetch(out.ptr, t * hv * d * 2)?, fetch(state.ptr, sbytes)?);
                    gpu.dtod(state.ptr, saved.ptr, sbytes)?;
                    launch(other, f, 7)?;
                    let (o2, s2) = (fetch(out.ptr, t * hv * d * 2)?, fetch(state.ptr, sbytes)?);
                    let rel = |a: &[f32], b: &[f32]| {
                        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs()));
                        let mut diffs: Vec<f32> = a
                            .iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).abs() / scale)
                            .collect();
                        diffs.sort_by(f32::total_cmp);
                        (
                            diffs[diffs.len() / 2],
                            diffs[diffs.len() * 99 / 100],
                            *diffs.last().unwrap(),
                        )
                    };
                    println!(
                        "{label}: outputs median/p99/max rel diff {:?}, state {:?}",
                        rel(&o1, &o2),
                        rel(&s1, &s2)
                    );
                }
                return Ok(());
            }
            let time = |which: &str, phases: u8| -> pocket_engine::Result<f64> {
                launch(which, fresh.ptr, phases)?;
                gpu.sync()?;
                let n = 50;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    launch(which, fresh.ptr, phases)?;
                }
                gpu.sync()?;
                Ok(t0.elapsed().as_secs_f64() / n as f64 * 1e6)
            };
            let us = time(which, 7)?;
            let kernel = if chunk { "chunked" } else { "gdn_decode" };
            println!(
                "{kernel}: {seqs} x {tokens} tokens, {hv} heads: {us:.1} us ({:.2} us per token per sequence)",
                us / tokens as f64
            );
            if chunk {
                println!(
                    "  gdn_prep {:.1} us, gdn_scan{split} {:.1} us, gdn_norm {:.1} us",
                    time(which, 1)?,
                    time(which, 2)?,
                    time(which, 4)?
                );
            } else {
                println!("  gdn_recurrent {:.1} us", time("recurrent", 2)?);
            }
        }
        Cmd::BenchAttn {
            kv_heads,
            group,
            seqs,
            tokens,
            context,
            check,
            shape,
            wg,
        } => {
            use pocket_engine::gpu::{Arg, Gpu};
            const D: usize = 256;
            let gpu = Gpu::new(0)?;
            let (hkv, hq) = (kv_heads, kv_heads * group);
            let qstride = (2 * hq + 2 * hkv) * D;
            let t = seqs * tokens;
            let len = context + tokens;
            let max_blocks = len.div_ceil(16);
            let rnd = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
                (0..n)
                    .flat_map(|i| {
                        let h = (i ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(2654435761);
                        let x = ((h % 2001) as f32 / 1000.0 - 1.0) * scale;
                        half::bf16::from_f32(x).to_le_bytes()
                    })
                    .collect()
            };
            let i32s =
                |v: Vec<i32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let q = gpu.upload(&rnd(t * hq * D, 1, 1.0))?;
            let proj = gpu.upload(&rnd(t * qstride, 2, 1.0))?;
            let slots = seqs * max_blocks * 16;
            let kc = gpu.upload(&rnd(slots * hkv * D, 3, 1.0))?;
            let vc = gpu.upload(&rnd(slots * hkv * D, 4, 1.0))?;
            // Blocks of a sequence in reverse order, as a paged cache may hold them.
            let bt = gpu.upload(&i32s(
                (0..seqs * max_blocks)
                    .map(|i| {
                        let (sq, b) = (i / max_blocks, i % max_blocks);
                        (sq * max_blocks + max_blocks - 1 - b) as i32
                    })
                    .collect(),
            ))?;
            let pos = gpu.upload(&i32s(
                (0..t).map(|i| (context + i % tokens) as i32).collect(),
            ))?;
            let tok_seq = gpu.upload(&i32s((0..t).map(|i| (i / tokens) as i32).collect()))?;
            let cu = gpu.upload(&i32s((0..=seqs).map(|i| (i * tokens) as i32).collect()))?;
            let out = gpu.alloc_zeroed(t * hq * D * 2)?;
            let scale = 1.0 / (D as f32).sqrt();
            let sh = match shape.as_deref() {
                Some(v) => pocket_engine::model::FaShape::parse(v).ok_or("--shape takes NT,BN")?,
                None => pocket_engine::model::fa_shape(group, gpu.arch, false),
            };
            let (tiles, smem) = (sh.tiles, sh.smem(group));
            let ws = if wg {
                Some(
                    pocket_engine::model::FaWs::get(group, gpu.arch, false)
                        .ok_or("--wg needs Hopper or B200 and a group of 4 or 6")?,
                )
            } else {
                None
            };
            let (name, grid, threads, smem) = if let Some(ws) = ws {
                (
                    ws.name(),
                    ws.grid(t, seqs, hq, gpu.sms),
                    ws.threads(),
                    ws.smem(),
                )
            } else {
                (
                    sh.kernel(group, false),
                    ((t.div_ceil(16 * tiles) + seqs) as u32, hkv as u32, 1),
                    sh.threads(group) as u32,
                    smem,
                )
            };
            let flash = gpu.kernel(&name)?;
            gpu.set_smem(flash, smem)?;
            // The wgmma kernel reads the cache through TMA maps:
            // [slot][kv head][256] in boxes of 16 slots x 64 columns.
            let maps = if wg {
                let map = |ptr| {
                    gpu.tensor_map_3d(
                        ptr,
                        [D as u64, hkv as u64, slots as u64],
                        [(D * 2) as u64, (hkv * D * 2) as u64],
                        [64, 1, 16],
                    )
                };
                Some((map(kc.ptr)?, map(vc.ptr)?))
            } else {
                None
            };
            let run = |o: u64| {
                let mut args = vec![
                    Arg::P(q.ptr),
                    Arg::P(kc.ptr),
                    Arg::P(vc.ptr),
                    Arg::P(0),
                    Arg::P(0),
                    Arg::P(proj.ptr),
                    Arg::I(qstride as i32),
                    Arg::I(hq as i32),
                    Arg::I(hkv as i32),
                    Arg::P(pos.ptr),
                    Arg::P(cu.ptr),
                    Arg::I(seqs as i32),
                    Arg::P(bt.ptr),
                    Arg::I(max_blocks as i32),
                    Arg::I(16),
                    Arg::P(o),
                    Arg::F(scale),
                    Arg::P(0),
                    Arg::I(1),
                ];
                if let Some((km, vm)) = maps {
                    args.extend([Arg::Map(km), Arg::Map(vm), Arg::I(slots as i32)]);
                }
                gpu.launch(flash, grid, threads, smem as u32, &args)
            };
            if check {
                let reference = gpu.alloc_zeroed(t * hq * D * 2)?;
                gpu.launch(
                    gpu.kernel(&format!("attn_paged_g{group}"))?,
                    (t as u32, hkv as u32, 1),
                    if group >= 6 { 128 } else { 256 },
                    0,
                    &[
                        Arg::P(q.ptr),
                        Arg::P(kc.ptr),
                        Arg::P(vc.ptr),
                        Arg::P(0),
                        Arg::P(0),
                        Arg::P(proj.ptr),
                        Arg::I(qstride as i32),
                        Arg::I(hq as i32),
                        Arg::I(hkv as i32),
                        Arg::P(pos.ptr),
                        Arg::P(tok_seq.ptr),
                        Arg::P(bt.ptr),
                        Arg::I(max_blocks as i32),
                        Arg::I(16),
                        Arg::P(reference.ptr),
                        Arg::F(scale),
                        Arg::P(0),
                        Arg::P(0),
                        Arg::I(0),
                        Arg::I(0),
                        Arg::P(cu.ptr),
                        Arg::I(0),
                        Arg::I(256),
                        Arg::P(0),
                        Arg::P(0),
                        Arg::I(1),
                    ],
                )?;
                run(out.ptr)?;
                let fetch = |ptr: u64| -> pocket_engine::Result<Vec<f32>> {
                    let mut h = gpu.pinned(t * hq * D * 2)?;
                    gpu.dtoh(&mut h, ptr, t * hq * D * 2)?;
                    gpu.sync()?;
                    Ok(h.as_slice::<u16>()
                        .iter()
                        .map(|&b| half::bf16::from_bits(b).to_f32())
                        .collect())
                };
                let (a, b) = (fetch(reference.ptr)?, fetch(out.ptr)?);
                if std::env::var_os("POCKET_CHECK_HEADS").is_some() {
                    // The largest difference by head, and the first token where it occurs.
                    for hh in 0..hq {
                        let (mut worst, mut at) = (0f32, 0);
                        for tt in 0..t {
                            for d in 0..D {
                                let i = (tt * hq + hh) * D + d;
                                let e = (a[i] - b[i]).abs();
                                if e > worst {
                                    (worst, at) = (e, tt);
                                }
                            }
                        }
                        println!("head {hh}: max |diff| {worst:.2e} at token {at}");
                        if worst > 5e-3 {
                            let bad: Vec<usize> = (0..t)
                                .filter(|&tt| {
                                    (0..D).any(|d| {
                                        (a[(tt * hq + hh) * D + d] - b[(tt * hq + hh) * D + d])
                                            .abs()
                                            > 5e-3
                                    })
                                })
                                .collect();
                            let cols: Vec<usize> = (0..D)
                                .filter(|&d| {
                                    (0..t).any(|tt| {
                                        (a[(tt * hq + hh) * D + d] - b[(tt * hq + hh) * D + d])
                                            .abs()
                                            > 5e-3
                                    })
                                })
                                .collect();
                            println!("  tokens {bad:?}\n  columns {cols:?}");
                        }
                    }
                }
                let mut diffs: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect();
                diffs.sort_by(f32::total_cmp);
                let amax = a.iter().fold(0f32, |m, x| m.max(x.abs()));
                println!(
                    "flash against paged: |diff| median {:.2e}, p99 {:.2e}, max {:.2e} (outputs up to {amax:.3})",
                    diffs[diffs.len() / 2],
                    diffs[diffs.len() * 99 / 100],
                    diffs.last().unwrap()
                );
                return Ok(());
            }
            run(out.ptr)?;
            gpu.sync()?;
            let n = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                run(out.ptr)?;
            }
            gpu.sync()?;
            let s = t0.elapsed().as_secs_f64() / n as f64;
            // Each query attends to the keys up to its position: 4 flops
            // per key and dimension (QK^T and PV).
            let attended: f64 = (0..tokens).map(|i| (context + i + 1) as f64).sum();
            let flops = 4.0 * attended * (D * hq * seqs) as f64;
            println!(
                "{name}: {seqs} x {tokens} tokens after {context}, {hq} heads: {:.2} ms, {:.1} TFLOPS",
                s * 1e3,
                flops / s / 1e12
            );
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
            adapter,
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
                        adapter: 0,
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
                        adapter: 0,
                    }
                };
                items.push(item);
            }
            let mut e = Engine::new(args.config())?;
            if let Some(name) = adapter {
                let id = *e.adapters.lock().unwrap().get(&name).ok_or(format!(
                    "no adapter {name}; load it with --lora {name}=PATH"
                ))?;
                items.iter_mut().for_each(|i| i.adapter = id);
            }
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
