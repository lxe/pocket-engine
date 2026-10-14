//! Runs scheduled steps on the GPU. Steps without prompt tokens replay a CUDA
//! graph captured at startup for the padded batch size; others run eagerly.

use crate::gpu::{Gpu, Graph};
use crate::image::Image;
use crate::model::{Act, BLOCK_SIZE, Config, Meta, Model, State};
use crate::sched::{Load, Pen, Request, Sampling, Scheduler, Scored, Seq};
use crate::vision::Vision;
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model: PathBuf,
    pub device: i32,
    /// Most sequences decoding at once.
    pub max_seqs: usize,
    /// Most tokens in one step (decode rows plus prompt chunks).
    pub max_batch_tokens: usize,
    /// Longest prompt plus output, in tokens.
    pub max_context: usize,
    /// Share of device memory the engine may hold.
    pub memory_fraction: f64,
    pub graphs: bool,
    /// Initialize the GPU, then wait for this path to exist before reading
    /// the model, so device setup overlaps a download.
    pub wait_for: Option<PathBuf>,
    /// Load the vision encoder when the checkpoint has one.
    pub vision: bool,
    /// Largest image, in pixels, after resizing.
    pub max_image_pixels: usize,
    /// Image embeddings that can wait at once for their prompt steps.
    pub image_pool: usize,
    /// Quantize bf16 weights to FP8 at load.
    pub fp8: bool,
    /// Quantize bf16 MLP weights to FP4 at load, the rest to FP8 (Blackwell).
    pub fp4: bool,
    /// Store the KV cache as E4M3 with a scale per token and head.
    pub kv_fp8: bool,
    /// Store the recurrent state as bf16 (math stays f32).
    pub state_bf16: bool,
    /// State slots for prefix-cache snapshots; None picks a quarter of
    /// max_seqs (2 to 64), 0 turns the cache off.
    pub prefix_cache: Option<usize>,
    /// Drafts per step from the multi-token-prediction layer; 0 turns
    /// speculative decoding off.
    pub mtp: usize,
    /// Quantize activations to int8 for 4- to 6-bit GGUF matrices in batches.
    pub int8: bool,
    /// LoRA adapters to load at startup: (name, PEFT directory or GGUF file).
    pub loras: Vec<(String, PathBuf)>,
    /// Adapters on the GPU at once; 0 turns adapters off.
    pub max_loras: usize,
    /// Largest adapter rank.
    pub max_lora_rank: usize,
}

impl EngineConfig {
    pub fn new(model: PathBuf) -> Self {
        Self {
            model,
            device: 0,
            max_seqs: 256,
            max_batch_tokens: 4096,
            max_context: 4096,
            memory_fraction: 0.9,
            graphs: true,
            wait_for: None,
            vision: true,
            max_image_pixels: 4 << 20,
            image_pool: 16_384,
            fp8: false,
            fp4: false,
            kv_fp8: false,
            state_bf16: true,
            prefix_cache: None,
            mtp: 0,
            int8: true,
            loras: Vec::new(),
            max_loras: 0,
            max_lora_rank: 16,
        }
    }
}

/// A teacher-forced scoring input: the prompt (with its images, if any) and
/// the continuation to feed.
pub struct Scoring {
    pub prompt: Vec<u32>,
    pub forced: Vec<u32>,
    pub images: Vec<Image>,
    pub rope: Option<crate::image::Rope>,
    /// The adapter to score with (an id from `Engine::load_adapter`), or 0.
    pub adapter: u32,
}

/// What the server sends the engine.
pub enum Msg {
    Request(Request),
    /// Loads an adapter under a name; the reply carries an error, if any.
    LoadAdapter {
        name: String,
        path: PathBuf,
        reply: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
    },
    UnloadAdapter {
        name: String,
        reply: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
    },
}

/// Adapter names the server may route requests to, by id.
pub type Adapters = Arc<Mutex<HashMap<String, u32>>>;

#[derive(Debug, Clone)]
pub struct Info {
    pub device: String,
    /// Whether the engine accepts images.
    pub vision: bool,
    pub cfg: Config,
    pub max_seqs: usize,
    pub kv_blocks: usize,
    pub max_context: usize,
}

/// Engine state shared with the server: capacity, and whether it still runs.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Health {
    pub load: Load,
    pub failed: Option<String>,
}

const BUCKETS: [usize; 16] = [
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256,
];

#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub graph_steps: u64,
    pub step_seconds: f64,
}

pub struct Engine {
    gpu: Gpu,
    model: Model,
    state: State,
    act: Act,
    cfg: EngineConfig,
    graphs: HashMap<usize, Graph>,
    /// Steps where every sequence verifies drafts, by padded sequence count.
    verify_graphs: HashMap<usize, Graph>,
    /// The same with the LoRA kernels, for steps where a sequence has an adapter.
    lora_graphs: HashMap<usize, Graph>,
    lora_verify_graphs: HashMap<usize, Graph>,
    lora: Option<crate::lora::Loras>,
    pub adapters: Adapters,
    vision: Option<Vision>,
    pub sched: Scheduler,
    pub stats: Stats,
}

fn wait_for(path: &std::path::Path) {
    let started = Instant::now();
    while !path.exists() {
        std::thread::sleep(Duration::from_millis(5));
    }
    eprintln!(
        "{} appeared after {:.2}s",
        path.display(),
        started.elapsed().as_secs_f64()
    );
}

/// State slots for prefix-cache snapshots beside `max_seqs` sequences.
fn snapshots(cfg: &EngineConfig, max_seqs: usize) -> usize {
    cfg.prefix_cache.unwrap_or((max_seqs / 4).clamp(2, 64))
}

impl Engine {
    pub fn new(mut cfg: EngineConfig) -> Result<Self> {
        let started = Instant::now();
        let gpu = Gpu::new(cfg.device)?;
        let t_gpu = started.elapsed().as_secs_f64();
        if let Some(path) = &cfg.wait_for {
            wait_for(path);
        }
        let load = crate::source::Load {
            fp8: gpu.fp8_mode(),
            quantize: cfg.fp8 || cfg.fp4,
            fp4: cfg.fp4,
            mtp: cfg.mtp > 0,
            int8: cfg.int8,
        };
        if cfg.mtp > 7 {
            return Err(Error::Config("--mtp takes 1 to 7 drafts per step".into()));
        }
        if cfg.fp4 && load.fp8 != Some(crate::gpu::Fp8Mode::Mx32) {
            return Err(Error::Config(
                "--fp4 needs a Blackwell GPU (RTX 50-series)".into(),
            ));
        }
        if cfg.fp8 && load.fp8.is_none() {
            return Err(Error::Config(
                "--fp8 needs a GPU with FP8 matmuls (Hopper or Blackwell)".into(),
            ));
        }
        let t_wait = started.elapsed().as_secs_f64();
        // One set of staging buffers serves the model and the vision tower.
        gpu.begin_uploads()?;
        let model = Model::load(&gpu, &cfg.model, load)?;
        let t_model = started.elapsed().as_secs_f64();
        let mc = model.cfg.clone();
        let vision = if cfg.vision {
            Vision::load(&gpu, &cfg.model, cfg.max_image_pixels / 256)?
        } else {
            None
        };
        gpu.end_uploads()?;
        let pool_rows = if vision.is_some() { cfg.image_pool } else { 0 };
        // A sequence verifying drafts feeds up to `spec` tokens a step.
        let spec = cfg.mtp + 1;
        // Each sequence holds a recurrent state slot (tens of MB on larger
        // models). Keep slots within half of what is left after the weights,
        // and leave the KV cache at least one full context.
        {
            let (free, total) = gpu.mem_info()?;
            let usable = free.saturating_sub((total as f64 * (1.0 - cfg.memory_fraction)) as usize);
            let one_context = cfg.max_context.div_ceil(BLOCK_SIZE) * mc.block_bytes(cfg.kv_fp8);
            let budget = (usable / 2).min(usable.saturating_sub(one_context + (1 << 30)))
                / mc.slot_bytes(cfg.state_bf16);
            // Slots: one per sequence, the scratch slot and the prefix
            // cache's snapshots.
            let mut fit = budget.saturating_sub(1).max(1);
            while fit > 1 && fit + 1 + snapshots(&cfg, fit) > budget {
                fit -= 1;
            }
            if fit < cfg.max_seqs {
                eprintln!(
                    "{} sequences fit beside the KV cache ({} MB of state each); --max-seqs {} lowered",
                    fit,
                    mc.slot_bytes(cfg.state_bf16) >> 20,
                    cfg.max_seqs
                );
                cfg.max_seqs = fit;
            }
        }
        let max_blocks = cfg.max_context.div_ceil(BLOCK_SIZE);
        let meta = Meta {
            // A multiple of 4, so FP8 matmuls padded to 4 rows stay in bounds.
            max_tokens: cfg
                .max_batch_tokens
                .max(cfg.max_seqs * spec)
                .next_multiple_of(4),
            max_seqs: cfg.max_seqs,
            max_blocks,
            spec,
        };
        let act = Act::new(&gpu, &mc, meta, pool_rows, model.fp8)?;
        let mut model = model;
        let lora = if cfg.max_loras > 0 || !cfg.loras.is_empty() {
            if spec > crate::lora::TILE {
                return Err(Error::Config(
                    "adapters take at most 15 drafts per step".into(),
                ));
            }
            let slots = cfg.max_loras.max(1);
            let l = crate::lora::Loras::new(
                &gpu,
                model.lora_sites(),
                model.gguf,
                (mc.lin_k_heads, mc.lin_v_heads),
                slots,
                cfg.max_lora_rank,
                meta.max_tokens,
            )?;
            model.attach_lora(&l);
            eprintln!(
                "adapters: {slots} GPU slots up to rank {} ({} MB)",
                l.max_rank,
                l.bytes() >> 20
            );
            Some(l)
        } else {
            None
        };
        // One extra slot absorbs the padding rows of graph replays; the
        // prefix cache's snapshots follow it.
        let snaps = snapshots(&cfg, cfg.max_seqs);
        let slots = cfg.max_seqs + 1 + snaps;
        let (free, total) = gpu.mem_info()?;
        let reserve = (total as f64 * (1.0 - cfg.memory_fraction)) as usize;
        let usable = free
            .saturating_sub(reserve)
            .saturating_sub(slots * mc.slot_bytes(cfg.state_bf16));
        let blocks = (usable / mc.block_bytes(cfg.kv_fp8)).min((cfg.max_seqs + snaps) * max_blocks);
        if blocks < max_blocks {
            return Err(Error::Config(format!(
                "not enough memory for one full context: {blocks} KV blocks, {max_blocks} needed"
            )));
        }
        let state = State::new(&gpu, &mc, slots, blocks, cfg.kv_fp8, cfg.state_bf16)?;
        gpu.sync()?;
        let mut sched = Scheduler::with_images(
            meta,
            cfg.max_context,
            blocks,
            mc.eos.clone(),
            mc.image_token,
            pool_rows,
        );
        // An H100 does about 2.5 times the matmul work per byte read of an
        // RTX 5090, so verification stays weight-bound to more rows.
        if gpu.arch == 90 {
            sched = sched.with_verify_rows(256);
        }
        if snaps > 0 {
            sched =
                sched.with_prefix_cache(cfg.max_seqs + 1..cfg.max_seqs + 1 + snaps, mc.turn_token);
        }
        if let Some(l) = &lora {
            sched = sched.with_lora_slots(l.max_slots);
        }
        let mut e = Self {
            gpu,
            model,
            state,
            act,
            graphs: HashMap::new(),
            verify_graphs: HashMap::new(),
            lora_graphs: HashMap::new(),
            lora_verify_graphs: HashMap::new(),
            lora,
            adapters: Arc::default(),
            vision,
            sched,
            stats: Stats::default(),
            cfg,
        };
        for (name, path) in e.cfg.loras.clone() {
            e.load_adapter(&name, &path)?;
        }
        let t_alloc = started.elapsed().as_secs_f64();
        if e.cfg.graphs {
            e.capture()?;
        }
        eprintln!(
            "ready in {:.2}s: GPU {:.2}s, model {:.2}s ({:.1} GB), vision and buffers {:.2}s, graphs {:.2}s",
            started.elapsed().as_secs_f64(),
            t_gpu,
            t_model - t_wait,
            e.model.weight_bytes as f64 / 1e9,
            t_alloc - t_model,
            started.elapsed().as_secs_f64() - t_alloc
        );
        Ok(e)
    }

    pub fn info(&self) -> Info {
        Info {
            device: self.gpu.name.clone(),
            vision: self.vision.is_some(),
            cfg: self.model.cfg.clone(),
            max_seqs: self.cfg.max_seqs,
            kv_blocks: self.state.blocks,
            max_context: self.cfg.max_context,
        }
    }

    fn capture(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let max = self.cfg.max_seqs;
        let spec = self.act.meta.spec;
        let shapes = if spec > 1 { vec![1, spec] } else { vec![1] };
        let lora_sets = if self.lora.is_some() {
            vec![false, true]
        } else {
            vec![false]
        };
        for with_lora in lora_sets {
            for &n in &shapes {
                for &b in BUCKETS.iter().filter(|&&b| b <= max) {
                    self.sched.pad(self.act.staging.as_mut_slice(), 0, b, n);
                    self.act.upload_meta(&self.gpu)?;
                    let t = b * n;
                    // A sequence's tokens fit one tile, so b bounds the tiles.
                    let lora = match (&self.lora, with_lora) {
                        (Some(l), true) => Some(self.act.lora_step(l, b)),
                        _ => None,
                    };
                    // A plain run first, so cuBLAS settles its kernels outside capture.
                    self.model
                        .forward(&self.gpu, &self.state, &self.act, t, b, t, lora)?;
                    self.gpu.sync()?;
                    self.gpu.begin_capture()?;
                    let r = self
                        .model
                        .forward(&self.gpu, &self.state, &self.act, t, b, t, lora);
                    let g = self.gpu.end_capture();
                    r?;
                    let set = match (with_lora, n == 1) {
                        (false, true) => &mut self.graphs,
                        (false, false) => &mut self.verify_graphs,
                        (true, true) => &mut self.lora_graphs,
                        (true, false) => &mut self.lora_verify_graphs,
                    };
                    set.insert(b, g?);
                }
            }
        }
        self.gpu.sync()?;
        eprintln!(
            "captured {} decode graphs in {:.2}s",
            self.graphs.len()
                + self.verify_graphs.len()
                + self.lora_graphs.len()
                + self.lora_verify_graphs.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Loads an adapter and makes it routable by name; returns its id.
    pub fn load_adapter(&mut self, name: &str, path: &std::path::Path) -> Result<u32> {
        let l = self.lora.as_mut().ok_or_else(|| {
            Error::Config("adapters are off; start with --max-loras or --lora".into())
        })?;
        let id = l.load(&self.gpu, name, path)?;
        self.adapters.lock().unwrap().insert(name.to_string(), id);
        eprintln!(
            "adapter {name}: rank {} from {}",
            l.adapter(id).map_or(0, |a| a.rank),
            path.display()
        );
        Ok(id)
    }

    /// Unloads an adapter no request is using.
    pub fn unload_adapter(&mut self, name: &str) -> Result<()> {
        let l = self
            .lora
            .as_mut()
            .ok_or_else(|| Error::Config(format!("no adapter named {name}")))?;
        let id = l
            .id(name)
            .ok_or_else(|| Error::Config(format!("no adapter named {name}")))?;
        if self.sched.uses_adapter(id) {
            return Err(Error::Config(format!("adapter {name} is in use")));
        }
        l.unload(name);
        self.sched.forget_adapter(id);
        self.adapters.lock().unwrap().remove(name);
        Ok(())
    }

    /// Queues a request, answering at once if its adapter is gone.
    pub fn submit(&mut self, r: Request) {
        if r.adapter != 0
            && self
                .lora
                .as_ref()
                .is_none_or(|l| l.adapter(r.adapter).is_none())
        {
            let _ = r.tx.send(crate::sched::Event::Error(
                "the adapter was unloaded".into(),
            ));
            return;
        }
        self.sched.submit(r);
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Request(r) => self.submit(r),
            Msg::LoadAdapter { name, path, reply } => {
                let _ = reply.send(
                    self.load_adapter(&name, &path)
                        .map(|_| ())
                        .map_err(|e| e.to_string()),
                );
            }
            Msg::UnloadAdapter { name, reply } => {
                let _ = reply.send(self.unload_adapter(&name).map_err(|e| e.to_string()));
            }
        }
    }

    /// Runs one step if there is work. Returns sequences that finished.
    pub fn step(&mut self) -> Result<Vec<Seq>> {
        let Some(step) = self.sched.plan() else {
            return Ok(Vec::new());
        };
        self.encode_images()?;
        for &(from, to) in &step.copies {
            self.state.copy_slot(&self.gpu, from, to)?;
        }
        if let Some(l) = self.lora.as_mut() {
            for &(slot, id) in &step.lora_loads {
                l.upload(&self.gpu, slot, id)?;
            }
        }
        let tiles = step.lora_tiles;
        let (t, s, r) = (step.tokens, step.seqs(), step.outs);
        let (pens, masks) = self.sched.fill(
            &step,
            self.act.staging.as_mut_slice(),
            &mut Pen::split(self.act.pen_staging.as_mut_slice()),
            self.act.mask_staging.as_mut_slice(),
        );
        self.act.upload_masks(&self.gpu, masks)?;
        let lp_rows = self.sched.logprob_rows(&step);
        let spec = self.act.meta.spec;
        let (graphs, n) = match (step.verify_only(spec), tiles > 0) {
            (true, false) => (&self.verify_graphs, spec),
            (false, false) => (&self.graphs, 1),
            (true, true) => (&self.lora_verify_graphs, spec),
            (false, true) => (&self.lora_graphs, 1),
        };
        let bucket = if (step.decode_only() || n > 1) && self.cfg.graphs {
            BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= s && graphs.contains_key(&b))
        } else {
            None
        };
        let t0 = Instant::now();
        match bucket {
            Some(b) => {
                self.sched.pad(self.act.staging.as_mut_slice(), s, b, n);
                self.act.upload_pen(&self.gpu, pens)?;
                self.act.upload_meta(&self.gpu)?;
                self.gpu.replay(&graphs[&b])?;
                self.stats.graph_steps += 1;
            }
            None => {
                self.act.upload_pen(&self.gpu, pens)?;
                self.act.upload_meta(&self.gpu)?;
                let lora = match &self.lora {
                    Some(l) if tiles > 0 => Some(self.act.lora_step(l, tiles)),
                    _ => None,
                };
                self.model
                    .forward(&self.gpu, &self.state, &self.act, t, s, r, lora)?;
            }
        }
        if !lp_rows.is_empty() {
            self.model.logprobs(&self.gpu, &mut self.act, &lp_rows)?;
        }
        self.act.download_ids(&self.gpu, r, s)?;
        self.gpu.sync()?;
        self.stats.step_seconds += t0.elapsed().as_secs_f64();
        let picked: Vec<u32> = self.act.ids_host.as_slice::<i32>()[..r]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let mut scored = Vec::with_capacity(s);
        for si in 0..s {
            let row = step.out_row(si);
            scored.push(
                if self.sched.is_forced(&step, si) && self.sched.yields(&step, si) {
                    Some(score(picked[row], &self.logits_row(row)?))
                } else {
                    None
                },
            );
        }
        let mut lps = self.act.logprobs(&lp_rows, r);
        let drafts = self.act.drafts(s);
        Ok(self.sched.finish(step, &picked, drafts, scored, &mut lps))
    }

    /// Runs the vision encoder for newly admitted requests, in batches the
    /// encoder's scratch holds, writing into their pool rows.
    fn encode_images(&mut self) -> Result<()> {
        let jobs = self.sched.take_vision();
        let Some(vision) = self.vision.as_mut() else {
            return Ok(());
        };
        let row_bytes = (self.model.cfg.hidden * 2) as u64;
        for job in jobs {
            let mut row = job.row;
            let mut batch: Vec<&Image> = Vec::new();
            let mut patches = 0;
            let mut batch_row = row;
            for img in &job.images {
                if patches + img.patches() > vision.max_patches() && !batch.is_empty() {
                    vision.encode(
                        &self.gpu,
                        &batch,
                        self.act.pool.ptr + batch_row as u64 * row_bytes,
                    )?;
                    batch.clear();
                    patches = 0;
                    batch_row = row;
                }
                batch.push(img);
                patches += img.patches();
                row += img.tokens();
            }
            if !batch.is_empty() {
                vision.encode(
                    &self.gpu,
                    &batch,
                    self.act.pool.ptr + batch_row as u64 * row_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        let vocab = self.model.cfg.vocab;
        let mut host = self.gpu.pinned(vocab * 4)?;
        self.gpu.dtoh(
            &mut host,
            self.act.logits.ptr + (row * vocab * 4) as u64,
            vocab * 4,
        )?;
        self.gpu.sync()?;
        Ok(host.as_slice::<f32>().to_vec())
    }

    /// Teacher-forced scoring of several prompts and continuations, batched
    /// together as a server would run them.
    pub fn score(&mut self, items: Vec<Scoring>) -> Result<Vec<Vec<Scored>>> {
        let n = items.len();
        let mut txs = Vec::new();
        for (i, item) in items.into_iter().enumerate() {
            let (tx, _) = tokio::sync::mpsc::unbounded_channel();
            txs.push(tx.clone());
            let sampling = Sampling {
                seed: i as u64,
                ..Sampling::default()
            };
            let mut req = Request::text(item.prompt, item.forced.len(), sampling, tx);
            req.images = item.images;
            req.rope = item.rope;
            req.adapter = item.adapter;
            self.sched.submit_forced(req, item.forced);
        }
        let mut out: Vec<Vec<Scored>> = vec![Vec::new(); n];
        // Sequences carry no id; match them back by their channel.
        while !self.sched.idle() {
            for s in self.step()? {
                let i = txs
                    .iter()
                    .position(|t| t.same_channel(&s.req.tx))
                    .expect("known channel");
                out[i] = s.scored;
            }
        }
        Ok(out)
    }
}

fn score(argmax: u32, logits: &[f32]) -> Scored {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max
        + logits
            .iter()
            .map(|&x| ((x - max) as f64).exp())
            .sum::<f64>()
            .ln() as f32;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(5, |&a, &b| logits[b].total_cmp(&logits[a]));
    let mut top: Vec<(u32, f32)> = idx[..5]
        .iter()
        .map(|&i| (i as u32, logits[i] - lse))
        .collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    Scored { argmax, top }
}

/// Runs the engine on its own thread. Returns once the model is loaded, with
/// the request channel and the shared health report.
/// What `spawn` hands the server: the message channel, the model's
/// description, the shared health report and the routable adapter names.
pub type Spawned = (mpsc::Sender<Msg>, Info, Arc<Mutex<Health>>, Adapters);

pub fn spawn(cfg: EngineConfig) -> Result<Spawned> {
    let (tx, rx) = mpsc::channel::<Msg>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let health = Arc::new(Mutex::new(Health::default()));
    let shared = health.clone();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut engine = match Engine::new(cfg) {
                Ok(e) => {
                    let _ = ready_tx.send(Ok((e.info(), e.adapters.clone())));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let publish = |e: &Engine, failed: Option<String>| {
                let mut h = shared.lock().unwrap();
                h.load = e.sched.load();
                if failed.is_some() {
                    h.failed = failed;
                }
            };
            publish(&engine, None);
            let mut last_report = Instant::now();
            loop {
                if engine.sched.idle() {
                    publish(&engine, None);
                    match rx.recv() {
                        Ok(m) => engine.handle(m),
                        Err(_) => return,
                    }
                }
                while let Ok(m) = rx.try_recv() {
                    engine.handle(m);
                }
                if let Err(e) = engine.step() {
                    eprintln!("engine step failed: {e}");
                    engine.sched.fail_all(&e.to_string());
                    publish(&engine, Some(e.to_string()));
                    // Answer later requests with the failure instead of hanging.
                    while let Ok(m) = rx.recv() {
                        match m {
                            Msg::Request(r) => {
                                let _ = r.tx.send(crate::sched::Event::Error(e.to_string()));
                            }
                            Msg::LoadAdapter { reply, .. } | Msg::UnloadAdapter { reply, .. } => {
                                let _ = reply.send(Err(e.to_string()));
                            }
                        }
                    }
                    return;
                }
                publish(&engine, None);
                if last_report.elapsed().as_secs() >= 10 {
                    let c = &engine.sched.counters;
                    let st = &engine.stats;
                    let load = engine.sched.load();
                    let drafts = if c.drafted > 0 {
                        format!(
                            ", drafts {:.0}% accepted",
                            100.0 * c.accepted as f64 / c.drafted as f64
                        )
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "steps {} (graph {}), decode {} tok, prompt {} tok (+{} cached){drafts}, {:.1} ms/step, running {}, waiting {}",
                        c.steps,
                        st.graph_steps,
                        c.decode_tokens,
                        c.prompt_tokens,
                        c.cached_tokens,
                        1e3 * st.step_seconds / c.steps.max(1) as f64,
                        load.running,
                        load.waiting
                    );
                    last_report = Instant::now();
                }
            }
        })?;
    let (info, adapters) = ready_rx
        .recv()
        .map_err(|_| Error::Config("engine thread exited".into()))??;
    Ok((tx, info, health, adapters))
}
