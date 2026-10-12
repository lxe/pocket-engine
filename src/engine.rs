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
}

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

impl Engine {
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let gpu = Gpu::new(cfg.device)?;
        if let Some(path) = &cfg.wait_for {
            wait_for(path);
        }
        let model = Model::load(&gpu, &cfg.model)?;
        let mc = model.cfg.clone();
        let vision = if cfg.vision {
            Vision::load(&gpu, &cfg.model, cfg.max_image_pixels / 256)?
        } else {
            None
        };
        let pool_rows = if vision.is_some() { cfg.image_pool } else { 0 };
        let max_blocks = cfg.max_context.div_ceil(BLOCK_SIZE);
        let meta = Meta {
            max_tokens: cfg.max_batch_tokens.max(cfg.max_seqs),
            max_seqs: cfg.max_seqs,
            max_blocks,
        };
        let act = Act::new(&gpu, &mc, meta, pool_rows)?;
        // One extra slot absorbs the padding rows of graph replays.
        let slots = cfg.max_seqs + 1;
        let (free, total) = gpu.mem_info()?;
        let reserve = (total as f64 * (1.0 - cfg.memory_fraction)) as usize;
        let usable = free
            .saturating_sub(reserve)
            .saturating_sub(slots * mc.slot_bytes());
        let blocks = (usable / mc.block_bytes()).min(cfg.max_seqs * max_blocks);
        if blocks < max_blocks {
            return Err(Error::Config(format!(
                "not enough memory for one full context: {blocks} KV blocks, {max_blocks} needed"
            )));
        }
        let state = State::new(&gpu, &mc, slots, blocks)?;
        gpu.sync()?;
        let sched = Scheduler::with_images(
            meta,
            cfg.max_context,
            blocks,
            mc.eos.clone(),
            mc.image_token,
            pool_rows,
        );
        let mut e = Self {
            gpu,
            model,
            state,
            act,
            graphs: HashMap::new(),
            vision,
            sched,
            stats: Stats::default(),
            cfg,
        };
        if e.cfg.graphs {
            e.capture()?;
        }
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
        for &b in BUCKETS.iter().filter(|&&b| b <= max) {
            self.sched.pad(self.act.staging.as_mut_slice(), 0, b);
            self.act.upload_meta(&self.gpu)?;
            // A plain run first, so cuBLAS settles its kernels outside capture.
            self.model
                .forward(&self.gpu, &self.state, &self.act, b, b)?;
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.model.forward(&self.gpu, &self.state, &self.act, b, b);
            let g = self.gpu.end_capture();
            r?;
            self.graphs.insert(b, g?);
        }
        self.gpu.sync()?;
        eprintln!(
            "captured {} decode graphs in {:.2}s",
            self.graphs.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Runs one step if there is work. Returns sequences that finished.
    pub fn step(&mut self) -> Result<Vec<Seq>> {
        let Some(step) = self.sched.plan() else {
            return Ok(Vec::new());
        };
        self.encode_images()?;
        let (t, s) = (step.tokens, step.seqs());
        let pens = self.sched.fill(
            &step,
            self.act.staging.as_mut_slice(),
            &mut Pen::split(self.act.pen_staging.as_mut_slice()),
        );
        let bucket = if step.decode_only() && self.cfg.graphs {
            BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= s && self.graphs.contains_key(&b))
        } else {
            None
        };
        let t0 = Instant::now();
        match bucket {
            Some(b) => {
                self.sched.pad(self.act.staging.as_mut_slice(), s, b);
                self.act.upload_pen(&self.gpu, pens)?;
                self.act.upload_meta(&self.gpu)?;
                self.gpu.replay(&self.graphs[&b])?;
                self.stats.graph_steps += 1;
            }
            None => {
                self.act.upload_pen(&self.gpu, pens)?;
                self.act.upload_meta(&self.gpu)?;
                self.model
                    .forward(&self.gpu, &self.state, &self.act, t, s)?;
            }
        }
        self.act.download_ids(&self.gpu, s)?;
        self.gpu.sync()?;
        self.stats.step_seconds += t0.elapsed().as_secs_f64();
        let picked: Vec<u32> = self.act.ids_host.as_slice::<i32>()[..s]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let mut scored = Vec::with_capacity(s);
        for (si, &pick) in picked.iter().enumerate() {
            scored.push(
                if self.sched.is_forced(&step, si) && self.sched.yields(&step, si) {
                    Some(score(pick, &self.logits_row(si)?))
                } else {
                    None
                },
            );
        }
        Ok(self.sched.finish(step, &picked, scored))
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
pub fn spawn(cfg: EngineConfig) -> Result<(mpsc::Sender<Request>, Info, Arc<Mutex<Health>>)> {
    let (tx, rx) = mpsc::channel::<Request>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let health = Arc::new(Mutex::new(Health::default()));
    let shared = health.clone();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut engine = match Engine::new(cfg) {
                Ok(e) => {
                    let _ = ready_tx.send(Ok(e.info()));
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
                        Ok(r) => engine.sched.submit(r),
                        Err(_) => return,
                    }
                }
                while let Ok(r) = rx.try_recv() {
                    engine.sched.submit(r);
                }
                if let Err(e) = engine.step() {
                    eprintln!("engine step failed: {e}");
                    engine.sched.fail_all(&e.to_string());
                    publish(&engine, Some(e.to_string()));
                    // Answer later requests with the failure instead of hanging.
                    while let Ok(r) = rx.recv() {
                        let _ = r.tx.send(crate::sched::Event::Error(e.to_string()));
                    }
                    return;
                }
                publish(&engine, None);
                if last_report.elapsed().as_secs() >= 10 {
                    let c = &engine.sched.counters;
                    let st = &engine.stats;
                    let load = engine.sched.load();
                    eprintln!(
                        "steps {} (graph {}), decode {} tok, prompt {} tok, {:.1} ms/step, running {}, waiting {}",
                        c.steps,
                        st.graph_steps,
                        c.decode_tokens,
                        c.prompt_tokens,
                        1e3 * st.step_seconds / c.steps.max(1) as f64,
                        load.running,
                        load.waiting
                    );
                    last_report = Instant::now();
                }
            }
        })?;
    let info = ready_rx
        .recv()
        .map_err(|_| Error::Config("engine thread exited".into()))??;
    Ok((tx, info, health))
}
