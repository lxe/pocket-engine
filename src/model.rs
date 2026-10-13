//! Qwen3.5 text decoder: configuration, fused weights, per-sequence state and
//! one forward step over a flat token batch.

use crate::gguf::Gguf;
use crate::gpu::{Arg, Buf, Gpu, Kernel, Pinned};
use crate::linear::{Kind, Linear, Part, Quants};
use crate::source::Source;
use crate::{Error, Result};
use std::path::Path;
use std::rc::Rc;

pub const GDN_DIM: usize = 128;
pub const HEAD_DIM: usize = 256;
pub const ROPE_DIM: usize = 64;
pub const BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub inter: usize,
    pub vocab: usize,
    pub hq: usize,
    pub hkv: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub attn_layers: Vec<bool>,
    pub eos: Vec<u32>,
    /// The image placeholder token.
    pub image_token: u32,
    /// Rotary frequencies per axis (time, height, width), interleaved.
    pub mrope_section: [usize; 3],
    /// Linear-attention value heads in GGUF's tiled order (see Source).
    pub tiled_v: bool,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(dir.join("config.json"))?)
    }

    /// Reads a Hugging Face config.json, rejecting shapes the kernels lack.
    pub fn parse(text: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let t = raw.get("text_config").unwrap_or(&raw);
        let u = |k: &str| -> Result<usize> {
            t[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| Error::Config(format!("config lacks {k}")))
        };
        let layers = u("num_hidden_layers")?;
        let attn_layers = match t["layer_types"].as_array() {
            Some(types) => types
                .iter()
                .map(|v| v.as_str() == Some("full_attention"))
                .collect(),
            None => {
                let every = u("full_attention_interval")?;
                (0..layers).map(|i| (i + 1) % every == 0).collect()
            }
        };
        let rope = &t["rope_parameters"];
        let cfg = Self {
            hidden: u("hidden_size")?,
            layers,
            inter: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            hq: u("num_attention_heads")?,
            hkv: u("num_key_value_heads")?,
            lin_k_heads: u("linear_num_key_heads")?,
            lin_v_heads: u("linear_num_value_heads")?,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: rope["rope_theta"]
                .as_f64()
                .or(t["rope_theta"].as_f64())
                .unwrap_or(1e7),
            attn_layers,
            // <|endoftext|> and <|im_end|>: chat turns end with the latter.
            eos: vec![t["eos_token_id"].as_u64().unwrap_or(248044) as u32, 248046],
            image_token: raw["image_token_id"].as_u64().unwrap_or(248056) as u32,
            mrope_section: match rope["mrope_section"].as_array() {
                Some(a) if a.len() == 3 => {
                    let v: Vec<usize> =
                        a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                    [v[0], v[1], v[2]]
                }
                _ => [11, 11, 10],
            },
            tiled_v: false,
        };
        let check = |ok: bool, what: &str| {
            if ok {
                Ok(())
            } else {
                Err(Error::Config(what.into()))
            }
        };
        check(u("head_dim")? == HEAD_DIM, "attention head_dim must be 256")?;
        check(
            u("linear_key_head_dim")? == GDN_DIM && u("linear_value_head_dim")? == GDN_DIM,
            "linear head dims must be 128",
        )?;
        check(
            cfg.hq.is_multiple_of(cfg.hkv) && matches!(cfg.hq / cfg.hkv, 1 | 2 | 4 | 6 | 8),
            "query heads per kv head must be 1, 2, 4, 6 or 8",
        )?;
        check(
            cfg.lin_v_heads.is_multiple_of(cfg.lin_k_heads),
            "value heads must be a multiple of key heads",
        )?;
        check(
            cfg.hidden.is_multiple_of(8) && cfg.hidden <= 8192 && cfg.inter.is_multiple_of(8),
            "unsupported widths",
        )?;
        let partial = rope["partial_rotary_factor"].as_f64().unwrap_or(0.25);
        check(
            (HEAD_DIM as f64 * partial) as usize == ROPE_DIM,
            "rotary dim must be 64",
        )?;
        check(
            t["attn_output_gate"].as_bool().unwrap_or(true),
            "expected gated attention",
        )?;
        Ok(cfg)
    }
    /// The configuration of a model directory or GGUF file.
    pub fn for_model(path: &Path) -> Result<Self> {
        match crate::source::gguf_path(path) {
            Some(g) => Self::from_gguf(&Gguf::open(&g)?),
            None => Self::load(path),
        }
    }

    /// Reads a GGUF file's metadata (architecture qwen35).
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        if arch != "qwen35" {
            return Err(Error::Config(format!(
                "unsupported GGUF architecture {arch}"
            )));
        }
        let u = |k: &str| -> Result<usize> {
            g.get(&format!("qwen35.{k}"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| Error::Config(format!("GGUF lacks qwen35.{k}")))
        };
        let tokens = g
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .unwrap_or(&[]);
        let find = |t: &str| {
            tokens
                .iter()
                .position(|v| v.as_str() == Some(t))
                .map(|i| i as u32)
        };
        // Multi-token-prediction layers (nextn) follow the decoder layers.
        let layers = u("block_count")? - u("nextn_predict_layers").unwrap_or(0);
        let every = u("full_attention_interval").unwrap_or(4);
        let vocab = g
            .tensors
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .ok_or_else(|| Error::Config("GGUF lacks token_embd.weight".into()))?;
        let sections: Vec<usize> = g
            .get("qwen35.rope.dimension_sections")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default();
        let mut eos: Vec<u32> = [find("<|im_end|>"), find("<|endoftext|>")]
            .into_iter()
            .flatten()
            .collect();
        if let Some(e) = g
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            && !eos.contains(&(e as u32))
        {
            eos.push(e as u32);
        }
        let cfg = Self {
            hidden: u("embedding_length")?,
            layers,
            inter: u("feed_forward_length")?,
            vocab,
            hq: u("attention.head_count")?,
            hkv: u("attention.head_count_kv")?,
            lin_k_heads: u("ssm.group_count")?,
            lin_v_heads: u("ssm.time_step_rank")?,
            eps: g
                .get("qwen35.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            rope_theta: g
                .get("qwen35.rope.freq_base")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e7),
            attn_layers: (0..layers).map(|i| (i + 1) % every == 0).collect(),
            eos,
            image_token: find("<|image_pad|>").unwrap_or(248056),
            mrope_section: if sections.len() >= 3 {
                [sections[0], sections[1], sections[2]]
            } else {
                [11, 11, 10]
            },
            tiled_v: true,
        };
        let ok = u("attention.key_length")? == HEAD_DIM
            && u("ssm.state_size")? == GDN_DIM
            && u("ssm.inner_size")? == cfg.lin_v_heads * GDN_DIM
            && u("rope.dimension_count")? == ROPE_DIM
            && u("ssm.conv_kernel")? == 4
            && cfg.hq.is_multiple_of(cfg.hkv)
            && matches!(cfg.hq / cfg.hkv, 1 | 2 | 4 | 6 | 8)
            && cfg.lin_v_heads.is_multiple_of(cfg.lin_k_heads)
            && cfg.hidden.is_multiple_of(256)
            && cfg.inter.is_multiple_of(256);
        if !ok {
            return Err(Error::Config("unsupported qwen35 GGUF shape".into()));
        }
        Ok(cfg)
    }

    /// Convolution channels: q, k (key heads) and v (value heads).
    pub fn conv_dim(&self) -> usize {
        (2 * self.lin_k_heads + self.lin_v_heads) * GDN_DIM
    }
    /// Fused linear-attention input projection width: [qkv | z | b | a].
    pub fn gdn_proj(&self) -> usize {
        self.conv_dim() + self.lin_v_heads * GDN_DIM + 2 * self.lin_v_heads
    }
    /// Fused attention input projection width: [q+gate | k | v].
    pub fn attn_proj(&self) -> usize {
        (2 * self.hq + 2 * self.hkv) * HEAD_DIM
    }
    pub fn gdn_layers(&self) -> usize {
        self.attn_layers.iter().filter(|a| !**a).count()
    }
    pub fn kv_layers(&self) -> usize {
        self.attn_layers.iter().filter(|a| **a).count()
    }
    /// Recurrent and convolution state bytes per sequence slot.
    pub fn slot_bytes(&self) -> usize {
        self.gdn_layers() * (self.lin_v_heads * GDN_DIM * GDN_DIM * 4 + 3 * self.conv_dim() * 2)
    }
    /// KV cache bytes per block of BLOCK_SIZE tokens.
    pub fn block_bytes(&self) -> usize {
        self.kv_layers() * 2 * BLOCK_SIZE * self.hkv * HEAD_DIM * 2
    }
}

enum Mixer {
    Gdn {
        w_in: Linear,
        conv_w: Buf,
        neg_a: Buf,
        dt_bias: Buf,
        norm_w: Buf,
        w_out: Linear,
        state: usize,
    },
    Attn {
        w_qkv: Linear,
        q_norm: Buf,
        k_norm: Buf,
        w_o: Linear,
        cache: usize,
    },
}

struct Layer {
    mixer: Mixer,
    in_norm: Buf,
    post_norm: Buf,
    w_gu: Linear,
    w_down: Linear,
}

struct Kernels {
    embed: Kernel,
    rmsnorm: Kernel,
    silu_mul: Kernel,
    gdn_conv: Kernel,
    gdn_recurrent: Kernel,
    attn_prep: Kernel,
    attn: Kernel,
    penalize: Kernel,
    sample: Kernel,
}

pub struct Model {
    pub cfg: Config,
    embed: Rc<Buf>,
    lm_head: Linear,
    final_norm: Buf,
    inv_freq: Buf,
    layers: Vec<Layer>,
    k: Kernels,
    quants: Quants,
    /// Bytes of weights on the device.
    pub weight_bytes: usize,
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Model {
    /// Loads a Hugging Face directory or a GGUF file (or a directory holding one).
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self> {
        let started = gpu.begin_uploads()?;
        let m = Self::load_weights(gpu, path);
        if started {
            gpu.end_uploads()?;
        }
        m
    }

    fn load_weights(gpu: &Gpu, path: &Path) -> Result<Self> {
        let src = Source::open(path)?;
        let mut cfg = match &src {
            Source::Gguf(g) => Config::from_gguf(g)?,
            Source::Safetensors(_) => Config::load(path)?,
        };
        cfg.tiled_v = src.tiled_value_heads() && cfg.lin_v_heads != cfg.lin_k_heads;
        let mut quants = Quants::default();
        let (h, i) = (cfg.hidden, cfg.inter);
        let up = |v: &[f32]| gpu.upload(&f32_bytes(v));
        let mut layers = Vec::with_capacity(cfg.layers);
        let (mut n_state, mut n_cache) = (0, 0);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let mixer = if cfg.attn_layers[l] {
                let a = format!("{p}self_attn.");
                let m = Mixer::Attn {
                    w_qkv: src.linear(
                        gpu,
                        &mut quants,
                        &[
                            (&format!("{a}q_proj.weight"), 2 * cfg.hq * HEAD_DIM),
                            (&format!("{a}k_proj.weight"), cfg.hkv * HEAD_DIM),
                            (&format!("{a}v_proj.weight"), cfg.hkv * HEAD_DIM),
                        ],
                        h,
                    )?,
                    q_norm: up(&src.norm(&format!("{a}q_norm.weight"), HEAD_DIM)?)?,
                    k_norm: up(&src.norm(&format!("{a}k_norm.weight"), HEAD_DIM)?)?,
                    w_o: src.linear(
                        gpu,
                        &mut quants,
                        &[(&format!("{a}o_proj.weight"), h)],
                        cfg.hq * HEAD_DIM,
                    )?,
                    cache: n_cache,
                };
                n_cache += 1;
                m
            } else {
                let a = format!("{p}linear_attn.");
                let hv = cfg.lin_v_heads;
                let m = Mixer::Gdn {
                    w_in: src.linear(
                        gpu,
                        &mut quants,
                        &[
                            (&format!("{a}in_proj_qkv.weight"), cfg.conv_dim()),
                            (&format!("{a}in_proj_z.weight"), hv * GDN_DIM),
                            (&format!("{a}in_proj_b.weight"), hv),
                            (&format!("{a}in_proj_a.weight"), hv),
                        ],
                        h,
                    )?,
                    conv_w: up(&src.vec_f32(&format!("{a}conv1d.weight"), cfg.conv_dim() * 4)?)?,
                    neg_a: up(&src.neg_a(l, hv)?)?,
                    dt_bias: up(&src.vec_f32(&format!("{a}dt_bias"), hv)?)?,
                    // The gated norm uses its weight directly.
                    norm_w: up(&src.vec_f32(&format!("{a}norm.weight"), GDN_DIM)?)?,
                    w_out: src.linear(
                        gpu,
                        &mut quants,
                        &[(&format!("{a}out_proj.weight"), h)],
                        hv * GDN_DIM,
                    )?,
                    state: n_state,
                };
                n_state += 1;
                m
            };
            layers.push(Layer {
                mixer,
                in_norm: up(&src.norm(&format!("{p}input_layernorm.weight"), h)?)?,
                post_norm: up(&src.norm(&format!("{p}post_attention_layernorm.weight"), h)?)?,
                w_gu: src.linear(
                    gpu,
                    &mut quants,
                    &[
                        (&format!("{p}mlp.gate_proj.weight"), i),
                        (&format!("{p}mlp.up_proj.weight"), i),
                    ],
                    h,
                )?,
                w_down: src.linear(
                    gpu,
                    &mut quants,
                    &[(&format!("{p}mlp.down_proj.weight"), h)],
                    i,
                )?,
            });
        }
        let (embed, embed_kind) =
            src.shared(gpu, &mut quants, "embed_tokens.weight", cfg.vocab, h)?;
        // The output layer is its own tensor, or tied to the embedding.
        let lm_head = if src.has("/lm_head.weight") {
            src.linear(gpu, &mut quants, &[("/lm_head.weight", cfg.vocab)], h)?
        } else {
            Linear {
                parts: vec![Part {
                    buf: embed.clone(),
                    kind: embed_kind,
                    rows: cfg.vocab,
                    off: 0,
                }],
                cols: h,
            }
        };
        let inv_freq: Vec<f32> = (0..ROPE_DIM / 2)
            .map(|d| 1f32 / cfg.rope_theta.powf((2 * d) as f64 / ROPE_DIM as f64) as f32)
            .collect();
        let embed_kernel = match embed_kind {
            Kind::Bf16 => gpu.kernel("embed")?,
            Kind::Quant(ty) => quants.get(gpu, ty)?.embed,
        };
        let k = Kernels {
            embed: embed_kernel,
            rmsnorm: gpu.kernel("rmsnorm")?,
            silu_mul: gpu.kernel("silu_mul")?,
            gdn_conv: gpu.kernel("gdn_conv")?,
            gdn_recurrent: gpu.kernel("gdn_recurrent")?,
            attn_prep: gpu.kernel("attn_prep")?,
            attn: gpu.kernel(&format!("attn_paged_g{}", cfg.hq / cfg.hkv))?,
            penalize: gpu.kernel("penalize")?,
            sample: gpu.kernel("sample")?,
        };
        let largest = layers
            .iter()
            .flat_map(|l| {
                let mut v = vec![&l.w_gu, &l.w_down];
                match &l.mixer {
                    Mixer::Gdn { w_in, w_out, .. } => v.extend([w_in, w_out]),
                    Mixer::Attn { w_qkv, w_o, .. } => v.extend([w_qkv, w_o]),
                }
                v
            })
            .chain(std::iter::once(&lm_head))
            .flat_map(|l| l.parts.iter().map(move |p| p.rows * l.cols))
            .max()
            .unwrap_or(0);
        quants.prepare(gpu, largest)?;
        let mut weight_bytes = embed.bytes;
        for l in &layers {
            weight_bytes += l.w_gu.bytes() + l.w_down.bytes();
            weight_bytes += match &l.mixer {
                Mixer::Gdn { w_in, w_out, .. } => w_in.bytes() + w_out.bytes(),
                Mixer::Attn { w_qkv, w_o, .. } => w_qkv.bytes() + w_o.bytes(),
            };
        }
        if src.has("/lm_head.weight") {
            weight_bytes += lm_head.bytes();
        }
        Ok(Self {
            final_norm: up(&src.norm("norm.weight", h)?)?,
            inv_freq: up(&inv_freq)?,
            embed,
            lm_head,
            layers,
            k,
            quants,
            weight_bytes,
            cfg,
        })
    }
}

/// Per-sequence recurrent state slots and the paged KV pool.
pub struct State {
    conv: Vec<Buf>,
    rec: Vec<Buf>,
    kc: Vec<Buf>,
    vc: Vec<Buf>,
    pub slots: usize,
    pub blocks: usize,
}

impl State {
    pub fn new(gpu: &Gpu, cfg: &Config, slots: usize, blocks: usize) -> Result<Self> {
        let conv_bytes = slots * 3 * cfg.conv_dim() * 2;
        let rec_bytes = slots * cfg.lin_v_heads * GDN_DIM * GDN_DIM * 4;
        let kv_bytes = blocks * BLOCK_SIZE * cfg.hkv * HEAD_DIM * 2;
        let mut s = Self {
            conv: vec![],
            rec: vec![],
            kc: vec![],
            vc: vec![],
            slots,
            blocks,
        };
        for _ in 0..cfg.gdn_layers() {
            s.conv.push(gpu.alloc_zeroed(conv_bytes)?);
            s.rec.push(gpu.alloc_zeroed(rec_bytes)?);
        }
        for _ in 0..cfg.kv_layers() {
            s.kc.push(gpu.alloc_zeroed(kv_bytes)?);
            s.vc.push(gpu.alloc_zeroed(kv_bytes)?);
        }
        Ok(s)
    }
}

/// Batch metadata layout (i32 words) in one device buffer, filled from a pinned
/// staging buffer with a single copy per step.
#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub max_blocks: usize,
}

impl Meta {
    pub fn ids(&self) -> usize {
        0
    }
    pub fn pos(&self) -> usize {
        self.max_tokens
    }
    pub fn slot_map(&self) -> usize {
        2 * self.max_tokens
    }
    pub fn tok_seq(&self) -> usize {
        3 * self.max_tokens
    }
    /// Image embedding pool row per token, or -1.
    pub fn emb_src(&self) -> usize {
        4 * self.max_tokens
    }
    /// Rotary positions: time, height, width rows of max_tokens each.
    pub fn rope(&self) -> usize {
        5 * self.max_tokens
    }
    pub fn cu(&self) -> usize {
        8 * self.max_tokens
    }
    pub fn slots(&self) -> usize {
        self.cu() + self.max_seqs + 1
    }
    pub fn fresh(&self) -> usize {
        self.slots() + self.max_seqs
    }
    pub fn last(&self) -> usize {
        self.fresh() + self.max_seqs
    }
    pub fn temps(&self) -> usize {
        self.last() + self.max_seqs
    }
    pub fn seeds(&self) -> usize {
        self.temps() + self.max_seqs
    }
    pub fn top_ks(&self) -> usize {
        self.seeds() + self.max_seqs
    }
    pub fn top_ps(&self) -> usize {
        self.top_ks() + self.max_seqs
    }
    pub fn min_ps(&self) -> usize {
        self.top_ps() + self.max_seqs
    }
    /// Offsets of each sequence's penalty entries (see [`Act::pen_capacity`]).
    pub fn pen_cu(&self) -> usize {
        self.min_ps() + self.max_seqs
    }
    pub fn block_table(&self) -> usize {
        self.pen_cu() + self.max_seqs + 1
    }
    pub fn words(&self) -> usize {
        self.block_table() + self.max_seqs * self.max_blocks
    }
}

/// Activation buffers sized for the largest step.
pub struct Act {
    pub meta: Meta,
    meta_dev: Buf,
    pub staging: Pinned,
    res: Buf,
    xn: Buf,
    h: Buf,
    proj: Buf,
    conv: Buf,
    core: Buf,
    q: Buf,
    ao: Buf,
    gu: Buf,
    act: Buf,
    fin: Buf,
    pub logits: Buf,
    /// Image embeddings awaiting their prompt step, `hidden` bf16 per row.
    pub pool: Buf,
    ids_out: Buf,
    pub ids_host: Pinned,
    /// Logit adjustments: [token i32 | add f32 | rep f32], each `pen_capacity` long.
    pen_dev: Buf,
    pub pen_staging: Pinned,
    pub pen_capacity: usize,
}

impl Act {
    /// `pool_rows` image embeddings can wait between the vision encoder and
    /// the prompt step that reads them.
    pub fn new(gpu: &Gpu, cfg: &Config, meta: Meta, pool_rows: usize) -> Result<Self> {
        let (t, s) = (meta.max_tokens, meta.max_seqs);
        let b = |n: usize| gpu.alloc_zeroed(n * 2);
        // Every token a sequence can hold, plus a logit_bias of up to 1,024.
        let pen_capacity = s * (meta.max_blocks * BLOCK_SIZE + 1024);
        Ok(Self {
            meta_dev: gpu.alloc_zeroed(meta.words() * 4)?,
            staging: gpu.pinned(meta.words() * 4)?,
            res: b(t * cfg.hidden)?,
            xn: b(t * cfg.hidden)?,
            h: b(t * cfg.hidden)?,
            proj: b(t * cfg.gdn_proj().max(cfg.attn_proj()))?,
            conv: b(t * cfg.conv_dim())?,
            core: b(t * cfg.lin_v_heads * GDN_DIM)?,
            q: b(t * cfg.hq * HEAD_DIM)?,
            ao: b(t * cfg.hq * HEAD_DIM)?,
            gu: b(t * 2 * cfg.inter)?,
            act: b(t * cfg.inter)?,
            fin: b(s * cfg.hidden)?,
            logits: gpu.alloc_zeroed(s * cfg.vocab * 4)?,
            pool: gpu.alloc_zeroed(pool_rows.max(1) * cfg.hidden * 2)?,
            ids_out: gpu.alloc_zeroed(s * 4)?,
            ids_host: gpu.pinned(s * 4)?,
            pen_dev: gpu.alloc_zeroed(3 * pen_capacity * 4)?,
            pen_staging: gpu.pinned(3 * pen_capacity * 4)?,
            pen_capacity,
            meta,
        })
    }

    fn m(&self, word: usize) -> u64 {
        self.meta_dev.ptr + 4 * word as u64
    }

    /// Copies the staged metadata to the device.
    pub fn upload_meta(&self, gpu: &Gpu) -> Result<()> {
        gpu.htod(self.meta_dev.ptr, &self.staging, 0, self.meta.words() * 4)
    }

    /// Copies the first `n` staged penalty entries to the device.
    pub fn upload_pen(&self, gpu: &Gpu, n: usize) -> Result<()> {
        let cap = 4 * self.pen_capacity;
        for region in 0..3 {
            let at = region * cap;
            gpu.htod(self.pen_dev.ptr + at as u64, &self.pen_staging, at, 4 * n)?;
        }
        Ok(())
    }

    /// Queues the copy of the picked tokens of the first `s` sequences.
    pub fn download_ids(&mut self, gpu: &Gpu, s: usize) -> Result<()> {
        gpu.dtoh(&mut self.ids_host, self.ids_out.ptr, s * 4)
    }
}

impl Model {
    /// Queues one step over `t` tokens from `s` sequences, as described by the
    /// uploaded metadata, ending with one picked token per sequence. Issues no
    /// host synchronization and no allocation, so it can be captured.
    pub fn forward(&self, gpu: &Gpu, st: &State, a: &Act, t: usize, s: usize) -> Result<()> {
        use Arg::{F, I, P};
        let c = &self.cfg;
        let k = &self.k;
        let q = &self.quants;
        let me = &a.meta;
        let (h, eps) = (c.hidden, c.eps);
        let (ti, si) = (t as u32, s as u32);
        gpu.launch(
            k.embed,
            (ti, 1, 1),
            256,
            0,
            &[
                P(a.m(me.ids())),
                P(self.embed.ptr),
                P(a.m(me.emb_src())),
                P(a.pool.ptr),
                P(a.res.ptr),
                I(h as i32),
            ],
        )?;
        let norm = |x: u64, w: &Buf, add: bool| -> Result<()> {
            gpu.launch(
                k.rmsnorm,
                (ti, 1, 1),
                256,
                0,
                &[
                    P(x),
                    P(a.res.ptr),
                    P(w.ptr),
                    P(a.xn.ptr),
                    P(0),
                    I(h as i32),
                    F(eps),
                    I(add as i32),
                    I(1),
                ],
            )
        };
        for (li, layer) in self.layers.iter().enumerate() {
            norm(a.h.ptr, &layer.in_norm, li > 0)?;
            match &layer.mixer {
                Mixer::Gdn {
                    w_in,
                    conv_w,
                    neg_a,
                    dt_bias,
                    norm_w,
                    w_out,
                    state,
                } => {
                    let (pw, cd, hv) = (c.gdn_proj(), c.conv_dim(), c.lin_v_heads);
                    w_in.forward(gpu, q, a.xn.ptr, t, a.proj.ptr, pw, false)?;
                    gpu.launch(
                        k.gdn_conv,
                        (cd.div_ceil(128) as u32, si, 1),
                        128,
                        0,
                        &[
                            P(a.proj.ptr),
                            I(pw as i32),
                            P(conv_w.ptr),
                            P(st.conv[*state].ptr),
                            P(a.conv.ptr),
                            P(a.m(me.cu())),
                            P(a.m(me.slots())),
                            P(a.m(me.fresh())),
                            I(cd as i32),
                        ],
                    )?;
                    gpu.launch(
                        k.gdn_recurrent,
                        (si, hv as u32, 1),
                        2 * GDN_DIM as u32,
                        0,
                        &[
                            P(a.conv.ptr),
                            P(a.proj.ptr),
                            I(pw as i32),
                            I(cd as i32),
                            I(c.lin_k_heads as i32),
                            I(hv as i32),
                            P(neg_a.ptr),
                            P(dt_bias.ptr),
                            P(norm_w.ptr),
                            P(st.rec[*state].ptr),
                            P(a.m(me.cu())),
                            P(a.m(me.slots())),
                            P(a.m(me.fresh())),
                            P(a.core.ptr),
                            F(eps),
                            I(c.tiled_v as i32),
                        ],
                    )?;
                    w_out.forward(gpu, q, a.core.ptr, t, a.h.ptr, h, false)?;
                }
                Mixer::Attn {
                    w_qkv,
                    q_norm,
                    k_norm,
                    w_o,
                    cache,
                } => {
                    let qs = c.attn_proj();
                    w_qkv.forward(gpu, q, a.xn.ptr, t, a.proj.ptr, qs, false)?;
                    gpu.launch(
                        k.attn_prep,
                        (ti, (c.hq + 2 * c.hkv) as u32, 1),
                        128,
                        0,
                        &[
                            P(a.proj.ptr),
                            I(qs as i32),
                            I(c.hq as i32),
                            I(c.hkv as i32),
                            P(q_norm.ptr),
                            P(k_norm.ptr),
                            P(a.m(me.rope())),
                            I(me.max_tokens as i32),
                            I(c.mrope_section[1] as i32),
                            I(c.mrope_section[2] as i32),
                            P(a.m(me.slot_map())),
                            P(self.inv_freq.ptr),
                            P(a.q.ptr),
                            P(st.kc[*cache].ptr),
                            P(st.vc[*cache].ptr),
                            F(eps),
                        ],
                    )?;
                    gpu.launch(
                        k.attn,
                        (ti, c.hkv as u32, 1),
                        if c.hq / c.hkv >= 6 { 128 } else { 256 },
                        0,
                        &[
                            P(a.q.ptr),
                            P(st.kc[*cache].ptr),
                            P(st.vc[*cache].ptr),
                            P(a.proj.ptr),
                            I(qs as i32),
                            I(c.hq as i32),
                            I(c.hkv as i32),
                            P(a.m(me.pos())),
                            P(a.m(me.tok_seq())),
                            P(a.m(me.block_table())),
                            I(me.max_blocks as i32),
                            I(BLOCK_SIZE as i32),
                            P(a.ao.ptr),
                            F(1.0 / (HEAD_DIM as f32).sqrt()),
                        ],
                    )?;
                    w_o.forward(gpu, q, a.ao.ptr, t, a.h.ptr, h, false)?;
                }
            }
            norm(a.h.ptr, &layer.post_norm, true)?;
            layer
                .w_gu
                .forward(gpu, q, a.xn.ptr, t, a.gu.ptr, 2 * c.inter, false)?;
            let blocks = (t * c.inter / 8).div_ceil(256).min(8192) as u32;
            gpu.launch(
                k.silu_mul,
                (blocks, 1, 1),
                256,
                0,
                &[P(a.gu.ptr), P(a.act.ptr), I(c.inter as i32), I(t as i32)],
            )?;
            layer
                .w_down
                .forward(gpu, q, a.act.ptr, t, a.h.ptr, h, false)?;
        }
        gpu.launch(
            k.rmsnorm,
            (si, 1, 1),
            256,
            0,
            &[
                P(a.h.ptr),
                P(a.res.ptr),
                P(self.final_norm.ptr),
                P(a.fin.ptr),
                P(a.m(me.last())),
                I(h as i32),
                F(eps),
                I(1),
                I(0),
            ],
        )?;
        self.lm_head
            .forward(gpu, q, a.fin.ptr, s, a.logits.ptr, c.vocab, true)?;
        let cap = (4 * a.pen_capacity) as u64;
        gpu.launch(
            k.penalize,
            (si, 1, 1),
            256,
            0,
            &[
                P(a.logits.ptr),
                I(c.vocab as i32),
                P(a.m(me.pen_cu())),
                P(a.pen_dev.ptr),
                P(a.pen_dev.ptr + cap),
                P(a.pen_dev.ptr + 2 * cap),
            ],
        )?;
        gpu.launch(
            k.sample,
            (si, 1, 1),
            1024,
            0,
            &[
                P(a.logits.ptr),
                I(c.vocab as i32),
                P(a.m(me.temps())),
                P(a.m(me.top_ks())),
                P(a.m(me.top_ps())),
                P(a.m(me.min_ps())),
                P(a.m(me.seeds())),
                P(a.ids_out.ptr),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Config {
        let path = format!(
            "{}/tests/fixtures/configs/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        Config::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn qwen35_sizes_parse_with_their_fused_widths() {
        let c = fixture("qwen3.5-2b");
        assert_eq!(
            (c.hidden, c.layers, c.inter, c.vocab),
            (2048, 24, 6144, 248320)
        );
        assert_eq!((c.gdn_layers(), c.kv_layers()), (18, 6));
        assert!(c.attn_layers[3] && !c.attn_layers[2]);
        assert_eq!(c.conv_dim(), 6144);
        assert_eq!(c.gdn_proj(), 6144 + 2048 + 32);
        assert_eq!(c.attn_proj(), 5120);
        assert_eq!(c.eos, vec![248044, 248046]);
        // 18 layers x (16 heads x 128 x 128 f32 + 3 x 6144 bf16).
        assert_eq!(c.slot_bytes(), 18 * (16 * 128 * 128 * 4 + 3 * 6144 * 2));
        let small = fixture("qwen3.5-0.8b");
        assert_eq!(small.hidden, 1024);
        // The 4B model has two value heads per key head.
        let big = fixture("qwen3.5-4b");
        assert_eq!((big.lin_k_heads, big.lin_v_heads, big.layers), (16, 32, 32));
        assert_eq!(big.conv_dim(), (2 * 16 + 32) * 128);
    }

    #[test]
    fn unsupported_shapes_are_rejected() {
        let path = format!(
            "{}/tests/fixtures/configs/qwen3.5-2b.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        v["text_config"]["head_dim"] = 128.into();
        assert!(Config::parse(&v.to_string()).is_err());
    }

    #[test]
    fn metadata_regions_do_not_overlap() {
        let m = Meta {
            max_tokens: 64,
            max_seqs: 8,
            max_blocks: 4,
        };
        let starts = [
            m.ids(),
            m.pos(),
            m.slot_map(),
            m.tok_seq(),
            m.emb_src(),
            m.rope(),
            m.cu(),
            m.slots(),
            m.fresh(),
            m.last(),
            m.temps(),
            m.seeds(),
            m.top_ks(),
            m.top_ps(),
            m.min_ps(),
            m.pen_cu(),
            m.block_table(),
            m.words(),
        ];
        assert!(starts.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(m.slots() - m.cu(), 9, "cu holds seqs + 1 offsets");
    }
}
