//! Qwen3.5 text decoder: configuration, fused weights, per-sequence state and
//! one forward step over a flat token batch.

use crate::gpu::{Arg, Buf, Gpu, Kernel, Pinned};
use crate::{Error, Result};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

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
            cfg.hq == 4 * cfg.hkv,
            "expected four query heads per kv head",
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
        w_in: Buf,
        conv_w: Buf,
        neg_a: Buf,
        dt_bias: Buf,
        norm_w: Buf,
        w_out: Buf,
        state: usize,
    },
    Attn {
        w_qkv: Buf,
        q_norm: Buf,
        k_norm: Buf,
        w_o: Buf,
        cache: usize,
    },
}

struct Layer {
    mixer: Mixer,
    in_norm: Buf,
    post_norm: Buf,
    w_gu: Buf,
    w_down: Buf,
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
    embed: Buf,
    final_norm: Buf,
    inv_freq: Buf,
    layers: Vec<Layer>,
    k: Kernels,
}

struct Checkpoint {
    maps: Vec<memmap2::Mmap>,
    index: HashMap<String, usize>,
    prefix: String,
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Config(format!(
                "no safetensors in {}",
                dir.display()
            )));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(dir.join(f))?;
            // SAFETY-free: memmap2::Mmap::map is unsafe only for concurrent file
            // mutation, which the model directory does not see while serving.
            let map = map_file(&file)?;
            let names: Vec<String> = SafeTensors::deserialize(&map)?
                .names()
                .into_iter()
                .map(String::from)
                .collect();
            for n in names {
                index.insert(n, i);
            }
            maps.push(map);
        }
        let prefix = if index.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        Ok(Self {
            maps,
            index,
            prefix: prefix.into(),
        })
    }

    fn view(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8])> {
        let full = format!("{}{name}", self.prefix);
        let i = *self
            .index
            .get(&full)
            .ok_or_else(|| Error::Config(format!("missing tensor {full}")))?;
        let st = SafeTensors::deserialize(&self.maps[i])?;
        let v = st.tensor(&full)?;
        let (dtype, shape) = (v.dtype(), v.shape().to_vec());
        let offset = v.data().as_ptr() as usize - self.maps[i].as_ptr() as usize;
        Ok((dtype, shape, &self.maps[i][offset..offset + v.data().len()]))
    }

    fn bf16(&self, name: &str, numel: usize) -> Result<Vec<u8>> {
        let (dtype, shape, data) = self.view(name)?;
        if shape.iter().product::<usize>() != numel {
            return Err(Error::Config(format!(
                "{name}: shape {shape:?}, expected {numel} elements"
            )));
        }
        match dtype {
            Dtype::BF16 => Ok(data.to_vec()),
            Dtype::F32 => Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| bf16::from_f32(f32::from_le_bytes(*c)).to_le_bytes())
                .collect()),
            d => Err(Error::Config(format!("{name}: unsupported dtype {d:?}"))),
        }
    }

    fn f32(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let (dtype, shape, data) = self.view(name)?;
        if shape.iter().product::<usize>() != numel {
            return Err(Error::Config(format!(
                "{name}: shape {shape:?}, expected {numel} elements"
            )));
        }
        match dtype {
            Dtype::BF16 => Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f32())
                .collect()),
            Dtype::F32 => Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()),
            d => Err(Error::Config(format!("{name}: unsupported dtype {d:?}"))),
        }
    }
}

#[allow(unsafe_code)]
fn map_file(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: the checkpoint is read-only while the engine runs.
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Model {
    pub fn load(gpu: &Gpu, dir: &Path) -> Result<Self> {
        let cfg = Config::load(dir)?;
        let ck = Checkpoint::open(dir)?;
        let (h, i) = (cfg.hidden, cfg.inter);
        // Zero-centred RMSNorm weights: the model multiplies by (1 + w).
        let norm = |name: &str, n: usize| -> Result<Buf> {
            let w: Vec<f32> = ck.f32(name, n)?.into_iter().map(|x| x + 1.0).collect();
            gpu.upload(&f32_bytes(&w))
        };
        let fused = |names: &[(&str, usize)]| -> Result<Buf> {
            let mut bytes = Vec::new();
            for (n, rows) in names {
                bytes.extend(ck.bf16(n, rows * h)?);
            }
            gpu.upload(&bytes)
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        let (mut n_state, mut n_cache) = (0, 0);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let mixer = if cfg.attn_layers[l] {
                let a = format!("{p}self_attn.");
                let m = Mixer::Attn {
                    w_qkv: fused(&[
                        (&format!("{a}q_proj.weight"), 2 * cfg.hq * HEAD_DIM),
                        (&format!("{a}k_proj.weight"), cfg.hkv * HEAD_DIM),
                        (&format!("{a}v_proj.weight"), cfg.hkv * HEAD_DIM),
                    ])?,
                    q_norm: norm(&format!("{a}q_norm.weight"), HEAD_DIM)?,
                    k_norm: norm(&format!("{a}k_norm.weight"), HEAD_DIM)?,
                    w_o: gpu
                        .upload(&ck.bf16(&format!("{a}o_proj.weight"), h * cfg.hq * HEAD_DIM)?)?,
                    cache: n_cache,
                };
                n_cache += 1;
                m
            } else {
                let a = format!("{p}linear_attn.");
                let hv = cfg.lin_v_heads;
                let conv_w = ck.f32(&format!("{a}conv1d.weight"), cfg.conv_dim() * 4)?;
                let a_log = ck.f32(&format!("{a}A_log"), hv)?;
                let neg_a: Vec<f32> = a_log.iter().map(|x| -x.exp()).collect();
                let m = Mixer::Gdn {
                    w_in: fused(&[
                        (&format!("{a}in_proj_qkv.weight"), cfg.conv_dim()),
                        (&format!("{a}in_proj_z.weight"), hv * GDN_DIM),
                        (&format!("{a}in_proj_b.weight"), hv),
                        (&format!("{a}in_proj_a.weight"), hv),
                    ])?,
                    conv_w: gpu.upload(&f32_bytes(&conv_w))?,
                    neg_a: gpu.upload(&f32_bytes(&neg_a))?,
                    dt_bias: gpu.upload(&f32_bytes(&ck.f32(&format!("{a}dt_bias"), hv)?))?,
                    // The gated norm uses its weight directly.
                    norm_w: gpu
                        .upload(&f32_bytes(&ck.f32(&format!("{a}norm.weight"), GDN_DIM)?))?,
                    w_out: gpu
                        .upload(&ck.bf16(&format!("{a}out_proj.weight"), h * hv * GDN_DIM)?)?,
                    state: n_state,
                };
                n_state += 1;
                m
            };
            layers.push(Layer {
                mixer,
                in_norm: norm(&format!("{p}input_layernorm.weight"), h)?,
                post_norm: norm(&format!("{p}post_attention_layernorm.weight"), h)?,
                w_gu: fused(&[
                    (&format!("{p}mlp.gate_proj.weight"), i),
                    (&format!("{p}mlp.up_proj.weight"), i),
                ])?,
                w_down: gpu.upload(&ck.bf16(&format!("{p}mlp.down_proj.weight"), h * i)?)?,
            });
        }
        let inv_freq: Vec<f32> = (0..ROPE_DIM / 2)
            .map(|d| 1f32 / cfg.rope_theta.powf((2 * d) as f64 / ROPE_DIM as f64) as f32)
            .collect();
        let k = Kernels {
            embed: gpu.kernel("embed")?,
            rmsnorm: gpu.kernel("rmsnorm")?,
            silu_mul: gpu.kernel("silu_mul")?,
            gdn_conv: gpu.kernel("gdn_conv")?,
            gdn_recurrent: gpu.kernel("gdn_recurrent")?,
            attn_prep: gpu.kernel("attn_prep")?,
            attn: gpu.kernel("attn_paged_g4")?,
            penalize: gpu.kernel("penalize")?,
            sample: gpu.kernel("sample")?,
        };
        Ok(Self {
            embed: gpu.upload(&ck.bf16("embed_tokens.weight", cfg.vocab * h)?)?,
            final_norm: norm("norm.weight", h)?,
            inv_freq: gpu.upload(&f32_bytes(&inv_freq))?,
            layers,
            k,
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
    pub fn cu(&self) -> usize {
        4 * self.max_tokens
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
    ids_out: Buf,
    pub ids_host: Pinned,
    /// Logit adjustments: [token i32 | add f32 | rep f32], each `pen_capacity` long.
    pen_dev: Buf,
    pub pen_staging: Pinned,
    pub pen_capacity: usize,
}

impl Act {
    pub fn new(gpu: &Gpu, cfg: &Config, meta: Meta) -> Result<Self> {
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
                    gpu.gemm(a.xn.ptr, w_in.ptr, a.proj.ptr, t, h, pw, false)?;
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
                        ],
                    )?;
                    gpu.gemm(a.core.ptr, w_out.ptr, a.h.ptr, t, hv * GDN_DIM, h, false)?;
                }
                Mixer::Attn {
                    w_qkv,
                    q_norm,
                    k_norm,
                    w_o,
                    cache,
                } => {
                    let qs = c.attn_proj();
                    gpu.gemm(a.xn.ptr, w_qkv.ptr, a.proj.ptr, t, h, qs, false)?;
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
                            P(a.m(me.pos())),
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
                        HEAD_DIM as u32,
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
                    gpu.gemm(a.ao.ptr, w_o.ptr, a.h.ptr, t, c.hq * HEAD_DIM, h, false)?;
                }
            }
            norm(a.h.ptr, &layer.post_norm, true)?;
            gpu.gemm(a.xn.ptr, layer.w_gu.ptr, a.gu.ptr, t, h, 2 * c.inter, false)?;
            let blocks = (t * c.inter / 8).div_ceil(256).min(8192) as u32;
            gpu.launch(
                k.silu_mul,
                (blocks, 1, 1),
                256,
                0,
                &[P(a.gu.ptr), P(a.act.ptr), I(c.inter as i32), I(t as i32)],
            )?;
            gpu.gemm(a.act.ptr, layer.w_down.ptr, a.h.ptr, t, c.inter, h, false)?;
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
        gpu.gemm(a.fin.ptr, self.embed.ptr, a.logits.ptr, s, h, c.vocab, true)?;
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
