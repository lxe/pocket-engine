//! Qwen3.5 text decoder: configuration, fused weights, per-sequence state and
//! one forward step over a flat token batch.

use crate::gguf::Gguf;
use crate::gpu::{Arg, Buf, Fp8Mode, Gpu, Kernel, Pinned, TensorMap};
use crate::linear::{Input, Kind, Linear, Part, Quants};
use crate::source::{Load, Source};
use crate::{Error, Result};
use std::path::Path;
use std::rc::Rc;

pub const GDN_DIM: usize = 128;
pub const HEAD_DIM: usize = 256;
pub const ROPE_DIM: usize = 64;
pub const BLOCK_SIZE: usize = 16;
/// Tokens drafts pick from (see Mtp::head): on Qwen3.5-9B, 65,536 keep
/// almost every accepted draft of the full 248,320 at a quarter of the cost.
const DRAFT_VOCAB: usize = 65_536;
/// Fewest keys per range of split decode attention.
const SPLIT_MIN_KEYS: usize = 256;
/// Most ranges one sequence's keys split into.
const SPLIT_MAX: usize = 64;

/// Key ranges per decode token for a step of `t` tokens: enough blocks to
/// give every SM about four, so one long sequence uses the whole GPU (1:
/// no split). Each sequence uses as many as its length fills with ranges of
/// at least SPLIT_MIN_KEYS (split_parts in qwen35.cu).
fn split_ranges(t: usize, hkv: usize, sms: usize) -> usize {
    (4 * sms).div_ceil(t * hkv).clamp(1, SPLIT_MAX)
}

/// Flash attention's shape for groups of `g` query heads on a device of
/// compute capability `arch`, for a bf16 or an E4M3 (`fp8`) cache: query
/// tiles of 16 tokens per block and keys per tile, naming the kernel
/// attn_group*_g<G>_t<NT>_k<BN> (qwen35.cu). Up to 99 KB of shared memory
/// (RTX 50-series): 8 warps of queries and 16-key tiles, or with a bf16 cache
/// for groups of 4 and 6 the queries in registers and 32-key tiles
/// (bench-attn on an RTX 5090: groups of 4 from 128 to 141 TFLOPS, of 6 from
/// 107 to 115). The 227 KB of Hopper and datacenter Blackwell (B200, B300)
/// take larger key tiles, and two query tiles for groups of 6 (bench-attn on
/// an H100: groups of 4 from 123 to 158 TFLOPS, of 6 from 97 to 168).
pub fn fa_shape(g: usize, arch: u32, fp8: bool) -> FaShape {
    let shape = |tiles, keys, qreg| FaShape { tiles, keys, qreg };
    match (arch, g, fp8) {
        (90 | 100 | 103, 4, _) => shape(2, 64, false),
        (90 | 100 | 103, 6, _) => shape(2, 32, false),
        (_, 4, false) => shape(2, 32, true),
        (_, 6, false) => shape(1, 32, true),
        _ => shape(if g == 6 { 1 } else { 8 / g }, 16, false),
    }
}

/// A flash kernel's shape: query tiles of 16 tokens per block and keys per
/// tile.
#[derive(Clone, Copy, Debug)]
pub struct FaShape {
    pub tiles: usize,
    pub keys: usize,
    /// Queries in registers (attn_groupr_; bf16 caches only).
    pub qreg: bool,
}

impl FaShape {
    /// Parses "NT,BN" or "NT,BN,r" (queries in registers).
    pub fn parse(v: &str) -> Option<Self> {
        let mut it = v.split(',');
        let (tiles, keys) = (it.next()?.parse().ok()?, it.next()?.parse().ok()?);
        let qreg = match it.next() {
            None => false,
            Some("r") => true,
            Some(_) => return None,
        };
        it.next().is_none().then_some(Self { tiles, keys, qreg })
    }
    /// The kernel: attn_group_g<G>_t<NT>_k<BN>, attn_group8_ for an E4M3
    /// cache, attn_groupr_ with the queries in registers.
    pub fn kernel(&self, g: usize, fp8: bool) -> String {
        let f = if fp8 {
            "8"
        } else if self.qreg {
            "r"
        } else {
            ""
        };
        format!("attn_group{f}_g{g}_t{}_k{}", self.tiles, self.keys)
    }
    /// Its dynamic shared memory: the tiles' queries for each of G heads and
    /// two buffers of BN keys and values, or, with the queries in registers,
    /// the larger of the two (the buffers reuse the queries' space).
    pub fn smem(&self, g: usize) -> usize {
        let (q, kv) = (g * 16 * self.tiles, 4 * self.keys);
        (if self.qreg { q.max(kv) } else { q + kv }) * (HEAD_DIM + 8) * 2
    }
}
/// Dynamic shared memory of attn_ws_g<G> (WS_SMEM in qwen35.cu): two
/// warpgroups' queries and two stages of keys and values, 64 x 256 bf16
/// each, the stages' mbarriers and 8 bytes per consumer thread.
pub const WS_SMEM: usize = 6 * 64 * HEAD_DIM * 2 + 128 + 8 * 256;
/// Whether prompts take the wgmma flash kernel: Hopper, a bf16 cache and
/// groups of 4 or 6.
pub fn fa_wgmma(g: usize, arch: u32, kv_fp8: bool) -> bool {
    arch == 90 && !kv_fp8 && matches!(g, 4 | 6)
}

/// Dynamic shared memory of the two-pass chunked gated delta rule
/// (GDN_PREP_SMEM and GDN_SCAN_SMEM(NS) in qwen35.cu).
pub const GDN_PREP_SMEM: usize = 3 * 32 * 136 * 2 + 32 * 36 * 4 + 6 * 32 * 4;
pub fn gdn_scan_smem(ns: usize) -> usize {
    let vs = 128 / ns + 8;
    let buffers = if ns > 1 { 2 } else { 1 };
    buffers * (3 * 32 * 136 * 2 + 32 * vs * 2 + 32 * 40 * 2 + 3 * 32 * 4)
        + 2 * 32 * vs * 2
        + 128 * vs * 2
}
/// Value-column shares of gdn_scan for a step of `s` sequences: the most
/// (up to 4) whose blocks are all resident at once.
pub fn gdn_split(s: usize, hv: usize, gpu: &Gpu) -> usize {
    let resident = |ns: usize| gpu.sms * (gpu.smem_per_sm / (gdn_scan_smem(ns) + 1024)).max(1);
    [4, 2]
        .into_iter()
        .find(|&ns| s * hv * ns <= resident(ns))
        .unwrap_or(1)
}

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
    /// The token that opens a chat message (<|im_start|>).
    pub turn_token: u32,
    /// Rotary frequencies per axis (time, height, width), interleaved.
    pub mrope_section: [usize; 3],
    /// Linear-attention value heads in GGUF's tiled order (see Source).
    pub tiled_v: bool,
    /// Whether the model runs its multi-token-prediction layer, whose
    /// attention takes one more KV cache layer.
    pub mtp: bool,
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
            turn_token: 248045,
            mrope_section: match rope["mrope_section"].as_array() {
                Some(a) if a.len() == 3 => {
                    let v: Vec<usize> =
                        a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                    [v[0], v[1], v[2]]
                }
                _ => [11, 11, 10],
            },
            tiled_v: false,
            mtp: false,
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
            turn_token: find("<|im_start|>").unwrap_or(248045),
            mrope_section: if sections.len() >= 3 {
                [sections[0], sections[1], sections[2]]
            } else {
                [11, 11, 10]
            },
            tiled_v: true,
            mtp: false,
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
        self.attn_layers.iter().filter(|a| **a).count() + self.mtp as usize
    }
    /// Recurrent and convolution state bytes per sequence slot, with the
    /// recurrent state in f32 or bf16.
    pub fn slot_bytes(&self, state_bf16: bool) -> usize {
        let elem = if state_bf16 { 2 } else { 4 };
        self.gdn_layers() * (self.lin_v_heads * GDN_DIM * GDN_DIM * elem + 3 * self.conv_dim() * 2)
    }
    /// KV cache bytes per block of BLOCK_SIZE tokens: bf16, or E4M3 with an
    /// f32 scale per token and head.
    pub fn block_bytes(&self, kv_fp8: bool) -> usize {
        self.kv_layers() * 2 * BLOCK_SIZE * self.hkv * kv_head_bytes(kv_fp8)
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

/// The multi-token-prediction layer: it drafts the token after next from a
/// token's embedding and the model's final hidden state before it.
struct Mtp {
    /// [hidden x 2 hidden], over [norm_e(embedding) | norm_h(hidden)].
    fc: Linear,
    norm_e: Buf,
    norm_h: Buf,
    layer: Layer,
    norm: Buf,
    /// The output layer over the first `vocab` tokens only, which drafts
    /// pick from: in BPE order these are the most common, and the full
    /// output layer is the costliest part of a draft.
    head: Linear,
    vocab: usize,
}

struct Kernels {
    embed: Kernel,
    mtp_tokens: Kernel,
    mtp_input: Kernel,
    mtp_accept: Kernel,
    mtp_next: Kernel,
    rmsnorm: Kernel,
    silu_mul: Kernel,
    gdn_conv: Kernel,
    gdn_recurrent: Kernel,
    /// The recurrence for steps with prompt tokens, chunked on tensor cores
    /// in two passes (see gdn_prep in qwen35.cu), gdn_scan with 1, 2 or 4
    /// blocks per head; and its gated norm.
    gdn_prep: Kernel,
    gdn_scan: [Kernel; 3],
    gdn_norm: Kernel,
    attn_prep: Kernel,
    attn: Kernel,
    /// Paged attention over an E4M3 cache.
    attn8: Kernel,
    /// Attention for prompt tokens on tensor cores, over a bf16 or E4M3 cache,
    /// with query tiles per block and keys per tile.
    attn_flash: Kernel,
    attn_flash8: Kernel,
    fa_shape: FaShape,
    fa_shape8: FaShape,
    /// On Hopper: flash attention on wgmma, warp-specialized with TMA loads,
    /// for a bf16 cache.
    attn_wg: Option<Kernel>,
    /// FP8 copies of bf16 rows (the flash kernel's output, for FP8 layers).
    fp8_rows: Kernel,
    /// Combines split decode attention's key ranges.
    attn_merge: Kernel,
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
    mtp: Option<Mtp>,
    k: Kernels,
    quants: Quants,
    /// Bytes of weights on the device.
    pub weight_bytes: usize,
    /// The FP8 layout of the model's FP8 layers, if any.
    pub fp8: Option<Fp8Mode>,
    /// Loaded from a GGUF file (linear-attention value heads tiled).
    pub gguf: bool,
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Model {
    /// Loads a Hugging Face directory or a GGUF file (or a directory holding one).
    pub fn load(gpu: &Gpu, path: &Path, load: Load) -> Result<Self> {
        let started = gpu.begin_uploads()?;
        let m = Self::load_weights(gpu, path, load);
        if started {
            gpu.end_uploads()?;
        }
        m
    }

    /// The linear layers adapters may target, in the order `attach_lora` takes.
    pub fn lora_sites(&self) -> Vec<crate::lora::Site> {
        use crate::lora::{Heads, Site, SitePart};
        let c = &self.cfg;
        let (hk, hv) = (c.lin_k_heads, c.lin_v_heads);
        let site = |layer: usize, cols: usize, parts: &[(&str, usize, Heads)], col_heads: Heads| {
            let mut off = 0;
            Site {
                layer,
                cols,
                parts: parts
                    .iter()
                    .map(|&(m, rows, heads)| {
                        let p = SitePart {
                            module: m.to_string(),
                            rows,
                            off,
                            heads,
                        };
                        off += rows;
                        p
                    })
                    .collect(),
                col_heads,
            }
        };
        let mut out = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            match layer.mixer {
                Mixer::Gdn { .. } => {
                    let v = Heads::From {
                        start: 0,
                        block: GDN_DIM,
                    };
                    out.push(site(
                        l,
                        c.hidden,
                        &[
                            (
                                "linear_attn.in_proj_qkv",
                                c.conv_dim(),
                                Heads::From {
                                    start: 2 * hk * GDN_DIM,
                                    block: GDN_DIM,
                                },
                            ),
                            ("linear_attn.in_proj_z", hv * GDN_DIM, v),
                            (
                                "linear_attn.in_proj_b",
                                hv,
                                Heads::From { start: 0, block: 1 },
                            ),
                            (
                                "linear_attn.in_proj_a",
                                hv,
                                Heads::From { start: 0, block: 1 },
                            ),
                        ],
                        Heads::None,
                    ));
                    out.push(site(
                        l,
                        hv * GDN_DIM,
                        &[("linear_attn.out_proj", c.hidden, Heads::None)],
                        v,
                    ));
                }
                Mixer::Attn { .. } => {
                    out.push(site(
                        l,
                        c.hidden,
                        &[
                            ("self_attn.q_proj", 2 * c.hq * HEAD_DIM, Heads::None),
                            ("self_attn.k_proj", c.hkv * HEAD_DIM, Heads::None),
                            ("self_attn.v_proj", c.hkv * HEAD_DIM, Heads::None),
                        ],
                        Heads::None,
                    ));
                    out.push(site(
                        l,
                        c.hq * HEAD_DIM,
                        &[("self_attn.o_proj", c.hidden, Heads::None)],
                        Heads::None,
                    ));
                }
            }
            out.push(site(
                l,
                c.hidden,
                &[
                    ("mlp.gate_proj", c.inter, Heads::None),
                    ("mlp.up_proj", c.inter, Heads::None),
                ],
                Heads::None,
            ));
            out.push(site(
                l,
                c.inter,
                &[("mlp.down_proj", c.hidden, Heads::None)],
                Heads::None,
            ));
        }
        out
    }

    /// Points each decoder linear layer at its LoRA descriptor table.
    pub fn attach_lora(&mut self, loras: &crate::lora::Loras) {
        let mut i = 0;
        for layer in &mut self.layers {
            let (a, b) = match &mut layer.mixer {
                Mixer::Gdn { w_in, w_out, .. } => (w_in, w_out),
                Mixer::Attn { w_qkv, w_o, .. } => (w_qkv, w_o),
            };
            for w in [a, b, &mut layer.w_gu, &mut layer.w_down] {
                w.lora = Some(loras.hook(i));
                i += 1;
            }
        }
    }

    fn load_weights(gpu: &Gpu, path: &Path, load: Load) -> Result<Self> {
        let src = Source::open(path)?;
        let mut cfg = match &src {
            Source::Gguf(g) => Config::from_gguf(g)?,
            Source::Safetensors(_) => Config::load(path)?,
        };
        cfg.tiled_v = src.tiled_value_heads() && cfg.lin_v_heads != cfg.lin_k_heads;
        let mut quants = Quants::default();
        if load.int8 {
            quants.int8_rows = crate::linear::int8_rows();
        }
        let (h, i) = (cfg.hidden, cfg.inter);
        let up = |v: &[f32]| gpu.upload(&f32_bytes(v));
        let mut layers = Vec::with_capacity(cfg.layers);
        let (mut n_state, mut n_cache) = (0, 0);
        let attn = |p: &str, quants: &mut Quants, cache: usize, load: Load| -> Result<Mixer> {
            let a = format!("{p}self_attn.");
            Ok(Mixer::Attn {
                w_qkv: src.linear(
                    gpu,
                    quants,
                    &[
                        (&format!("{a}q_proj.weight"), 2 * cfg.hq * HEAD_DIM),
                        (&format!("{a}k_proj.weight"), cfg.hkv * HEAD_DIM),
                        (&format!("{a}v_proj.weight"), cfg.hkv * HEAD_DIM),
                    ],
                    h,
                    load,
                )?,
                q_norm: up(&src.norm(&format!("{a}q_norm.weight"), HEAD_DIM)?)?,
                k_norm: up(&src.norm(&format!("{a}k_norm.weight"), HEAD_DIM)?)?,
                w_o: src.linear(
                    gpu,
                    quants,
                    &[(&format!("{a}o_proj.weight"), h)],
                    cfg.hq * HEAD_DIM,
                    load,
                )?,
                cache,
            })
        };
        let layer = |p: &str, mixer: Mixer, quants: &mut Quants, load: Load| -> Result<Layer> {
            Ok(Layer {
                mixer,
                in_norm: up(&src.norm(&format!("{p}input_layernorm.weight"), h)?)?,
                post_norm: up(&src.norm(&format!("{p}post_attention_layernorm.weight"), h)?)?,
                w_gu: src.linear(
                    gpu,
                    quants,
                    &[
                        (&format!("{p}mlp.gate_proj.weight"), i),
                        (&format!("{p}mlp.up_proj.weight"), i),
                    ],
                    h,
                    load,
                )?,
                w_down: src.linear(
                    gpu,
                    quants,
                    &[(&format!("{p}mlp.down_proj.weight"), h)],
                    i,
                    load,
                )?,
            })
        };
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let mixer = if cfg.attn_layers[l] {
                n_cache += 1;
                attn(&p, &mut quants, n_cache - 1, load)?
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
                        load,
                    )?,
                    conv_w: up(&src.vec_f32(&format!("{a}conv1d.weight"), cfg.conv_dim() * 4)?)?,
                    neg_a: up(&src.neg_a(l, hv)?)?,
                    dt_bias: up(&src.vec_f32(&format!("{a}dt_bias"), hv)?)?,
                    // The gated norm uses its weight directly.
                    norm_w: up(&src.vec_f32(&format!("{a}norm.weight"), GDN_DIM)?)?,
                    w_out: Linear {
                        // Its gated, normed input loses too much to int8.
                        int8: false,
                        ..src.linear(
                            gpu,
                            &mut quants,
                            &[(&format!("{a}out_proj.weight"), h)],
                            hv * GDN_DIM,
                            load,
                        )?
                    },
                    state: n_state,
                };
                n_state += 1;
                m
            };
            layers.push(layer(&p, mixer, &mut quants, load)?);
        }
        let mtp = if load.mtp {
            if !src.has("/mtp.fc.weight") {
                return Err(Error::Config(
                    "--mtp: this checkpoint has no multi-token-prediction layer".into(),
                ));
            }
            let p = "/mtp.layers.0.";
            // The fc input has no FP8 copy, so it stays in bf16.
            let plain = Load {
                quantize: false,
                fp4: false,
                ..load
            };
            let fc = src.linear(gpu, &mut quants, &[("/mtp.fc.weight", h)], 2 * h, plain)?;
            if fc.fp8().is_some() {
                return Err(Error::Config(
                    "--mtp needs a bf16 or GGUF-quantized mtp.fc".into(),
                ));
            }
            let mixer = attn(p, &mut quants, n_cache, load)?;
            Some((
                fc,
                up(&src.norm("/mtp.pre_fc_norm_embedding.weight", h)?)?,
                up(&src.norm("/mtp.pre_fc_norm_hidden.weight", h)?)?,
                layer(p, mixer, &mut quants, load)?,
                up(&src.norm("/mtp.norm.weight", h)?)?,
            ))
        } else {
            None
        };
        cfg.mtp = mtp.is_some();
        let (embed, embed_kind) =
            src.shared(gpu, &mut quants, "embed_tokens.weight", cfg.vocab, h)?;
        // The output layer is its own tensor, or tied to the embedding.
        let lm_head = if src.has("/lm_head.weight") {
            src.linear(gpu, &mut quants, &[("/lm_head.weight", cfg.vocab)], h, load)?
        } else {
            Linear {
                parts: vec![Part {
                    buf: embed.clone(),
                    kind: embed_kind,
                    scales: None,
                    rows: cfg.vocab,
                    off: 0,
                }],
                cols: h,
                int8: true,
                lora: None,
            }
        };
        let mtp = match mtp {
            Some((fc, norm_e, norm_h, layer, norm)) => {
                let vocab = DRAFT_VOCAB.min(cfg.vocab);
                // Rows of a bf16 or row-blocked quantized matrix are a prefix.
                let sliceable = lm_head.parts.len() == 1
                    && matches!(lm_head.parts[0].kind, Kind::Bf16 | Kind::Quant(_));
                let vocab = if sliceable { vocab } else { cfg.vocab };
                let head = Linear {
                    parts: lm_head
                        .parts
                        .iter()
                        .map(|p| Part {
                            buf: p.buf.clone(),
                            kind: p.kind,
                            scales: p.scales.clone(),
                            rows: if sliceable { vocab } else { p.rows },
                            off: p.off,
                        })
                        .collect(),
                    cols: lm_head.cols,
                    int8: lm_head.int8,
                    lora: None,
                };
                Some(Mtp {
                    fc,
                    norm_e,
                    norm_h,
                    layer,
                    norm,
                    head,
                    vocab,
                })
            }
            None => None,
        };
        let inv_freq: Vec<f32> = (0..ROPE_DIM / 2)
            .map(|d| 1f32 / cfg.rope_theta.powf((2 * d) as f64 / ROPE_DIM as f64) as f32)
            .collect();
        let embed_kernel = match embed_kind {
            Kind::Bf16 => gpu.kernel("embed")?,
            Kind::Quant(ty) => quants.get(gpu, ty)?.embed,
            Kind::Fp8(_) | Kind::Tiled(_) | Kind::Fp4 => {
                return Err(Error::Config(
                    "FP8, FP4 and tiled embeddings are not supported".into(),
                ));
            }
        };
        let k = Kernels {
            embed: embed_kernel,
            mtp_tokens: gpu.kernel("mtp_tokens")?,
            mtp_input: gpu.kernel("mtp_input")?,
            mtp_accept: gpu.kernel("mtp_accept")?,
            mtp_next: gpu.kernel("mtp_next")?,
            rmsnorm: gpu.kernel("rmsnorm")?,
            silu_mul: gpu.kernel("silu_mul")?,
            gdn_conv: gpu.kernel("gdn_conv")?,
            gdn_recurrent: gpu.kernel("gdn_recurrent")?,
            gdn_prep: {
                let k = gpu.kernel("gdn_prep")?;
                gpu.set_smem(k, GDN_PREP_SMEM)?;
                k
            },
            gdn_scan: {
                let mut ks = Vec::new();
                for ns in [1, 2, 4] {
                    let k = gpu.kernel(&format!("gdn_scan{ns}"))?;
                    gpu.set_smem(k, gdn_scan_smem(ns))?;
                    ks.push(k);
                }
                [ks[0], ks[1], ks[2]]
            },
            gdn_norm: gpu.kernel("gdn_norm")?,
            attn_prep: gpu.kernel("attn_prep")?,
            attn: gpu.kernel(&format!("attn_paged_g{}", cfg.hq / cfg.hkv))?,
            attn8: gpu.kernel(&format!("attn_paged8_g{}", cfg.hq / cfg.hkv))?,
            attn_flash: {
                let g = cfg.hq / cfg.hkv;
                let sh = fa_shape(g, gpu.arch, false);
                let k = gpu.kernel(&sh.kernel(g, false))?;
                gpu.set_smem(k, sh.smem(g))?;
                k
            },
            attn_flash8: {
                let g = cfg.hq / cfg.hkv;
                let sh = fa_shape(g, gpu.arch, true);
                let k = gpu.kernel(&sh.kernel(g, true))?;
                gpu.set_smem(k, sh.smem(g))?;
                k
            },
            fa_shape: fa_shape(cfg.hq / cfg.hkv, gpu.arch, false),
            fa_shape8: fa_shape(cfg.hq / cfg.hkv, gpu.arch, true),
            attn_wg: if fa_wgmma(cfg.hq / cfg.hkv, gpu.arch, false) {
                let k = gpu.kernel(&format!("attn_ws_g{}", cfg.hq / cfg.hkv))?;
                gpu.set_smem(k, WS_SMEM)?;
                Some(k)
            } else {
                None
            },
            fp8_rows: gpu.kernel("fp8_activations")?,
            attn_merge: gpu.kernel("attn_merge")?,
            penalize: gpu.kernel("penalize")?,
            sample: gpu.kernel("sample")?,
        };
        let largest = layers
            .iter()
            .chain(mtp.as_ref().map(|m| &m.layer))
            .flat_map(|l| {
                let mut v = vec![&l.w_gu, &l.w_down];
                match &l.mixer {
                    Mixer::Gdn { w_in, w_out, .. } => v.extend([w_in, w_out]),
                    Mixer::Attn { w_qkv, w_o, .. } => v.extend([w_qkv, w_o]),
                }
                v
            })
            .chain(std::iter::once(&lm_head))
            .chain(mtp.as_ref().map(|m| &m.fc))
            .flat_map(|l| l.parts.iter().map(move |p| p.rows * l.cols))
            .max()
            .unwrap_or(0);
        quants.prepare(gpu, largest)?;
        let mut weight_bytes = embed.bytes + mtp.as_ref().map_or(0, |m| m.fc.bytes());
        for l in layers.iter().chain(mtp.as_ref().map(|m| &m.layer)) {
            weight_bytes += l.w_gu.bytes() + l.w_down.bytes();
            weight_bytes += match &l.mixer {
                Mixer::Gdn { w_in, w_out, .. } => w_in.bytes() + w_out.bytes(),
                Mixer::Attn { w_qkv, w_o, .. } => w_qkv.bytes() + w_o.bytes(),
            };
        }
        if src.has("/lm_head.weight") {
            weight_bytes += lm_head.bytes();
        }
        let fp8 = layers.iter().find_map(|l| l.w_gu.fp8().or(l.w_down.fp8()));
        Ok(Self {
            fp8,
            gguf: matches!(src, Source::Gguf(_)),
            final_norm: up(&src.norm("norm.weight", h)?)?,
            inv_freq: up(&inv_freq)?,
            embed,
            lm_head,
            layers,
            mtp,
            k,
            quants,
            weight_bytes,
            cfg,
        })
    }
}

/// Bytes of one cached key or value head.
fn kv_head_bytes(kv_fp8: bool) -> usize {
    if kv_fp8 { HEAD_DIM + 4 } else { HEAD_DIM * 2 }
}

/// Per-sequence recurrent state slots and the paged KV pool.
pub struct State {
    conv: Vec<Buf>,
    rec: Vec<Buf>,
    kc: Vec<Buf>,
    vc: Vec<Buf>,
    /// With an E4M3 cache: f32 scales per [slot][kv head] of each layer.
    ks: Vec<Buf>,
    vs: Vec<Buf>,
    /// On Hopper with a bf16 cache: TMA maps of each layer's keys and values
    /// ([slot][kv head][256], boxes of 16 slots x 64 columns).
    maps: Vec<(&'static TensorMap, &'static TensorMap)>,
    pub kv_fp8: bool,
    /// Recurrent state stored as bf16 (math stays f32).
    pub state_bf16: bool,
    pub slots: usize,
    pub blocks: usize,
    /// Bytes of one slot in each layer's convolution and recurrent buffers.
    conv_slot: usize,
    rec_slot: usize,
}

impl State {
    pub fn new(
        gpu: &Gpu,
        cfg: &Config,
        slots: usize,
        blocks: usize,
        kv_fp8: bool,
        state_bf16: bool,
    ) -> Result<Self> {
        let conv_bytes = slots * 3 * cfg.conv_dim() * 2;
        let rec_bytes =
            slots * cfg.lin_v_heads * GDN_DIM * GDN_DIM * if state_bf16 { 2 } else { 4 };
        let heads = blocks * BLOCK_SIZE * cfg.hkv;
        let kv_bytes = heads * HEAD_DIM * if kv_fp8 { 1 } else { 2 };
        let mut s = Self {
            conv: vec![],
            rec: vec![],
            kc: vec![],
            vc: vec![],
            ks: vec![],
            vs: vec![],
            maps: vec![],
            kv_fp8,
            state_bf16,
            slots,
            blocks,
            conv_slot: conv_bytes / slots,
            rec_slot: rec_bytes / slots,
        };
        for _ in 0..cfg.gdn_layers() {
            s.conv.push(gpu.alloc_zeroed(conv_bytes)?);
            s.rec.push(gpu.alloc_zeroed(rec_bytes)?);
        }
        for _ in 0..cfg.kv_layers() {
            s.kc.push(gpu.alloc_zeroed(kv_bytes)?);
            s.vc.push(gpu.alloc_zeroed(kv_bytes)?);
            if kv_fp8 {
                s.ks.push(gpu.alloc_zeroed(heads * 4)?);
                s.vs.push(gpu.alloc_zeroed(heads * 4)?);
            }
        }
        if fa_wgmma(cfg.hq / cfg.hkv, gpu.arch, kv_fp8) {
            for (k, v) in s.kc.iter().zip(&s.vc) {
                let map = |ptr| {
                    gpu.tensor_map_3d(
                        ptr,
                        [
                            HEAD_DIM as u64,
                            cfg.hkv as u64,
                            (blocks * BLOCK_SIZE) as u64,
                        ],
                        [(HEAD_DIM * 2) as u64, (cfg.hkv * HEAD_DIM * 2) as u64],
                        [64, 1, BLOCK_SIZE as u32],
                    )
                };
                s.maps.push((map(k.ptr)?, map(v.ptr)?));
            }
        }
        Ok(s)
    }

    /// Copies slot `from`'s recurrent and convolution state to slot `to`,
    /// ordered on the stream.
    pub fn copy_slot(&self, gpu: &Gpu, from: usize, to: usize) -> Result<()> {
        for (bufs, n) in [(&self.conv, self.conv_slot), (&self.rec, self.rec_slot)] {
            for b in bufs {
                gpu.dtod(b.ptr + (to * n) as u64, b.ptr + (from * n) as u64, n)?;
            }
        }
        Ok(())
    }

    /// Scale pointers of an attention layer's cache, 0 for a bf16 cache.
    fn kv_scales(&self, cache: usize) -> (u64, u64) {
        match (self.ks.get(cache), self.vs.get(cache)) {
            (Some(k), Some(v)) => (k.ptr, v.ptr),
            _ => (0, 0),
        }
    }
}

/// Batch metadata layout (i32 words) in one device buffer, filled from a pinned
/// staging buffer with a single copy per step.
///
/// Each sequence yields one output row (logits and a pick), or one per token
/// when it verifies drafts: up to `spec` (drafts + 1), so per-output arrays
/// hold `max_outs()`.
#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub max_tokens: usize,
    pub max_seqs: usize,
    pub max_blocks: usize,
    /// Tokens a sequence verifying drafts feeds per step (drafts + 1); 1
    /// without multi-token prediction.
    pub spec: usize,
}

impl Meta {
    pub fn max_outs(&self) -> usize {
        self.max_seqs * self.spec
    }
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
    /// Per token, the token after it for the draft layer: an id, or -(r + 1)
    /// for the pick of output row r.
    pub fn mtp_ids(&self) -> usize {
        8 * self.max_tokens
    }
    /// Per token, the image pool row of the token after it, or -1.
    pub fn mtp_src(&self) -> usize {
        9 * self.max_tokens
    }
    pub fn cu(&self) -> usize {
        10 * self.max_tokens
    }
    pub fn slots(&self) -> usize {
        self.cu() + self.max_seqs + 1
    }
    pub fn fresh(&self) -> usize {
        self.slots() + self.max_seqs
    }
    /// Per sequence, `spec` state slots to keep its state in after each of
    /// its tokens when it verifies drafts; -1 first otherwise.
    pub fn outs(&self) -> usize {
        self.fresh() + self.max_seqs
    }
    /// Offsets of each sequence's output rows.
    pub fn ocu(&self) -> usize {
        self.outs() + self.max_outs()
    }
    /// Per output row, its token row.
    pub fn last(&self) -> usize {
        self.ocu() + self.max_seqs + 1
    }
    pub fn temps(&self) -> usize {
        self.last() + self.max_outs()
    }
    pub fn seeds(&self) -> usize {
        self.temps() + self.max_outs()
    }
    pub fn top_ks(&self) -> usize {
        self.seeds() + self.max_outs()
    }
    pub fn top_ps(&self) -> usize {
        self.top_ks() + self.max_outs()
    }
    pub fn min_ps(&self) -> usize {
        self.top_ps() + self.max_outs()
    }
    /// Offsets of each output row's penalty entries (see [`Act::pen_capacity`]).
    pub fn pen_cu(&self) -> usize {
        self.min_ps() + self.max_outs()
    }
    /// Per sequence, the last position it holds KV for (-1 for padding).
    pub fn limits(&self) -> usize {
        self.pen_cu() + self.max_outs() + 1
    }
    /// Per sequence, up to two prefix-cache snapshots to write in the step:
    /// (tokens into its row, state slot), -1 when unused.
    pub fn snaps(&self) -> usize {
        self.limits() + self.max_seqs
    }
    pub fn block_table(&self) -> usize {
        self.snaps() + 4 * self.max_seqs
    }
    /// Adapter tiles: [count, (first token, tokens, slot) x max_lora_tiles()].
    pub fn lora(&self) -> usize {
        self.block_table() + self.max_seqs * self.max_blocks
    }
    pub fn max_lora_tiles(&self) -> usize {
        self.max_tokens / crate::lora::TILE + self.max_seqs
    }
    pub fn words(&self) -> usize {
        self.lora() + 1 + 3 * self.max_lora_tiles()
    }
}

/// Activation buffers sized for the largest step.
pub struct Act {
    pub meta: Meta,
    meta_dev: Buf,
    /// Partial results of split decode attention: per sequence, query head
    /// and key range, the max, the sum and 256 accumulators. None when one
    /// range covers the context.
    attn_part: Option<Buf>,
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
    /// FP8 copies of the projections' inputs, when the model has FP8 layers.
    fp8: Option<Fp8Acts>,
    /// The draft layer's buffers, with multi-token prediction.
    mtp: Option<MtpActs>,
    /// gdn_prep's per-token rows for gdn_scan: U, W (bf16, 128 per head),
    /// q's and k's inverse norms (f32, one per head), P (bf16, 32 per head)
    /// and G (f32, one per head).
    gdn: [Buf; 6],
}

struct MtpActs {
    /// Per token row: the token its draft input embeds.
    tok: Buf,
    /// Per token row: [norm_e(embedding) | norm_h(hidden)].
    cat: Buf,
    /// Per sequence: the token row it drafts from.
    rows: Buf,
    /// Zero temperatures: drafts are greedy.
    zeros: Buf,
    logits: Buf,
    /// `spec - 1` drafts per sequence, draft k of sequence s at k * S + s.
    drafts: Buf,
    pub drafts_host: Pinned,
    /// Later draft passes' metadata: positions, KV slots, rotary positions
    /// (three rows of S), 0..=S (token-to-sequence map and offsets), and -1s.
    dpos: Buf,
    dslot: Buf,
    drope: Buf,
    iota: Buf,
    neg: Buf,
}

/// FP8 values and scales of each projection input: the normalized hidden
/// state, the MLP activation, the linear-attention output and the gated
/// attention output.
struct Fp8Acts {
    mode: Fp8Mode,
    xn: (Buf, Buf),
    act: (Buf, Buf),
    core: (Buf, Buf),
    ao: (Buf, Buf),
    /// The final norm's output, one per output row, for an FP8 or FP4
    /// output layer.
    fin: (Buf, Buf),
}

impl Act {
    /// `pool_rows` image embeddings can wait between the vision encoder and
    /// the prompt step that reads them.
    pub fn new(
        gpu: &Gpu,
        cfg: &Config,
        meta: Meta,
        pool_rows: usize,
        fp8: Option<Fp8Mode>,
    ) -> Result<Self> {
        let (t, s) = (meta.max_tokens, meta.max_seqs);
        let outs = meta.max_outs();
        let fp8 = match fp8 {
            None => None,
            Some(mode) => {
                // Scale tiles cover whole 128-row tiles (and padding to 4 rows).
                let rows = (t + 4).next_multiple_of(128);
                let pair = |k: usize| -> Result<(Buf, Buf)> {
                    Ok((
                        gpu.alloc_zeroed(rows * k)?,
                        gpu.alloc_zeroed(mode.act_scale_bytes(rows, k))?,
                    ))
                };
                let fin_rows = (outs + 4).next_multiple_of(128);
                Some(Fp8Acts {
                    mode,
                    xn: pair(cfg.hidden)?,
                    act: pair(cfg.inter)?,
                    core: pair(cfg.lin_v_heads * GDN_DIM)?,
                    ao: pair(cfg.hq * HEAD_DIM)?,
                    fin: (
                        gpu.alloc_zeroed(fin_rows * cfg.hidden)?,
                        gpu.alloc_zeroed(mode.act_scale_bytes(fin_rows, cfg.hidden))?,
                    ),
                })
            }
        };
        let b = |n: usize| gpu.alloc_zeroed(n * 2);
        // Every token a sequence can hold, plus a logit_bias of up to 1,024.
        let pen_capacity = s * (meta.max_blocks * BLOCK_SIZE + 1024);
        // Partials are per decode token (up to `spec` per sequence) and range.
        let part_rows = (1..=t)
            .map(|n| match split_ranges(n, cfg.hkv, gpu.sms) {
                1 => 0,
                z => n.min(s) * meta.spec * z,
            })
            .max()
            .unwrap_or(0);
        let attn_part = if part_rows > 0 {
            Some(gpu.alloc(part_rows * cfg.hq * (HEAD_DIM + 2) * 4)?)
        } else {
            None
        };
        Ok(Self {
            attn_part,
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
            fin: b(outs * cfg.hidden)?,
            logits: gpu.alloc_zeroed(outs * cfg.vocab * 4)?,
            pool: gpu.alloc_zeroed(pool_rows.max(1) * cfg.hidden * 2)?,
            ids_out: gpu.alloc_zeroed(outs * 4)?,
            ids_host: gpu.pinned(outs * 4)?,
            pen_dev: gpu.alloc_zeroed(3 * pen_capacity * 4)?,
            pen_staging: gpu.pinned(3 * pen_capacity * 4)?,
            pen_capacity,
            fp8,
            gdn: {
                let rows = t * cfg.lin_v_heads;
                [
                    b(rows * GDN_DIM)?,
                    b(rows * GDN_DIM)?,
                    gpu.alloc_zeroed(rows * 4)?,
                    gpu.alloc_zeroed(rows * 4)?,
                    b(rows * 32)?,
                    gpu.alloc_zeroed(rows * 4)?,
                ]
            },
            mtp: if cfg.mtp {
                Some(MtpActs {
                    tok: gpu.alloc_zeroed(t * 4)?,
                    cat: b(t * 2 * cfg.hidden)?,
                    rows: gpu.alloc_zeroed(s * 4)?,
                    zeros: gpu.alloc_zeroed(s * 4)?,
                    logits: gpu.alloc_zeroed(s * cfg.vocab * 4)?,
                    drafts: gpu.alloc_zeroed(outs * 4)?,
                    drafts_host: gpu.pinned(outs * 4)?,
                    dpos: gpu.alloc_zeroed(s * 4)?,
                    dslot: gpu.alloc_zeroed(s * 4)?,
                    drope: gpu.alloc_zeroed(3 * s * 4)?,
                    iota: gpu.upload(
                        &(0..=s as i32)
                            .flat_map(|i| i.to_le_bytes())
                            .collect::<Vec<u8>>(),
                    )?,
                    neg: gpu.upload(&vec![0xffu8; outs * 4])?,
                })
            } else {
                None
            },
            meta,
        })
    }

    /// Kernel arguments for writing FP8 copies into `which` (values, scales,
    /// mode, rows), or zeros when the model has no FP8 layers.
    fn fp8_args(&self, which: fn(&Fp8Acts) -> &(Buf, Buf), rows: usize) -> [Arg; 4] {
        match &self.fp8 {
            Some(f) => {
                let (q, s) = which(f);
                let mode = match f.mode {
                    Fp8Mode::Block128 => 1,
                    Fp8Mode::Mx32 => 2,
                };
                [
                    Arg::P(q.ptr),
                    Arg::P(s.ptr),
                    Arg::I(mode),
                    Arg::I(crate::linear::fp8_rows(f.mode, rows) as i32),
                ]
            }
            None => [Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0)],
        }
    }

    /// A projection input: bf16 rows, plus their FP8 copy if there is one.
    fn input(&self, bf16: &Buf, which: fn(&Fp8Acts) -> &(Buf, Buf)) -> Input {
        match &self.fp8 {
            Some(f) => {
                let (q, s) = which(f);
                Input {
                    bf16: bf16.ptr,
                    fp8: q.ptr,
                    scales: s.ptr,
                    lora: None,
                }
            }
            None => Input::bf16(bf16.ptr),
        }
    }

    /// The step's LoRA work for the linear layers: `max_tiles` bounds the
    /// tiles the step's metadata lists.
    pub fn lora_step(&self, l: &crate::lora::Loras, max_tiles: usize) -> crate::lora::StepLora {
        crate::lora::StepLora {
            tiles: self.m(self.meta.lora()),
            max_tiles,
            s: l.s.ptr,
            s_stride: (self.meta.max_tokens * l.lds * 4) as u64,
            lds: l.lds,
            shrink: l.shrink,
            expand: l.expand,
        }
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

    /// Queues the copy of the picks of the first `r` output rows, and of the
    /// first `s` sequences' drafts.
    pub fn download_ids(&mut self, gpu: &Gpu, r: usize, s: usize) -> Result<()> {
        gpu.dtoh(&mut self.ids_host, self.ids_out.ptr, r * 4)?;
        let k = self.meta.spec - 1;
        if let Some(m) = &mut self.mtp {
            gpu.dtoh(&mut m.drafts_host, m.drafts.ptr, k * s * 4)?;
        }
        Ok(())
    }

    /// The drafts downloaded with the last step's picks, if the model drafts:
    /// draft k of sequence i at k * s + i.
    pub fn drafts(&self, s: usize) -> Option<&[i32]> {
        let k = self.meta.spec - 1;
        self.mtp
            .as_ref()
            .map(|m| &m.drafts_host.as_slice::<i32>()[..k * s])
    }
}

/// Where a layer reads its batch metadata: the step's (see [`Meta`]), or a
/// draft pass's, which the GPU computes.
#[derive(Clone, Copy)]
struct View {
    pos: u64,
    slot_map: u64,
    tok_seq: u64,
    rope: u64,
    rope_stride: usize,
    cu: u64,
    slots: u64,
    fresh: u64,
    outs: u64,
    block_table: u64,
    /// Prefix-cache snapshots to write in the step (Meta::snaps), or 0.
    snaps: u64,
    /// The step's adapters, for the decoder layers.
    lora: Option<crate::lora::StepLora>,
}

impl View {
    fn of(a: &Act) -> Self {
        let me = &a.meta;
        Self {
            pos: a.m(me.pos()),
            slot_map: a.m(me.slot_map()),
            tok_seq: a.m(me.tok_seq()),
            rope: a.m(me.rope()),
            rope_stride: me.max_tokens,
            cu: a.m(me.cu()),
            slots: a.m(me.slots()),
            fresh: a.m(me.fresh()),
            outs: a.m(me.outs()),
            block_table: a.m(me.block_table()),
            snaps: a.m(me.snaps()),
            lora: None,
        }
    }
}

impl Model {
    /// Queues one step over `t` tokens from `s` sequences, as described by the
    /// uploaded metadata, ending with picks for `r` output rows (one per
    /// sequence, or one per token of a sequence verifying drafts) and, with
    /// multi-token prediction, one draft per sequence. Issues no host
    /// synchronization and no allocation, so it can be captured.
    /// One step over `t` tokens of `s` sequences with `r` output rows.
    /// `lora` holds the step's adapter work, when any sequence has one.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        st: &State,
        a: &Act,
        t: usize,
        s: usize,
        r: usize,
        lora: Option<crate::lora::StepLora>,
    ) -> Result<()> {
        use Arg::{F, I, P};
        let c = &self.cfg;
        let k = &self.k;
        let me = &a.meta;
        let (h, eps) = (c.hidden, c.eps);
        let (ti, ri) = (t as u32, r as u32);
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
        let v = View {
            lora,
            ..View::of(a)
        };
        for (li, layer) in self.layers.iter().enumerate() {
            self.layer(gpu, st, a, &v, layer, li > 0, t, s)?;
        }
        gpu.launch(
            k.rmsnorm,
            (ri, 1, 1),
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
            ]
            .into_iter()
            .chain(a.fp8_args(|f| &f.fin, r))
            .collect::<Vec<_>>(),
        )?;
        self.lm_head.forward(
            gpu,
            &self.quants,
            a.input(&a.fin, |f| &f.fin),
            r,
            a.logits.ptr,
            c.vocab,
            true,
        )?;
        let cap = (4 * a.pen_capacity) as u64;
        gpu.launch(
            k.penalize,
            (ri, 1, 1),
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
            (ri, 1, 1),
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
        )?;
        if let (Some(m), Some(ma)) = (&self.mtp, &a.mtp) {
            self.draft(gpu, st, a, m, ma, t, s)?;
        }
        Ok(())
    }

    /// The draft layer over every token row: each row's input is the token
    /// after it (known, or just picked) and the model's final hidden state
    /// at it, so its KV cache follows the model's. Each sequence then drafts
    /// greedily from one row: its last, or its last accepted one.
    #[allow(clippy::too_many_arguments)]
    fn draft(
        &self,
        gpu: &Gpu,
        st: &State,
        a: &Act,
        m: &Mtp,
        ma: &MtpActs,
        t: usize,
        s: usize,
    ) -> Result<()> {
        use Arg::{F, I, P};
        let c = &self.cfg;
        let k = &self.k;
        let me = &a.meta;
        let (h, eps) = (c.hidden, c.eps);
        let (ti, si) = (t as u32, s as u32);
        gpu.launch(
            k.mtp_tokens,
            (ti.div_ceil(256), 1, 1),
            256,
            0,
            &[
                P(a.m(me.mtp_ids())),
                P(a.ids_out.ptr),
                P(ma.tok.ptr),
                I(t as i32),
            ],
        )?;
        gpu.launch(
            k.embed,
            (ti, 1, 1),
            256,
            0,
            &[
                P(ma.tok.ptr),
                P(self.embed.ptr),
                P(a.m(me.mtp_src())),
                P(a.pool.ptr),
                P(a.xn.ptr),
                I(h as i32),
            ],
        )?;
        gpu.launch(
            k.mtp_input,
            (ti, 1, 1),
            256,
            0,
            &[
                P(a.xn.ptr),
                P(a.h.ptr),
                P(a.res.ptr),
                P(self.final_norm.ptr),
                P(m.norm_e.ptr),
                P(m.norm_h.ptr),
                P(ma.cat.ptr),
                I(h as i32),
                F(eps),
            ],
        )?;
        m.fc.forward(
            gpu,
            &self.quants,
            Input::bf16(ma.cat.ptr),
            t,
            a.res.ptr,
            h,
            false,
        )?;
        self.layer(gpu, st, a, &View::of(a), &m.layer, false, t, s)?;
        gpu.launch(
            k.mtp_accept,
            (si.div_ceil(128), 1, 1),
            128,
            0,
            &[
                P(a.m(me.ids())),
                P(a.m(me.cu())),
                P(a.m(me.ocu())),
                P(a.ids_out.ptr),
                P(a.m(me.outs())),
                I(me.spec as i32),
                P(ma.rows.ptr),
                I(s as i32),
            ],
        )?;
        gpu.launch(
            k.rmsnorm,
            (si, 1, 1),
            256,
            0,
            &[
                P(a.h.ptr),
                P(a.res.ptr),
                P(m.norm.ptr),
                P(a.fin.ptr),
                P(ma.rows.ptr),
                I(h as i32),
                F(eps),
                I(1),
                I(0),
            ]
            .into_iter()
            .chain(a.fp8_args(|f| &f.fin, s))
            .collect::<Vec<_>>(),
        )?;
        m.head.forward(
            gpu,
            &self.quants,
            a.input(&a.fin, |f| &f.fin),
            s,
            ma.logits.ptr,
            m.vocab,
            true,
        )?;
        gpu.launch(
            k.sample,
            (si, 1, 1),
            1024,
            0,
            &[
                P(ma.logits.ptr),
                I(m.vocab as i32),
                P(ma.zeros.ptr),
                P(a.m(me.top_ks())),
                P(a.m(me.top_ps())),
                P(a.m(me.min_ps())),
                P(a.m(me.seeds())),
                P(ma.drafts.ptr),
            ],
        )?;
        // Further drafts: each from the last draft and the draft layer's own
        // output, one position on.
        let view = View {
            pos: ma.dpos.ptr,
            slot_map: ma.dslot.ptr,
            tok_seq: ma.iota.ptr,
            rope: ma.drope.ptr,
            rope_stride: s,
            cu: ma.iota.ptr,
            slots: a.m(me.slots()),
            fresh: a.m(me.fresh()),
            outs: ma.neg.ptr,
            block_table: a.m(me.block_table()),
            snaps: 0,
            lora: None,
        };
        for d in 1..me.spec - 1 {
            gpu.launch(
                k.mtp_next,
                (si.div_ceil(128), 1, 1),
                128,
                0,
                &[
                    P(ma.rows.ptr),
                    P(a.m(me.pos())),
                    P(a.m(me.rope())),
                    I(me.max_tokens as i32),
                    P(a.m(me.block_table())),
                    I(me.max_blocks as i32),
                    I(BLOCK_SIZE as i32),
                    P(a.m(me.limits())),
                    I(d as i32),
                    P(ma.dpos.ptr),
                    P(ma.dslot.ptr),
                    P(ma.drope.ptr),
                    I(s as i32),
                ],
            )?;
            gpu.launch(
                k.embed,
                (si, 1, 1),
                256,
                0,
                &[
                    P(ma.drafts.ptr + (4 * (d - 1) * s) as u64),
                    P(self.embed.ptr),
                    P(ma.neg.ptr),
                    P(a.pool.ptr),
                    P(a.xn.ptr),
                    I(h as i32),
                ],
            )?;
            gpu.launch(
                k.mtp_input,
                (si, 1, 1),
                256,
                0,
                &[
                    P(a.xn.ptr),
                    P(0),
                    P(a.fin.ptr),
                    P(self.final_norm.ptr),
                    P(m.norm_e.ptr),
                    P(m.norm_h.ptr),
                    P(ma.cat.ptr),
                    I(h as i32),
                    F(eps),
                ],
            )?;
            m.fc.forward(
                gpu,
                &self.quants,
                Input::bf16(ma.cat.ptr),
                s,
                a.res.ptr,
                h,
                false,
            )?;
            self.layer(gpu, st, a, &view, &m.layer, false, s, s)?;
            gpu.launch(
                k.rmsnorm,
                (si, 1, 1),
                256,
                0,
                &[
                    P(a.h.ptr),
                    P(a.res.ptr),
                    P(m.norm.ptr),
                    P(a.fin.ptr),
                    P(0),
                    I(h as i32),
                    F(eps),
                    I(1),
                    I(0),
                ]
                .into_iter()
                .chain(a.fp8_args(|f| &f.fin, s))
                .collect::<Vec<_>>(),
            )?;
            m.head.forward(
                gpu,
                &self.quants,
                a.input(&a.fin, |f| &f.fin),
                s,
                ma.logits.ptr,
                m.vocab,
                true,
            )?;
            gpu.launch(
                k.sample,
                (si, 1, 1),
                1024,
                0,
                &[
                    P(ma.logits.ptr),
                    I(m.vocab as i32),
                    P(ma.zeros.ptr),
                    P(a.m(me.top_ks())),
                    P(a.m(me.top_ps())),
                    P(a.m(me.min_ps())),
                    P(a.m(me.seeds())),
                    P(ma.drafts.ptr + (4 * d * s) as u64),
                ],
            )?;
        }
        Ok(())
    }

    /// One decoder layer over the step's tokens. The first layer's input
    /// norm reads the residual stream alone (`add` false).
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        gpu: &Gpu,
        st: &State,
        a: &Act,
        v: &View,
        layer: &Layer,
        add: bool,
        t: usize,
        s: usize,
    ) -> Result<()> {
        use Arg::{F, I, P};
        let c = &self.cfg;
        let k = &self.k;
        let q = &self.quants;
        let me = &a.meta;
        let (h, eps) = (c.hidden, c.eps);
        let (ti, si) = (t as u32, s as u32);
        let outs = [P(v.outs), I(me.spec as i32)];
        let xn_in = a.input(&a.xn, |f| &f.xn).with_lora(v.lora);
        let norm = |x: u64, w: &Buf, add: bool| -> Result<()> {
            let mut args = vec![
                P(x),
                P(a.res.ptr),
                P(w.ptr),
                P(a.xn.ptr),
                P(0),
                I(h as i32),
                F(eps),
                I(add as i32),
                I(1),
            ];
            args.extend(a.fp8_args(|f| &f.xn, t));
            gpu.launch(k.rmsnorm, (ti, 1, 1), 256, 0, &args)
        };
        {
            norm(a.h.ptr, &layer.in_norm, add)?;
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
                    w_in.forward(gpu, q, xn_in, t, a.proj.ptr, pw, false)?;
                    gpu.launch(
                        k.gdn_conv,
                        // Chunks of CONV_CHUNK tokens; a sequence has at most t.
                        (cd.div_ceil(128) as u32, si, t.div_ceil(64) as u32),
                        128,
                        0,
                        &[
                            P(a.proj.ptr),
                            I(pw as i32),
                            P(conv_w.ptr),
                            P(st.conv[*state].ptr),
                            P(a.conv.ptr),
                            P(v.cu),
                            P(v.slots),
                            P(v.fresh),
                            I(cd as i32),
                            outs[0],
                            outs[1],
                            P(v.snaps),
                        ],
                    )?;
                    let args = [
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
                        I(st.state_bf16 as i32),
                        P(v.cu),
                        P(v.slots),
                        P(v.fresh),
                        P(a.core.ptr),
                        F(eps),
                        I(c.tiled_v as i32),
                    ];
                    if t > s {
                        // Prompt tokens: chunked on tensor cores, in two
                        // passes, then the norm.
                        let scratch = a.gdn.each_ref().map(|b| P(b.ptr));
                        gpu.launch(
                            k.gdn_prep,
                            ((t.div_ceil(32) + s) as u32, hv as u32, 1),
                            2 * GDN_DIM as u32,
                            GDN_PREP_SMEM as u32,
                            &[
                                [
                                    P(a.conv.ptr),
                                    P(a.proj.ptr),
                                    I(pw as i32),
                                    I(cd as i32),
                                    I(c.lin_k_heads as i32),
                                    I(hv as i32),
                                    P(neg_a.ptr),
                                    P(dt_bias.ptr),
                                    P(v.cu),
                                    I(s as i32),
                                    outs[0],
                                    outs[1],
                                    I(c.tiled_v as i32),
                                ]
                                .as_slice(),
                                &scratch,
                            ]
                            .concat(),
                        )?;
                        let split = gdn_split(s, hv, gpu);
                        gpu.launch(
                            k.gdn_scan[split.trailing_zeros() as usize],
                            (si, hv as u32, split as u32),
                            2 * GDN_DIM as u32,
                            gdn_scan_smem(split) as u32,
                            &[
                                args.as_slice(),
                                &a.fp8_args(|f| &f.core, t),
                                &outs,
                                &scratch,
                                &[P(v.snaps)],
                            ]
                            .concat(),
                        )?;
                        gpu.launch(
                            k.gdn_norm,
                            (t as u32, 1, 1),
                            256,
                            0,
                            &[
                                [
                                    P(a.proj.ptr),
                                    I(pw as i32),
                                    I(cd as i32),
                                    I(hv as i32),
                                    P(norm_w.ptr),
                                    P(a.core.ptr),
                                    I(t as i32),
                                    F(eps),
                                    P(v.cu),
                                    P(v.tok_seq),
                                ]
                                .as_slice(),
                                &a.fp8_args(|f| &f.core, t),
                                &outs,
                            ]
                            .concat(),
                        )?;
                    } else {
                        gpu.launch(
                            k.gdn_recurrent,
                            (si, hv as u32, 1),
                            2 * GDN_DIM as u32,
                            0,
                            &[args.as_slice(), &a.fp8_args(|f| &f.core, t), &outs].concat(),
                        )?;
                    }
                    w_out.forward(
                        gpu,
                        q,
                        a.input(&a.core, |f| &f.core).with_lora(v.lora),
                        t,
                        a.h.ptr,
                        h,
                        false,
                    )?;
                }
                Mixer::Attn {
                    w_qkv,
                    q_norm,
                    k_norm,
                    w_o,
                    cache,
                } => {
                    let qs = c.attn_proj();
                    let (kscale, vscale) = st.kv_scales(*cache);
                    w_qkv.forward(gpu, q, xn_in, t, a.proj.ptr, qs, false)?;
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
                            P(v.rope),
                            I(v.rope_stride as i32),
                            I(c.mrope_section[1] as i32),
                            I(c.mrope_section[2] as i32),
                            P(v.slot_map),
                            P(self.inv_freq.ptr),
                            P(a.q.ptr),
                            P(st.kc[*cache].ptr),
                            P(st.vc[*cache].ptr),
                            P(kscale),
                            P(vscale),
                            F(eps),
                        ],
                    )?;
                    let ranges = split_ranges(t, c.hkv, gpu.sms);
                    let part = match ranges {
                        1 => 0,
                        _ => a.attn_part.as_ref().map_or(0, |b| b.ptr),
                    };
                    gpu.launch(
                        if st.kv_fp8 { k.attn8 } else { k.attn },
                        (ti, c.hkv as u32, ranges as u32),
                        if c.hq / c.hkv >= 6 { 128 } else { 256 },
                        0,
                        &[
                            P(a.q.ptr),
                            P(st.kc[*cache].ptr),
                            P(st.vc[*cache].ptr),
                            P(kscale),
                            P(vscale),
                            P(a.proj.ptr),
                            I(qs as i32),
                            I(c.hq as i32),
                            I(c.hkv as i32),
                            P(v.pos),
                            P(v.tok_seq),
                            P(v.block_table),
                            I(me.max_blocks as i32),
                            I(BLOCK_SIZE as i32),
                            P(a.ao.ptr),
                            F(1.0 / (HEAD_DIM as f32).sqrt()),
                        ]
                        .into_iter()
                        .chain(if part == 0 {
                            a.fp8_args(|f| &f.ao, t)
                        } else {
                            [P(0), P(0), I(0), I(0)]
                        })
                        .chain([
                            P(v.cu),
                            I((t > s) as i32),
                            I(SPLIT_MIN_KEYS as i32),
                            P(part),
                            outs[0],
                            outs[1],
                        ])
                        .collect::<Vec<_>>(),
                    )?;
                    if part != 0 {
                        // Long contexts: combine the key ranges.
                        gpu.launch(
                            k.attn_merge,
                            (ti, c.hq as u32, 1),
                            HEAD_DIM as u32,
                            0,
                            &[
                                [
                                    P(part),
                                    I(ranges as i32),
                                    I(SPLIT_MIN_KEYS as i32),
                                    P(a.proj.ptr),
                                    I(qs as i32),
                                    I(c.hq as i32),
                                    P(v.pos),
                                    P(v.tok_seq),
                                    P(v.cu),
                                    I((t > s) as i32),
                                    P(a.ao.ptr),
                                ]
                                .as_slice(),
                                &a.fp8_args(|f| &f.ao, t),
                                &outs,
                            ]
                            .concat(),
                        )?;
                    }
                    if t > s {
                        // Prompt sequences: flash attention.
                        let group = c.hq / c.hkv;
                        let sh = if st.kv_fp8 { k.fa_shape8 } else { k.fa_shape };
                        let maps = st.maps.get(*cache);
                        let (kern, grid, threads, smem) = match (k.attn_wg, maps) {
                            (Some(wg), Some(_)) => (wg, (gpu.sms as u32, 1, 1), 384, WS_SMEM),
                            _ => (
                                if st.kv_fp8 {
                                    k.attn_flash8
                                } else {
                                    k.attn_flash
                                },
                                ((t.div_ceil(16 * sh.tiles) + s) as u32, c.hkv as u32, 1),
                                (32 * group * sh.tiles) as u32,
                                sh.smem(group),
                            ),
                        };
                        gpu.launch(
                            kern,
                            grid,
                            threads,
                            smem as u32,
                            &[
                                P(a.q.ptr),
                                P(st.kc[*cache].ptr),
                                P(st.vc[*cache].ptr),
                                P(kscale),
                                P(vscale),
                                P(a.proj.ptr),
                                I(qs as i32),
                                I(c.hq as i32),
                                I(c.hkv as i32),
                                P(v.pos),
                                P(v.cu),
                                I(s as i32),
                                P(v.block_table),
                                I(me.max_blocks as i32),
                                I(BLOCK_SIZE as i32),
                                P(a.ao.ptr),
                                F(1.0 / (HEAD_DIM as f32).sqrt()),
                                outs[0],
                                outs[1],
                            ]
                            .into_iter()
                            .chain(maps.into_iter().flat_map(|&(km, vm)| {
                                [
                                    Arg::Map(km),
                                    Arg::Map(vm),
                                    I((st.blocks * BLOCK_SIZE) as i32),
                                ]
                            }))
                            .collect::<Vec<_>>(),
                        )?;
                        if a.fp8.is_some() {
                            // FP8 copies for the output projection.
                            let [fq, fs, fmode, frows] = a.fp8_args(|f| &f.ao, t);
                            let width = c.hq * HEAD_DIM;
                            gpu.launch(
                                k.fp8_rows,
                                ((t * width / 8).div_ceil(256).min(65_535) as u32, 1, 1),
                                256,
                                0,
                                &[
                                    P(a.ao.ptr),
                                    I(t as i32),
                                    I(width as i32),
                                    fq,
                                    fs,
                                    fmode,
                                    frows,
                                ],
                            )?;
                        }
                    }
                    w_o.forward(
                        gpu,
                        q,
                        a.input(&a.ao, |f| &f.ao).with_lora(v.lora),
                        t,
                        a.h.ptr,
                        h,
                        false,
                    )?;
                }
            }
            norm(a.h.ptr, &layer.post_norm, true)?;
            layer
                .w_gu
                .forward(gpu, q, xn_in, t, a.gu.ptr, 2 * c.inter, false)?;
            let blocks = (t * c.inter / 8).div_ceil(256).min(8192) as u32;
            gpu.launch(
                k.silu_mul,
                (blocks, 1, 1),
                256,
                0,
                &[
                    [P(a.gu.ptr), P(a.act.ptr), I(c.inter as i32), I(t as i32)].as_slice(),
                    &a.fp8_args(|f| &f.act, t),
                ]
                .concat(),
            )?;
            layer.w_down.forward(
                gpu,
                q,
                a.input(&a.act, |f| &f.act).with_lora(v.lora),
                t,
                a.h.ptr,
                h,
                false,
            )?;
        }
        Ok(())
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
        assert_eq!(
            c.slot_bytes(false),
            18 * (16 * 128 * 128 * 4 + 3 * 6144 * 2)
        );
        assert_eq!(c.slot_bytes(true), 18 * (16 * 128 * 128 * 2 + 3 * 6144 * 2));
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
            spec: 3,
        };
        let starts = [
            m.ids(),
            m.pos(),
            m.slot_map(),
            m.tok_seq(),
            m.emb_src(),
            m.rope(),
            m.mtp_ids(),
            m.mtp_src(),
            m.cu(),
            m.slots(),
            m.fresh(),
            m.outs(),
            m.ocu(),
            m.last(),
            m.temps(),
            m.seeds(),
            m.top_ks(),
            m.top_ps(),
            m.min_ps(),
            m.pen_cu(),
            m.limits(),
            m.snaps(),
            m.block_table(),
            m.words(),
        ];
        assert!(starts.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(m.slots() - m.cu(), 9, "cu holds seqs + 1 offsets");
    }
}
