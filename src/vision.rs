//! The Qwen3.5 vision encoder on the GPU: patch embedding with interpolated
//! learned positions, ViT blocks with 2D rotary attention, and the merger that
//! turns each 2x2 block of patches into one text-model embedding.

use crate::gpu::{Arg, Buf, Gpu, Kernel, Mat, Pinned};
use crate::image::{Image, PATCH_VALUES, position_taps};
use crate::model::f32_bytes;
use crate::source::Checkpoint;
use crate::{Error, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub out: usize,
    /// Side of the square learned position table.
    pub side: usize,
    /// Attention head width (64 or 72).
    pub head: usize,
}

impl VisionConfig {
    pub fn parse(raw: &serde_json::Value) -> Result<Option<Self>> {
        let v = &raw["vision_config"];
        if v.is_null() {
            return Ok(None);
        }
        let u = |k: &str| {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| Error::Config(format!("vision_config lacks {k}")))
        };
        let cfg = Self {
            depth: u("depth")?,
            hidden: u("hidden_size")?,
            inter: u("intermediate_size")?,
            heads: u("num_heads")?,
            out: u("out_hidden_size")?,
            side: (u("num_position_embeddings")? as f64).sqrt() as usize,
            head: u("hidden_size")? / u("num_heads")?,
        };
        let ok = cfg.hidden == cfg.heads * cfg.head
            && cfg.head.is_multiple_of(4)
            && cfg.head <= 128
            && u("patch_size")? == 16
            && u("temporal_patch_size")? == 2
            && u("spatial_merge_size")? == 2
            && v["deepstack_visual_indexes"]
                .as_array()
                .is_none_or(Vec::is_empty)
            && cfg.side * cfg.side == u("num_position_embeddings")?;
        if !ok {
            return Err(Error::Config("unsupported vision encoder shape".into()));
        }
        Ok(Some(cfg))
    }
}

struct Block {
    norm1: (Buf, Buf),
    qkv: (Buf, Buf),
    proj: (Buf, Buf),
    norm2: (Buf, Buf),
    fc1: (Buf, Buf),
    fc2: (Buf, Buf),
}

struct Kernels {
    embed: Kernel,
    layernorm: Kernel,
    qkv_rope: Kernel,
    softmax: Kernel,
    bias: Kernel,
}

/// Scratch for up to `max_patches` patches per encoder run.
struct Scratch {
    max_patches: usize,
    staging: Pinned,
    pix: Buf,
    taps: Buf,
    x: Buf,
    xn: Buf,
    y: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    ao: Buf,
    mid: Buf,
    scores: Buf,
    probs: Buf,
    score_elems: usize,
}

pub struct Vision {
    pub cfg: VisionConfig,
    patch: (Buf, Buf),
    pos: Buf,
    inv_freq: Buf,
    blocks: Vec<Block>,
    merger_norm: (Buf, Buf),
    merger_fc1: (Buf, Buf),
    merger_fc2: (Buf, Buf),
    k: Kernels,
    s: Scratch,
}

/// Bytes of host staging per patch: pixels, position taps and row/column.
fn staging_per_patch() -> usize {
    PATCH_VALUES * 2 + 4 * 4 + 4 * 4 + 2 * 4
}

impl Vision {
    pub fn load(gpu: &Gpu, dir: &Path, max_patches: usize) -> Result<Option<Self>> {
        let started = gpu.begin_uploads()?;
        let v = Self::load_weights(gpu, dir, max_patches);
        if started {
            gpu.end_uploads()?;
        }
        v
    }

    fn load_weights(gpu: &Gpu, dir: &Path, max_patches: usize) -> Result<Option<Self>> {
        if crate::source::gguf_path(dir).is_some() {
            return Ok(None); // GGUF text models; projector files are not read yet
        }
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let Some(cfg) = VisionConfig::parse(&raw)? else {
            return Ok(None);
        };
        let ck = Checkpoint::open(dir)?;
        if !ck.has("model.visual.patch_embed.proj.weight") {
            return Ok(None);
        }
        let (d, i) = (cfg.hidden, cfg.inter);
        let w = |name: &str, n: usize| -> Result<Buf> {
            let b = ck.bf16_bytes(&format!("/model.visual.{name}"), n)?;
            gpu.upload_pieces(&[b.piece()])
        };
        let f = |name: &str, n: usize| -> Result<Buf> {
            gpu.upload(&f32_bytes(&ck.f32(&format!("/model.visual.{name}"), n)?))
        };
        let pair = |name: &str, rows: usize, cols: usize| -> Result<(Buf, Buf)> {
            Ok((
                w(&format!("{name}.weight"), rows * cols)?,
                f(&format!("{name}.bias"), rows)?,
            ))
        };
        let norm = |name: &str, n: usize| -> Result<(Buf, Buf)> {
            Ok((
                f(&format!("{name}.weight"), n)?,
                f(&format!("{name}.bias"), n)?,
            ))
        };
        let mut blocks = Vec::with_capacity(cfg.depth);
        for b in 0..cfg.depth {
            let p = format!("blocks.{b}");
            blocks.push(Block {
                norm1: norm(&format!("{p}.norm1"), d)?,
                qkv: pair(&format!("{p}.attn.qkv"), 3 * d, d)?,
                proj: pair(&format!("{p}.attn.proj"), d, d)?,
                norm2: norm(&format!("{p}.norm2"), d)?,
                fc1: pair(&format!("{p}.mlp.linear_fc1"), i, d)?,
                fc2: pair(&format!("{p}.mlp.linear_fc2"), d, i)?,
            });
        }
        let head = cfg.head;
        let inv_freq: Vec<f32> = (0..head / 4)
            .map(|j| 1.0 / 10000f32.powf((2 * j) as f32 / (head / 2) as f32))
            .collect();
        let pm = max_patches;
        let score_elems = 64 << 20;
        let s = Scratch {
            max_patches: pm,
            staging: gpu.pinned(pm * staging_per_patch())?,
            pix: gpu.alloc(pm * PATCH_VALUES * 2)?,
            taps: gpu.alloc(pm * (4 * 4 + 4 * 4 + 2 * 4))?,
            x: gpu.alloc(pm * d * 2)?,
            xn: gpu.alloc(pm * d * 2)?,
            y: gpu.alloc(pm * (3 * d).max(i).max(d) * 4)?,
            q: gpu.alloc(pm * d * 2)?,
            k: gpu.alloc(pm * d * 2)?,
            v: gpu.alloc(pm * d * 2)?,
            ao: gpu.alloc(pm * d * 2)?,
            mid: gpu.alloc(pm * i.max(d) * 2)?,
            scores: gpu.alloc(score_elems * 4)?,
            probs: gpu.alloc(score_elems * 2)?,
            score_elems,
        };
        Ok(Some(Self {
            patch: pair("patch_embed.proj", d, PATCH_VALUES)?,
            pos: w("pos_embed.weight", cfg.side * cfg.side * d)?,
            inv_freq: gpu.upload(&f32_bytes(&inv_freq))?,
            blocks,
            merger_norm: norm("merger.norm", d)?,
            merger_fc1: pair("merger.linear_fc1", 4 * d, 4 * d)?,
            merger_fc2: pair("merger.linear_fc2", cfg.out, 4 * d)?,
            k: Kernels {
                embed: gpu.kernel("vit_embed")?,
                layernorm: gpu.kernel("layernorm")?,
                qkv_rope: gpu.kernel("vit_qkv_rope")?,
                softmax: gpu.kernel("softmax_rows")?,
                bias: gpu.kernel("vit_bias")?,
            },
            s,
            cfg,
        }))
    }

    pub fn max_patches(&self) -> usize {
        self.s.max_patches
    }

    /// Encodes `images` (at most `max_patches` patches together) and writes
    /// their embeddings, image after image, as bf16 rows starting at `out`.
    /// Synchronizes before it reuses its staging buffer.
    pub fn encode(&mut self, gpu: &Gpu, images: &[&Image], out: u64) -> Result<()> {
        use Arg::{F, I, P};
        let total: usize = images.iter().map(|i| i.patches()).sum();
        if total > self.s.max_patches {
            return Err(Error::Request(format!(
                "{total} image patches exceed the encoder's {}",
                self.s.max_patches
            )));
        }
        gpu.sync()?;
        let (d, hd, heads) = (self.cfg.hidden, self.cfg.inter, self.cfg.heads);
        let head = self.cfg.head;
        // Stage pixels, then taps (idx, weights) and row/column per patch.
        {
            let words = self.s.staging.as_mut_slice::<u16>();
            let mut at = 0;
            for img in images {
                for (dst, v) in words[at..].iter_mut().zip(&img.pixels) {
                    *dst = v.to_bits();
                }
                at += img.pixels.len();
            }
        }
        let pix_bytes = total * PATCH_VALUES * 2;
        gpu.htod(self.s.pix.ptr, &self.s.staging, 0, pix_bytes)?;
        let (mut idx, mut wts, mut hw) = (Vec::new(), Vec::new(), Vec::new());
        for img in images {
            let (i, w) = position_taps(img.grid_h, img.grid_w, self.cfg.side);
            idx.extend(i);
            wts.extend(w);
            for (r, c) in crate::image::patch_order(img.grid_h, img.grid_w) {
                hw.extend([r as i32, c as i32]);
            }
        }
        let aux_at = pix_bytes.next_multiple_of(16);
        {
            let bytes = self.s.staging.as_mut_slice::<u8>();
            let mut o = aux_at;
            for v in idx
                .iter()
                .map(|x| x.to_le_bytes())
                .chain(wts.iter().map(|x| x.to_le_bytes()))
                .chain(hw.iter().map(|x| x.to_le_bytes()))
            {
                bytes[o..o + 4].copy_from_slice(&v);
                o += 4;
            }
        }
        let aux_bytes = (idx.len() + wts.len() + hw.len()) * 4;
        gpu.htod(self.s.taps.ptr, &self.s.staging, aux_at, aux_bytes)?;
        let (idx_ptr, w_ptr) = (self.s.taps.ptr, self.s.taps.ptr + (idx.len() * 4) as u64);
        let hw_ptr = w_ptr + (wts.len() * 4) as u64;
        let s = &self.s;
        let k = &self.k;
        let p = total;
        let rows = |n: usize| (n as u32, 1, 1);
        let bias = |y: u64, b: &Buf, out: u64, width: usize, n: usize, mode: i32| -> Result<()> {
            let blocks = n.div_ceil(256).min(65_535) as u32;
            gpu.launch(
                k.bias,
                (blocks, 1, 1),
                256,
                0,
                &[
                    P(y),
                    P(b.ptr),
                    P(out),
                    I(width as i32),
                    P(n as u64),
                    I(mode),
                ],
            )
        };
        // Patch embedding plus interpolated position.
        gpu.gemm(
            s.pix.ptr,
            self.patch.0.ptr,
            s.y.ptr,
            p,
            PATCH_VALUES,
            d,
            true,
        )?;
        gpu.launch(
            k.embed,
            rows(p),
            256,
            0,
            &[
                P(s.y.ptr),
                P(self.patch.1.ptr),
                P(self.pos.ptr),
                P(idx_ptr),
                P(w_ptr),
                P(s.x.ptr),
                I(d as i32),
            ],
        )?;
        let ln = |x: u64, n: &(Buf, Buf), out: u64, r: usize| -> Result<()> {
            gpu.launch(
                k.layernorm,
                rows(r),
                128,
                0,
                &[P(x), P(n.0.ptr), P(n.1.ptr), P(out), I(d as i32), F(1e-6)],
            )
        };
        for b in &self.blocks {
            ln(s.x.ptr, &b.norm1, s.xn.ptr, p)?;
            gpu.gemm(s.xn.ptr, b.qkv.0.ptr, s.y.ptr, p, d, 3 * d, true)?;
            gpu.launch(
                k.qkv_rope,
                (p as u32, heads as u32, 1),
                head as u32,
                0,
                &[
                    P(s.y.ptr),
                    P(b.qkv.1.ptr),
                    P(hw_ptr),
                    P(self.inv_freq.ptr),
                    P(s.q.ptr),
                    P(s.k.ptr),
                    P(s.v.ptr),
                    I(p as i32),
                    I(heads as i32),
                ],
            )?;
            // Attention within each image, in query chunks that fit the scratch.
            let mut off = 0;
            for img in images {
                let n = img.patches();
                let chunk = (s.score_elems / (heads * n)).clamp(1, n);
                let mut c0 = 0;
                while c0 < n {
                    let cq = chunk.min(n - c0);
                    let head_stride = p * head;
                    gpu.gemm_batched(
                        n,
                        cq,
                        head,
                        Mat {
                            ptr: s.k.ptr + (off * head * 2) as u64,
                            ld: head,
                            stride: head_stride,
                            trans: true,
                            f32: false,
                        },
                        Mat {
                            ptr: s.q.ptr + ((off + c0) * head * 2) as u64,
                            ld: head,
                            stride: head_stride,
                            trans: false,
                            f32: false,
                        },
                        Mat {
                            ptr: s.scores.ptr,
                            ld: n,
                            stride: cq * n,
                            trans: false,
                            f32: true,
                        },
                        heads,
                    )?;
                    gpu.launch(
                        k.softmax,
                        rows(heads * cq),
                        256,
                        0,
                        &[
                            P(s.scores.ptr),
                            P(s.probs.ptr),
                            I(n as i32),
                            F(1.0 / (head as f32).sqrt()),
                        ],
                    )?;
                    gpu.gemm_batched(
                        head,
                        cq,
                        n,
                        Mat {
                            ptr: s.v.ptr + (off * head * 2) as u64,
                            ld: head,
                            stride: head_stride,
                            trans: false,
                            f32: false,
                        },
                        Mat {
                            ptr: s.probs.ptr,
                            ld: n,
                            stride: cq * n,
                            trans: false,
                            f32: false,
                        },
                        Mat {
                            ptr: s.ao.ptr + ((off + c0) * d * 2) as u64,
                            ld: d,
                            stride: head,
                            trans: false,
                            f32: false,
                        },
                        heads,
                    )?;
                    c0 += cq;
                }
                off += n;
            }
            gpu.gemm(s.ao.ptr, b.proj.0.ptr, s.y.ptr, p, d, d, true)?;
            bias(s.y.ptr, &b.proj.1, s.x.ptr, d, p * d, 0)?;
            ln(s.x.ptr, &b.norm2, s.xn.ptr, p)?;
            gpu.gemm(s.xn.ptr, b.fc1.0.ptr, s.y.ptr, p, d, hd, true)?;
            bias(s.y.ptr, &b.fc1.1, s.mid.ptr, hd, p * hd, 1)?;
            gpu.gemm(s.mid.ptr, b.fc2.0.ptr, s.y.ptr, p, hd, d, true)?;
            bias(s.y.ptr, &b.fc2.1, s.x.ptr, d, p * d, 0)?;
        }
        // Merger: norm each patch, then each 2x2 block (4 consecutive patches).
        ln(s.x.ptr, &self.merger_norm, s.xn.ptr, p)?;
        let m = p / 4;
        gpu.gemm(
            s.xn.ptr,
            self.merger_fc1.0.ptr,
            s.y.ptr,
            m,
            4 * d,
            4 * d,
            true,
        )?;
        bias(s.y.ptr, &self.merger_fc1.1, s.mid.ptr, 4 * d, m * 4 * d, 2)?;
        gpu.gemm(
            s.mid.ptr,
            self.merger_fc2.0.ptr,
            s.y.ptr,
            m,
            4 * d,
            self.cfg.out,
            true,
        )?;
        bias(
            s.y.ptr,
            &self.merger_fc2.1,
            out,
            self.cfg.out,
            m * self.cfg.out,
            3,
        )
    }
}
