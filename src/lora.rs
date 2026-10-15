//! LoRA adapters: PEFT directories and llama.cpp GGUF adapters, prepared on
//! the host in the layout the kernels read and swapped into a fixed set of
//! GPU slots. A request names its adapter; every step's tokens of one adapter
//! go through `lora_shrink` (S = x A^T for all of a layer's parts at once) and
//! `lora_expand` (y += S B^T per part) after the layer's own matrix multiply.

use crate::gguf::Gguf;
use crate::gpu::{Buf, Gpu, Kernel, Pinned, Pinner};
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Parts a linear layer fuses at most (linear attention's qkv, z, b and a).
pub const MAX_PARTS: usize = 4;
/// Ranks are padded to whole tensor-core steps.
const RANK_STEP: usize = 16;
/// Tokens per kernel tile; a tile holds one adapter's tokens.
pub const TILE: usize = 16;

/// Where a part's rows or a layer's input columns are linear-attention value
/// heads, which GGUF files store tiled (value head h uses key head h % key
/// heads) instead of grouped by key head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heads {
    None,
    /// Rows (or columns) from `start` on, `block` per value head.
    From {
        start: usize,
        block: usize,
    },
}

/// One HF module of a fused linear layer, in output order.
#[derive(Debug, Clone)]
pub struct SitePart {
    /// Module path under the layer, e.g. "linear_attn.in_proj_qkv".
    pub module: String,
    pub rows: usize,
    pub off: usize,
    pub heads: Heads,
}

/// A linear layer adapters may target.
#[derive(Debug, Clone)]
pub struct Site {
    pub layer: usize,
    pub cols: usize,
    pub parts: Vec<SitePart>,
    pub col_heads: Heads,
}

impl Site {
    fn max_rows(&self) -> usize {
        self.parts.iter().map(|p| p.rows).max().unwrap_or(0)
    }
}

/// What the kernels read per (layer, slot); mirrors `LoraDesc` in lora.cuh.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Desc {
    a: u64,
    b: [u64; MAX_PARTS],
    rank: i32,
    r: [i32; MAX_PARTS],
    r_off: [i32; MAX_PARTS],
    off: [i32; MAX_PARTS],
    rows: [i32; MAX_PARTS],
    _pad: i32,
}

/// An adapter's data for one site: A for all targeted parts, stacked, then
/// each part's B, as byte offsets into the adapter's buffer.
#[derive(Debug, Clone, Default)]
struct SiteData {
    a: usize,
    rank: usize,
    parts: [(usize, usize, usize); MAX_PARTS], // (b offset, r offset, r)
}

/// An adapter on the host, laid out exactly as its GPU slot holds it.
pub struct Adapter {
    pub name: String,
    pub rank: usize,
    data: Pinned,
    len: usize,
    sites: Vec<SiteData>,
}

/// One module's factors: A [r, cols] and B [rows, r] (scale folded in), f32.
struct Factors {
    r: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

/// Grouped-to-tiled order of `hv` value heads: tiled head t holds grouped head g(t).
fn tiled_order(hk: usize, hv: usize) -> Vec<usize> {
    let per = hv / hk;
    (0..hv).map(|t| (t % hk) * per + t / hk).collect()
}

/// Reorders `n` blocks of `block` elements, starting at block index `start /
/// block`, from grouped to tiled order (or back).
fn permute_blocks(v: &mut [f32], start: usize, block: usize, order: &[usize], inverse: bool) {
    let n = order.len();
    let span = &mut v[start..start + n * block];
    let old = span.to_vec();
    for (t, &g) in order.iter().enumerate() {
        let (dst, src) = if inverse { (g, t) } else { (t, g) };
        span[dst * block..(dst + 1) * block].copy_from_slice(&old[src * block..(src + 1) * block]);
    }
}

impl Factors {
    /// Moves value heads between HF's grouped order and GGUF's tiled order.
    fn permute(
        &mut self,
        part: &SitePart,
        col_heads: Heads,
        cols: usize,
        order: &[usize],
        inverse: bool,
    ) {
        if let Heads::From { start, block } = part.heads {
            // B's rows are output features, r values each.
            permute_blocks(&mut self.b, start * self.r, block * self.r, order, inverse);
        }
        if let Heads::From { start, block } = col_heads {
            // A's columns are input features: permute within every row.
            for row in self.a.chunks_mut(cols) {
                permute_blocks(row, start, block, order, inverse);
            }
        }
    }
}

/// An adapter's modules by (layer, module path).
type Modules = HashMap<(usize, String), Factors>;

fn bf16_bytes(dst: &mut [u8], src: impl Iterator<Item = f32>) {
    for (d, x) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        *d = half::bf16::from_f32(x).to_le_bytes();
    }
}

/// Reads PEFT's `adapter_config.json` and `adapter_model.safetensors`.
fn read_peft(dir: &Path) -> Result<(Modules, bool)> {
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("adapter_config.json")).map_err(|e| {
            Error::Config(format!(
                "{}: {e}",
                dir.join("adapter_config.json").display()
            ))
        })?,
    )
    .map_err(|e| Error::Config(format!("adapter_config.json: {e}")))?;
    let flag = |k: &str| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    if flag("use_dora") {
        return Err(Error::Config("DoRA adapters are not supported".into()));
    }
    if cfg
        .get("modules_to_save")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err(Error::Config(
            "adapters with modules_to_save (full embedding or output layers) are not supported"
                .into(),
        ));
    }
    if cfg
        .get("bias")
        .and_then(|v| v.as_str())
        .is_some_and(|b| b != "none")
    {
        return Err(Error::Config(
            "adapters with trained biases are not supported".into(),
        ));
    }
    if cfg
        .get("alpha_pattern")
        .and_then(|v| v.as_object())
        .is_some_and(|o| !o.is_empty())
    {
        return Err(Error::Config(
            "adapters with alpha_pattern are not supported".into(),
        ));
    }
    let alpha = cfg
        .get("lora_alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(8.0);
    let rslora = flag("use_rslora");
    let ck = crate::source::Checkpoint::open(dir)?;
    let mut out = Modules::new();
    for name in ck.names() {
        let Some(stem) = name.strip_suffix(".lora_A.weight") else {
            continue;
        };
        if stem.contains("visual") {
            return Err(Error::Config(
                "adapters for the vision encoder are not supported".into(),
            ));
        }
        if stem.contains(".mtp.") {
            continue;
        }
        let Some((_, rest)) = stem.rsplit_once("layers.") else {
            return Err(Error::Config(format!("{name}: not a decoder layer")));
        };
        let (layer, module) = rest
            .split_once('.')
            .ok_or_else(|| Error::Config(format!("{name}: unexpected name")))?;
        let layer: usize = layer
            .parse()
            .map_err(|_| Error::Config(format!("{name}: unexpected name")))?;
        let (ashape, bshape) = (ck.shape(name)?, ck.shape(&format!("{stem}.lora_B.weight"))?);
        let (r, cols, rows) = (ashape[0], ashape[1], bshape[0]);
        if bshape[1] != r {
            return Err(Error::Config(format!(
                "{stem}: A is rank {r}, B {:?}",
                bshape
            )));
        }
        let scale = (alpha / if rslora { (r as f64).sqrt() } else { r as f64 }) as f32;
        let a = ck.f32_full(name, r * cols)?;
        let mut b = ck.f32_full(&format!("{stem}.lora_B.weight"), rows * r)?;
        b.iter_mut().for_each(|x| *x *= scale);
        out.insert((layer, module.to_string()), Factors { r, a, b });
    }
    // PEFT files hold HF's grouped value-head order.
    Ok((out, false))
}

/// Reads a llama.cpp LoRA (`general.type` = "adapter"), whose tensors are
/// named after the GGUF base (`blk.N.attn_q.weight.lora_a`).
fn read_gguf(path: &Path, sites: &[Site]) -> Result<(Modules, bool)> {
    let g = Gguf::open(path)?;
    if g.get("general.type").and_then(|v| v.as_str()) != Some("adapter") {
        return Err(Error::Config(format!(
            "{} is not a LoRA adapter",
            path.display()
        )));
    }
    let alpha = g.get("adapter.lora.alpha").and_then(|v| v.as_f64());
    let mut out = Modules::new();
    for site in sites {
        for part in &site.parts {
            let hf = format!("layers.{}.{}.weight", site.layer, part.module);
            let base = crate::source::gguf_name(&g, &hf);
            let (Some((ta, da)), Some((tb, db))) = (
                g.tensor(&format!("{base}.lora_a")),
                g.tensor(&format!("{base}.lora_b")),
            ) else {
                continue;
            };
            // ggml lists the innermost dimension first: A is (cols, r), B (r, rows).
            let (cols, r, rows) = (ta.dims[0], ta.dims[1], tb.dims[1]);
            if tb.dims[0] != r || cols != site.cols || rows != part.rows {
                return Err(Error::Config(format!("{base}: unexpected LoRA shapes")));
            }
            let f = |ty, d: &[u8]| {
                crate::source::to_f32(ty, d).ok_or_else(|| {
                    Error::Config(format!("{base}: LoRA tensors must be F32, F16 or BF16"))
                })
            };
            let a = f(ta.ty, da)?;
            let scale = alpha.map_or(1.0, |al| (al / r as f64) as f32);
            let b = f(tb.ty, db)?.into_iter().map(|x| x * scale).collect();
            out.insert((site.layer, part.module.clone()), Factors { r, a, b });
        }
    }
    if out.is_empty() {
        return Err(Error::Config(format!(
            "{}: no LoRA tensors for this model",
            path.display()
        )));
    }
    // llama.cpp converts adapters like their base: value heads tiled.
    Ok((out, true))
}

/// Everything preparing an adapter needs, apart from the engine: reading
/// and converting one takes tens of milliseconds, so the engine hands this
/// to a worker thread and keeps serving (see [`Loras::prep`]).
#[derive(Clone)]
pub struct Prep {
    sites: Vec<Site>,
    base_gguf: bool,
    heads: (usize, usize),
    max_rank: usize,
    pinner: Pinner,
}

impl Prep {
    /// Reads an adapter from a PEFT directory or a GGUF file, laid out as its
    /// GPU slot will hold it.
    pub fn read(&self, name: &str, path: &Path) -> Result<Adapter> {
        let (modules, tiled_file) = if path.is_dir() {
            read_peft(path)?
        } else {
            read_gguf(path, &self.sites)?
        };
        self.prepare(name, modules, tiled_file)
    }

    fn prepare(&self, name: &str, mut modules: Modules, tiled_file: bool) -> Result<Adapter> {
        if tiled_file && !self.base_gguf {
            return Err(Error::Config(format!(
                "{name}: GGUF adapters need a GGUF base (their value heads are reordered)"
            )));
        }
        let (hk, hv) = self.heads;
        // PEFT adapters on a GGUF base move value heads to the base's order.
        let order = (self.base_gguf && !tiled_file && hk != hv).then(|| tiled_order(hk, hv));
        let mut rank = 0;
        // Layout first, then fill.
        let mut sites = Vec::with_capacity(self.sites.len());
        let mut len = 0;
        for site in &self.sites {
            let mut sd = SiteData::default();
            let mut r_total = 0;
            for (pi, part) in site.parts.iter().enumerate() {
                if let Some(f) = modules.get(&(site.layer, part.module.clone())) {
                    if f.a.len() != f.r * site.cols || f.b.len() != part.rows * f.r {
                        return Err(Error::Config(format!(
                            "{name}: layer {} {}: shapes do not match the model",
                            site.layer, part.module
                        )));
                    }
                    let rp = f.r.next_multiple_of(RANK_STEP);
                    if rp > self.max_rank {
                        return Err(Error::Config(format!(
                            "{name}: rank {} is over --max-lora-rank {}",
                            f.r, self.max_rank
                        )));
                    }
                    rank = rank.max(f.r);
                    sd.parts[pi] = (0, r_total, rp);
                    r_total += rp;
                }
            }
            sd.rank = r_total;
            if r_total > 0 {
                sd.a = len;
                len += r_total * site.cols * 2;
                for (pi, part) in site.parts.iter().enumerate() {
                    if sd.parts[pi].2 > 0 {
                        sd.parts[pi].0 = len;
                        len += part.rows * sd.parts[pi].2 * 2;
                    }
                }
            }
            sites.push(sd);
        }
        let used = modules.len();
        let placed: usize = self
            .sites
            .iter()
            .map(|s| {
                s.parts
                    .iter()
                    .filter(|p| modules.contains_key(&(s.layer, p.module.clone())))
                    .count()
            })
            .sum();
        if placed == 0 || placed < used {
            return Err(Error::Config(format!(
                "{name}: {} of its {used} modules match this model's layers",
                placed
            )));
        }
        let mut data = self.pinner.pinned(len.max(1))?;
        let bytes = data.as_mut_slice::<u8>();
        bytes.fill(0);
        for (site, sd) in self.sites.iter().zip(&sites) {
            for (pi, part) in site.parts.iter().enumerate() {
                let (b_off, r_off, rp) = sd.parts[pi];
                if rp == 0 {
                    continue;
                }
                let mut f = modules
                    .remove(&(site.layer, part.module.clone()))
                    .expect("placed above");
                if let Some(order) = &order {
                    f.permute(part, site.col_heads, site.cols, order, false);
                }
                // A rows r_off..r_off + r of the stacked A; padding stays zero.
                let a0 = sd.a + r_off * site.cols * 2;
                bf16_bytes(
                    &mut bytes[a0..a0 + f.r * site.cols * 2],
                    f.a.iter().copied(),
                );
                // B [rows, rp], the last rp - r columns zero.
                for row in 0..part.rows {
                    let at = b_off + row * rp * 2;
                    bf16_bytes(
                        &mut bytes[at..at + f.r * 2],
                        f.b[row * f.r..(row + 1) * f.r].iter().copied(),
                    );
                }
            }
        }
        Ok(Adapter {
            name: name.to_string(),
            rank,
            data,
            len,
            sites,
        })
    }
}

/// The LoRA state of an engine: adapters by id, the GPU slots and their
/// per-layer descriptor tables.
pub struct Loras {
    sites: Vec<Site>,
    /// Whether the base is a GGUF file (value heads tiled), and its linear
    /// attention's (key heads, value heads).
    base_gguf: bool,
    heads: (usize, usize),
    pub max_slots: usize,
    pub max_rank: usize,
    slot_bytes: usize,
    pool: Buf,
    tables: Buf,
    host_tables: Pinned,
    adapters: HashMap<u32, Adapter>,
    names: HashMap<String, u32>,
    next_id: u32,
    /// Intermediate S, two buffers of [tokens, MAX_PARTS * max_rank] f32 that
    /// consecutive layers alternate between (see lora_expand).
    pub s: Buf,
    pub lds: usize,
    pub shrink: Kernel,
    pub expand: Kernel,
    /// The pair for decoding steps (tiles of a few tokens): on AMD,
    /// amd/lora.cuh's, faster there (1 token: 12.6 against 16.4 us a layer);
    /// the originals stay faster for tiles of 16 prompt tokens.
    pub shrink_dec: Kernel,
    pub expand_dec: Kernel,
}

/// Per-layer hook a linear layer keeps: its descriptor table and launch shape.
#[derive(Debug, Clone, Copy)]
pub struct Hook {
    pub table: u64,
    pub parts: usize,
    pub max_rows: usize,
    /// Which of the two S buffers this layer accumulates into.
    pub parity: usize,
}

/// The step's LoRA work, passed to each linear layer.
#[derive(Clone, Copy)]
pub struct StepLora {
    /// The step's tiles in device metadata: [count, (start, len, slot) x n].
    pub tiles: u64,
    /// Upper bound on the tile count (the launch grid).
    pub max_tiles: usize,
    /// The two S buffers, `s_stride` bytes apart.
    pub s: u64,
    pub s_stride: u64,
    pub lds: usize,
    pub shrink: Kernel,
    pub expand: Kernel,
}

impl Loras {
    pub fn new(
        gpu: &Gpu,
        sites: Vec<Site>,
        base_gguf: bool,
        heads: (usize, usize),
        max_slots: usize,
        max_rank: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        // lora_expand's buffer alternation needs an even number of layers.
        assert!(sites.len().is_multiple_of(2), "LoRA sites come in pairs");
        let max_rank = max_rank.next_multiple_of(RANK_STEP);
        let slot_bytes: usize = sites
            .iter()
            .map(|s| {
                s.parts.len() * max_rank * s.cols * 2
                    + s.parts.iter().map(|p| p.rows * max_rank * 2).sum::<usize>()
            })
            .sum();
        let entries = sites.len() * (max_slots + 1);
        let table_bytes = entries * std::mem::size_of::<Desc>();
        let lds = MAX_PARTS * max_rank;
        let mut host_tables = gpu.pinned(table_bytes)?;
        host_tables.as_mut_slice::<u8>().fill(0);
        let tables = gpu.alloc_zeroed(table_bytes)?;
        Ok(Self {
            pool: gpu.alloc(slot_bytes * max_slots)?,
            tables,
            host_tables,
            sites,
            base_gguf,
            heads,
            max_slots,
            max_rank,
            slot_bytes,
            adapters: HashMap::new(),
            names: HashMap::new(),
            next_id: 1,
            s: gpu.alloc_zeroed(2 * max_tokens * lds * 4)?,
            lds,
            shrink: gpu.kernel("lora_shrink")?,
            expand: gpu.kernel("lora_expand")?,
            shrink_dec: gpu.kernel(if amd_lora() {
                "lora_shrink_amd"
            } else {
                "lora_shrink"
            })?,
            expand_dec: gpu.kernel(if amd_lora() {
                "lora_expand_amd"
            } else {
                "lora_expand"
            })?,
        })
    }

    /// GPU memory the slots and buffers take.
    pub fn bytes(&self) -> usize {
        self.pool.bytes + self.tables.bytes + self.s.bytes
    }

    pub fn hook(&self, site: usize) -> Hook {
        Hook {
            table: self.tables.ptr
                + (site * (self.max_slots + 1) * std::mem::size_of::<Desc>()) as u64,
            parts: self.sites[site].parts.len(),
            max_rows: self.sites[site].max_rows(),
            parity: site % 2,
        }
    }

    pub fn id(&self, name: &str) -> Option<u32> {
        self.names.get(name).copied()
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.names.keys().cloned().collect();
        v.sort();
        v
    }

    /// What preparing an adapter needs, to do it off the engine thread.
    pub fn prep(&self, gpu: &Gpu) -> Prep {
        Prep {
            sites: self.sites.clone(),
            base_gguf: self.base_gguf,
            heads: self.heads,
            max_rank: self.max_rank,
            pinner: gpu.pinner(),
        }
    }

    /// Registers an adapter [`Prep::read`] made; returns its id.
    pub fn insert(&mut self, adapter: Adapter) -> Result<u32> {
        if self.names.contains_key(&adapter.name) {
            return Err(Error::Config(format!(
                "adapter {} is already loaded",
                adapter.name
            )));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.names.insert(adapter.name.clone(), id);
        self.adapters.insert(id, adapter);
        Ok(id)
    }

    /// Drops an adapter no running request uses.
    pub fn unload(&mut self, name: &str) -> Option<u32> {
        let id = self.names.remove(name)?;
        self.adapters.remove(&id);
        Some(id)
    }

    /// Copies adapter `id` into GPU slot `slot` (1-based) and points that
    /// slot's descriptors at it. Ordered on the stream before the next step.
    pub fn upload(&mut self, gpu: &Gpu, slot: usize, id: u32) -> Result<()> {
        let a = self
            .adapters
            .get(&id)
            .ok_or_else(|| Error::Config(format!("adapter {id} is not loaded")))?;
        let base = self.pool.ptr + ((slot - 1) * self.slot_bytes) as u64;
        gpu.htod(base, &a.data, 0, a.len)?;
        let descs = self.host_tables.as_mut_slice::<Desc>();
        for (si, (site, sd)) in self.sites.iter().zip(&a.sites).enumerate() {
            let mut d = Desc {
                a: base + sd.a as u64,
                rank: sd.rank as i32,
                ..Default::default()
            };
            for (pi, part) in site.parts.iter().enumerate() {
                let (b_off, r_off, rp) = sd.parts[pi];
                d.b[pi] = base + b_off as u64;
                d.r[pi] = rp as i32;
                d.r_off[pi] = r_off as i32;
                d.off[pi] = part.off as i32;
                d.rows[pi] = part.rows as i32;
            }
            if sd.rank == 0 {
                d = Desc::default();
            }
            descs[si * (self.max_slots + 1) + slot] = d;
        }
        let size = std::mem::size_of::<Desc>();
        // One copy per site's entry for this slot.
        for si in 0..self.sites.len() {
            let at = (si * (self.max_slots + 1) + slot) * size;
            gpu.htod(self.tables.ptr + at as u64, &self.host_tables, at, size)?;
        }
        Ok(())
    }

    pub fn adapter(&self, id: u32) -> Option<&Adapter> {
        self.adapters.get(&id)
    }
}

/// Whether decoding steps run amd/lora.cuh's kernels: on AMD, unless
/// POCKET_LORA_AMD=0.
fn amd_lora() -> bool {
    cfg!(feature = "rocm") && std::env::var_os("POCKET_LORA_AMD").is_none_or(|v| v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_order_matches_the_kernels() {
        // 2 key heads, 6 value heads: tiled head t uses key head t % 2.
        let o = tiled_order(2, 6);
        assert_eq!(o, vec![0, 3, 1, 4, 2, 5]);
        for (t, &g) in o.iter().enumerate() {
            assert_eq!(g / 3, t % 2);
        }
    }

    #[test]
    fn permutation_round_trips() {
        let order = tiled_order(2, 4);
        let orig: Vec<f32> = (0..8 * 3).map(|x| x as f32).collect();
        let mut v = orig.clone();
        permute_blocks(&mut v, 0, 6, &order, false);
        assert_ne!(v, orig);
        permute_blocks(&mut v, 0, 6, &order, true);
        assert_eq!(v, orig);
    }
}
