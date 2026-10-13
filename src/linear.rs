//! Linear layers whose weights are bf16 (cuBLAS) or quantized in a ggml
//! format (this crate's kernels). A layer may be several parts writing to
//! column ranges of one output, when fused projections come from tensors of
//! different formats.

use crate::gguf::GgmlType;
use crate::gpu::{Arg, Buf, Fp8Mode, Gpu, Kernel};
use crate::{Error, Result};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Bf16,
    Quant(GgmlType),
    /// E4M3 weights with scales in the given layout (see `Part::scales`).
    Fp8(Fp8Mode),
}

pub struct Part {
    pub buf: Rc<Buf>,
    pub kind: Kind,
    /// FP8 weight scales.
    pub scales: Option<Rc<Buf>>,
    /// Output features of this part and their first column in the output.
    pub rows: usize,
    pub off: usize,
}

pub struct Linear {
    pub parts: Vec<Part>,
    /// Input features.
    pub cols: usize,
}

/// Rows an FP8 matmul runs: Hopper's block-scaled FP8 needs a multiple of 4.
/// The extra rows read stale inputs and write rows of the output buffers that
/// this step does not use; every activation buffer has room for them.
pub fn fp8_rows(mode: Fp8Mode, rows: usize) -> usize {
    match mode {
        Fp8Mode::Block128 => rows.next_multiple_of(4),
        Fp8Mode::Mx32 => rows,
    }
}

/// A layer's input: bf16 rows, and the same rows as FP8 with their scales
/// when FP8 parts read them (0 otherwise).
#[derive(Clone, Copy)]
pub struct Input {
    pub bf16: u64,
    pub fp8: u64,
    pub scales: u64,
}

impl Input {
    pub fn bf16(ptr: u64) -> Self {
        Self {
            bf16: ptr,
            fp8: 0,
            scales: 0,
        }
    }
}

/// Kernels of one quantized format.
#[derive(Clone)]
pub struct QuantKernels {
    pub dequant: Kernel,
    pub embed: Kernel,
    /// Tensor-core kernels for tiles of 8, 16, 32 and 64 rows.
    pub mma: [Kernel; 4],
    /// Matrix-vector kernels for 1 to 8 rows.
    pub gemv: [Kernel; 8],
}

const MMA_TILES: [usize; 4] = [8, 16, 32, 64];

/// Split-K partial sums, one f32 slice per split.
const SPLIT_WORKSPACE: usize = 64 << 20;

fn args_mma(base: &[Arg], ws: u64, split: usize) -> Vec<Arg> {
    let mut v = base.to_vec();
    v.extend_from_slice(&[Arg::P(ws), Arg::I(split as i32)]);
    v
}

/// Rows at or below which the matrix-vector kernel beats tensor-core tiles.
const GEMV_ROWS: usize = 4;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn gemv_rows() -> usize {
    env_usize("POCKET_GEMV_ROWS", GEMV_ROWS).min(8)
}
fn split_tuning() -> (usize, usize) {
    (
        env_usize("POCKET_SPLIT_WARPS", 32),
        env_usize("POCKET_SPLIT_MIN_STEPS", 8),
    )
}

#[derive(Default)]
pub struct Quants {
    kernels: HashMap<GgmlType, QuantKernels>,
    finish: Option<Kernel>,
    workspace: Option<Buf>,
    sms: usize,
    /// bf16 copy of a weight matrix (or a slice of its rows) for large batches.
    scratch: Option<Buf>,
}

/// Rows above which a quantized matrix is dequantized to bf16 and handed to
/// cuBLAS: decoding once then amortizes over many rows.
const DENSE_ROWS: usize = 160;
/// Largest dequantized slice, in weights (bf16).
const SCRATCH_WEIGHTS: usize = 192 << 20;

impl Quants {
    pub fn get(&mut self, gpu: &Gpu, ty: GgmlType) -> Result<QuantKernels> {
        if let Some(k) = self.kernels.get(&ty) {
            return Ok(k.clone());
        }
        let name = ty.name();
        if !ty.is_quantized() || name == "unsupported" {
            return Err(Error::Config(format!(
                "no kernels for tensor type {}",
                ty.0
            )));
        }
        let mut gemv = Vec::with_capacity(8);
        for r in 1..=8 {
            gemv.push(gpu.kernel(&format!("qgemv_{name}_{r}"))?);
        }
        let mut mma = Vec::with_capacity(4);
        for t in MMA_TILES {
            mma.push(gpu.kernel(&format!("qmma_{name}_{t}"))?);
        }
        if self.workspace.is_none() {
            self.workspace = Some(gpu.alloc(SPLIT_WORKSPACE)?);
            self.finish = Some(gpu.kernel("qmma_finish")?);
            self.sms = gpu.sms;
        }
        let k = QuantKernels {
            dequant: gpu.kernel(&format!("dequant_{name}"))?,
            embed: gpu.kernel(&format!("embed_{name}"))?,
            mma: mma.try_into().unwrap_or_else(|_| unreachable!()),
            gemv: gemv.try_into().unwrap_or_else(|_| unreachable!()),
        };
        self.kernels.insert(ty, k.clone());
        Ok(k)
    }

    /// Allocates the dequantization scratch once any quantized layer exists.
    /// Call after loading, before running (allocation cannot be captured).
    pub fn prepare(&mut self, gpu: &Gpu, largest: usize) -> Result<()> {
        if !self.kernels.is_empty() && self.scratch.is_none() {
            self.scratch = Some(gpu.alloc(largest.min(SCRATCH_WEIGHTS) * 2)?);
        }
        Ok(())
    }

    pub fn loaded(&self, ty: GgmlType) -> Option<&QuantKernels> {
        self.kernels.get(&ty)
    }
}

impl Linear {
    pub fn out(&self) -> usize {
        self.parts.iter().map(|p| p.off + p.rows).max().unwrap_or(0)
    }

    /// Bytes of weights on the device.
    pub fn bytes(&self) -> usize {
        self.parts
            .iter()
            .map(|p| p.buf.bytes + p.scales.as_ref().map_or(0, |s| s.bytes))
            .sum()
    }

    /// The FP8 layout of this layer's FP8 parts, if any.
    pub fn fp8(&self) -> Option<Fp8Mode> {
        self.parts.iter().find_map(|p| match p.kind {
            Kind::Fp8(m) => Some(m),
            _ => None,
        })
    }

    /// y[rows, ldy] (columns from 0) = x[rows, cols] . W^T. Issues no host
    /// synchronization or allocation, so it can be captured in a graph.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        q: &Quants,
        input: Input,
        rows: usize,
        y: u64,
        ldy: usize,
        f32_out: bool,
    ) -> Result<()> {
        use Arg::{I, P};
        let x = input.bf16;
        let elem = if f32_out { 4 } else { 2 };
        for p in &self.parts {
            match p.kind {
                Kind::Fp8(mode) => {
                    if input.fp8 == 0 {
                        return Err(Error::Config("an FP8 layer got no FP8 input".into()));
                    }
                    let s = p
                        .scales
                        .as_ref()
                        .ok_or_else(|| Error::Config("FP8 part without scales".into()))?;
                    gpu.fp8_matmul(
                        mode,
                        p.buf.ptr,
                        s.ptr,
                        input.fp8,
                        input.scales,
                        y + (p.off * elem) as u64,
                        p.rows,
                        self.cols,
                        fp8_rows(mode, rows),
                        ldy,
                        f32_out,
                    )?;
                }
                Kind::Bf16 => gpu.gemm_ld(
                    x,
                    p.buf.ptr,
                    y + (p.off * elem) as u64,
                    rows,
                    self.cols,
                    p.rows,
                    ldy,
                    f32_out,
                )?,
                Kind::Quant(ty) => {
                    let k = q.loaded(ty).ok_or_else(|| {
                        Error::Config(format!("kernels for {} not loaded", ty.name()))
                    })?;
                    let args = |extra: &[Arg]| -> Vec<Arg> {
                        let mut v = vec![
                            P(p.buf.ptr),
                            I(p.rows as i32),
                            I(self.cols as i32),
                            P(x),
                            I(self.cols as i32),
                        ];
                        v.extend_from_slice(extra);
                        v.extend_from_slice(&[
                            P(y),
                            I(ldy as i32),
                            I(p.off as i32),
                            I(f32_out as i32),
                        ]);
                        v
                    };
                    if rows > DENSE_ROWS
                        && let Some(sc) = &q.scratch
                    {
                        // Dequantize slices of rows into scratch, then cuBLAS.
                        let per = (sc.bytes / 2 / self.cols).min(p.rows).max(1);
                        let mut r0 = 0;
                        while r0 < p.rows {
                            let n = per.min(p.rows - r0);
                            gpu.launch(
                                k.dequant,
                                ((n * self.cols / 8).div_ceil(256).min(65_535) as u32, 1, 1),
                                256,
                                0,
                                &[
                                    P(p.buf.ptr),
                                    I(r0 as i32),
                                    I(n as i32),
                                    I(self.cols as i32),
                                    P(sc.ptr),
                                ],
                            )?;
                            gpu.gemm_ld(
                                x,
                                sc.ptr,
                                y + ((p.off + r0) * elem) as u64,
                                rows,
                                self.cols,
                                n,
                                ldy,
                                f32_out,
                            )?;
                            r0 += n;
                        }
                    } else if rows <= gemv_rows() {
                        gpu.launch(
                            k.gemv[rows - 1],
                            (p.rows.div_ceil(4) as u32, 1, 1),
                            128,
                            0,
                            &args(&[]),
                        )?;
                    } else {
                        let ti = MMA_TILES.iter().position(|&t| rows <= t).unwrap_or(3);
                        let tile = MMA_TILES[ti];
                        let warps = p.rows.div_ceil(16) * rows.div_ceil(tile);
                        // Split K until about 32 warps per SM stream weights,
                        // keeping at least 8 steps of 32 per slice.
                        let steps = self.cols / 32;
                        let (target, min_steps) = split_tuning();
                        let mut split = (target * q.sms)
                            .div_ceil(warps)
                            .clamp(1, (steps / min_steps).clamp(1, 32));
                        // Every slice keeps its own partial sums.
                        split = split.min(SPLIT_WORKSPACE / (rows * p.rows * 4)).max(1);
                        let ws = q.workspace.as_ref().map_or(0, |b| b.ptr);
                        gpu.launch(
                            k.mma[ti],
                            (
                                p.rows.div_ceil(64) as u32,
                                rows.div_ceil(tile) as u32,
                                split as u32,
                            ),
                            128,
                            0,
                            &args_mma(&args(&[I(rows as i32)]), ws, split),
                        )?;
                        if split > 1 {
                            let n = rows * p.rows;
                            gpu.launch(
                                q.finish.expect("loaded with the workspace"),
                                (n.div_ceil(256).min(4096) as u32, 1, 1),
                                256,
                                0,
                                &[
                                    P(ws),
                                    I(rows as i32),
                                    I(p.rows as i32),
                                    I(split as i32),
                                    P(y),
                                    I(ldy as i32),
                                    I(p.off as i32),
                                    I(f32_out as i32),
                                ],
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
