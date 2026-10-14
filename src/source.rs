//! Where weights come from: a Hugging Face safetensors checkpoint or a GGUF
//! file. Tensors are named here in Hugging Face terms relative to the text
//! model ("layers.3.mlp.up_proj.weight"); GGUF names are mapped to them, and
//! GGUF's own conventions (norm weights stored with the +1 folded in, A
//! stored as -exp(A_log)) are undone in one place.

use crate::gguf::{GgmlType, Gguf};
use crate::gpu::{Arg, Buf, Fp8Mode, Gpu, Piece};
use crate::linear::{Kind, Linear, Part, Quants};
use crate::model::f32_bytes;
use crate::{Error, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// A tensor's place in a checkpoint file.
struct Entry {
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Byte range in the file.
    start: usize,
    end: usize,
}

pub struct Checkpoint {
    files: Vec<std::fs::File>,
    maps: Vec<memmap2::Mmap>,
    index: HashMap<String, Entry>,
    prefix: String,
}

impl Checkpoint {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let mut names: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".safetensors"))
            .collect();
        names.sort();
        if names.is_empty() {
            return Err(Error::Config(format!(
                "no safetensors in {}",
                dir.display()
            )));
        }
        let (mut files, mut maps) = (Vec::new(), Vec::new());
        let mut index = HashMap::new();
        for (i, f) in names.iter().enumerate() {
            let file = std::fs::File::open(dir.join(f))?;
            let map = map_file(&file)?;
            // Headers are parsed once; tensors are then found by offset.
            let (n, meta) = SafeTensors::read_metadata(&map)?;
            for (name, t) in meta.tensors() {
                let base = 8 + n;
                index.insert(
                    name,
                    Entry {
                        file: i,
                        dtype: t.dtype,
                        shape: t.shape.clone(),
                        start: base + t.data_offsets.0,
                        end: base + t.data_offsets.1,
                    },
                );
            }
            files.push(file);
            maps.push(map);
        }
        let prefix = if index.contains_key("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        Ok(Self {
            files,
            maps,
            index,
            prefix: prefix.into(),
        })
    }

    /// Whether a tensor exists, by its full name.
    pub(crate) fn has(&self, full: &str) -> bool {
        self.index.contains_key(full)
    }

    /// Every tensor's full name.
    pub(crate) fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.index.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// A tensor's shape, by its full name.
    pub(crate) fn shape(&self, full: &str) -> Result<Vec<usize>> {
        self.index
            .get(full)
            .map(|e| e.shape.clone())
            .ok_or_else(|| Error::Config(format!("missing tensor {full}")))
    }

    /// A tensor as f32, by its full name.
    pub(crate) fn f32_full(&self, full: &str, numel: usize) -> Result<Vec<f32>> {
        self.f32(&format!("/{full}"), numel)
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        let full = match name.strip_prefix('/') {
            Some(full) => full.to_string(),
            None => format!("{}{name}", self.prefix),
        };
        self.index
            .get(&full)
            .ok_or_else(|| Error::Config(format!("missing tensor {full}")))
    }

    fn view(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8])> {
        let e = self.entry(name)?;
        let data = self.maps[e.file]
            .get(e.start..e.end)
            .ok_or_else(|| Error::Config(format!("{name} runs past the end of its file")))?;
        Ok((e.dtype, e.shape.clone(), data))
    }

    /// A tensor in a form ready to upload: bf16 read in place from the file,
    /// or converted from f32. Same naming as [`Checkpoint::bf16`].
    pub(crate) fn bf16_bytes(&self, name: &str, numel: usize) -> Result<Bytes<'_>> {
        let e = self.entry(name)?;
        if e.dtype == Dtype::BF16 && e.shape.iter().product::<usize>() == numel {
            self.view(name)?;
            return Ok(Bytes::File {
                file: &self.files[e.file],
                offset: e.start as u64,
                len: e.end - e.start,
            });
        }
        Ok(Bytes::Owned(self.bf16(name, numel)?))
    }

    /// A tensor as bf16 bytes. Names are relative to the text model; a leading
    /// '/' names a tensor in full.
    pub(crate) fn bf16(&self, name: &str, numel: usize) -> Result<Vec<u8>> {
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

    pub(crate) fn f32(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
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

pub enum Source {
    Safetensors(Checkpoint),
    Gguf(Gguf),
}

/// A matrix in device-ready form: bf16 bytes, a quantized format's raw blocks,
/// or E4M3 bytes with f32 scales per 128 x 128 tile (Qwen's FP8 checkpoints).
pub struct Mat<'a> {
    pub kind: Kind,
    /// As stored: quantized formats are repacked for the device on upload.
    pub bytes: Bytes<'a>,
    pub scales: Vec<u8>,
}

/// Tensor bytes: a range of a model file, read straight into pinned staging
/// at upload, or bytes converted on the host.
pub enum Bytes<'a> {
    File {
        file: &'a std::fs::File,
        offset: u64,
        len: usize,
    },
    Owned(Vec<u8>),
}

impl Bytes<'_> {
    pub fn piece(&self) -> Piece<'_> {
        match self {
            Bytes::File { file, offset, len } => Piece::File {
                file,
                offset: *offset,
                len: *len,
            },
            Bytes::Owned(v) => Piece::Mem(v),
        }
    }
    pub fn len(&self) -> usize {
        self.piece().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Uploads matrices stored back to back as one buffer of `kind`, repacking
/// quantized blocks into the device layout on the GPU.
fn upload_kind(gpu: &Gpu, kind: Kind, pieces: &[Piece]) -> Result<Buf> {
    let Kind::Quant(ty) = kind else {
        return gpu.upload_pieces(pieces);
    };
    let (Some((_, bs)), Some(dbs)) = (ty.block(), ty.device_block()) else {
        return gpu.upload_pieces(pieces);
    };
    if bs == dbs {
        return gpu.upload_pieces(pieces);
    }
    let raw: usize = pieces.iter().map(Piece::len).sum();
    let tmp = gpu.temp(0, raw)?;
    let mut at = 0;
    for p in pieces {
        gpu.put(tmp + at as u64, p)?;
        at += p.len();
    }
    let blocks = raw / bs;
    let out = gpu.alloc(blocks * dbs)?;
    gpu.launch(
        gpu.kernel("repack_blocks")?,
        ((blocks * dbs).div_ceil(256).min(65_535) as u32, 1, 1),
        256,
        0,
        &[
            Arg::P(tmp),
            Arg::P(out.ptr),
            Arg::I(blocks as i32),
            Arg::I(bs as i32),
            Arg::I(dbs as i32),
            Arg::I(ty.repack_head() as i32),
        ],
    )?;
    Ok(out)
}

/// A GGUF tensor on the device in its device layout, by its GGUF name.
pub fn upload_tensor(gpu: &Gpu, g: &Gguf, name: &str) -> Result<Buf> {
    let (t, _) = g
        .tensor(name)
        .ok_or_else(|| Error::Config(format!("missing tensor {name}")))?;
    let kind = if t.ty.is_quantized() {
        Kind::Quant(t.ty)
    } else {
        Kind::Bf16
    };
    if !t.ty.is_quantized() && t.ty != GgmlType::BF16 {
        return Err(Error::Config(format!(
            "{name}: {} is not uploaded as is",
            t.ty.name()
        )));
    }
    let (offset, len) = g.file_range(name).expect("found above");
    let started = gpu.begin_uploads()?;
    let b = upload_kind(
        gpu,
        kind,
        &[Piece::File {
            file: &g.file,
            offset,
            len,
        }],
    );
    if started {
        gpu.end_uploads()?;
    }
    b
}

/// A GGUF tensor in the tiled layout of its format, by its GGUF name; None
/// when the format or shape has none.
pub fn upload_tiled(gpu: &Gpu, g: &Gguf, name: &str) -> Result<Option<Buf>> {
    let (t, _) = g
        .tensor(name)
        .ok_or_else(|| Error::Config(format!("missing tensor {name}")))?;
    let (cols, rows) = (t.dims[0], t.numel() / t.dims[0]);
    let Some(chunk) = t.ty.tile_chunk() else {
        return Ok(None);
    };
    if !rows.is_multiple_of(16) || !cols.is_multiple_of(256) {
        return Ok(None);
    }
    let (offset, len) = g.file_range(name).expect("found above");
    let started = gpu.begin_uploads()?;
    let raw = gpu.temp(0, len)?;
    gpu.put(
        raw,
        &Piece::File {
            file: &g.file,
            offset,
            len,
        },
    )?;
    let out = gpu.alloc(rows / 16 * (cols / 256) * chunk)?;
    gpu.launch(
        gpu.kernel(&format!("tile_{}", t.ty.name()))?,
        ((rows * cols / 32).div_ceil(256).min(65_535) as u32, 1, 1),
        256,
        0,
        &[
            Arg::P(raw),
            Arg::I(rows as i32),
            Arg::I(cols as i32),
            Arg::P(out.ptr),
        ],
    )?;
    if started {
        gpu.end_uploads()?;
    }
    Ok(Some(out))
}

impl Mat<'_> {
    /// The matrix on the device, in its kind's device layout (FP8 scales are
    /// left to the caller).
    pub fn upload(&self, gpu: &Gpu) -> Result<Buf> {
        upload_kind(gpu, self.kind, &[self.bytes.piece()])
    }
}

/// How a layer's matrices are loaded.
#[derive(Debug, Clone, Copy, Default)]
pub struct Load {
    /// The FP8 layout of this device, for FP8 checkpoints and `quantize`.
    pub fp8: Option<Fp8Mode>,
    /// Quantize bf16 matrices to FP8 at load.
    pub quantize: bool,
    /// Quantize bf16 MLP matrices to FP4 at load (the rest to FP8, so
    /// `quantize` is set too). Blackwell only.
    pub fp4: bool,
    /// Load the multi-token-prediction layer.
    pub mtp: bool,
    /// Run 4-bit GGUF matrices on int8 tensor cores for batches.
    pub int8: bool,
}

/// What a part of a layer becomes on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Bf16,
    Quant(crate::gguf::GgmlType),
    /// E4M3 from the checkpoint, with block scales.
    Fp8Checkpoint,
    /// bf16 from the checkpoint, quantized to FP8 on the GPU.
    Fp8Quantized,
    /// NVFP4 from the checkpoint, re-quantized to MXFP4 tiles.
    Fp4Checkpoint,
    /// bf16 from the checkpoint, quantized to MXFP4 tiles.
    Fp4Quantized,
}

/// Matrices Qwen's FP8 checkpoints keep in bf16, kept so when quantizing.
fn keeps_bf16(name: &str) -> bool {
    name.ends_with("in_proj_a.weight")
        || name.ends_with("in_proj_b.weight")
        || name == "/lm_head.weight"
}

/// The GGUF file a model path names: the path itself, or the single .gguf
/// (other than projector files) in a directory without safetensors.
pub fn gguf_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return (path.extension().and_then(|e| e.to_str()) == Some("gguf"))
            .then(|| path.to_path_buf());
    }
    let entries: Vec<PathBuf> = std::fs::read_dir(path)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .collect();
    let has = |ext: &str| {
        entries
            .iter()
            .any(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
    };
    if has("safetensors") {
        return None;
    }
    let ggufs: Vec<&PathBuf> = entries
        .iter()
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("gguf")
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"))
        })
        .collect();
    (ggufs.len() == 1).then(|| ggufs[0].clone())
}

/// The GGUF name of a text-model tensor. The multi-token-prediction layer
/// (`mtp.*`) is block `nextn` of the file, with its own names for the
/// projection and norms that join the embedding and the hidden state.
pub(crate) fn gguf_name(g: &Gguf, name: &str) -> String {
    match name {
        "embed_tokens.weight" => return "token_embd.weight".into(),
        "norm.weight" => return "output_norm.weight".into(),
        "/lm_head.weight" => return "output.weight".into(),
        _ => {}
    }
    if let Some(tail) = name.strip_prefix("/mtp.") {
        let Some(n) = g.nextn_block() else {
            return name.into();
        };
        return match tail {
            "fc.weight" => format!("blk.{n}.nextn.eh_proj.weight"),
            "pre_fc_norm_embedding.weight" => format!("blk.{n}.nextn.enorm.weight"),
            "pre_fc_norm_hidden.weight" => format!("blk.{n}.nextn.hnorm.weight"),
            "norm.weight" => format!("blk.{n}.nextn.shared_head_norm.weight"),
            _ => match tail.strip_prefix("layers.0.") {
                Some(t) => gguf_name(g, &format!("layers.{n}.{t}")),
                None => name.into(),
            },
        };
    }
    let Some(rest) = name.strip_prefix("layers.") else {
        return name.into();
    };
    let (layer, tail) = rest.split_once('.').unwrap_or((rest, ""));
    let mapped = match tail {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        other => other,
    };
    format!("blk.{layer}.{mapped}")
}

pub(crate) fn to_f32(ty: GgmlType, data: &[u8]) -> Option<Vec<f32>> {
    Some(match ty {
        GgmlType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        GgmlType::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        _ => return None,
    })
}

impl Source {
    pub fn open(path: &Path) -> Result<Self> {
        match gguf_path(path) {
            Some(g) => Ok(Source::Gguf(Gguf::open(&g)?)),
            None => Ok(Source::Safetensors(Checkpoint::open(path)?)),
        }
    }

    /// A tensor's shape, outermost first (safetensors order).
    pub fn dims(&self, name: &str) -> Option<Vec<usize>> {
        match self {
            Source::Safetensors(ck) => {
                let full = format!("{}{name}", ck.prefix);
                Some(ck.index.get(&full)?.shape.clone())
            }
            Source::Gguf(g) => g
                .tensors
                .get(&gguf_name(g, name))
                .map(|t| t.dims.iter().rev().copied().collect()),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match self {
            Source::Safetensors(ck) => match name.strip_prefix('/') {
                Some(full) => ck.has(full),
                None => ck.has(&format!("{}{name}", ck.prefix)),
            },
            Source::Gguf(g) => g.tensors.contains_key(&gguf_name(g, name)),
        }
    }

    fn gguf_tensor<'g>(
        &self,
        g: &'g Gguf,
        name: &str,
        numel: usize,
    ) -> Result<(GgmlType, Vec<usize>, &'g [u8])> {
        let gname = gguf_name(g, name);
        let (t, data) = g
            .tensor(&gname)
            .ok_or_else(|| Error::Config(format!("missing tensor {gname}")))?;
        if t.numel() != numel {
            return Err(Error::Config(format!(
                "{gname}: {:?}, expected {numel} elements",
                t.dims
            )));
        }
        Ok((t.ty, t.dims.clone(), data))
    }

    /// A small tensor as f32, from any float format.
    pub fn vec_f32(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        match self {
            Source::Safetensors(ck) => ck.f32(name, n),
            Source::Gguf(g) => {
                let (ty, _, data) = self.gguf_tensor(g, name, n)?;
                to_f32(ty, data).ok_or_else(|| {
                    Error::Config(format!(
                        "{name}: expected a float tensor, found {}",
                        ty.name()
                    ))
                })
            }
        }
    }

    /// An RMSNorm weight as the multiplier: Qwen3.5 stores it zero-centred
    /// (the model multiplies by 1 + w); GGUF files store 1 + w already.
    pub fn norm(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let w = self.vec_f32(name, n)?;
        Ok(match self {
            Source::Safetensors(_) => w.into_iter().map(|x| x + 1.0).collect(),
            Source::Gguf(_) => w,
        })
    }

    /// The linear-attention decay rate A = -exp(A_log); GGUF stores A itself.
    pub fn neg_a(&self, layer: usize, n: usize) -> Result<Vec<f32>> {
        let w = self.vec_f32(&format!("layers.{layer}.linear_attn.A_log"), n)?;
        Ok(match self {
            Source::Safetensors(_) => w.into_iter().map(|x| -x.exp()).collect(),
            Source::Gguf(_) => w,
        })
    }

    /// Whether linear-attention value heads are in GGUF's tiled order (value
    /// head h uses key head h % key_heads) instead of grouped by key head.
    pub fn tiled_value_heads(&self) -> bool {
        matches!(self, Source::Gguf(_))
    }

    /// A `rows x cols` matrix (output features x input features).
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Mat<'_>> {
        match self {
            Source::Safetensors(ck) => {
                let (dtype, shape, _) = ck.view(name)?;
                let has = |suffix: &str| match name.strip_prefix('/') {
                    Some(full) => ck.has(&format!("{full}{suffix}")),
                    None => ck.has(&format!("{}{name}{suffix}", ck.prefix)),
                };
                if dtype == Dtype::U8 && has("_scale_2") {
                    // NVFP4 (ModelOpt): two E2M1 per byte, an E4M3 scale per
                    // 16 and a global scale.
                    if shape.iter().product::<usize>() * 2 != rows * cols {
                        return Err(Error::Config(format!(
                            "{name}: shape {shape:?}, expected {rows} x {cols} FP4"
                        )));
                    }
                    let e = ck.entry(name)?;
                    let (_, _, ws) = ck.view(&format!("{name}_scale"))?;
                    if ws.len() != rows * cols / 16 {
                        return Err(Error::Config(format!("{name}_scale: expected one per 16")));
                    }
                    let global = ck.f32(&format!("{name}_scale_2"), 1)?[0];
                    let mut scales = ws.to_vec();
                    scales.extend(global.to_le_bytes());
                    return Ok(Mat {
                        kind: Kind::Fp4,
                        bytes: Bytes::File {
                            file: &ck.files[e.file],
                            offset: e.start as u64,
                            len: e.end - e.start,
                        },
                        scales,
                    });
                }
                if dtype == Dtype::F8_E4M3 && !has("_scale_inv") && has("_scale") {
                    // FP8 with one scale per tensor: as block scales, all equal.
                    let e = ck.entry(name)?;
                    let sc = ck.f32(&format!("{name}_scale"), 1)?[0];
                    let n = rows.div_ceil(128) * cols.div_ceil(128);
                    return Ok(Mat {
                        kind: Kind::Fp8(Fp8Mode::Block128),
                        bytes: Bytes::File {
                            file: &ck.files[e.file],
                            offset: e.start as u64,
                            len: e.end - e.start,
                        },
                        scales: f32_bytes(&vec![sc; n]),
                    });
                }
                if dtype == Dtype::F8_E4M3 {
                    if shape.iter().product::<usize>() != rows * cols {
                        return Err(Error::Config(format!(
                            "{name}: shape {shape:?}, expected {rows} x {cols}"
                        )));
                    }
                    let e = ck.entry(name)?;
                    let scale_name = format!("{name}_scale_inv");
                    let n = rows.div_ceil(128) * cols.div_ceil(128);
                    let scales = ck.f32(&scale_name, n)?;
                    return Ok(Mat {
                        kind: Kind::Fp8(Fp8Mode::Block128),
                        bytes: Bytes::File {
                            file: &ck.files[e.file],
                            offset: e.start as u64,
                            len: e.end - e.start,
                        },
                        scales: f32_bytes(&scales),
                    });
                }
                Ok(Mat {
                    kind: Kind::Bf16,
                    bytes: ck.bf16_bytes(name, rows * cols)?,
                    scales: Vec::new(),
                })
            }
            Source::Gguf(g) => {
                let (ty, dims, data) = self.gguf_tensor(g, name, rows * cols)?;
                if dims.first() != Some(&cols) {
                    return Err(Error::Config(format!(
                        "{}: dims {dims:?}, expected rows of {cols}",
                        gguf_name(g, name)
                    )));
                }
                if ty.is_quantized() || ty == GgmlType::BF16 {
                    let (offset, len) = g.file_range(&gguf_name(g, name)).expect("found above");
                    Ok(Mat {
                        kind: if ty == GgmlType::BF16 {
                            Kind::Bf16
                        } else {
                            Kind::Quant(ty)
                        },
                        bytes: Bytes::File {
                            file: &g.file,
                            offset,
                            len,
                        },
                        scales: Vec::new(),
                    })
                } else {
                    let v = to_f32(ty, data).expect("float type");
                    Ok(Mat {
                        kind: Kind::Bf16,
                        bytes: Bytes::Owned(
                            v.iter()
                                .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
                                .collect(),
                        ),
                        scales: Vec::new(),
                    })
                }
            }
        }
    }

    /// A linear layer from matrices stacked by output feature, all with
    /// `cols` inputs. Neighbours that become the same format share a buffer.
    pub fn linear(
        &self,
        gpu: &Gpu,
        quants: &mut Quants,
        parts: &[(&str, usize)],
        cols: usize,
        load: Load,
    ) -> Result<Linear> {
        struct Pending<'a> {
            target: Target,
            kind: Kind,
            bytes: Vec<Bytes<'a>>,
            scales: Vec<u8>,
            rows: usize,
            off: usize,
        }
        let mut out: Vec<Part> = Vec::new();
        let mut pending: Option<Pending> = None;
        let mut off = 0;
        for &(name, rows) in parts {
            let m = self.matrix(name, rows, cols)?;
            let target = match m.kind {
                Kind::Quant(ty) | Kind::Tiled(ty) => {
                    quants.get(gpu, ty)?;
                    Target::Quant(ty)
                }
                Kind::Fp8(_) => Target::Fp8Checkpoint,
                Kind::Fp4 => {
                    quants.fp4(gpu)?;
                    Target::Fp4Checkpoint
                }
                Kind::Bf16
                    if load.fp4
                        && name.contains("mlp.")
                        && rows.is_multiple_of(16)
                        && cols.is_multiple_of(256) =>
                {
                    quants.fp4(gpu)?;
                    Target::Fp4Quantized
                }
                Kind::Bf16
                    if load.quantize
                        && load.fp8.is_some()
                        && !keeps_bf16(name)
                        && cols.is_multiple_of(128) =>
                {
                    Target::Fp8Quantized
                }
                Kind::Bf16 => Target::Bf16,
            };
            // Block scales stack only in whole 128-row tiles.
            // NVFP4 matrices each carry their own global scale.
            let stacks = target != Target::Fp4Checkpoint
                && (target != Target::Fp8Checkpoint
                    || pending.as_ref().is_some_and(|p| p.rows.is_multiple_of(128)));
            match &mut pending {
                Some(p) if p.target == target && stacks => {
                    p.bytes.push(m.bytes);
                    p.scales.extend(m.scales);
                    p.rows += rows;
                }
                _ => {
                    if let Some(p) = pending.take() {
                        out.push(self.place(
                            gpu, p.target, p.kind, &p.bytes, p.scales, p.rows, p.off, cols, load,
                        )?);
                    }
                    pending = Some(Pending {
                        target,
                        kind: m.kind,
                        bytes: vec![m.bytes],
                        scales: m.scales,
                        rows,
                        off,
                    });
                }
            }
            off += rows;
        }
        if let Some(p) = pending {
            out.push(self.place(
                gpu, p.target, p.kind, &p.bytes, p.scales, p.rows, p.off, cols, load,
            )?);
        }
        Ok(Linear {
            parts: out,
            cols,
            int8: true,
            lora: None,
        })
    }

    /// Uploads a part, converting FP8 to this device's layout. Everything is
    /// ordered on the stream, so nothing here waits for the GPU.
    #[allow(clippy::too_many_arguments)]
    fn place(
        &self,
        gpu: &Gpu,
        target: Target,
        kind: Kind,
        bytes: &[Bytes],
        scales: Vec<u8>,
        rows: usize,
        off: usize,
        cols: usize,
        load: Load,
    ) -> Result<Part> {
        let part = |buf: Buf, kind: Kind, scales: Option<Buf>| Part {
            buf: Rc::new(buf),
            kind,
            scales: scales.map(Rc::new),
            rows,
            off,
        };
        let pieces: Vec<Piece> = bytes.iter().map(Bytes::piece).collect();
        // Pieces into device scratch, for a conversion kernel to read.
        let to_temp = |slot: usize, pieces: &[Piece]| -> Result<u64> {
            let t = gpu.temp(slot, pieces.iter().map(Piece::len).sum())?;
            let mut at = 0;
            for p in pieces {
                gpu.put(t + at as u64, p)?;
                at += p.len();
            }
            Ok(t)
        };
        let grid = |n: usize| ((n.div_ceil(256)).min(65_535) as u32, 1, 1);
        Ok(match target {
            Target::Quant(ty)
                if ty.tile_chunk().is_some()
                    && rows.is_multiple_of(16)
                    && cols.is_multiple_of(256) =>
            {
                // Rewritten for the tensor-core path, from the raw blocks.
                let raw = to_temp(0, &pieces)?;
                let chunk = ty.tile_chunk().expect("checked");
                let out = gpu.alloc(rows / 16 * (cols / 256) * chunk)?;
                gpu.launch(
                    gpu.kernel(&format!("tile_{}", ty.name()))?,
                    grid(rows * cols / 32),
                    256,
                    0,
                    &[
                        Arg::P(raw),
                        Arg::I(rows as i32),
                        Arg::I(cols as i32),
                        Arg::P(out.ptr),
                    ],
                )?;
                part(out, Kind::Tiled(ty), None)
            }
            Target::Bf16 | Target::Quant(_) => part(upload_kind(gpu, kind, &pieces)?, kind, None),
            Target::Fp8Checkpoint => {
                let mode = load.fp8.ok_or_else(|| {
                    Error::Config(
                        "FP8 weights need a GPU with FP8 matmuls (Hopper or Blackwell)".into(),
                    )
                })?;
                match mode {
                    Fp8Mode::Block128 => part(
                        gpu.upload_pieces(&pieces)?,
                        Kind::Fp8(mode),
                        Some(gpu.upload(&scales)?),
                    ),
                    Fp8Mode::Mx32 => {
                        let w = to_temp(0, &pieces)?;
                        let s = to_temp(1, &[Piece::Mem(&scales)])?;
                        let q = gpu.alloc(rows * cols)?;
                        let sx = gpu.alloc_zeroed(mode.weight_scale_bytes(rows, cols))?;
                        gpu.launch(
                            gpu.kernel("fp8_block_to_mx")?,
                            grid(rows * cols / 32),
                            256,
                            0,
                            &[
                                Arg::P(w),
                                Arg::P(s),
                                Arg::I(rows as i32),
                                Arg::I(cols as i32),
                                Arg::P(q.ptr),
                                Arg::P(sx.ptr),
                            ],
                        )?;
                        part(q, Kind::Fp8(mode), Some(sx))
                    }
                }
            }
            Target::Fp4Checkpoint | Target::Fp4Quantized => {
                if load.fp8 != Some(Fp8Mode::Mx32) {
                    return Err(Error::Config(
                        "FP4 weights need a Blackwell GPU (RTX 50-series)".into(),
                    ));
                }
                let out = gpu.alloc(rows / 16 * (cols / 256) * 2176)?;
                let w = to_temp(0, &pieces)?;
                if target == Target::Fp4Checkpoint {
                    let (ws, global) = scales.split_at(scales.len() - 4);
                    let global = f32::from_le_bytes(global.try_into().expect("4 bytes"));
                    let s = to_temp(1, &[Piece::Mem(ws)])?;
                    gpu.launch(
                        gpu.kernel("nvfp4_to_tiles")?,
                        grid(rows * cols / 32),
                        256,
                        0,
                        &[
                            Arg::P(w),
                            Arg::P(s),
                            Arg::F(global),
                            Arg::I(rows as i32),
                            Arg::I(cols as i32),
                            Arg::P(out.ptr),
                        ],
                    )?;
                } else {
                    gpu.launch(
                        gpu.kernel("bf16_to_fp4_tiles")?,
                        grid(rows * cols / 32),
                        256,
                        0,
                        &[
                            Arg::P(w),
                            Arg::I(rows as i32),
                            Arg::I(cols as i32),
                            Arg::P(out.ptr),
                        ],
                    )?;
                }
                part(out, Kind::Fp4, None)
            }
            Target::Fp8Quantized => {
                let mode = load.fp8.expect("checked when chosen");
                let w = to_temp(0, &pieces)?;
                let q = gpu.alloc(rows * cols)?;
                let sx = gpu.alloc_zeroed(mode.weight_scale_bytes(rows, cols))?;
                let args = [
                    Arg::P(w),
                    Arg::I(rows as i32),
                    Arg::I(cols as i32),
                    Arg::P(q.ptr),
                    Arg::P(sx.ptr),
                ];
                match mode {
                    Fp8Mode::Mx32 => gpu.launch(
                        gpu.kernel("fp8_weights_mx")?,
                        grid(rows * cols / 32),
                        256,
                        0,
                        &args,
                    )?,
                    Fp8Mode::Block128 => gpu.launch(
                        gpu.kernel("fp8_weights_block")?,
                        (cols.div_ceil(128) as u32, rows.div_ceil(128) as u32, 1),
                        256,
                        0,
                        &args,
                    )?,
                }
                part(q, Kind::Fp8(mode), Some(sx))
            }
        })
    }

    /// A single matrix kept as one buffer, for sharing (tied embeddings).
    pub fn shared(
        &self,
        gpu: &Gpu,
        quants: &mut Quants,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Rc<Buf>, Kind)> {
        let m = self.matrix(name, rows, cols)?;
        if let Kind::Quant(ty) = m.kind {
            quants.get(gpu, ty)?;
        }
        Ok((Rc::new(m.upload(gpu)?), m.kind))
    }
}
