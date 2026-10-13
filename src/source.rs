//! Where weights come from: a Hugging Face safetensors checkpoint or a GGUF
//! file. Tensors are named here in Hugging Face terms relative to the text
//! model ("layers.3.mlp.up_proj.weight"); GGUF names are mapped to them, and
//! GGUF's own conventions (norm weights stored with the +1 folded in, A
//! stored as -exp(A_log)) are undone in one place.

use crate::gguf::{GgmlType, Gguf};
use crate::gpu::{Arg, Buf, Gpu, Piece};
use crate::linear::{Kind, Linear, Part, Quants};
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

/// A matrix: bf16 bytes, or a quantized format's blocks.
pub struct Mat<'a> {
    pub kind: Kind,
    /// As stored: quantized formats are repacked for the device on upload.
    pub bytes: Bytes<'a>,
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

impl Mat<'_> {
    /// The matrix on the device, in its kind's device layout (FP8 scales are
    /// left to the caller).
    pub fn upload(&self, gpu: &Gpu) -> Result<Buf> {
        upload_kind(gpu, self.kind, &[self.bytes.piece()])
    }
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

/// The GGUF name of a text-model tensor.
fn gguf_name(name: &str) -> String {
    match name {
        "embed_tokens.weight" => return "token_embd.weight".into(),
        "norm.weight" => return "output_norm.weight".into(),
        "/lm_head.weight" => return "output.weight".into(),
        _ => {}
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

fn to_f32(ty: GgmlType, data: &[u8]) -> Option<Vec<f32>> {
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

    pub fn has(&self, name: &str) -> bool {
        match self {
            Source::Safetensors(ck) => match name.strip_prefix('/') {
                Some(full) => ck.has(full),
                None => ck.has(&format!("{}{name}", ck.prefix)),
            },
            Source::Gguf(g) => g.tensors.contains_key(&gguf_name(name)),
        }
    }

    fn gguf_tensor<'g>(
        &self,
        g: &'g Gguf,
        name: &str,
        numel: usize,
    ) -> Result<(GgmlType, Vec<usize>, &'g [u8])> {
        let gname = gguf_name(name);
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
            Source::Safetensors(ck) => Ok(Mat {
                kind: Kind::Bf16,
                bytes: ck.bf16_bytes(name, rows * cols)?,
            }),
            Source::Gguf(g) => {
                let (ty, dims, data) = self.gguf_tensor(g, name, rows * cols)?;
                if dims.first() != Some(&cols) {
                    return Err(Error::Config(format!(
                        "{}: dims {dims:?}, expected rows of {cols}",
                        gguf_name(name)
                    )));
                }
                if ty.is_quantized() || ty == GgmlType::BF16 {
                    let (offset, len) = g.file_range(&gguf_name(name)).expect("found above");
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
                    })
                }
            }
        }
    }

    /// A linear layer from matrices stacked by output feature, all with
    /// `cols` inputs. Neighbours of one format share a buffer.
    pub fn linear(
        &self,
        gpu: &Gpu,
        quants: &mut Quants,
        parts: &[(&str, usize)],
        cols: usize,
    ) -> Result<Linear> {
        let mut out: Vec<Part> = Vec::new();
        let mut pending: Option<(Kind, Vec<Bytes>, usize, usize)> = None; // kind, bytes, rows, off
        let mut off = 0;
        let flush =
            |p: Option<(Kind, Vec<Bytes>, usize, usize)>, out: &mut Vec<Part>| -> Result<()> {
                if let Some((kind, bytes, rows, off)) = p {
                    let pieces: Vec<Piece> = bytes.iter().map(Bytes::piece).collect();
                    out.push(Part {
                        buf: Rc::new(upload_kind(gpu, kind, &pieces)?),
                        kind,
                        rows,
                        off,
                    });
                }
                Ok(())
            };
        for &(name, rows) in parts {
            let m = self.matrix(name, rows, cols)?;
            if let Kind::Quant(ty) = m.kind {
                quants.get(gpu, ty)?;
            }
            match &mut pending {
                Some((kind, bytes, r, _)) if *kind == m.kind => {
                    bytes.push(m.bytes);
                    *r += rows;
                }
                _ => {
                    flush(pending.take(), &mut out)?;
                    pending = Some((m.kind, vec![m.bytes], rows, off));
                }
            }
            off += rows;
        }
        flush(pending, &mut out)?;
        Ok(Linear { parts: out, cols })
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
