//! Where weights come from: a Hugging Face safetensors checkpoint or a GGUF
//! file. Tensors are named here in Hugging Face terms relative to the text
//! model ("layers.3.mlp.up_proj.weight"); GGUF names are mapped to them, and
//! GGUF's own conventions (norm weights stored with the +1 folded in, A
//! stored as -exp(A_log)) are undone in one place.

use crate::gguf::{GgmlType, Gguf};
use crate::gpu::{Buf, Gpu};
use crate::linear::{Kind, Linear, Part, Quants};
use crate::{Error, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub struct Checkpoint {
    maps: Vec<memmap2::Mmap>,
    index: HashMap<String, usize>,
    prefix: String,
}

impl Checkpoint {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
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

    /// Whether a tensor exists, by its full name.
    pub(crate) fn has(&self, full: &str) -> bool {
        self.index.contains_key(full)
    }

    fn view(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8])> {
        let full = match name.strip_prefix('/') {
            Some(full) => full.to_string(),
            None => format!("{}{name}", self.prefix),
        };
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

/// A matrix in device-ready form: bf16 bytes, or a quantized format's raw blocks.
pub struct Mat {
    pub kind: Kind,
    pub bytes: Vec<u8>,
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

    fn gguf_tensor(
        &self,
        g: &Gguf,
        name: &str,
        numel: usize,
    ) -> Result<(GgmlType, Vec<usize>, Vec<u8>)> {
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
        Ok((t.ty, t.dims.clone(), data.to_vec()))
    }

    /// A small tensor as f32, from any float format.
    pub fn vec_f32(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        match self {
            Source::Safetensors(ck) => ck.f32(name, n),
            Source::Gguf(g) => {
                let (ty, _, data) = self.gguf_tensor(g, name, n)?;
                to_f32(ty, &data).ok_or_else(|| {
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
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Mat> {
        match self {
            Source::Safetensors(ck) => Ok(Mat {
                kind: Kind::Bf16,
                bytes: ck.bf16(name, rows * cols)?,
            }),
            Source::Gguf(g) => {
                let (ty, dims, data) = self.gguf_tensor(g, name, rows * cols)?;
                if dims.first() != Some(&cols) {
                    return Err(Error::Config(format!(
                        "{}: dims {dims:?}, expected rows of {cols}",
                        gguf_name(name)
                    )));
                }
                if ty.is_quantized() {
                    Ok(Mat {
                        kind: Kind::Quant(ty),
                        bytes: crate::gguf::repack(ty, &data),
                    })
                } else {
                    let v = to_f32(ty, &data).expect("float type");
                    Ok(Mat {
                        kind: Kind::Bf16,
                        bytes: v
                            .iter()
                            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
                            .collect(),
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
        let mut pending: Option<(Kind, Vec<u8>, usize, usize)> = None; // kind, bytes, rows, off
        let mut off = 0;
        let flush = |p: Option<(Kind, Vec<u8>, usize, usize)>, out: &mut Vec<Part>| -> Result<()> {
            if let Some((kind, bytes, rows, off)) = p {
                out.push(Part {
                    buf: Rc::new(gpu.upload(&bytes)?),
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
                    bytes.extend(m.bytes);
                    *r += rows;
                }
                _ => {
                    flush(pending.take(), &mut out)?;
                    pending = Some((m.kind, m.bytes, rows, off));
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
        Ok((Rc::new(gpu.upload(&m.bytes)?), m.kind))
    }
}
