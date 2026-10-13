//! GGUF files (llama.cpp's format): metadata, tensor directory and data,
//! memory-mapped. Tensors keep ggml's dimension order, innermost first.

use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// A ggml tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const BF16: Self = Self(30);

    /// (values per block, bytes per block) for the types this crate reads.
    pub fn block(self) -> Option<(usize, usize)> {
        Some(match self.0 {
            0 => (1, 4),
            1 => (1, 2),
            30 => (1, 2),
            2 => (32, 18),
            3 => (32, 20),
            6 => (32, 22),
            7 => (32, 24),
            8 => (32, 34),
            10 => (256, 84),
            11 => (256, 110),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            16 => (256, 66),
            17 => (256, 74),
            18 => (256, 98),
            19 => (256, 50),
            20 => (32, 18),
            21 => (256, 110),
            22 => (256, 82),
            23 => (256, 136),
            29 => (256, 56),
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "f32",
            1 => "f16",
            30 => "bf16",
            2 => "q4_0",
            3 => "q4_1",
            6 => "q5_0",
            7 => "q5_1",
            8 => "q8_0",
            10 => "q2_k",
            11 => "q3_k",
            12 => "q4_k",
            13 => "q5_k",
            14 => "q6_k",
            16 => "iq2_xxs",
            17 => "iq2_xs",
            18 => "iq3_xxs",
            19 => "iq1_s",
            20 => "iq4_nl",
            21 => "iq3_s",
            22 => "iq2_s",
            23 => "iq4_xs",
            29 => "iq1_m",
            _ => "unsupported",
        }
    }

    /// Bytes per block on the device. Formats whose blocks would leave the
    /// weight bytes misaligned are repacked with padding at load: the first
    /// [`GgmlType::repack_head`] bytes stay, padding follows, then the rest.
    pub fn device_block(self) -> Option<usize> {
        Some(match self.0 {
            8 => 36,   // q8_0: d, pad, qs
            2 => 20,   // q4_0: d, pad, qs
            20 => 20,  // iq4_nl: d, pad, qs
            6 => 24,   // q5_0: d, pad, qh, qs
            14 => 216, // q6_k: ql, qh, scales, d, pad
            11 => 112, // q3_k: hmask, qs, scales, d, pad
            _ => return self.block().map(|b| b.1),
        })
    }

    /// Bytes of a block kept in place before the device padding.
    pub fn repack_head(self) -> usize {
        match self.0 {
            8 | 2 | 20 | 6 => 2,
            _ => self.block().map_or(0, |b| b.1),
        }
    }

    /// Bytes per chunk of 16 rows x 256 inputs in the tiled layout the
    /// tensor-core path reads (quant.cuh), for the formats that have one.
    pub fn tile_chunk(self) -> Option<usize> {
        match self.0 {
            2 => Some(2304),
            12 => Some(2368),
            _ => None,
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self.0, 0 | 1 | 30)
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(v) => u64::try_from(*v).ok(),
            Value::Float(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ty: GgmlType,
    /// ggml order: dims[0] is the row length.
    pub dims: Vec<usize>,
    offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn bytes(&self) -> usize {
        let (qk, bs) = self.ty.block().unwrap_or((1, 0));
        self.numel() / qk * bs
    }
}

pub struct Gguf {
    map: memmap2::Mmap,
    pub file: std::fs::File,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    data: usize,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self
            .b
            .get(self.at..self.at + n)
            .ok_or_else(|| Error::Config("truncated GGUF".into()))?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(ety)?);
                }
                Value::Array(v)
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            t => return Err(Error::Config(format!("unknown GGUF value type {t}"))),
        })
    }
}

#[allow(unsafe_code)]
fn map(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: the model file is read-only while the engine runs.
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = map(&file)?;
        let mut r = Reader { b: &map, at: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(Error::Config(format!(
                "{} is not a GGUF file",
                path.display()
            )));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Config(format!("unsupported GGUF version {version}")));
        }
        let (n_tensors, n_kv) = (r.u64()? as usize, r.u64()? as usize);
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            meta.insert(key, r.value(ty)?);
        }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()? as usize);
            }
            let ty = GgmlType(r.u32()?);
            let offset = r.u64()? as usize;
            tensors.insert(name, TensorInfo { ty, dims, offset });
        }
        let align = meta
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data = r.at.next_multiple_of(align);
        let g = Self {
            map,
            file,
            meta,
            tensors,
            data,
        };
        for (name, t) in &g.tensors {
            if t.ty.block().is_none() {
                return Err(Error::Config(format!(
                    "tensor {name}: unsupported type {}",
                    t.ty.0
                )));
            }
            if g.data + t.offset + t.bytes() > g.map.len() {
                return Err(Error::Config(format!(
                    "tensor {name} runs past the end of the file"
                )));
            }
        }
        Ok(g)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let t = self.tensors.get(name)?;
        let start = self.data + t.offset;
        Some((t, &self.map[start..start + t.bytes()]))
    }

    /// A tensor's data as a range of the file, for reading it in place.
    pub fn file_range(&self, name: &str) -> Option<(u64, usize)> {
        let t = self.tensors.get(name)?;
        Some(((self.data + t.offset) as u64, t.bytes()))
    }
}
