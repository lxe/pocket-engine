//! Settings measured on each GPU by `pocket-engine tune`, kept in
//! `tuning/cuda/` and built into the binary. A table only replaces a built-in
//! choice where tuning measured a gain on that GPU; anything it does not list
//! keeps the built-in rule. `POCKET_TUNING=0` turns tables off and
//! `POCKET_TUNING=FILE` reads one from disk instead.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A cuBLASLt algorithm (`cublasLtMatmulAlgo_t`), as tables store it.
pub type Algo = [u64; 8];

/// Tuned algorithms by shape (kind, n, k, ldy): first and last rows, and
/// the algorithm.
type GemmIndex = HashMap<(String, usize, usize, usize), Vec<(usize, usize, Algo)>>;

/// Tables built into the binary: (file name, contents).
const BUILT_IN: &[(&str, &str)] = &[];

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Table {
    /// The device name, as the driver reports it.
    pub gpu: String,
    pub sms: usize,
    /// cuBLASLt's version (`cublasLtGetVersion`); its algorithms are valid
    /// for this major and minor version only.
    pub cublaslt: usize,
    /// Models whose shapes were tuned.
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub gemm: Vec<Gemm>,
    #[serde(default)]
    pub quant: Vec<QuantRows>,
    #[serde(default)]
    pub gdn_split: Vec<Split>,
    #[serde(default)]
    pub gdn_decode: Vec<DecodeShape>,
    #[serde(skip)]
    index: GemmIndex,
}

/// The algorithm for one matrix-multiply shape over a range of rows: y[rows,
/// ldy] = x[rows, k] . w[n, k]^T.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Gemm {
    /// "bf16", "bf16 f32" (f32 output), "fp8 block128", "fp8 mx32", ...
    pub kind: String,
    pub n: usize,
    pub k: usize,
    pub ldy: usize,
    /// First and last row count, inclusive.
    pub rows: [usize; 2],
    /// The algorithm's 64 bytes, in hex.
    pub algo: String,
    /// The default's time over this algorithm's, as measured.
    pub speedup: f32,
}

/// Where GGUF matrices change path (see `linear::Quants`), for models of
/// width `hidden` (0 while unset).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct QuantRows {
    #[serde(default)]
    pub hidden: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8_max: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub big_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_rows: Option<usize>,
}

/// gdn_scan's column shares for steps of `seqs` prompt sequences of a model
/// with `heads` linear-attention value heads.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Split {
    pub heads: usize,
    pub seqs: [usize; 2],
    pub split: usize,
}

/// gdn_decode's state buffers per block and blocks per SM, for models with
/// `heads` linear-attention value heads.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct DecodeShape {
    pub heads: usize,
    pub nbuf: usize,
    pub per_sm: usize,
}

pub fn algo_hex(a: &Algo) -> String {
    a.iter().map(|w| format!("{w:016x}")).collect()
}

pub fn algo_from_hex(s: &str) -> Option<Algo> {
    if s.len() != 128 {
        return None;
    }
    let mut a = [0u64; 8];
    for (i, w) in a.iter_mut().enumerate() {
        *w = u64::from_str_radix(&s[16 * i..16 * i + 16], 16).ok()?;
    }
    Some(a)
}

/// The file name of a GPU's table: its name in lowercase words and its SM
/// count, as in `nvidia-h100-80gb-hbm3-132.json`.
pub fn file_name(gpu: &str, sms: usize) -> String {
    let mut s = String::new();
    for w in gpu
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        s += w;
        s.push('-');
    }
    format!("{s}{sms}.json")
}

impl Table {
    pub fn new(gpu: &str, sms: usize, cublaslt: usize) -> Self {
        Self {
            gpu: gpu.to_string(),
            sms,
            cublaslt,
            ..Self::default()
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut t: Table = serde_json::from_str(text).map_err(|e| e.to_string())?;
        t.reindex();
        Ok(t)
    }

    /// The table for this GPU: from `POCKET_TUNING`, or built in. `cublaslt`
    /// is the loaded library's version; matrix-multiply entries from another
    /// major or minor version are dropped.
    pub fn for_gpu(gpu: &str, sms: usize, cublaslt: usize) -> Option<Self> {
        let (source, mut t) = match std::env::var("POCKET_TUNING") {
            Ok(v) if v == "0" => return None,
            Ok(path) => {
                let t = std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|s| Self::parse(&s));
                match t {
                    Ok(t) => (path, t),
                    Err(e) => {
                        eprintln!("POCKET_TUNING={path}: {e}; not tuned");
                        return None;
                    }
                }
            }
            Err(_) => {
                let name = file_name(gpu, sms);
                let (_, text) = BUILT_IN.iter().find(|(f, _)| *f == name)?;
                (format!("tuning/cuda/{name}"), Self::parse(text).ok()?)
            }
        };
        if t.cublaslt / 100 != cublaslt / 100 && !t.gemm.is_empty() {
            eprintln!(
                "tuning: {source} has cuBLASLt {} algorithms, this is {cublaslt}; using cuBLAS's own",
                t.cublaslt
            );
            t.gemm.clear();
            t.reindex();
        }
        eprintln!("tuning: {source}");
        Some(t)
    }

    pub fn reindex(&mut self) {
        self.index.clear();
        for g in &self.gemm {
            if let Some(a) = algo_from_hex(&g.algo) {
                self.index
                    .entry((g.kind.clone(), g.n, g.k, g.ldy))
                    .or_default()
                    .push((g.rows[0], g.rows[1], a));
            }
        }
    }

    /// The tuned algorithm for a shape at `rows`, if any.
    pub fn gemm_algo(
        &self,
        kind: &str,
        n: usize,
        k: usize,
        ldy: usize,
        rows: usize,
    ) -> Option<Algo> {
        self.index
            .get(&(kind.to_string(), n, k, ldy))?
            .iter()
            .find(|(lo, hi, _)| (*lo..=*hi).contains(&rows))
            .map(|e| e.2)
    }

    /// The tuned column shares for `seqs` sequences, if any.
    pub fn split(&self, heads: usize, seqs: usize) -> Option<usize> {
        self.gdn_split
            .iter()
            .find(|s| s.heads == heads && (s.seqs[0]..=s.seqs[1]).contains(&seqs))
            .map(|s| s.split)
    }

    /// The GGUF thresholds for models of width `hidden`, if tuned.
    pub fn quant(&self, hidden: usize) -> Option<&QuantRows> {
        self.quant.iter().find(|q| q.hidden == hidden)
    }

    /// gdn_decode's shape for `heads` value heads, if tuned.
    pub fn decode(&self, heads: usize) -> Option<DecodeShape> {
        self.gdn_decode.iter().find(|d| d.heads == heads).copied()
    }

    /// Adds another run's results: its entries replace this table's for the
    /// same shapes and settings.
    pub fn merge(&mut self, other: Table) {
        for m in other.models {
            if !self.models.contains(&m) {
                self.models.push(m);
            }
        }
        self.gemm.retain(|g| {
            !other
                .gemm
                .iter()
                .any(|o| (&o.kind, o.n, o.k, o.ldy) == (&g.kind, g.n, g.k, g.ldy))
        });
        self.gemm.extend(other.gemm);
        self.gemm.sort_by(|a, b| {
            (&a.kind, a.n, a.k, a.ldy, a.rows).cmp(&(&b.kind, b.n, b.k, b.ldy, b.rows))
        });
        self.gdn_split
            .retain(|s| !other.gdn_split.iter().any(|o| o.heads == s.heads));
        self.gdn_split.extend(other.gdn_split);
        self.quant
            .retain(|q| !other.quant.iter().any(|o| o.hidden == q.hidden));
        self.quant.extend(other.quant);
        self.gdn_decode
            .retain(|d| !other.gdn_decode.iter().any(|o| o.heads == d.heads));
        self.gdn_decode.extend(other.gdn_decode);
        self.cublaslt = other.cublaslt;
        self.reindex();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithms_round_trip_through_hex() {
        let a: Algo = [1, u64::MAX, 3, 0, 5, 6, 7, 0x0123_4567_89ab_cdef];
        assert_eq!(algo_from_hex(&algo_hex(&a)), Some(a));
        assert_eq!(algo_from_hex("12"), None);
    }

    #[test]
    fn lookups_follow_row_ranges_and_merges_replace_shapes() {
        let a = algo_hex(&[7; 8]);
        let entry = |rows: [usize; 2]| Gemm {
            kind: "bf16".into(),
            n: 8,
            k: 4,
            ldy: 8,
            rows,
            algo: a.clone(),
            speedup: 1.1,
        };
        let mut t = Table::new("NVIDIA RTX 5090", 170, 130_100);
        t.gemm = vec![entry([1, 1]), entry([9, 16])];
        t.reindex();
        assert_eq!(t.gemm_algo("bf16", 8, 4, 8, 12), Some([7; 8]));
        assert_eq!(t.gemm_algo("bf16", 8, 4, 8, 4), None);
        assert_eq!(t.gemm_algo("bf16 f32", 8, 4, 8, 1), None);
        let mut other = Table::new("NVIDIA RTX 5090", 170, 130_100);
        other.gemm = vec![entry([2, 2])];
        other.models = vec!["m".into()];
        t.merge(other);
        assert_eq!(t.gemm_algo("bf16", 8, 4, 8, 12), None);
        assert_eq!(t.gemm_algo("bf16", 8, 4, 8, 2), Some([7; 8]));
        assert_eq!(
            file_name("NVIDIA H100 80GB HBM3", 132),
            "nvidia-h100-80gb-hbm3-132.json"
        );
    }
}
