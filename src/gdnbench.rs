//! The linear-attention kernels on synthetic inputs, for `bench-gdn` and
//! `pocket-engine tune`: `seqs` sequences of `tokens` tokens, `hv` value
//! heads and `hk` key heads of width 128.

use crate::Result;
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use crate::model::{GDN_PREP_SMEM, gdn_scan_smem};

const D: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Which {
    /// gdn_prep, gdn_scan and gdn_norm, as prompts run.
    Chunk,
    /// gdn_decode, as decode steps run.
    Decode,
    /// gdn_recurrent, the reference.
    Recurrent,
}

pub struct GdnRig {
    pub hv: usize,
    pub hk: usize,
    pub seqs: usize,
    pub tokens: usize,
    conv: Buf,
    proj: Buf,
    neg_a: Buf,
    dt: Buf,
    nw: Buf,
    pub state: Buf,
    cu: Buf,
    slots: Buf,
    /// All zero: every sequence carries its state.
    pub fresh: Buf,
    tok_seq: Buf,
    pub out: Buf,
    scratch: [Buf; 6],
    norm: Kernel,
    prep: Kernel,
    /// gdn_scan1, gdn_scan2 and gdn_scan4.
    scan: [Kernel; 3],
    decode: Option<Kernel>,
    recurrent: Kernel,
}

impl GdnRig {
    pub fn new(gpu: &Gpu, hv: usize, hk: usize, seqs: usize, tokens: usize) -> Result<Self> {
        let c = 2 * hk * D + hv * D;
        let pw = c + hv * D + 2 * hv;
        let t = seqs * tokens;
        let rnd = |n: usize, scale: f32| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let x = ((i.wrapping_mul(2654435761) % 2001) as f32 / 1000.0 - 1.0) * scale;
                    half::bf16::from_f32(x).to_le_bytes()
                })
                .collect()
        };
        let f32s = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i32s = |v: Vec<i32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let prep = gpu.kernel("gdn_prep")?;
        gpu.set_smem(prep, GDN_PREP_SMEM)?;
        let mut scan = Vec::new();
        for split in [1, 2, 4] {
            let k = gpu.kernel(&format!("gdn_scan{split}"))?;
            gpu.set_smem(k, gdn_scan_smem(split))?;
            scan.push(k);
        }
        // gdn_decode's state buffers: as many as fit, up to three.
        let decode = match crate::model::gdn_decode_shape(true, hv, gpu) {
            Some(_) => {
                let k = gpu.kernel("gdn_decode")?;
                let most = (2..=3)
                    .rev()
                    .find(|&n| Self::decode_fits(gpu, n, 1))
                    .unwrap_or(2);
                gpu.set_smem(k, most * 2 * D * D)?;
                Some(k)
            }
            None => None,
        };
        Ok(Self {
            hv,
            hk,
            seqs,
            tokens,
            conv: gpu.upload(&rnd(t * c, 1.0))?,
            proj: gpu.upload(&rnd(t * pw, 1.0))?,
            neg_a: gpu.upload(&f32s(vec![-0.01; hv]))?,
            dt: gpu.upload(&f32s(vec![0.1; hv]))?,
            nw: gpu.upload(&f32s(vec![1.0; D]))?,
            state: gpu.alloc_zeroed(seqs * hv * D * D * 2)?,
            cu: gpu.upload(&i32s((0..=seqs).map(|i| (i * tokens) as i32).collect()))?,
            slots: gpu.upload(&i32s((0..seqs as i32).collect()))?,
            fresh: gpu.upload(&i32s(vec![0; seqs]))?,
            tok_seq: gpu.upload(&i32s((0..t).map(|i| (i / tokens) as i32).collect()))?,
            out: gpu.alloc(t * hv * D * 2)?,
            scratch: [
                gpu.alloc(t * hv * D * 2)?,
                gpu.alloc(t * hv * D * 2)?,
                gpu.alloc(t * hv * 4)?,
                gpu.alloc(t * hv * 4)?,
                gpu.alloc(t * hv * 32 * 2)?,
                gpu.alloc(t * hv * 4)?,
            ],
            norm: gpu.kernel("gdn_norm")?,
            prep,
            scan: [scan[0], scan[1], scan[2]],
            decode,
            recurrent: gpu.kernel("gdn_recurrent")?,
        })
    }

    /// Whether `per_sm` blocks of gdn_decode with `nbuf` bf16 state buffers
    /// fit an SM's shared memory.
    pub fn decode_fits(gpu: &Gpu, nbuf: usize, per_sm: usize) -> bool {
        per_sm * (nbuf * 2 * D * D + crate::model::GDN_DECODE_STATIC + 1024) <= gpu.smem_per_sm
    }

    /// Runs `which`: phases 1 the prep, 2 the scan or recurrence, 4 the norm
    /// (prompts only). `split` picks gdn_scan's column shares; `decode`
    /// gdn_decode's buffers and blocks per SM, where it runs at all (else
    /// gdn_recurrent stands in, as the engine does).
    pub fn launch(
        &self,
        gpu: &Gpu,
        which: Which,
        fresh: u64,
        phases: u8,
        split: usize,
        decode: Option<(usize, usize)>,
    ) -> Result<()> {
        let (hv, hk, seqs) = (self.hv, self.hk, self.seqs);
        let c = 2 * hk * D + hv * D;
        let pw = c + hv * D + 2 * hv;
        let t = seqs * self.tokens;
        let s = &self.scratch;
        if which == Which::Chunk && phases & 1 != 0 {
            gpu.launch(
                self.prep,
                ((t.div_ceil(32) + seqs) as u32, hv as u32, 1),
                2 * D as u32,
                GDN_PREP_SMEM as u32,
                &[
                    Arg::P(self.conv.ptr),
                    Arg::P(self.proj.ptr),
                    Arg::I(pw as i32),
                    Arg::I(c as i32),
                    Arg::I(hk as i32),
                    Arg::I(hv as i32),
                    Arg::P(self.neg_a.ptr),
                    Arg::P(self.dt.ptr),
                    Arg::P(self.cu.ptr),
                    Arg::I(seqs as i32),
                    Arg::P(0),
                    Arg::I(1),
                    Arg::I(0),
                    Arg::P(s[0].ptr),
                    Arg::P(s[1].ptr),
                    Arg::P(s[2].ptr),
                    Arg::P(s[3].ptr),
                    Arg::P(s[4].ptr),
                    Arg::P(s[5].ptr),
                ],
            )?;
        }
        if phases & 2 != 0 {
            let decoding = match (which, self.decode, decode) {
                (Which::Decode, Some(k), Some(shape)) => Some((k, shape)),
                _ => None,
            };
            let (kern, smem, grid, extra) = match (which, decoding) {
                (Which::Chunk, _) => (
                    self.scan[split.trailing_zeros() as usize],
                    gdn_scan_smem(split) as u32,
                    (seqs as u32, hv as u32, split as u32),
                    vec![
                        Arg::P(s[0].ptr),
                        Arg::P(s[1].ptr),
                        Arg::P(s[2].ptr),
                        Arg::P(s[3].ptr),
                        Arg::P(s[4].ptr),
                        Arg::P(s[5].ptr),
                        Arg::P(0),
                    ],
                ),
                (_, Some((k, (nbuf, per_sm)))) => (
                    k,
                    (nbuf * 2 * D * D) as u32,
                    ((seqs * hv).min(gpu.sms * per_sm) as u32, 1, 1),
                    vec![Arg::I(seqs as i32), Arg::I(nbuf as i32)],
                ),
                _ => (self.recurrent, 0, (seqs as u32, hv as u32, 1), vec![]),
            };
            let threads = if decoding.is_some() {
                256
            } else {
                2 * D as u32
            };
            gpu.launch(
                kern,
                grid,
                threads,
                smem,
                &[
                    Arg::P(self.conv.ptr),
                    Arg::P(self.proj.ptr),
                    Arg::I(pw as i32),
                    Arg::I(c as i32),
                    Arg::I(hk as i32),
                    Arg::I(hv as i32),
                    Arg::P(self.neg_a.ptr),
                    Arg::P(self.dt.ptr),
                    Arg::P(self.nw.ptr),
                    Arg::P(self.state.ptr),
                    Arg::I(1),
                    Arg::P(self.cu.ptr),
                    Arg::P(self.slots.ptr),
                    Arg::P(fresh),
                    Arg::P(self.out.ptr),
                    Arg::F(1e-6),
                    Arg::I(0),
                    Arg::P(0),
                    Arg::P(0),
                    Arg::I(0),
                    Arg::I(0),
                    Arg::P(0),
                    Arg::I(1),
                ]
                .into_iter()
                .chain(extra)
                .collect::<Vec<_>>(),
            )?;
        }
        if which == Which::Chunk && phases & 4 != 0 {
            gpu.launch(
                self.norm,
                (t as u32, 1, 1),
                256,
                0,
                &[
                    Arg::P(self.proj.ptr),
                    Arg::I(pw as i32),
                    Arg::I(c as i32),
                    Arg::I(hv as i32),
                    Arg::P(self.nw.ptr),
                    Arg::P(self.out.ptr),
                    Arg::I(t as i32),
                    Arg::F(1e-6),
                    Arg::P(self.cu.ptr),
                    Arg::P(self.tok_seq.ptr),
                    Arg::P(0),
                    Arg::P(0),
                    Arg::I(0),
                    Arg::I(0),
                    Arg::P(0),
                    Arg::I(1),
                ],
            )?;
        }
        Ok(())
    }
}
