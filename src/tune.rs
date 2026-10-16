//! `pocket-engine tune`: measures this GPU's settings for one model and adds
//! them to a tuning table (see `tuning`).
//!
//! The engine loads the model with tables off and records every matrix
//! product it runs while capturing its decode graphs and running one
//! step-sized prompt. Each recorded shape is then timed on the model's own
//! weights, cycling through every layer's copy as serving does, over ranges
//! of rows: cuBLAS's default against cuBLASLt's own candidates. A candidate
//! is kept for a range only when it is no slower at either end of the range,
//! at least 3% faster on average, and gives the default's output.

use crate::engine::{Engine, EngineConfig};
use crate::gpu::{GemmShape, GemmUse, Gpu};
use crate::sched::{Event, Request, Sampling};
use crate::tuning::{Algo, DecodeShape, Gemm, QuantRows, Split, Table, algo_hex};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// Candidates asked of cuBLASLt for each shape and row count.
const CANDIDATES: usize = 16;
/// A kept setting's least average gain.
const MIN_GAIN: f64 = 0.03;
/// GPU time per timing round, in microseconds.
const ROUND_US: f64 = 3000.0;

/// Row ranges tuned apart: 1, 2, 3-4, 5-8, ..., up to `max`.
pub fn ranges(max: usize) -> Vec<[usize; 2]> {
    let mut v = Vec::new();
    if max >= 1 {
        v.push([1, 1]);
    }
    if max >= 2 {
        v.push([2, 2]);
    }
    let mut hi = 4;
    while hi / 2 < max {
        v.push([hi / 2 + 1, hi.min(max)]);
        hi *= 2;
    }
    v
}

/// One product on a recorded shape's buffers, with weights `i` (cycling
/// through the layers) and algorithm `a`, or the engine's default path.
fn run(
    gpu: &Gpu,
    s: &GemmShape,
    u: &GemmUse,
    rows: usize,
    i: usize,
    a: Option<&Algo>,
) -> Result<bool> {
    let (w, ws) = u.weights[i % u.weights.len()];
    match a {
        Some(a) => gpu.lt_run(s, rows, (w, ws), u.x, u.x_scale, u.y, a),
        None => {
            match u.fp8 {
                None => gpu.gemm_ld(u.x, w, u.y, rows, s.k, s.n, s.ldy, u.f32_out)?,
                Some(m) => gpu.fp8_matmul(
                    m, w, ws, u.x, u.x_scale, u.y, s.n, s.k, rows, s.ldy, u.f32_out,
                )?,
            }
            Ok(true)
        }
    }
}

/// Median microseconds of one call of `f(i)`, replayed from a CUDA graph as
/// decode steps are, so host time per call does not count. At least `min`
/// calls go in a graph.
fn graph_us(gpu: &Gpu, min: usize, f: &dyn Fn(usize) -> Result<()>) -> Result<f64> {
    let graph = |reps: usize| -> Result<crate::gpu::Graph> {
        gpu.begin_capture()?;
        let r = (0..reps).try_for_each(f);
        let g = gpu.end_capture();
        r.and(g)
    };
    let g = graph(min)?;
    let once = gpu.time_us(1, 3, |_| gpu.replay(&g))? / min as f64;
    let reps = ((ROUND_US / once.max(0.5)).ceil() as usize).clamp(min, 2000);
    let g = graph(reps)?;
    Ok(gpu.time_us(1, 5, |_| gpu.replay(&g))? / reps as f64)
}

/// Median microseconds of one product.
fn time(gpu: &Gpu, s: &GemmShape, u: &GemmUse, rows: usize, a: Option<&Algo>) -> Result<f64> {
    graph_us(gpu, u.weights.len(), &|i| {
        run(gpu, s, u, rows, i, a).map(|_| ())
    })
}

/// Whether `ratio(c)` (a time over the default's) shows a gain twice over:
/// the first pass picks, a second confirms against noise.
fn confirmed(ratios: impl Fn() -> Result<Vec<f64>>) -> Result<Option<f64>> {
    let mut mean = 0.0;
    for _ in 0..2 {
        let r = ratios()?;
        let worst = r.iter().cloned().fold(0f64, f64::max);
        mean = r.iter().sum::<f64>() / r.len() as f64;
        if worst > 1.0 || mean > 1.0 - MIN_GAIN {
            return Ok(None);
        }
    }
    Ok(Some(mean))
}

/// The first and last output rows of the latest product, as f32.
fn sample(gpu: &Gpu, s: &GemmShape, u: &GemmUse, rows: usize) -> Result<Vec<f32>> {
    let elem = if u.f32_out { 4 } else { 2 };
    let mut out = Vec::with_capacity(2 * s.n);
    for r in [0, rows - 1] {
        let mut h = gpu.pinned(s.n * elem)?;
        gpu.dtoh(&mut h, u.y + (r * s.ldy * elem) as u64, s.n * elem)?;
        gpu.sync()?;
        if u.f32_out {
            out.extend_from_slice(h.as_slice::<f32>());
        } else {
            out.extend(
                h.as_slice::<u16>()
                    .iter()
                    .map(|&b| half::bf16::from_bits(b).to_f32()),
            );
        }
    }
    Ok(out)
}

/// Whether two outputs agree to rounding: the largest difference within 2% of
/// the largest value.
fn agree(a: &[f32], b: &[f32]) -> bool {
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 0.02 * scale)
}

/// Tunes the recorded products' algorithms.
pub fn gemms(gpu: &Gpu, uses: &BTreeMap<GemmShape, GemmUse>) -> Result<Vec<Gemm>> {
    let mut found = Vec::new();
    for (s, u) in uses {
        let Some(&max) = u.rows.last() else { continue };
        let pad = |r: usize| crate::linear::fp8_rows_of(u.fp8, r);
        let mut line = format!("{} {}x{} (ldy {}):", s.kind, s.n, s.k, s.ldy);
        for [lo, hi] in ranges(max) {
            let mut points: Vec<usize> = vec![pad(lo), pad(hi)];
            points.dedup();
            if points.iter().any(|&p| p > max) {
                continue;
            }
            let mut base = Vec::new();
            let mut want = Vec::new();
            for &p in &points {
                base.push(time(gpu, s, u, p, None)?);
                run(gpu, s, u, p, 0, None)?;
                want.push(sample(gpu, s, u, p)?);
            }
            let mut cands: Vec<Algo> = Vec::new();
            for &p in &points {
                let (_, ws) = u.weights[0];
                for a in gpu.lt_algos(s, p, ws, u.x_scale, CANDIDATES)? {
                    if !cands.contains(&a) {
                        cands.push(a);
                    }
                }
            }
            let mut best: Option<(f64, Algo)> = None;
            'cand: for a in &cands {
                let mut ratios = Vec::new();
                for (j, &p) in points.iter().enumerate() {
                    if !run(gpu, s, u, p, 0, Some(a))? || !agree(&want[j], &sample(gpu, s, u, p)?) {
                        continue 'cand;
                    }
                    ratios.push(time(gpu, s, u, p, Some(a))? / base[j]);
                }
                let worst = ratios.iter().cloned().fold(0f64, f64::max);
                let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
                if worst <= 1.0 && mean <= 1.0 - MIN_GAIN && best.is_none_or(|b| mean < b.0) {
                    best = Some((mean, *a));
                }
            }
            // The pick again, against the default timed afresh.
            let best = match best {
                Some((_, a)) => confirmed(|| {
                    points
                        .iter()
                        .map(|&p| Ok(time(gpu, s, u, p, Some(&a))? / time(gpu, s, u, p, None)?))
                        .collect()
                })?
                .map(|m| (m, a)),
                None => None,
            };
            match best {
                Some((mean, a)) => {
                    line += &format!(" {lo}-{hi} x{:.2}", 1.0 / mean);
                    found.push(Gemm {
                        kind: s.kind.clone(),
                        n: s.n,
                        k: s.k,
                        ldy: s.ldy,
                        rows: [lo, hi],
                        algo: algo_hex(&a),
                        speedup: (1.0 / mean) as f32,
                    });
                }
                None => line += &format!(" {lo}-{hi} -"),
            }
        }
        eprintln!("{line}");
    }
    Ok(found)
}

/// Prompt sequences per step at which gdn_scan's shares are compared, each
/// with a step's 4,096 tokens split among them.
const SPLIT_SEQS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// gdn_scan's column shares, where another than the built-in rule's measured
/// faster.
pub fn splits(gpu: &Gpu, hv: usize, hk: usize, max_seqs: usize) -> Result<Vec<Split>> {
    use crate::gdnbench::{GdnRig, Which};
    let mut out = Vec::new();
    let mut lo = 1;
    let mut line = format!("gdn_scan, {hv} heads:");
    for s in SPLIT_SEQS.into_iter().filter(|&s| s <= max_seqs) {
        let rig = GdnRig::new(gpu, hv, hk, s, (4096 / s).max(32))?;
        let t = |split: usize| -> Result<f64> {
            let f = |_: usize| rig.launch(gpu, Which::Chunk, rig.fresh.ptr, 2, split, None);
            f(0)?;
            gpu.time_us(4, 3, f)
        };
        let rule = crate::model::gdn_split(s, hv, gpu);
        // The scan reads the prep's outputs.
        rig.launch(gpu, Which::Chunk, rig.fresh.ptr, 7, rule, None)?;
        let base = t(rule)?;
        let mut best = (1.0, rule);
        for split in [1, 2, 4] {
            if split != rule {
                let r = t(split)? / base;
                if r < best.0 {
                    best = (r, split);
                }
            }
        }
        // Confirmed against the rule's choice timed afresh.
        if best.0 <= 1.0 - MIN_GAIN && confirmed(|| Ok(vec![t(best.1)? / t(rule)?]))?.is_some() {
            line += &format!(" {lo}-{s} {}(rule {rule}) x{:.2}", best.1, 1.0 / best.0);
            out.push(Split {
                heads: hv,
                seqs: [
                    lo,
                    if s == *SPLIT_SEQS.last().unwrap() {
                        1 << 20
                    } else {
                        s
                    },
                ],
                split: best.1,
            });
        } else {
            line += &format!(" {lo}-{s} {rule}");
        }
        lo = s + 1;
    }
    eprintln!("{line}");
    Ok(out)
}

/// gdn_decode's buffers and blocks per SM, if a shape other than the
/// built-in one measured faster over 1 to 256 sequences.
pub fn decode_shape(
    gpu: &Gpu,
    hv: usize,
    hk: usize,
    max_seqs: usize,
) -> Result<Option<DecodeShape>> {
    use crate::gdnbench::{GdnRig, Which};
    let Some(rule) = crate::model::gdn_decode_shape(true, hv, gpu) else {
        return Ok(None);
    };
    let mut shapes = vec![rule];
    for nbuf in 2..=3 {
        for per_sm in 1..=2 {
            if GdnRig::decode_fits(gpu, nbuf, per_sm) && !shapes.contains(&(nbuf, per_sm)) {
                shapes.push((nbuf, per_sm));
            }
        }
    }
    let rigs: Vec<GdnRig> = [1, 8, 32, 128, 256]
        .into_iter()
        .filter(|&s| s <= max_seqs.max(1))
        .map(|s| GdnRig::new(gpu, hv, hk, s, 1))
        .collect::<Result<_>>()?;
    let t = |shape: (usize, usize), rig: &GdnRig| {
        graph_us(gpu, 4, &|_| {
            rig.launch(gpu, Which::Decode, rig.fresh.ptr, 2, 1, Some(shape))
        })
    };
    let ratios = |shape| -> Result<Vec<f64>> {
        rigs.iter()
            .map(|r| Ok(t(shape, r)? / t(rule, r)?))
            .collect()
    };
    let mut best: Option<(f64, (usize, usize))> = None;
    for &shape in &shapes[1..] {
        let r = ratios(shape)?;
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        if r.iter().all(|&x| x <= 1.0) && mean <= 1.0 - MIN_GAIN && best.is_none_or(|b| mean < b.0)
        {
            best = Some((mean, shape));
        }
    }
    let found = match best {
        Some((_, shape)) => confirmed(|| ratios(shape))?.map(|m| (m, shape)),
        None => None,
    };
    match found {
        Some((m, (nbuf, per_sm))) => {
            eprintln!(
                "gdn_decode: {nbuf} buffers, {per_sm} blocks per SM (rule {rule:?}) x{:.2}",
                1.0 / m
            );
            Ok(Some(DecodeShape {
                heads: hv,
                nbuf,
                per_sm,
            }))
        }
        None => {
            eprintln!("gdn_decode: {rule:?}, the rule's");
            Ok(None)
        }
    }
}

/// Rows at which GGUF paths are compared (1 to 4 always stream the weights).
const QUANT_ROWS: [usize; 18] = [
    5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 512, 1024, 2048, 4096,
];
const OFF: usize = usize::MAX;

/// Where GGUF matrices should change path, if thresholds other than the
/// built-in ones measured faster. Times the model's own quantized matrices,
/// as it runs them (fused parts included) and cycling through the layers so
/// the weights come from memory, on each path at each of `QUANT_ROWS`; then
/// picks the thresholds whose choices sum to the least time without making
/// any row count slower than the built-in ones do.
pub fn quant_rows(
    gpu: &Gpu,
    linears: &[&crate::linear::Linear],
    hidden: usize,
) -> Result<Option<QuantRows>> {
    use crate::linear::{Input, Kind, Quants};
    // The model's matrices by shape and format.
    let mut groups: BTreeMap<String, Vec<&crate::linear::Linear>> = BTreeMap::new();
    for l in linears {
        if l.parts
            .iter()
            .any(|p| matches!(p.kind, Kind::Quant(_) | Kind::Tiled(_)))
        {
            let key = format!(
                "{} {:?}",
                l.cols,
                l.parts.iter().map(|p| (p.rows, p.kind)).collect::<Vec<_>>()
            );
            groups.entry(key).or_default().push(l);
        }
    }
    if groups.is_empty() {
        return Ok(None);
    }
    let mut types: Vec<crate::gguf::GgmlType> = Vec::new();
    for l in linears {
        for p in &l.parts {
            if let Kind::Quant(ty) | Kind::Tiled(ty) = p.kind
                && !types.contains(&ty)
            {
                types.push(ty);
            }
        }
    }
    let largest = linears
        .iter()
        .flat_map(|l| l.parts.iter().map(|p| p.rows * l.cols))
        .max()
        .unwrap_or(0);
    // Paths, by thresholds that force each: bf16 tiles, int8 tiles, int8
    // large tiles, dequantized for cuBLAS.
    let q = |int8_rows, big_rows, dense_rows| QuantRows {
        hidden,
        int8_rows: Some(int8_rows),
        int8_max: Some(4096),
        big_rows: Some(big_rows),
        dense_rows: Some(dense_rows),
    };
    let paths = [
        q(OFF, OFF, OFF),
        q(1, OFF, OFF),
        q(1, 1, OFF),
        q(OFF, OFF, 0),
    ];
    let mut defaults = Quants::default();
    defaults.hidden = hidden;
    defaults.int8_rows = crate::linear::int8_rows(gpu, hidden);
    for &ty in &types {
        defaults.get(gpu, ty)?;
    }
    let mut qs = Vec::new();
    for p in &paths {
        let mut qq = Quants::default();
        qq.hidden = hidden;
        qq.int8_rows = 1;
        for &ty in &types {
            qq.get(gpu, ty)?;
        }
        qq.set_rows(p);
        qq.prepare(gpu, largest)?;
        qs.push(qq);
    }
    let max = *QUANT_ROWS.last().unwrap();
    let cols = linears.iter().map(|l| l.cols).max().unwrap_or(0);
    let outs = linears.iter().map(|l| l.out()).max().unwrap_or(0);
    let x: Vec<u8> = (0..max * cols)
        .flat_map(|i| {
            half::bf16::from_f32(((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0).to_le_bytes()
        })
        .collect();
    let x = gpu.upload(&x)?;
    let y = gpu.alloc(max * outs * 2)?;
    let mut time = vec![[0f64; 4]; QUANT_ROWS.len()];
    for ls in groups.values() {
        // Every layer runs each matrix once a step; up to 16 copies are cycled.
        let cycle = &ls[..ls.len().min(16)];
        for (ri, &rows) in QUANT_ROWS.iter().enumerate() {
            for (pi, qq) in qs.iter().enumerate() {
                let f = |i: usize| {
                    let l = cycle[i % cycle.len()];
                    l.forward(gpu, qq, Input::bf16(x.ptr), rows, y.ptr, l.out(), false)
                };
                time[ri][pi] += ls.len() as f64 * graph_us(gpu, cycle.len(), &f)?;
            }
        }
    }
    // The path thresholds `r` pick at row count `rows`, as Linear::forward
    // picks it.
    let path = |r: &QuantRows, rows: usize| -> usize {
        let int8 = (r.int8_rows.unwrap_or(OFF)..=r.int8_max.unwrap_or(0)).contains(&rows);
        if int8 {
            if rows >= r.big_rows.unwrap_or(OFF) {
                2
            } else {
                1
            }
        } else if rows > r.dense_rows.unwrap_or(OFF) {
            3
        } else {
            0
        }
    };
    // Their cost: each row count's time relative to the fastest path there.
    let cost = |r: &QuantRows| -> f64 {
        QUANT_ROWS
            .iter()
            .enumerate()
            .map(|(ri, &rows)| {
                let best = time[ri].iter().cloned().fold(f64::MAX, f64::min);
                time[ri][path(r, rows)] / best
            })
            .sum()
    };
    let current = defaults.rows();
    // No row count may get slower than the built-in thresholds make it: a
    // gain at some batch sizes must not cost another.
    let no_worse = |r: &QuantRows| {
        QUANT_ROWS
            .iter()
            .enumerate()
            .all(|(ri, &rows)| time[ri][path(r, rows)] <= time[ri][path(&current, rows)] * 1.02)
    };
    let grid: Vec<usize> = QUANT_ROWS.iter().cloned().chain([OFF]).collect();
    let mut best = (cost(&current), current.clone());
    for &ir in &grid {
        for &im in QUANT_ROWS.iter().filter(|&&m| ir == OFF || m >= ir) {
            for &br in &grid {
                for &dr in grid.iter().chain(&[4]) {
                    let r = QuantRows {
                        hidden,
                        int8_rows: Some(ir),
                        int8_max: Some(im),
                        big_rows: Some(br),
                        dense_rows: Some(dr),
                    };
                    let c = cost(&r);
                    if c < best.0 - 1e-9 && no_worse(&r) {
                        best = (c, r);
                    }
                }
            }
        }
    }
    let gain = cost(&current) / best.0;
    let mut line = format!("GGUF paths ({} kinds of matrix):", groups.len());
    for (ri, &rows) in QUANT_ROWS.iter().enumerate() {
        let names = ["tiles", "int8", "int8 big", "dense"];
        let (pi, _) = time[ri]
            .iter()
            .enumerate()
            .fold((0, f64::MAX), |b, (i, &t)| if t < b.1 { (i, t) } else { b });
        line += &format!(" {rows}:{}", names[pi]);
    }
    eprintln!("{line}");
    if gain < 1.0 + MIN_GAIN {
        eprintln!(
            "GGUF thresholds: built-in {current:?} within {:.1}%",
            (gain - 1.0) * 100.0
        );
        return Ok(None);
    }
    eprintln!(
        "GGUF thresholds: {:?} (built-in {current:?}) x{gain:.2} over the row counts",
        best.1
    );
    // Only what differs from the built-in.
    let b = best.1;
    let pick = |t: Option<usize>, c: Option<usize>| if t != c { t } else { None };
    Ok(Some(QuantRows {
        hidden,
        int8_rows: pick(b.int8_rows, current.int8_rows),
        int8_max: pick(b.int8_max, current.int8_max),
        big_rows: pick(b.big_rows, current.big_rows),
        dense_rows: pick(b.dense_rows, current.dense_rows),
    }))
}

/// Loads the model as `cfg` describes, tunes it and returns its table.
#[allow(unsafe_code)]
pub fn run_model(cfg: EngineConfig) -> Result<Table> {
    // SAFETY: set before the engine starts any thread.
    unsafe {
        std::env::set_var("POCKET_TUNING", "0");
        std::env::set_var("POCKET_RECORD_GEMMS", "1");
    }
    let model = cfg
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut e = Engine::new(cfg.clone())?;
    // One prompt as long as a step allows, so prompt-sized products run too.
    let n = cfg
        .max_batch_tokens
        .min(cfg.max_context.saturating_sub(8))
        .max(1);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let prompt = (0..n).map(|i| 1000 + (i % 5000) as u32).collect();
    e.submit(Request::text(prompt, 2, Sampling::default(), tx));
    loop {
        e.step()?;
        match rx.try_recv() {
            Ok(Event::Done { .. }) | Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
            Ok(Event::Error(m)) => return Err(Error::Config(m)),
            _ => {}
        }
    }
    let uses = e.gpu.recorded();
    eprintln!(
        "tuning {model} on {} ({} SMs): {} matrix shapes",
        e.gpu.name,
        e.gpu.sms,
        uses.len()
    );
    // SAFETY: plain library query.
    let version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
    let mut t = Table::new(&e.gpu.name, e.gpu.sms, version);
    t.models.push(model);
    t.gemm = gemms(&e.gpu, &uses)?;
    let info = e.info();
    let (hv, hk, hidden) = (info.cfg.lin_v_heads, info.cfg.lin_k_heads, info.cfg.hidden);
    if e.model().gguf {
        t.quant = quant_rows(&e.gpu, &e.model().linears(), hidden)?
            .into_iter()
            .collect();
    }
    drop(e);
    // The rest on a fresh device, with the products tuned so far (GGUF's
    // dequantized path runs cuBLAS).
    let mut gpu = Gpu::new(cfg.device)?;
    t.reindex();
    gpu.tuning = Some(t.clone());
    t.gdn_split = splits(&gpu, hv, hk, cfg.max_seqs)?;
    t.gdn_decode = decode_shape(&gpu, hv, hk, cfg.max_seqs)?
        .into_iter()
        .collect();
    Ok(t)
}
