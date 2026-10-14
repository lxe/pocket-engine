//! End-to-end checks on a GPU. They run when POCKET_ENGINE_MODEL names a
//! Qwen3.5 0.8B, 2B, 4B or 9B or Qwen3.8-27B directory, or a
//! GGUF file of one, optionally with FP8 weights (POCKET_ENGINE_FP8=1), and the crate is built with `--features cuda`;
//! otherwise they pass without doing anything.

use pocket_engine::engine::{self, Engine, EngineConfig, Scoring};
use pocket_engine::sched::{Event, Request, Sampling};
use pocket_engine::server::{self, ChatTokenizer};
use std::path::PathBuf;

/// One engine at a time: three copies of a larger model need not fit together.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn model() -> Option<PathBuf> {
    let dir = std::env::var_os("POCKET_ENGINE_MODEL").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("POCKET_ENGINE_MODEL is not set: skipping");
    }
    dir
}

fn config(model: PathBuf) -> EngineConfig {
    EngineConfig {
        max_seqs: 16,
        max_batch_tokens: 64,
        max_context: 1024,
        // KV is capped by max_seqs x max_context, so tests stay small.
        memory_fraction: 0.95,
        image_pool: 4096,
        max_image_pixels: 1 << 20,
        fp8: fp8(),
        fp4: fp4(),
        kv_fp8: kv_fp8(),
        state_bf16: !std::env::var("POCKET_ENGINE_STATE_F32").is_ok_and(|v| v == "1"),
        mtp: std::env::var("POCKET_ENGINE_MTP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        int8: int8(),
        ..EngineConfig::new(model)
    }
}

fn fixture<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[derive(serde::Deserialize)]
struct Pair {
    prompt: Vec<u32>,
    forced: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct Reference {
    argmax: u32,
    top: Vec<(u32, f32)>,
}

/// The fixture name for the model under test, from its hidden width.
fn size(dir: &std::path::Path) -> &'static str {
    match pocket_engine::model::Config::for_model(dir).unwrap().hidden {
        1024 => "qwen3.5-0.8b",
        2048 => "qwen3.5-2b",
        2560 => "qwen3.5-4b",
        4096 => "qwen3.5-9b",
        // Qwen3.5, 3.6 and 3.8-27B share this shape; the fixture is 3.8's.
        5120 => "qwen3.8-27b",
        h => panic!("no fixture for hidden size {h}"),
    }
}

/// Teacher-forced against Transformers (bf16): the engine picks the reference
/// token wherever the reference's top two are more than one bf16 rounding
/// step (0.125 at these logit sizes) apart, and its log-probability of that
/// token stays close. 64-token steps split the prompts into chunks.
#[test]
fn matches_transformers_teacher_forced() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let name = size(&dir);
    let pairs: Vec<Pair> = fixture(&format!("{name}.pairs.json"));
    let reference: Vec<Vec<Reference>> = fixture(&format!("{name}.ref.json"));
    let gguf = quantized(&dir);
    let mut e = Engine::new(config(dir)).unwrap();
    let got = e
        .score(
            pairs
                .into_iter()
                .map(|p| Scoring {
                    prompt: p.prompt,
                    forced: p.forced,
                    images: vec![],
                    rope: None,
                    adapter: 0,
                })
                .collect(),
        )
        .unwrap();
    assert!(reference.iter().map(Vec::len).sum::<usize>() > 200);
    if gguf || fp4() {
        // 4-bit weights carry their own error against the bf16 reference.
        close_enough(&reference, &got, 85, 0.1);
    } else if fp8() || kv_fp8() {
        close_enough(&reference, &got, 93, 0.03);
    } else {
        assert!(agree(&reference, &got) < 0.01);
    }
}

/// For quantized weights: at least `pct`% of picks agree with the bf16
/// reference and the median log-probability difference stays under `median`.
fn close_enough(
    reference: &[Vec<Reference>],
    got: &[Vec<pocket_engine::sched::Scored>],
    pct: usize,
    median: f32,
) {
    let (mut same, mut total, mut diffs) = (0, 0, Vec::new());
    for (r_seq, g_seq) in reference.iter().zip(got) {
        for (r, g) in r_seq.iter().zip(g_seq) {
            total += 1;
            same += (r.argmax == g.argmax) as usize;
            let lp = g
                .top
                .iter()
                .find(|t| t.0 == r.argmax)
                .map_or(-99.0, |t| t.1);
            diffs.push((lp - r.top[0].1).abs());
        }
    }
    diffs.sort_by(f32::total_cmp);
    assert!(same * 100 >= total * pct, "{same} of {total} picks agree");
    let m = diffs[diffs.len() / 2];
    assert!(m < median, "median {m}");
}

/// POCKET_ENGINE_FP8=1 runs the suite with weights quantized to FP8 at load.
fn fp8() -> bool {
    std::env::var("POCKET_ENGINE_FP8").is_ok_and(|v| v == "1")
}

/// POCKET_ENGINE_FP4=1 runs the suite with MLP weights quantized to FP4 at
/// load and the other projections to FP8 (Blackwell).
fn fp4() -> bool {
    std::env::var("POCKET_ENGINE_FP4").is_ok_and(|v| v == "1")
}

/// POCKET_ENGINE_NO_INT8=1 keeps GGUF activations in bf16.
fn int8() -> bool {
    !std::env::var("POCKET_ENGINE_NO_INT8").is_ok_and(|v| v == "1")
}

/// POCKET_ENGINE_KV_FP8=1 runs the suite with an FP8 KV cache.
fn kv_fp8() -> bool {
    std::env::var("POCKET_ENGINE_KV_FP8").is_ok_and(|v| v == "1")
}

/// How far apart the reference's top two may be for a miss to count as a
/// near tie: one bf16 step.
const TIE_GAP: f32 = 0.126;

fn quantized(dir: &std::path::Path) -> bool {
    pocket_engine::source::gguf_path(dir).is_some()
}

#[derive(serde::Deserialize)]
struct ImagePair {
    messages: Vec<serde_json::Value>,
    forced: Vec<u32>,
}

/// Checks teacher-forced picks against the reference, allowing ties within
/// one bf16 step; returns the median log-probability difference.
fn agree(reference: &[Vec<Reference>], got: &[Vec<pocket_engine::sched::Scored>]) -> f32 {
    let mut diffs = Vec::new();
    for (r_seq, g_seq) in reference.iter().zip(got) {
        assert_eq!(r_seq.len(), g_seq.len());
        for (r, g) in r_seq.iter().zip(g_seq) {
            let tied = r.top[0].1 - r.top[1].1 <= TIE_GAP;
            assert!(
                tied || r.argmax == g.argmax,
                "pick {} vs reference {} (reference top {:?}, ours {:?})",
                g.argmax,
                r.argmax,
                &r.top[..2],
                &g.top[..2]
            );
            let lp = g
                .top
                .iter()
                .find(|t| t.0 == r.argmax)
                .map_or(-99.0, |t| t.1);
            diffs.push((lp - r.top[0].1).abs());
        }
    }
    diffs.sort_by(f32::total_cmp);
    diffs[diffs.len() / 2]
}

/// Image prompts: the same prompt tokens as Transformers' processor, and the
/// same picks. Two images keep their sides; one is resized.
#[test]
fn images_match_transformers() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if quantized(&dir) {
        eprintln!("GGUF models are served without images: skipping");
        return;
    }
    let name = size(&dir);
    let pairs: Vec<ImagePair> = fixture(&format!("{name}.images.pairs.json"));
    let reference: Vec<Vec<Reference>> = fixture(&format!("{name}.images.ref.json"));
    let prompts: Vec<Vec<u32>> = fixture(&format!("{name}.images.prompts.json"));
    let tok = ChatTokenizer::load(&dir).unwrap();
    let spec = pocket_engine::image::Spec::load(&dir, 1 << 20).unwrap();
    let placeholder = pocket_engine::model::Config::load(&dir)
        .unwrap()
        .image_token;
    let mut items = Vec::new();
    for (p, want) in pairs.into_iter().zip(&prompts) {
        let (ids, urls) = tok.chat(&p.messages).unwrap();
        let encoded: Vec<Vec<u8>> = urls
            .iter()
            .map(|u| server::data_url(u).unwrap().unwrap())
            .collect();
        let (prompt, images, rope) =
            server::with_images(ids, &encoded, &spec, placeholder).unwrap();
        assert_eq!(&prompt, want, "prompt tokens");
        items.push(Scoring {
            prompt,
            forced: p.forced,
            images,
            rope,
            adapter: 0,
        });
    }
    let mut e = Engine::new(config(dir)).unwrap();
    let got = e.score(items).unwrap();
    if fp4() {
        close_enough(&reference, &got, 90, 0.05);
        return;
    }
    if fp8() {
        // Image tokens pass through every FP8 projection.
        close_enough(&reference, &got, 95, 0.01);
        return;
    }
    assert!(agree(&reference, &got) < 0.01);
}

fn generate(e: &mut Engine, prompts: &[Vec<u32>], n: usize) -> Vec<Vec<u32>> {
    let runs: Vec<_> = prompts
        .iter()
        .map(|p| (p.clone(), Sampling::default()))
        .collect();
    generate_with(e, &runs, n)
}

fn generate_with(e: &mut Engine, runs: &[(Vec<u32>, Sampling)], n: usize) -> Vec<Vec<u32>> {
    let mut rxs = Vec::new();
    for (p, sampling) in runs {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        e.sched
            .submit(Request::text(p.clone(), n, sampling.clone(), tx));
        rxs.push(rx);
    }
    while !e.sched.idle() {
        e.step().unwrap();
    }
    rxs.into_iter()
        .map(|mut rx| {
            let mut out = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                if let Event::Token(t) = ev {
                    out.push(t);
                }
            }
            out
        })
        .collect()
}

/// A request's predictions do not depend on what else shares its steps,
/// beyond rounding: the same tokens are scored alone and among the others.
/// GEMM kernels are chosen by batch size, so sums differ in the last bit;
/// bf16 holds picks to near ties and a median log-probability difference
/// under 0.01. FP8 and FP4 activations, and an FP8 KV cache, magnify those
/// bits about as much as their own error against bf16, so they take the FP8
/// bar.
#[test]
fn batching_does_not_change_predictions() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let pairs: Vec<Pair> = fixture(&format!("{}.pairs.json", size(&dir)));
    // Without the prefix cache: a cached prefix would move the first prompt's
    // remaining tokens into a step shared with the others.
    let mut e = Engine::new(EngineConfig {
        prefix_cache: Some(0),
        ..config(dir)
    })
    .unwrap();
    let items = |pairs: &[Pair]| -> Vec<Scoring> {
        pairs
            .iter()
            .map(|p| Scoring {
                prompt: p.prompt.clone(),
                forced: p.forced.clone(),
                images: vec![],
                rope: None,
                adapter: 0,
            })
            .collect()
    };
    let alone = e.score(items(&pairs[..1])).unwrap();
    let together = e.score(items(&pairs)).unwrap();
    let alone: Vec<Vec<Reference>> = alone
        .into_iter()
        .map(|seq| {
            seq.into_iter()
                .map(|s| Reference {
                    argmax: s.argmax,
                    top: s.top,
                })
                .collect()
        })
        .collect();
    if fp8() || fp4() || kv_fp8() {
        close_enough(&alone, &together[..1], 90, 0.03);
    } else {
        assert!(agree(&alone, &together[..1]) < 0.01);
    }
    // And greedy generation runs alongside.
    let prompts: Vec<Vec<u32>> = pairs.into_iter().map(|p| p.prompt).collect();
    let together = generate(&mut e, &prompts, 8);
    assert!(together.iter().all(|t| !t.is_empty()));
}

/// Whether a checkpoint directory has a multi-token-prediction layer.
fn has_mtp(dir: &std::path::Path) -> bool {
    if dir.extension().is_some_and(|e| e == "gguf") {
        return pocket_engine::gguf::Gguf::open(dir).is_ok_and(|g| g.nextn_block().is_some());
    }
    std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .is_ok_and(|s| s.contains("\"mtp.fc.weight\""))
}

/// With multi-token prediction each step verifies drafted tokens, and
/// greedy output stays the model's: where it departs from the model's pick
/// with that output fed back (teacher-forced, which never drafts), the two
/// were within rounding of a tie. Most drafts are accepted.
#[test]
fn drafts_keep_the_models_greedy_choices() {
    let Some(dir) = model() else { return };
    if !has_mtp(&dir) {
        eprintln!("no multi-token-prediction layer: skipping");
        return;
    }
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let pairs: Vec<Pair> = fixture(&format!("{}.pairs.json", size(&dir)));
    let mut e = Engine::new(EngineConfig {
        mtp: 3,
        ..config(dir.clone())
    })
    .unwrap();
    let prompts: Vec<Vec<u32>> = pairs.into_iter().map(|p| p.prompt).collect();
    let out = generate(&mut e, &prompts, 64);
    let c = e.sched.counters.clone();
    // Three drafts a step: about half accepted.
    assert!(
        c.drafted > 100 && c.accepted * 3 > c.drafted,
        "{} of {} drafts accepted",
        c.accepted,
        c.drafted
    );
    let scored = e
        .score(
            prompts
                .iter()
                .zip(&out)
                .map(|(p, o)| Scoring {
                    prompt: p.clone(),
                    forced: o.clone(),
                    images: vec![],
                    rope: None,
                    adapter: 0,
                })
                .collect(),
        )
        .unwrap();
    let (mut positions, mut departures, mut worst) = (0, 0, (0.0f32, String::new()));
    for (o, sc) in out.iter().zip(&scored) {
        for (i, (&tok, s)) in o.iter().zip(sc).enumerate() {
            positions += 1;
            if tok == s.argmax {
                continue;
            }
            departures += 1;
            let lp = |t: u32| s.top.iter().find(|x| x.0 == t).map(|x| x.1);
            let gap = lp(s.argmax).unwrap() - lp(tok).unwrap_or(f32::INFINITY);
            if gap > worst.0 {
                worst = (
                    gap,
                    format!("position {i}: picked {tok}, the model {}", s.argmax),
                );
            }
        }
    }
    if fp8() || fp4() || kv_fp8() || (quantized(&dir) && int8()) {
        // Verification runs several rows per sequence, so its matmuls take
        // other kernels, and FP8 or int8 activations magnify the last-bit
        // differences (see batching_does_not_change_predictions).
        assert!(
            departures * 20 <= positions,
            "{departures} of {positions} positions depart; worst {} by {}",
            worst.1,
            worst.0
        );
    } else {
        assert!(worst.0 < 0.15, "{} by {}", worst.1, worst.0);
    }
}

/// The tokens a sampler should keep, given a row's top five log-probabilities.
type KeepSet<'a> = dyn Fn(&[(u32, f32)]) -> Vec<u32> + 'a;

/// A prompt that extends a cached one starts from the cached prefix (keys,
/// values and recurrent state) and produces what it would without the cache.
#[test]
fn prefix_cache_reuses_without_changing_output() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tok = ChatTokenizer::load(&dir).unwrap();
    let text = "Here is a long note about trains, tracks and timetables. ".repeat(24);
    let first = tok
        .chat(&[serde_json::json!({ "role": "user", "content": text })])
        .unwrap()
        .0;
    let second = tok
        .chat(&[serde_json::json!({ "role": "user", "content": format!("{text} Summarize it in one line.") })])
        .unwrap()
        .0;
    assert!(first.len() > 2 * pocket_engine::sched::CACHE_UNIT);
    let mut cached = Engine::new(config(dir.clone())).unwrap();
    generate(&mut cached, std::slice::from_ref(&first), 8);
    let with = generate(&mut cached, std::slice::from_ref(&second), 24);
    assert!(
        cached.sched.counters.cached_tokens >= pocket_engine::sched::CACHE_UNIT as u64,
        "the second prompt reused the first's prefix"
    );
    drop(cached);
    let mut plain = Engine::new(EngineConfig {
        prefix_cache: Some(0),
        ..config(dir.clone())
    })
    .unwrap();
    let without = generate(&mut plain, std::slice::from_ref(&second), 24);
    assert_eq!(plain.sched.counters.cached_tokens, 0);
    assert_eq!(with, without);
    drop(plain);
    // With whole prompts in one step, the state at a boundary inside the
    // step is written by the step itself. Prompts then split differently
    // with and without the cache, so matmuls round differently: the scores
    // of the same continuation agree as batching's do.
    let big = |prefix_cache| EngineConfig {
        max_batch_tokens: 1024,
        prefix_cache,
        ..config(dir.clone())
    };
    let scoring = |prompt: &Vec<u32>, forced: &Vec<u32>| Scoring {
        prompt: prompt.clone(),
        forced: forced.clone(),
        images: vec![],
        rope: None,
        adapter: 0,
    };
    let mut plain = Engine::new(big(Some(0))).unwrap();
    let cont = generate(&mut plain, std::slice::from_ref(&second), 16).remove(0);
    let reference: Vec<Vec<Reference>> = plain
        .score(vec![scoring(&second, &cont)])
        .unwrap()
        .into_iter()
        .map(|seq| {
            seq.into_iter()
                .map(|s| Reference {
                    argmax: s.argmax,
                    top: s.top,
                })
                .collect()
        })
        .collect();
    drop(plain);
    let mut cached = Engine::new(big(None)).unwrap();
    cached.score(vec![scoring(&first, &vec![cont[0]])]).unwrap();
    let got = cached.score(vec![scoring(&second, &cont)]).unwrap();
    assert!(cached.sched.counters.cached_tokens >= pocket_engine::sched::CACHE_UNIT as u64);
    // Both runs are this engine; the chunked recurrence's bf16 roundings
    // fall on different chunk boundaries in each, so the bar is looser than
    // the one against Transformers. GGUF matmuls also take different kernels
    // for the two splits (tensor-core tiles or dequantize and cuBLAS), so
    // near ties may flip.
    if fp8() || fp4() || kv_fp8() {
        close_enough(&reference, &got, 90, 0.03);
    } else if quantized(&dir) {
        close_enough(&reference, &got, 90, 0.02);
    } else {
        assert!(agree(&reference, &got) < 0.02);
    }
}

/// Draws one token `n` times from the first position of a prompt.
fn draws(e: &mut Engine, prompt: &[u32], sampling: Sampling, n: usize) -> Vec<u32> {
    let runs: Vec<_> = (0..n)
        .map(|i| {
            let s = Sampling {
                seed: 1000 + i as u64,
                ..sampling.clone()
            };
            (prompt.to_vec(), s)
        })
        .collect();
    generate_with(e, &runs, 1)
        .into_iter()
        .map(|t| t[0])
        .collect()
}

/// Top-k, top-p and min-p keep exactly the tokens they should, and the kept
/// tokens are drawn in proportion to the model's probabilities.
#[test]
fn samplers_follow_the_model_distribution() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tok = ChatTokenizer::load(&dir).unwrap();
    let prompt = tok
        .chat(&[serde_json::json!({ "role": "user", "content": "Name one color. One word." })])
        .unwrap()
        .0;
    let mut e = Engine::new(config(dir)).unwrap();
    // The model's top five at the first position, with log-probabilities.
    let top = e
        .score(vec![Scoring {
            prompt: prompt.clone(),
            forced: vec![0],
            images: vec![],
            rope: None,
            adapter: 0,
        }])
        .unwrap()[0][0]
        .top
        .clone();
    let p: Vec<f64> = top.iter().map(|t| (t.1 as f64).exp()).collect();
    let n = 400;
    // Each draw's own top five: the same n requests scored in the same steps
    // get the same batch layout as the draws. Logits move by a last bit with
    // batch layout, and FP8 magnifies that, so keep sets are judged per row.
    let rows: Vec<Vec<(u32, f32)>> = e
        .score(
            (0..n)
                .map(|_| Scoring {
                    prompt: prompt.clone(),
                    forced: vec![0],
                    images: vec![],
                    rope: None,
                    adapter: 0,
                })
                .collect(),
        )
        .unwrap()
        .into_iter()
        .map(|s| s[0].top.clone())
        .collect();
    let count = |d: &[u32], t: u32| d.iter().filter(|&&x| x == t).count();
    let close = |got: usize, want: f64| {
        let sd = (n as f64 * want * (1.0 - want)).sqrt().max(1.0);
        (got as f64 - n as f64 * want).abs() < 4.0 * sd + 2.0
    };
    let all_kept = |d: &[u32], keep: &KeepSet<'_>, what: &str| {
        for (i, (t, row)) in d.iter().zip(&rows).enumerate() {
            assert!(
                keep(row).contains(t),
                "{what}: draw {i} is {t}, row {row:?}"
            );
        }
    };
    // top_k = 2: only the top two, in proportion.
    let d = draws(
        &mut e,
        &prompt,
        Sampling {
            temperature: 1.0,
            top_k: 2,
            ..Sampling::default()
        },
        n,
    );
    all_kept(&d, &|row| vec![row[0].0, row[1].0], "top_k");
    assert!(
        close(count(&d, top[0].0), p[0] / (p[0] + p[1])),
        "top_k split"
    );
    // top_p between the second and third cumulative mass: the top three.
    let c: Vec<f64> = p
        .iter()
        .scan(0.0, |a, x| {
            *a += x;
            Some(*a)
        })
        .collect();
    let top_p = ((c[1] + c[2]) / 2.0) as f32;
    let d = draws(
        &mut e,
        &prompt,
        Sampling {
            temperature: 1.0,
            top_p,
            ..Sampling::default()
        },
        n,
    );
    all_kept(
        &d,
        &|row| {
            // The smallest prefix holding top_p of the probability.
            let mut mass = 0.0;
            let mut keep = Vec::new();
            for &(t, lp) in row {
                keep.push(t);
                mass += lp.exp();
                if mass >= top_p {
                    break;
                }
            }
            keep
        },
        "top_p",
    );
    assert!(close(count(&d, top[0].0), p[0] / c[2]), "top_p split");
    // min_p at the second token's ratio keeps the top two.
    let min_p = ((p[1] / p[0] + p[2] / p[0]) / 2.0) as f32;
    let d = draws(
        &mut e,
        &prompt,
        Sampling {
            temperature: 1.0,
            min_p,
            ..Sampling::default()
        },
        n,
    );
    all_kept(
        &d,
        &|row| {
            let floor = row[0].1 + min_p.ln();
            row.iter().filter(|x| x.1 >= floor).map(|x| x.0).collect()
        },
        "min_p",
    );
    // Plain temperature sampling reaches beyond the top token, reproducibly.
    let plain = Sampling {
        temperature: 1.0,
        ..Sampling::default()
    };
    let d = draws(&mut e, &prompt, plain.clone(), 64);
    assert_eq!(
        d,
        draws(&mut e, &prompt, plain, 64),
        "same seeds, same draws"
    );
    // A -100 bias removes the greedy pick; +100 forces any token.
    let greedy = draws(&mut e, &prompt, Sampling::default(), 1)[0];
    assert_eq!(greedy, top[0].0);
    let banned = Sampling {
        logit_bias: vec![(greedy, -100.0)],
        ..Sampling::default()
    };
    assert_eq!(draws(&mut e, &prompt, banned, 1)[0], top[1].0);
    let forced = Sampling {
        logit_bias: vec![(12345, 100.0)],
        ..Sampling::default()
    };
    assert_eq!(draws(&mut e, &prompt, forced, 1)[0], 12345);
}

/// Penalties discourage repeating generated tokens.
#[test]
fn penalties_reduce_repetition() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tok = ChatTokenizer::load(&dir).unwrap();
    let prompt = tok
        .chat(&[serde_json::json!({ "role": "user", "content": "Write the word yes 30 times, separated by spaces, nothing else." })])
        .unwrap()
        .0;
    let mut e = Engine::new(config(dir)).unwrap();
    // The most repeated token's count: penalties cut it, whether the model
    // then varies its words or stops early.
    let most = |t: &[u32]| {
        let mut c = std::collections::HashMap::new();
        t.iter().for_each(|x| *c.entry(x).or_insert(0) += 1);
        c.into_values().max().unwrap_or(0)
    };
    let plain = generate_with(&mut e, &[(prompt.clone(), Sampling::default())], 48);
    let penalized = Sampling {
        frequency_penalty: 2.0,
        presence_penalty: 2.0,
        ..Sampling::default()
    };
    let varied = generate_with(&mut e, &[(prompt, penalized)], 48);
    assert!(
        most(&plain[0]) >= 20,
        "the prompt repeats without penalties"
    );
    assert!(
        most(&varied[0]) * 2 < most(&plain[0]),
        "{:?} vs {:?}",
        varied[0],
        plain[0]
    );
}

/// The OpenAI-compatible server streams text and reports usage.
#[test]
fn serves_chat_completions() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (tx, info, health, _) = engine::spawn(config(dir.clone())).unwrap();
    let tok = ChatTokenizer::load(&dir).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let bind = "127.0.0.1:18197";
        tokio::spawn(server::serve(
            tok,
            None,
            tx,
            info,
            health,
            None,
            "test".into(),
            bind,
        ));
        let client = reqwest::Client::new();
        let url = format!("http://{bind}");
        for _ in 0..100 {
            if client.get(format!("{url}/health")).send().await.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let body = serde_json::json!({ "model": "test", "max_tokens": 8, "temperature": 0,
            "messages": [{ "role": "user", "content": "Count to five." }] });
        let whole: serde_json::Value = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let text = whole["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!text.is_empty());
        assert_eq!(whole["usage"]["completion_tokens"], 8);
        let mut stream_body = body.clone();
        stream_body["stream"] = true.into();
        stream_body["stream_options"] = serde_json::json!({ "include_usage": true });
        let sse = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&stream_body)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let mut streamed = String::new();
        let mut usage = None;
        for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if line == "[DONE]" {
                continue;
            }
            let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
                streamed.push_str(c);
            }
            if chunk["usage"].is_object() {
                usage = Some(chunk["usage"]["completion_tokens"].clone());
            }
        }
        assert_eq!(streamed, text, "streamed and whole greedy outputs agree");
        assert_eq!(usage, Some(8.into()));
        assert!(sse.trim_end().ends_with("data: [DONE]"));
    });
}

/// Writes a PEFT adapter of rank `r` for every linear layer of the model at
/// `dir`, with A ~ N(0, 1/cols) and B ~ N(0, `b_std`) from `seed`.
fn synthetic_adapter(
    dir: &std::path::Path,
    out: &std::path::Path,
    r: usize,
    b_std: f32,
    seed: u64,
) {
    use pocket_engine::model::{Config, GDN_DIM, HEAD_DIM};
    let c = Config::for_model(dir).unwrap();
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut normal = move || {
        // Sum of uniforms: close enough to normal for a test.
        let mut s = 0.0f32;
        for _ in 0..4 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            s += (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }
        s * 1.732
    };
    let (hk, hv) = (c.lin_k_heads, c.lin_v_heads);
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for l in 0..c.layers {
        let mut mods: Vec<(String, usize, usize)> = if c.attn_layers[l] {
            vec![
                ("self_attn.q_proj".into(), c.hidden, 2 * c.hq * HEAD_DIM),
                ("self_attn.k_proj".into(), c.hidden, c.hkv * HEAD_DIM),
                ("self_attn.v_proj".into(), c.hidden, c.hkv * HEAD_DIM),
                ("self_attn.o_proj".into(), c.hq * HEAD_DIM, c.hidden),
            ]
        } else {
            vec![
                (
                    "linear_attn.in_proj_qkv".into(),
                    c.hidden,
                    (2 * hk + hv) * GDN_DIM,
                ),
                ("linear_attn.in_proj_z".into(), c.hidden, hv * GDN_DIM),
                ("linear_attn.in_proj_b".into(), c.hidden, hv),
                ("linear_attn.in_proj_a".into(), c.hidden, hv),
                ("linear_attn.out_proj".into(), hv * GDN_DIM, c.hidden),
            ]
        };
        mods.push(("mlp.gate_proj".into(), c.hidden, c.inter));
        mods.push(("mlp.up_proj".into(), c.hidden, c.inter));
        mods.push(("mlp.down_proj".into(), c.inter, c.hidden));
        for (m, cols, rows) in mods {
            let k = format!("base_model.model.model.language_model.layers.{l}.{m}");
            let a: Vec<u8> = (0..r * cols)
                .flat_map(|_| (normal() / (cols as f32).sqrt()).to_le_bytes())
                .collect();
            let b: Vec<u8> = (0..rows * r)
                .flat_map(|_| (normal() * b_std).to_le_bytes())
                .collect();
            tensors.push((format!("{k}.lora_A.weight"), vec![r, cols], a));
            tensors.push((format!("{k}.lora_B.weight"), vec![rows, r], b));
        }
    }
    std::fs::create_dir_all(out).unwrap();
    let views: Vec<(String, safetensors::tensor::TensorView)> = tensors
        .iter()
        .map(|(n, s, d)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), d)
                    .unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &out.join("adapter_model.safetensors")).unwrap();
    std::fs::write(
        out.join("adapter_config.json"),
        serde_json::json!({ "r": r, "lora_alpha": r, "peft_type": "LORA", "bias": "none" })
            .to_string(),
    )
    .unwrap();
}

fn scorings(pairs: &[Pair], adapters: &[u32]) -> Vec<Scoring> {
    pairs
        .iter()
        .zip(adapters.iter().cycle())
        .map(|(p, &adapter)| Scoring {
            prompt: p.prompt.clone(),
            forced: p.forced.clone(),
            images: vec![],
            rope: None,
            adapter,
        })
        .collect()
}

fn as_reference(scored: Vec<Vec<pocket_engine::sched::Scored>>) -> Vec<Vec<Reference>> {
    scored
        .into_iter()
        .map(|seq| {
            seq.into_iter()
                .map(|s| Reference {
                    argmax: s.argmax,
                    top: s.top,
                })
                .collect()
        })
        .collect()
}

/// An adapter whose B is zero leaves every output as the base model's.
#[test]
fn lora_zero_adapter_matches_base() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = std::env::temp_dir().join(format!("pocket-lora-zero-{}", std::process::id()));
    synthetic_adapter(&dir, &tmp, 16, 0.0, 1);
    let pairs: Vec<Pair> = fixture(&format!("{}.pairs.json", size(&dir)));
    let mut e = Engine::new(EngineConfig {
        prefix_cache: Some(0),
        max_loras: 1,
        ..config(dir)
    })
    .unwrap();
    let id = e.load_adapter("zero", &tmp).unwrap();
    let base = e.score(scorings(&pairs, &[0])).unwrap();
    let zero = e.score(scorings(&pairs, &[id])).unwrap();
    for (b, z) in base.iter().zip(&zero) {
        for (x, y) in b.iter().zip(z) {
            assert_eq!(x.argmax, y.argmax);
            assert_eq!(x.top, y.top);
        }
    }
    let _ = std::fs::remove_dir_all(tmp);
}

/// Requests for different adapters and the base model in one batch, with
/// more adapters than GPU slots, score as each does alone; and the adapters
/// change the outputs.
#[test]
fn lora_mixed_batches_match_alone() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = std::env::temp_dir().join(format!("pocket-lora-mix-{}", std::process::id()));
    for k in 0..3 {
        synthetic_adapter(&dir, &tmp.join(format!("a{k}")), 8, 0.02, 10 + k);
    }
    let pairs: Vec<Pair> = fixture(&format!("{}.pairs.json", size(&dir)));
    let mut e = Engine::new(EngineConfig {
        prefix_cache: Some(0),
        max_loras: 2,
        max_lora_rank: 8,
        ..config(dir.clone())
    })
    .unwrap();
    let ids: Vec<u32> = (0..3)
        .map(|k| {
            e.load_adapter(&format!("a{k}"), &tmp.join(format!("a{k}")))
                .unwrap()
        })
        .collect();
    // Each prompt with each choice, alone, then all of them in one batch.
    let choices = [ids[0], ids[1], 0, ids[2]];
    let mut alone = Vec::new();
    let mut mixed = Vec::new();
    for (i, p) in pairs.iter().enumerate() {
        let a = choices[i % choices.len()];
        alone.extend(as_reference(
            e.score(scorings(std::slice::from_ref(p), &[a])).unwrap(),
        ));
        mixed.push(a);
    }
    let together = e.score(scorings(&pairs, &mixed)).unwrap();
    if fp8() || fp4() || kv_fp8() || (quantized(&dir) && int8()) {
        close_enough(&alone, &together, 90, 0.03);
    } else {
        assert!(agree(&alone, &together) < 0.01);
    }
    let base = as_reference(e.score(scorings(&pairs, &[0])).unwrap());
    let adapted = e.score(scorings(&pairs, &[ids[0]])).unwrap();
    let moved = base
        .iter()
        .zip(&adapted)
        .flat_map(|(b, a)| b.iter().zip(a))
        .filter(|(b, a)| b.argmax != a.argmax || (b.top[0].1 - a.top[0].1).abs() > 0.05)
        .count();
    assert!(moved > 10, "the adapter moved only {moved} positions");
    let _ = std::fs::remove_dir_all(tmp);
}

/// With POCKET_ENGINE_LORA naming a PEFT adapter of the model under test and
/// fixtures `<size>.lora.{pairs,ref}.json` from `scripts/accuracy.py
/// reference ... --adapter`, outputs match Transformers with the adapter
/// merged into the weights.
#[test]
fn lora_matches_merged_reference() {
    let Some(dir) = model() else { return };
    let Some(adapter) = std::env::var_os("POCKET_ENGINE_LORA")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("POCKET_ENGINE_LORA is not set: skipping");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let pairs: Vec<Pair> = fixture(&format!("{}.lora.pairs.json", size(&dir)));
    let reference: Vec<Vec<Reference>> = fixture(&format!("{}.lora.ref.json", size(&dir)));
    let mut e = Engine::new(EngineConfig {
        max_loras: 1,
        max_lora_rank: 64,
        ..config(dir.clone())
    })
    .unwrap();
    let id = e.load_adapter("under-test", &adapter).unwrap();
    let got = e.score(scorings(&pairs, &[id])).unwrap();
    if quantized(&dir) || fp4() {
        close_enough(&reference, &got, 85, 0.1);
    } else if fp8() || kv_fp8() {
        close_enough(&reference, &got, 93, 0.03);
    } else {
        assert!(agree(&reference, &got) < 0.01);
    }
}

/// vLLM's adapter API: load at runtime, list, route by `model`, reject
/// unknown names, refuse nothing in use, unload.
#[test]
fn serves_adapters() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = std::env::temp_dir().join(format!("pocket-lora-http-{}", std::process::id()));
    // Strong enough to change even a greeting's first tokens.
    synthetic_adapter(&dir, &tmp, 8, 0.2, 3);
    let (tx, info, health, adapters) = engine::spawn(EngineConfig {
        max_loras: 1,
        ..config(dir.clone())
    })
    .unwrap();
    let tok = ChatTokenizer::load(&dir).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let bind = "127.0.0.1:18198";
        tokio::spawn(server::serve(
            tok,
            None,
            tx,
            info,
            health,
            Some(adapters),
            "base".into(),
            bind,
        ));
        let client = reqwest::Client::new();
        let url = format!("http://{bind}");
        for _ in 0..100 {
            if client.get(format!("{url}/health")).send().await.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let post = |path: &str, body: serde_json::Value| {
            client.post(format!("{url}{path}")).json(&body).send()
        };
        let r = post(
            "/v1/load_lora_adapter",
            serde_json::json!({ "lora_name": "syn", "lora_path": tmp }),
        )
        .await
        .unwrap();
        assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
        let models: serde_json::Value = client
            .get(format!("{url}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let ids: Vec<&str> = models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["base", "syn"]);
        assert_eq!(models["data"][1]["parent"], "base");
        let chat = |model: &str| {
            post(
                "/v1/chat/completions",
                serde_json::json!({ "model": model, "max_tokens": 8, "temperature": 0,
                    "messages": [{ "role": "user", "content": "Hello!" }] }),
            )
        };
        let a: serde_json::Value = chat("syn").await.unwrap().json().await.unwrap();
        assert_eq!(a["model"], "syn");
        let b: serde_json::Value = chat("base").await.unwrap().json().await.unwrap();
        assert_eq!(b["model"], "base");
        assert_ne!(
            a["choices"][0]["message"]["content"],
            b["choices"][0]["message"]["content"]
        );
        assert_eq!(chat("nope").await.unwrap().status(), 404);
        let r = post(
            "/v1/unload_lora_adapter",
            serde_json::json!({ "lora_name": "syn" }),
        )
        .await
        .unwrap();
        assert_eq!(r.status(), 200);
        assert_eq!(chat("syn").await.unwrap().status(), 404);
    });
    let _ = std::fs::remove_dir_all(tmp);
}
