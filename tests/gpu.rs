//! End-to-end checks on a GPU. They run when POCKET_ENGINE_MODEL names a
//! Qwen3.5 0.8B, 2B or 4B directory and the crate is built with `--features cuda`;
//! otherwise they pass without doing anything.

use pocket_engine::engine::{self, Engine, EngineConfig};
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
        model,
        device: 0,
        max_seqs: 16,
        max_batch_tokens: 64,
        max_context: 1024,
        // KV is capped by max_seqs x max_context, so tests stay small and can
        // share the device.
        memory_fraction: 0.95,
        graphs: true,
        wait_for: None,
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
    match pocket_engine::model::Config::load(dir).unwrap().hidden {
        1024 => "qwen3.5-0.8b",
        2048 => "qwen3.5-2b",
        2560 => "qwen3.5-4b",
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
    let mut e = Engine::new(config(dir)).unwrap();
    let got = e
        .score(pairs.into_iter().map(|p| (p.prompt, p.forced)).collect())
        .unwrap();
    let (mut positions, mut diffs) = (0, Vec::new());
    for (r_seq, g_seq) in reference.iter().zip(&got) {
        assert_eq!(r_seq.len(), g_seq.len());
        for (r, g) in r_seq.iter().zip(g_seq) {
            positions += 1;
            let tied = r.top[0].1 - r.top[1].1 <= 0.125;
            assert!(
                tied || r.argmax == g.argmax,
                "pick {} vs reference {}",
                g.argmax,
                r.argmax
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
    assert!(positions > 200);
    assert!(
        diffs[diffs.len() / 2] < 0.01,
        "median {}",
        diffs[diffs.len() / 2]
    );
    assert!(diffs[diffs.len() * 95 / 100] < 0.1);
}

fn generate(e: &mut Engine, prompts: &[Vec<u32>], n: usize) -> Vec<Vec<u32>> {
    let mut rxs = Vec::new();
    for p in prompts {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        e.sched.submit(Request {
            prompt: p.clone(),
            max_new: n,
            sampling: Sampling::default(),
            tx,
        });
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

/// Greedy output of a request does not depend on what else shares its steps.
#[test]
fn batching_does_not_change_greedy_output() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let pairs: Vec<Pair> = fixture(&format!("{}.pairs.json", size(&dir)));
    let prompts: Vec<Vec<u32>> = pairs.into_iter().map(|p| p.prompt).collect();
    let mut e = Engine::new(config(dir)).unwrap();
    let alone = generate(&mut e, &prompts[..1], 24);
    let together = generate(&mut e, &prompts, 24);
    assert_eq!(alone[0], together[0]);
    assert!(together.iter().all(|t| !t.is_empty()));
}

/// The OpenAI-compatible server streams text and reports usage.
#[test]
fn serves_chat_completions() {
    let Some(dir) = model() else { return };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (tx, info, health) = engine::spawn(config(dir.clone())).unwrap();
    let tok = ChatTokenizer::load(&dir).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let bind = "127.0.0.1:18197";
        tokio::spawn(server::serve(tok, tx, info, health, "test".into(), bind));
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
