//! OpenAI-compatible chat completions over the engine thread.

use crate::engine::{Health, Info};
use crate::sched::{Event, Finish, Request, Sampling};
use crate::{Error, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as Sse, KeepAlive};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedReceiver;

pub struct ChatTokenizer {
    tok: tokenizers::Tokenizer,
    template: String,
}

impl ChatTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            tok: tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| Error::Config(e.to_string()))?,
            template: std::fs::read_to_string(dir.join("chat_template.jinja"))?,
        })
    }

    /// Renders messages with the model's chat template (thinking off) and
    /// tokenizes the result.
    pub fn chat(&self, messages: &[Value]) -> Result<Vec<u32>> {
        let mut norm = Vec::with_capacity(messages.len());
        for m in messages {
            let role = m["role"].as_str().unwrap_or_default();
            if !matches!(role, "system" | "user" | "assistant") {
                return Err(Error::Request(format!("unsupported role {role:?}")));
            }
            let content = match &m["content"] {
                Value::String(s) => s.clone(),
                Value::Array(parts) => parts
                    .iter()
                    .map(|p| match p["type"].as_str() {
                        Some("text") => Ok(p["text"].as_str().unwrap_or_default().to_string()),
                        _ => Err(Error::Request("only text content is supported".into())),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join(""),
                Value::Null => String::new(),
                _ => return Err(Error::Request("message content must be text".into())),
            };
            norm.push(json!({ "role": role, "content": content }));
        }
        if norm.is_empty() {
            return Err(Error::Request("messages are empty".into()));
        }
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("chat", &self.template)
            .map_err(|e| Error::Config(e.to_string()))?;
        let text = env
            .get_template("chat")
            .map_err(|e| Error::Config(e.to_string()))?
            .render(minijinja::context! {
                messages => norm, add_generation_prompt => true, enable_thinking => false,
                tools => false, add_vision_id => false
            })
            .map_err(|e| Error::Request(e.to_string()))?;
        self.encode(&text)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Config(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, true).unwrap_or_default()
    }
}

/// Incremental detokenization: emits text once it no longer ends inside a
/// multi-byte character, decoding only a short window of recent tokens.
struct Detok {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    fn push(&mut self, tok: &ChatTokenizer, id: u32) -> Option<String> {
        self.ids.push(id);
        let before = tok.decode(&self.ids[self.prefix..self.read]);
        let after = tok.decode(&self.ids[self.prefix..]);
        if after.len() > before.len() && !after.ends_with('\u{FFFD}') {
            let delta = after.get(before.len()..).unwrap_or_default().to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            Some(delta)
        } else {
            None
        }
    }
}

struct App {
    tok: ChatTokenizer,
    engine: Mutex<mpsc::Sender<Request>>,
    model: String,
    info: Info,
    health: Arc<Mutex<Health>>,
    next: AtomicU64,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<Value>,
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": msg.into(), "type": "invalid_request_error" } })),
    )
        .into_response()
}

fn finish_name(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn chat(State(app): State<Arc<App>>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = match app.tok.chat(&req.messages) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let n = app.next.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{n}");
    let max_new = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(app.info.max_context);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let request = Request {
        prompt,
        max_new,
        sampling: Sampling {
            temperature: req.temperature.unwrap_or(1.0),
            seed: req.seed.unwrap_or(n.wrapping_mul(0x2545_f491_4f6c_dd1d)),
        },
        tx,
    };
    if app.engine.lock().unwrap().send(request).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    if req.stream {
        let usage = req
            .stream_options
            .as_ref()
            .and_then(|o| o["include_usage"].as_bool())
            .unwrap_or(false);
        stream_response(app, rx, id, usage).into_response()
    } else {
        whole_response(app, rx, id).await
    }
}

async fn whole_response(app: Arc<App>, mut rx: UnboundedReceiver<Event>, id: String) -> Response {
    let mut ids = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token(t) => ids.push(t),
            Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": now(), "model": app.model,
                    "choices": [{ "index": 0, "finish_reason": finish_name(finish),
                        "message": { "role": "assistant", "content": app.tok.decode(&ids) } }],
                    "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens }
                }))
                .into_response();
            }
            Event::Error(e) => return err(StatusCode::BAD_REQUEST, e),
        }
    }
    err(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine dropped the request",
    )
}

fn stream_response(
    app: Arc<App>,
    rx: UnboundedReceiver<Event>,
    id: String,
    usage: bool,
) -> axum::response::Sse<
    impl futures_util::Stream<Item = std::result::Result<Sse, std::convert::Infallible>>,
> {
    let created = now();
    let chunk = move |app: &App, id: &str, delta: Value, finish: Option<&str>| {
        Sse::default().data(
            json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": app.model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
            .to_string(),
        )
    };
    enum Phase {
        Start,
        Run,
        End,
    }
    let detok = Detok {
        ids: Vec::new(),
        prefix: 0,
        read: 0,
    };
    let s = stream::unfold(
        (rx, detok, Phase::Start, app, id),
        move |(mut rx, mut detok, phase, app, id)| async move {
            let mut out: Vec<Sse> = Vec::new();
            let next_phase = match phase {
                Phase::End => return None,
                Phase::Start => {
                    out.push(chunk(
                        &app,
                        &id,
                        json!({ "role": "assistant", "content": "" }),
                        None,
                    ));
                    Phase::Run
                }
                Phase::Run => match rx.recv().await {
                    Some(Event::Token(t)) => {
                        if let Some(text) = detok.push(&app.tok, t) {
                            out.push(chunk(&app, &id, json!({ "content": text }), None));
                        }
                        Phase::Run
                    }
                    Some(Event::Done {
                        finish,
                        prompt_tokens,
                        completion_tokens,
                    }) => {
                        let tail = app.tok.decode(&detok.ids[detok.prefix..]);
                        let seen = app.tok.decode(&detok.ids[detok.prefix..detok.read]);
                        if let Some(rest) = tail.get(seen.len()..).filter(|r| !r.is_empty()) {
                            out.push(chunk(&app, &id, json!({ "content": rest }), None));
                        }
                        out.push(chunk(&app, &id, json!({}), Some(finish_name(finish))));
                        if usage {
                            out.push(Sse::default().data(
                            json!({ "id": id, "object": "chat.completion.chunk", "created": created,
                                "model": app.model, "choices": [],
                                "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens } })
                            .to_string(),
                        ));
                        }
                        out.push(Sse::default().data("[DONE]"));
                        Phase::End
                    }
                    Some(Event::Error(e)) => {
                        out.push(
                            Sse::default().data(json!({ "error": { "message": e } }).to_string()),
                        );
                        out.push(Sse::default().data("[DONE]"));
                        Phase::End
                    }
                    None => Phase::End,
                },
            };
            Some((
                stream::iter(out.into_iter().map(Ok)),
                (rx, detok, next_phase, app, id),
            ))
        },
    );
    axum::response::Sse::new(futures_util::StreamExt::flatten(s)).keep_alive(KeepAlive::default())
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({ "object": "list", "data": [{ "id": app.model, "object": "model", "owned_by": "pocket-engine",
        "max_model_len": app.info.max_context }] }),
    )
}

async fn health_report(State(app): State<Arc<App>>) -> Response {
    let h = app.health.lock().unwrap().clone();
    let status = if h.failed.is_some() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    let body = json!({ "status": if h.failed.is_some() { "failed" } else { "ok" }, "error": h.failed,
        "device": app.info.device, "max_context": app.info.max_context, "load": h.load });
    (status, Json(body)).into_response()
}

pub async fn serve(
    tok: ChatTokenizer,
    engine: mpsc::Sender<Request>,
    info: Info,
    health: Arc<Mutex<Health>>,
    model: String,
    bind: &str,
) -> Result<()> {
    let app = Arc::new(App {
        tok,
        engine: Mutex::new(engine),
        model,
        info,
        health,
        next: AtomicU64::new(0),
    });
    let router = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/health", get(health_report))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("listening on {bind}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
