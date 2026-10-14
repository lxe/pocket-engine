//! OpenAI-compatible chat completions over the engine thread.

use crate::engine::{Adapters, Health, Info, Msg};
use crate::sched::{Event, Finish, Request, Sampling, TokenLogprobs};
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

/// Qwen3.5's pre-tokenizer, normalizer and decoder, as in its tokenizer.json;
/// GGUF files name this setup "qwen35".
const QWEN_TOKENIZER_PARTS: &str = r#"{
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
}"#;

impl ChatTokenizer {
    /// The tokenizer as tokenizer.json.
    pub fn json(&self) -> String {
        self.tok.to_string(false).unwrap_or_default()
    }

    /// From a model directory (tokenizer.json, chat_template.jinja) or a GGUF
    /// file, whose metadata carries the vocabulary, merges and chat template.
    pub fn load(path: &Path) -> Result<Self> {
        match crate::source::gguf_path(path) {
            Some(g) => Self::from_gguf(&crate::gguf::Gguf::open(&g)?),
            None => Self::from_dir(path),
        }
    }

    pub fn from_gguf(g: &crate::gguf::Gguf) -> Result<Self> {
        let arr = |k: &str| {
            g.get(k)
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Config(format!("GGUF lacks {k}")))
        };
        let tokens = arr("tokenizer.ggml.tokens")?;
        let types = arr("tokenizer.ggml.token_type")?;
        let merges = arr("tokenizer.ggml.merges")?;
        let model = g
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if model != "gpt2" {
            return Err(Error::Config(format!(
                "unsupported GGUF tokenizer {model:?}"
            )));
        }
        let mut vocab = serde_json::Map::new();
        let mut added = Vec::new();
        for (id, (t, ty)) in tokens.iter().zip(types).enumerate() {
            let t = t.as_str().unwrap_or_default();
            match ty.as_u64() {
                // Normal tokens form the BPE vocabulary; control (3) and
                // user-defined (4) tokens are added tokens; 5 is padding.
                Some(1) | None => {
                    vocab.insert(t.to_string(), json!(id));
                }
                Some(k @ (3 | 4)) => added
                    .push(json!({ "id": id, "content": t, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": k == 3 })),
                _ => {}
            }
        }
        let merges: Vec<&str> = merges.iter().filter_map(|m| m.as_str()).collect();
        let mut doc: Value = serde_json::from_str(QWEN_TOKENIZER_PARTS)?;
        doc["version"] = json!("1.0");
        doc["truncation"] = Value::Null;
        doc["padding"] = Value::Null;
        doc["added_tokens"] = Value::Array(added);
        doc["model"] = json!({ "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges });
        let tok: tokenizers::Tokenizer = doc
            .to_string()
            .parse()
            .map_err(|e: tokenizers::Error| Error::Config(e.to_string()))?;
        let template = g
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Config("GGUF lacks tokenizer.chat_template".into()))?
            .to_string();
        Ok(Self { tok, template })
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Ok(Self {
            tok: tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| Error::Config(e.to_string()))?,
            template: std::fs::read_to_string(dir.join("chat_template.jinja"))?,
        })
    }

    /// Renders messages with the model's chat template (thinking off) and
    /// tokenizes the result. Returns the token ids, with one image
    /// placeholder per image, and the images' URLs in order.
    pub fn chat(&self, messages: &[Value]) -> Result<(Vec<u32>, Vec<String>)> {
        self.chat_with(messages, false)
    }

    /// `chat`, with the template's thinking switched on or off.
    pub fn chat_with(&self, messages: &[Value], thinking: bool) -> Result<(Vec<u32>, Vec<String>)> {
        let mut norm = Vec::with_capacity(messages.len());
        let mut images = Vec::new();
        for m in messages {
            let role = m["role"].as_str().unwrap_or_default();
            if !matches!(role, "system" | "user" | "assistant") {
                return Err(Error::Request(format!("unsupported role {role:?}")));
            }
            let content = match &m["content"] {
                Value::String(s) => json!(s),
                Value::Array(parts) => {
                    let mut out = Vec::with_capacity(parts.len());
                    for p in parts {
                        match p["type"].as_str() {
                            Some("text") => out.push(json!({ "type": "text", "text": p["text"].as_str().unwrap_or_default() })),
                            Some("image_url") => {
                                let url = p["image_url"]["url"].as_str().or(p["image_url"].as_str());
                                images.push(url.ok_or_else(|| Error::Request("image_url needs a url".into()))?.to_string());
                                out.push(json!({ "type": "image" }));
                            }
                            other => return Err(Error::Request(format!("unsupported content part {other:?}"))),
                        }
                    }
                    Value::Array(out)
                }
                Value::Null => json!(""),
                _ => {
                    return Err(Error::Request(
                        "message content must be text or parts".into(),
                    ));
                }
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
                messages => norm, add_generation_prompt => true, enable_thinking => thinking,
                tools => false, add_vision_id => false
            })
            .map_err(|e| Error::Request(e.to_string()))?;
        Ok((self.encode(&text)?, images))
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

/// The bytes of a `data:` URL (base64 only), or None for other URLs.
pub fn data_url(url: &str) -> Option<Result<Vec<u8>>> {
    use base64::Engine as _;
    let rest = url.strip_prefix("data:")?;
    Some(match rest.split_once(";base64,") {
        Some((_, data)) => base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .map_err(|e| Error::Request(format!("bad base64 image: {e}"))),
        None => Err(Error::Request("image data URLs must be base64".into())),
    })
}

/// Preprocesses encoded images and completes a prompt that refers to them:
/// placeholders expanded per embedding, and rotary positions for every token.
pub fn with_images(
    ids: Vec<u32>,
    encoded: &[Vec<u8>],
    spec: &crate::image::Spec,
    placeholder: u32,
) -> Result<(
    Vec<u32>,
    Vec<crate::image::Image>,
    Option<crate::image::Rope>,
)> {
    if encoded.is_empty() {
        return Ok((ids, Vec::new(), None));
    }
    let images = encoded
        .iter()
        .map(|b| crate::image::prepare(b, spec))
        .collect::<Result<Vec<_>>>()?;
    let ids = crate::image::expand_placeholders(&ids, &images, placeholder)?;
    let rope = crate::image::rope_positions(&ids, &images, placeholder);
    Ok((ids, images, Some(rope)))
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
    /// Image preprocessing, when the engine accepts images.
    spec: Option<crate::image::Spec>,
    http: reqwest::Client,
    engine: Mutex<mpsc::Sender<Msg>>,
    model: String,
    /// Adapter names routable through `model`, when adapters are on.
    adapters: Option<Adapters>,
    /// Compiles structured-output grammars against the tokenizer.
    grammars: Arc<llguidance::ParserFactory>,
    info: Info,
    health: Arc<Mutex<Health>>,
    next: AtomicU64,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<Value>,
    #[serde(default)]
    logprobs: bool,
    #[serde(default)]
    top_logprobs: Option<usize>,
    /// OpenAI's structured output: {"type": "json_object"} or {"type":
    /// "json_schema", "json_schema": {"schema": ...}}.
    #[serde(default)]
    response_format: Option<Value>,
    /// vLLM's guided decoding.
    #[serde(default)]
    guided_json: Option<Value>,
    #[serde(default)]
    guided_regex: Option<String>,
    #[serde(default)]
    guided_choice: Option<Vec<String>>,
    #[serde(default)]
    guided_grammar: Option<String>,
    /// vLLM's template arguments; only `enable_thinking` (default false) is read.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Map<String, Value>>,
}

impl ChatRequest {
    /// The grammar the output must follow, if the request sets one.
    fn grammar(&self) -> std::result::Result<Option<llguidance::api::TopLevelGrammar>, String> {
        use llguidance::api::TopLevelGrammar as G;
        let schema = |v: &Value| -> std::result::Result<Value, String> {
            match v {
                Value::String(s) => {
                    serde_json::from_str(s).map_err(|e| format!("guided_json: {e}"))
                }
                v => Ok(v.clone()),
            }
        };
        let mut found = Vec::new();
        if let Some(f) = &self.response_format {
            match f["type"].as_str() {
                Some("text") | None => {}
                Some("json_object") => found.push(G::from_json_schema(json!({ "type": "object" }))),
                Some("json_schema") => {
                    let s = f["json_schema"]
                        .get("schema")
                        .ok_or("response_format json_schema needs json_schema.schema")?;
                    found.push(G::from_json_schema(s.clone()));
                }
                Some(t) => return Err(format!("unsupported response_format type {t}")),
            }
        }
        if let Some(v) = &self.guided_json {
            found.push(G::from_json_schema(schema(v)?));
        }
        if let Some(r) = &self.guided_regex {
            found.push(G::from_regex(r));
        }
        if let Some(c) = &self.guided_choice {
            if c.is_empty() {
                return Err("guided_choice needs at least one choice".into());
            }
            let alts: Vec<String> = c.iter().map(|x| regex_escape(x)).collect();
            found.push(G::from_regex(&format!("({})", alts.join("|"))));
        }
        if let Some(g) = &self.guided_grammar {
            found.push(G::from_lark(g.clone()));
        }
        if found.len() > 1 {
            return Err("set one of response_format, guided_json, guided_regex, guided_choice, guided_grammar".into());
        }
        Ok(found.pop())
    }

    /// The top-token count for log-probabilities, if asked for.
    fn logprobs(&self) -> std::result::Result<Option<usize>, String> {
        match (self.logprobs, self.top_logprobs) {
            (false, Some(n)) if n > 0 => Err("top_logprobs needs logprobs: true".into()),
            (false, _) => Ok(None),
            (true, Some(n)) if n > crate::model::LOGPROBS_TOP => Err(format!(
                "top_logprobs takes at most {}",
                crate::model::LOGPROBS_TOP
            )),
            (true, n) => Ok(Some(n.unwrap_or(0))),
        }
    }

    /// Validates the sampling fields the way OpenAI and vLLM accept them.
    fn sampling(&self, vocab: usize, default_seed: u64) -> std::result::Result<Sampling, String> {
        let d = Sampling::default();
        let in_range = |name: &str, v: Option<f32>, lo: f32, hi: f32, def: f32| match v {
            None => Ok(def),
            Some(x) if x.is_finite() && (lo..=hi).contains(&x) => Ok(x),
            Some(x) => Err(format!("{name} must be in [{lo}, {hi}], got {x}")),
        };
        let top_p = in_range("top_p", self.top_p, 0.0, 1.0, d.top_p)?;
        if top_p == 0.0 {
            return Err("top_p must be greater than 0".into());
        }
        let top_k = match self.top_k {
            None | Some(-1) | Some(0) => 0,
            Some(k) if k > 0 => k.min(vocab as i64) as u32,
            Some(k) => return Err(format!("top_k must be -1, 0 or positive, got {k}")),
        };
        let repetition_penalty = in_range(
            "repetition_penalty",
            self.repetition_penalty,
            0.0,
            10.0,
            1.0,
        )?;
        if repetition_penalty == 0.0 {
            return Err("repetition_penalty must be greater than 0".into());
        }
        let mut logit_bias = Vec::new();
        for (k, v) in self.logit_bias.iter().flatten() {
            let token: u32 = k
                .parse()
                .map_err(|_| format!("logit_bias key {k:?} is not a token id"))?;
            let bias = v.as_f64().ok_or("logit_bias values must be numbers")? as f32;
            if token as usize >= vocab || !(-100.0..=100.0).contains(&bias) {
                return Err(format!("logit_bias {k}: {bias} is out of range"));
            }
            logit_bias.push((token, bias));
        }
        if logit_bias.len() > 1024 {
            return Err("logit_bias holds at most 1,024 tokens".into());
        }
        Ok(Sampling {
            temperature: in_range("temperature", self.temperature, 0.0, 100.0, 1.0)?,
            top_k,
            top_p,
            min_p: in_range("min_p", self.min_p, 0.0, 1.0, d.min_p)?,
            presence_penalty: in_range("presence_penalty", self.presence_penalty, -2.0, 2.0, 0.0)?,
            frequency_penalty: in_range(
                "frequency_penalty",
                self.frequency_penalty,
                -2.0,
                2.0,
                0.0,
            )?,
            repetition_penalty,
            logit_bias,
            seed: self.seed.unwrap_or(default_seed),
        })
    }

    fn stops(&self) -> std::result::Result<Vec<String>, String> {
        let stops: Vec<String> = match &self.stop {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(String::from)
                        .ok_or("stop entries must be strings")
                })
                .collect::<std::result::Result<_, _>>()?,
            Some(_) => return Err("stop must be a string or a list of strings".into()),
        };
        if stops.len() > 16 || stops.iter().any(String::is_empty) {
            return Err("stop takes up to 16 non-empty strings".into());
        }
        Ok(stops)
    }
}

/// Downloads an image, up to 32 MB.
async fn fetch(http: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let fail = |e: reqwest::Error| Error::Request(format!("cannot fetch image: {e}"));
    let resp = http
        .get(url)
        .send()
        .await
        .map_err(fail)?
        .error_for_status()
        .map_err(fail)?;
    if resp.content_length().is_some_and(|n| n > 32 << 20) {
        return Err(Error::Request("image is larger than 32 MB".into()));
    }
    let bytes = resp.bytes().await.map_err(fail)?;
    if bytes.len() > 32 << 20 {
        return Err(Error::Request("image is larger than 32 MB".into()));
    }
    Ok(bytes.to_vec())
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
    if req.n.is_some_and(|n| n != 1) {
        return err(StatusCode::BAD_REQUEST, "only n = 1 is supported");
    }
    // `model` names the base or an adapter. Without adapters any name means
    // the base, as before.
    let (served, adapter) = match (&req.model, &app.adapters) {
        (Some(m), Some(a)) if *m != app.model => match a.lock().unwrap().get(m) {
            Some(&id) => (m.clone(), id),
            None => {
                return err(
                    StatusCode::NOT_FOUND,
                    format!("The model `{m}` does not exist."),
                );
            }
        },
        _ => (app.model.clone(), 0),
    };
    let n = app.next.fetch_add(1, Ordering::Relaxed);
    let sampling = match req.sampling(app.info.cfg.vocab, n.wrapping_mul(0x2545_f491_4f6c_dd1d)) {
        Ok(s) => s,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let stops = match req.stops() {
        Ok(s) => s,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let logprobs = match req.logprobs() {
        Ok(l) => l,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let constraint = match req.grammar() {
        Ok(None) => None,
        Ok(Some(g)) => {
            let m = llguidance::Matcher::new(app.grammars.create_parser(g));
            if let Some(e) = m.get_error() {
                return err(
                    StatusCode::BAD_REQUEST,
                    format!("invalid structured output: {e}"),
                );
            }
            Some(Box::new(m))
        }
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let thinking = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let (ids, urls) = match app.tok.chat_with(&req.messages, thinking) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let mut encoded = Vec::with_capacity(urls.len());
    if !urls.is_empty() && app.spec.is_none() {
        return err(
            StatusCode::BAD_REQUEST,
            "this model is served without images",
        );
    }
    for url in &urls {
        let bytes = match data_url(url) {
            Some(b) => b,
            None if url.starts_with("http://") || url.starts_with("https://") => {
                fetch(&app.http, url).await
            }
            None => Err(Error::Request(
                "images must be data: or http(s) URLs".into(),
            )),
        };
        match bytes {
            Ok(b) => encoded.push(b),
            Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
        }
    }
    let spec = app.spec.clone().unwrap_or_default();
    let placeholder = app.info.cfg.image_token;
    let built =
        tokio::task::spawn_blocking(move || with_images(ids, &encoded, &spec, placeholder)).await;
    let (prompt, images, rope) = match built {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, e.to_string()),
        Err(_) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "image preprocessing failed",
            );
        }
    };
    let id = format!("chatcmpl-{n}");
    let max_new = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(app.info.max_context);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let out = Output::new(served, stops, prompt.len());
    let request = Request {
        prompt,
        max_new,
        sampling,
        tx,
        images,
        rope,
        adapter,
        logprobs,
        constraint,
    };
    if app
        .engine
        .lock()
        .unwrap()
        .send(Msg::Request(request))
        .is_err()
    {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    if req.stream {
        let usage = req
            .stream_options
            .as_ref()
            .and_then(|o| o["include_usage"].as_bool())
            .unwrap_or(false);
        stream_response(app, rx, out, id, usage).into_response()
    } else {
        whole_response(app, rx, out, id).await
    }
}

/// Text produced so far, released only once it cannot be the start of a stop
/// string. Finding a stop string ends the response; dropping the event
/// receiver then cancels the request in the engine.
struct Output {
    /// The model name to answer with: the base's or the adapter's.
    model: String,
    /// Log-probabilities of the next token, as they arrive before it.
    pending: Option<TokenLogprobs>,
    detok: Detok,
    text: String,
    sent: usize,
    stops: Vec<String>,
    tokens: usize,
    prompt_tokens: usize,
}

impl Output {
    fn new(model: String, stops: Vec<String>, prompt_tokens: usize) -> Self {
        Self {
            model,
            pending: None,
            prompt_tokens,
            detok: Detok {
                ids: Vec::new(),
                prefix: 0,
                read: 0,
            },
            text: String::new(),
            sent: 0,
            stops,
            tokens: 0,
        }
    }

    /// Adds a token; returns the text now safe to send and whether a stop
    /// string ended the output.
    fn push(&mut self, tok: &ChatTokenizer, id: u32) -> (String, bool) {
        self.tokens += 1;
        if let Some(d) = self.detok.push(tok, id) {
            self.text.push_str(&d);
        }
        self.release(false)
    }

    /// The remaining text when generation ends.
    fn end(&mut self, tok: &ChatTokenizer) -> (String, bool) {
        let tail = tok.decode(&self.detok.ids[self.detok.prefix..]);
        let seen = tok.decode(&self.detok.ids[self.detok.prefix..self.detok.read]);
        if let Some(rest) = tail.get(seen.len()..) {
            self.text.push_str(rest);
        }
        self.detok.prefix = self.detok.ids.len();
        self.detok.read = self.detok.ids.len();
        self.release(true)
    }

    fn release(&mut self, all: bool) -> (String, bool) {
        let longest = self.stops.iter().map(String::len).max().unwrap_or(0);
        let from = self.sent.saturating_sub(longest);
        let from = (0..=from)
            .rev()
            .find(|&i| self.text.is_char_boundary(i))
            .unwrap_or(0);
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.text[from..].find(s.as_str()).map(|i| from + i))
            .min();
        let (end, stopped) = match found {
            Some(at) => (at.max(self.sent), true),
            None if all => (self.text.len(), false),
            None => {
                // Hold back what could still grow into a stop string.
                let mut end = self
                    .text
                    .len()
                    .saturating_sub(longest.saturating_sub(1))
                    .max(self.sent);
                while !self.text.is_char_boundary(end) {
                    end -= 1;
                }
                (end.max(self.sent), false)
            }
        };
        let out = self.text[self.sent..end].to_string();
        self.sent = end;
        (out, stopped)
    }
}

/// A literal as a regular expression.
fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\.+*?()|[]{}^$#&-~/\"".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// One token of OpenAI's `logprobs.content`: its text, raw bytes,
/// log-probability and the most likely alternatives.
fn logprob_entry(app: &App, tok: u32, lp: &TokenLogprobs) -> Value {
    let trie = app.grammars.tok_env().tok_trie();
    let one = |id: u32, logprob: f32| {
        let bytes = trie.token(id).to_vec();
        json!({ "token": String::from_utf8_lossy(&bytes), "bytes": bytes,
            "logprob": if logprob.is_finite() { logprob } else { -9999.0 } })
    };
    let mut e = one(tok, lp.logprob);
    e["top_logprobs"] = Value::Array(lp.top.iter().map(|&(id, l)| one(id, l)).collect());
    e
}

async fn whole_response(
    app: Arc<App>,
    mut rx: UnboundedReceiver<Event>,
    mut out: Output,
    id: String,
) -> Response {
    let mut content = String::new();
    let mut lp_content: Vec<Value> = Vec::new();
    let (finish, prompt_tokens, completion_tokens) = loop {
        match rx.recv().await {
            Some(Event::Logprobs(lp)) => out.pending = Some(lp),
            Some(Event::Token(t)) => {
                if let Some(lp) = out.pending.take() {
                    lp_content.push(logprob_entry(&app, t, &lp));
                }
                let (text, stopped) = out.push(&app.tok, t);
                content.push_str(&text);
                if stopped {
                    break ("stop", out.prompt_tokens, out.tokens);
                }
            }
            Some(Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            }) => {
                let (text, stopped) = out.end(&app.tok);
                content.push_str(&text);
                let finish = if stopped { "stop" } else { finish_name(finish) };
                break (finish, prompt_tokens, completion_tokens);
            }
            Some(Event::Error(e)) => return err(StatusCode::BAD_REQUEST, e),
            None => {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "engine dropped the request",
                );
            }
        }
    };
    drop(rx);
    Json(json!({
        "id": id, "object": "chat.completion", "created": now(), "model": out.model,
        "choices": [{ "index": 0, "finish_reason": finish,
            "message": { "role": "assistant", "content": content },
            "logprobs": if lp_content.is_empty() { Value::Null } else { json!({ "content": lp_content }) } }],
        "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens }
    }))
    .into_response()
}

fn stream_response(
    app: Arc<App>,
    rx: UnboundedReceiver<Event>,
    out: Output,
    id: String,
    usage: bool,
) -> axum::response::Sse<
    impl futures_util::Stream<Item = std::result::Result<Sse, std::convert::Infallible>>,
> {
    let created = now();
    let chunk_lp = move |model: &str, id: &str, delta: Value, finish: Option<&str>, lp: Value| {
        Sse::default().data(
            json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{ "index": 0, "delta": delta, "logprobs": lp, "finish_reason": finish }] })
            .to_string(),
        )
    };
    let chunk = move |model: &str, id: &str, delta: Value, finish: Option<&str>| {
        chunk_lp(model, id, delta, finish, Value::Null)
    };
    let closing = move |model: &str, id: &str, finish: &str, prompt: usize, completion: usize| {
        let mut events = vec![chunk(model, id, json!({}), Some(finish))];
        if usage {
            events.push(
                Sse::default().data(
                    json!({ "id": id, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [],
                    "usage": { "prompt_tokens": prompt, "completion_tokens": completion,
                        "total_tokens": prompt + completion } })
                    .to_string(),
                ),
            );
        }
        events.push(Sse::default().data("[DONE]"));
        events
    };
    enum Phase {
        Start,
        Run,
        End,
    }
    let s = stream::unfold(
        (Some(rx), out, Phase::Start, app, id),
        move |(mut rx, mut out, phase, app, id)| async move {
            let mut events: Vec<Sse> = Vec::new();
            let next = match phase {
                Phase::End => return None,
                Phase::Start => {
                    events.push(chunk(
                        &out.model,
                        &id,
                        json!({ "role": "assistant", "content": "" }),
                        None,
                    ));
                    Phase::Run
                }
                Phase::Run => match rx.as_mut()?.recv().await {
                    Some(Event::Logprobs(lp)) => {
                        out.pending = Some(lp);
                        Phase::Run
                    }
                    Some(Event::Token(t)) => {
                        let lp = out.pending.take().map(|lp| logprob_entry(&app, t, &lp));
                        let (text, stopped) = out.push(&app.tok, t);
                        if let Some(lp) = lp {
                            // Every token with log-probabilities gets a chunk, its
                            // text held back if it may start a stop string.
                            events.push(chunk_lp(
                                &out.model,
                                &id,
                                json!({ "content": text }),
                                None,
                                json!({ "content": [lp] }),
                            ));
                        } else if !text.is_empty() {
                            events.push(chunk(&out.model, &id, json!({ "content": text }), None));
                        }
                        if stopped {
                            rx = None; // cancels the request in the engine
                            events.extend(closing(
                                &out.model,
                                &id,
                                "stop",
                                out.prompt_tokens,
                                out.tokens,
                            ));
                            Phase::End
                        } else {
                            Phase::Run
                        }
                    }
                    Some(Event::Done {
                        finish,
                        prompt_tokens,
                        completion_tokens,
                    }) => {
                        let (text, stopped) = out.end(&app.tok);
                        if !text.is_empty() {
                            events.push(chunk(&out.model, &id, json!({ "content": text }), None));
                        }
                        let finish = if stopped { "stop" } else { finish_name(finish) };
                        events.extend(closing(
                            &out.model,
                            &id,
                            finish,
                            prompt_tokens,
                            completion_tokens,
                        ));
                        Phase::End
                    }
                    Some(Event::Error(e)) => {
                        events.push(
                            Sse::default().data(json!({ "error": { "message": e } }).to_string()),
                        );
                        events.push(Sse::default().data("[DONE]"));
                        Phase::End
                    }
                    None => Phase::End,
                },
            };
            Some((
                stream::iter(events.into_iter().map(Ok)),
                (rx, out, next, app, id),
            ))
        },
    );
    axum::response::Sse::new(futures_util::StreamExt::flatten(s)).keep_alive(KeepAlive::default())
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let mut data = vec![
        json!({ "id": app.model, "object": "model", "owned_by": "pocket-engine",
        "root": app.model, "parent": null, "max_model_len": app.info.max_context }),
    ];
    if let Some(a) = &app.adapters {
        let mut names: Vec<String> = a.lock().unwrap().keys().cloned().collect();
        names.sort();
        for n in names {
            data.push(
                json!({ "id": n, "object": "model", "owned_by": "pocket-engine",
                "root": n, "parent": app.model, "max_model_len": app.info.max_context }),
            );
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

/// vLLM's runtime adapter endpoints.
#[derive(Deserialize)]
struct AdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_path: Option<String>,
}

async fn adapter_call(
    app: &App,
    msg: impl FnOnce(tokio::sync::oneshot::Sender<std::result::Result<(), String>>) -> Msg,
) -> Response {
    if app.adapters.is_none() {
        return err(
            StatusCode::BAD_REQUEST,
            "adapters are off; start with --max-loras or --lora",
        );
    }
    let (reply, rx) = tokio::sync::oneshot::channel();
    if app.engine.lock().unwrap().send(msg(reply)).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(Ok(())) => StatusCode::OK.into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, e),
        Err(_) => err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

async fn load_adapter(State(app): State<Arc<App>>, Json(req): Json<AdapterRequest>) -> Response {
    let Some(path) = req.lora_path else {
        return err(StatusCode::BAD_REQUEST, "lora_path is required");
    };
    let name = req.lora_name;
    let r = adapter_call(&app, |reply| Msg::LoadAdapter {
        name: name.clone(),
        path: path.into(),
        reply,
    })
    .await;
    if r.status() == StatusCode::OK {
        return (
            StatusCode::OK,
            format!("Success: LoRA adapter '{name}' added successfully."),
        )
            .into_response();
    }
    r
}

async fn unload_adapter(State(app): State<Arc<App>>, Json(req): Json<AdapterRequest>) -> Response {
    let name = req.lora_name;
    let r = adapter_call(&app, |reply| Msg::UnloadAdapter {
        name: name.clone(),
        reply,
    })
    .await;
    if r.status() == StatusCode::OK {
        return (
            StatusCode::OK,
            format!("Success: LoRA adapter '{name}' removed successfully."),
        )
            .into_response();
    }
    r
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

/// The structured-output compiler for this tokenizer.
/// The structured-output parser factory over the model's vocabulary (about a
/// second for Qwen's 248k tokens; `serve` callers build it beside the engine).
pub fn grammars(
    tok: &ChatTokenizer,
    cfg: &crate::model::Config,
) -> Result<llguidance::ParserFactory> {
    fn e(e: impl std::fmt::Display) -> Error {
        Error::Config(format!("structured output setup: {e}"))
    }
    let mut bt =
        toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(tok.json().as_bytes()).map_err(e)?;
    bt.set_eos_tokens(&cfg.eos);
    let env = bt.into_tok_env(Some(cfg.vocab)).map_err(e)?;
    let mut f = llguidance::ParserFactory::new_simple(&env).map_err(e)?;
    f.quiet();
    Ok(f)
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    tok: ChatTokenizer,
    grammars: llguidance::ParserFactory,
    spec: Option<crate::image::Spec>,
    engine: mpsc::Sender<Msg>,
    info: Info,
    health: Arc<Mutex<Health>>,
    adapters: Option<Adapters>,
    model: String,
    bind: &str,
) -> Result<()> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Config(e.to_string()))?;
    let grammars = Arc::new(grammars);
    let app = Arc::new(App {
        tok,
        spec,
        http,
        grammars,
        engine: Mutex::new(engine),
        model,
        adapters,
        info,
        health,
        next: AtomicU64::new(0),
    });
    let router = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/v1/load_lora_adapter", post(load_adapter))
        .route("/v1/unload_lora_adapter", post(unload_adapter))
        .route("/health", get(health_report))
        .with_state(app);
    // Tokens are small writes: without TCP_NODELAY, Nagle's algorithm holds one
    // back until the previous is acknowledged, which a delayed ACK can stall.
    let listener =
        axum::serve::ListenerExt::tap_io(tokio::net::TcpListener::bind(bind).await?, |tcp| {
            let _ = tcp.set_nodelay(true);
        });
    eprintln!("listening on {bind}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(out: &mut Output, parts: &[&str]) -> (String, bool) {
        let mut sent = String::new();
        for p in parts {
            out.text.push_str(p);
            let (t, stopped) = out.release(false);
            sent.push_str(&t);
            if stopped {
                return (sent, true);
            }
        }
        let (t, stopped) = out.release(true);
        sent.push_str(&t);
        (sent, stopped)
    }

    #[test]
    fn stop_strings_end_the_text_even_across_tokens() {
        let mut out = Output::new(String::new(), vec!["</a>".into(), "STOP".into()], 0);
        assert_eq!(
            feed(&mut out, &["one <", "/a", "> two"]),
            ("one ".into(), true)
        );
        let mut out = Output::new(String::new(), vec!["STOP".into()], 0);
        assert_eq!(
            feed(&mut out, &["no stop ", "here"]),
            ("no stop here".into(), false)
        );
    }

    #[test]
    fn held_back_text_is_released_whole_and_on_char_boundaries() {
        let mut out = Output::new(String::new(), vec!["xyz".into()], 0);
        out.text.push_str("caf\u{e9}s");
        let (first, _) = out.release(false);
        assert!(first.len() <= out.text.len() - 2 && out.text.starts_with(&first));
        let (rest, stopped) = out.release(true);
        assert_eq!(first + &rest, "caf\u{e9}s");
        assert!(!stopped);
    }

    fn request(v: Value) -> ChatRequest {
        serde_json::from_value(
            json!({ "messages": [] })
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(v.as_object().unwrap().clone())
                .collect::<serde_json::Map<_, _>>()
                .into(),
        )
        .unwrap()
    }

    #[test]
    fn sampling_fields_are_validated() {
        let s = request(json!({ "top_k": -1, "top_p": 0.9, "logit_bias": { "5": -100 } }))
            .sampling(10, 7)
            .unwrap();
        assert_eq!((s.top_k, s.top_p, s.seed, s.temperature), (0, 0.9, 7, 1.0));
        assert_eq!(s.logit_bias, vec![(5, -100.0)]);
        for bad in [
            json!({ "top_p": 0 }),
            json!({ "top_k": -2 }),
            json!({ "presence_penalty": 3 }),
            json!({ "logit_bias": { "10": 1 } }),
            json!({ "repetition_penalty": 0 }),
        ] {
            assert!(request(bad.clone()).sampling(10, 0).is_err(), "{bad}");
        }
        assert!(request(json!({ "stop": ["a", ""] })).stops().is_err());
        assert_eq!(request(json!({ "stop": "a" })).stops().unwrap(), vec!["a"]);
    }
}
