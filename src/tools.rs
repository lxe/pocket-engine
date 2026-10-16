//! OpenAI-style tool calling. The chat template teaches the model a call
//! format; this module reads calls back out of its output, as a whole or as
//! it streams, and builds the grammar that forces a call when `tool_choice`
//! asks for one.
//!
//! Two formats, picked by what the template teaches:
//!
//! | Format | Templates | A call |
//! |---|---|---|
//! | `Xml` | Qwen3.5, Qwen3.8 | `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>` |
//! | `Hermes` | Qwen3 and others | `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` |

use serde_json::{Map, Value, json};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const THINK_END: &str = "</think>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Xml,
    Hermes,
}

impl Format {
    /// The format a chat template teaches.
    pub fn of_template(template: &str) -> Self {
        if template.contains("<function=") {
            Format::Xml
        } else {
            Format::Hermes
        }
    }
}

/// JSON the way Python's `json.dumps(x, ensure_ascii=False)` writes it, as
/// chat templates (through transformers' `tojson`) and vLLM's parsers do:
/// `{"a": 1, "b": [2, 3]}`.
pub fn py_json(v: &impl serde::Serialize) -> String {
    struct Py;
    impl serde_json::ser::Formatter for Py {
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Py);
    if v.serialize(&mut ser).is_err() {
        return String::new();
    }
    String::from_utf8(out).unwrap_or_default()
}

/// A request's tools and what it asked of them.
#[derive(Clone, Debug)]
pub struct Tools {
    /// As the request sent them: `{"type": "function", "function": {...}}`.
    pub list: Vec<Value>,
    /// `tool_choice` "required" (None) or a named function: the output must
    /// be calls.
    pub forced: Option<Option<String>>,
    /// Whether one turn may make several calls.
    pub parallel: bool,
}

impl Tools {
    /// From a request's `tools`, `tool_choice` and `parallel_tool_calls`.
    /// None when there are no tools or `tool_choice` is "none".
    pub fn from_request(
        tools: Option<&[Value]>,
        choice: Option<&Value>,
        parallel: Option<bool>,
    ) -> Result<Option<Self>, String> {
        let list = tools.unwrap_or_default();
        let mut names = std::collections::HashSet::new();
        for t in list {
            let name = t["function"]["name"].as_str();
            if t["type"] != "function" || name.is_none_or(str::is_empty) {
                return Err(
                    "tools entries must be {\"type\": \"function\", \"function\": {\"name\": ...}}"
                        .into(),
                );
            }
            if !names.insert(name.unwrap_or_default()) {
                return Err(format!("tool {} appears twice", name.unwrap_or_default()));
            }
        }
        let forced = match choice {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => match s.as_str() {
                "auto" => None,
                "none" => return Ok(None),
                "required" => Some(None),
                _ => return Err(format!("unsupported tool_choice {s:?}")),
            },
            Some(v) => match v["function"]["name"].as_str() {
                Some(n) if v["type"] == "function" => {
                    if !names.contains(n) {
                        return Err(format!("tool_choice names {n:?}, which is not in tools"));
                    }
                    Some(Some(n.to_string()))
                }
                _ => {
                    return Err(
                        "tool_choice must be auto, none, required or {\"type\": \"function\", \"function\": {\"name\": ...}}"
                            .into(),
                    );
                }
            },
        };
        if list.is_empty() {
            return match forced {
                Some(_) => Err("tool_choice needs tools".into()),
                None => Ok(None),
            };
        }
        Ok(Some(Self {
            list: list.to_vec(),
            forced,
            parallel: parallel.unwrap_or(true),
        }))
    }

    fn function(&self, name: &str) -> Option<&Value> {
        self.list
            .iter()
            .map(|t| &t["function"])
            .find(|f| f["name"] == name)
    }

    /// The Lark grammar of a forced turn: one call, or with `parallel` one
    /// or more, of the named function or any tool. With `thinking` the turn
    /// first finishes its reasoning. `token` gives the id of a tag that is a
    /// single token of the vocabulary (Qwen's `<tool_call>`, `</think>`),
    /// which the grammar must name as a token rather than spell.
    pub fn grammar(
        &self,
        format: Format,
        thinking: bool,
        token: &dyn Fn(&str) -> Option<u32>,
    ) -> Option<String> {
        let only = self.forced.as_ref()?;
        let lit = |s: &str| serde_json::to_string(s).unwrap_or_default();
        let tag = |s: &str| token(s).map_or_else(|| lit(s), |id| format!("<[{id}]>"));
        let (open, close) = (tag(OPEN), tag(CLOSE));
        let mut g = String::new();
        let calls = if self.parallel && only.is_none() {
            "call (\"\\n\" call)*"
        } else {
            "call"
        };
        if thinking {
            // A tag token cannot be part of the reasoning's text, which ends
            // where the tag comes; a spelled tag ends it lazily.
            g += &match token(THINK_END) {
                Some(id) => format!(
                    "start: think <[{id}]> {} {calls}\nthink: /(?s:.*)/\n",
                    lit("\n\n")
                ),
                None => format!(
                    "start: think {calls}\nthink[suffix={}]: /(?s:.*)/\n",
                    lit("</think>\n\n")
                ),
            };
        } else {
            g += &format!("start: {calls}\n");
        }
        let fns: Vec<&Value> = match only {
            Some(n) => self.function(n).into_iter().collect(),
            None => self.list.iter().map(|t| &t["function"]).collect(),
        };
        let alts: Vec<String> = (0..fns.len()).map(|i| format!("call_{i}")).collect();
        g += &format!("call: {}\n", alts.join(" | "));
        for (i, f) in fns.iter().enumerate() {
            let name = f["name"].as_str().unwrap_or_default();
            let params = &f["parameters"];
            match format {
                Format::Hermes => {
                    let schema = json!({ "type": "object", "properties": {
                        "name": { "const": name },
                        "arguments": if params.is_object() { params.clone() } else { json!({ "type": "object" }) } },
                        "required": ["name", "arguments"], "additionalProperties": false });
                    g += &format!(
                        "call_{i}: {open} {} %json {} {} {close}\n",
                        lit("\n"),
                        schema,
                        lit("\n")
                    );
                }
                Format::Xml => {
                    let required: Vec<&str> = params["required"]
                        .as_array()
                        .map(|r| r.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
                    let mut body = Vec::new();
                    if let Some(props) = params["properties"].as_object() {
                        for (j, (key, schema)) in props.iter().enumerate() {
                            let rule = format!("p_{i}_{j}");
                            let open = lit(&format!("<parameter={key}>\n"));
                            if takes_string(schema) {
                                g += &format!("{rule}: {open} v_{i}_{j}\n");
                                g += &format!(
                                    "v_{i}_{j}[suffix={}]: /(?s:.*)/\n",
                                    lit("\n</parameter>\n")
                                );
                            } else {
                                g += &format!(
                                    "{rule}: {open} %json {} {}\n",
                                    schema,
                                    lit("\n</parameter>\n")
                                );
                            }
                            let opt = if required.contains(&key.as_str()) {
                                ""
                            } else {
                                "?"
                            };
                            body.push(format!("{rule}{opt}"));
                        }
                    }
                    g += &format!(
                        "call_{i}: {open} {} {} {} {close}\n",
                        lit(&format!("\n<function={name}>\n")),
                        body.join(" "),
                        lit("</function>\n")
                    );
                }
            }
        }
        Some(g)
    }
}

/// The JSON types a parameter's schema allows.
fn types(schema: &Value) -> Vec<&str> {
    match &schema["type"] {
        Value::String(s) => vec![s.as_str()],
        Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Whether a parameter is written as raw text: strings, and anything without
/// a type, as vLLM's qwen3_coder parser reads them.
fn takes_string(schema: &Value) -> bool {
    let t = types(schema);
    t.is_empty() || t.contains(&"string")
}

/// A parameter's text as JSON, by its schema.
fn convert(raw: &str, schema: Option<&Value>) -> Value {
    let Some(schema) = schema.filter(|s| !takes_string(s)) else {
        return Value::String(raw.to_string());
    };
    let t = raw.trim();
    for ty in types(schema) {
        let v = match ty {
            "integer" => t.parse::<i64>().ok().map(Value::from),
            "number" => serde_json::from_str::<serde_json::Number>(t)
                .ok()
                .map(Value::Number),
            "boolean" => match t.to_ascii_lowercase().as_str() {
                "true" => Some(Value::Bool(true)),
                "false" => Some(Value::Bool(false)),
                _ => None,
            },
            "null" => matches!(t, "null" | "None").then_some(Value::Null),
            _ => serde_json::from_str::<Value>(t).ok(),
        };
        if let Some(v) = v {
            return v;
        }
    }
    serde_json::from_str(t).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// A call the model made.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    /// The arguments as a JSON object, in a string.
    pub arguments: String,
}

impl Call {
    /// The call in `message.tool_calls` and in streamed deltas.
    pub fn to_json(&self, id: &str) -> Value {
        json!({ "id": id, "type": "function",
            "function": { "name": self.name, "arguments": self.arguments } })
    }
}

/// One call block, `<tool_call>` to `</tool_call>` exclusive. None when it
/// is not a well-formed call.
fn parse_call(inner: &str, tools: &Tools) -> Option<Call> {
    let inner = inner.trim();
    if let Some(rest) = inner.strip_prefix("<function=") {
        let (name, mut rest) = rest.split_once('>')?;
        let name = name.trim();
        let props = tools.function(name).map(|f| &f["parameters"]["properties"]);
        let mut args = Map::new();
        while let Some(at) = rest.find("<parameter=") {
            let (key, after) = rest[at + "<parameter=".len()..].split_once('>')?;
            let end = after
                .find("</parameter>")
                .or_else(|| after.find("<parameter="))
                .or_else(|| after.find("</function>"))
                .unwrap_or(after.len());
            let raw = &after[..end];
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let key = key.trim();
            args.insert(
                key.to_string(),
                convert(raw, props.and_then(|p| p.get(key))),
            );
            rest = after[end..]
                .strip_prefix("</parameter>")
                .unwrap_or(&after[end..]);
        }
        return Some(Call {
            name: name.to_string(),
            arguments: py_json(&args),
        });
    }
    let v: Value = serde_json::from_str(inner).ok()?;
    let name = v["name"].as_str()?.to_string();
    let arguments = match &v["arguments"] {
        Value::String(s) => s.clone(),
        Value::Null => "{}".into(),
        a => py_json(a),
    };
    Some(Call { name, arguments })
}

/// A piece of the output: text for `content`, or a call.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    Call(Call),
}

/// Reads calls out of the output as it arrives. Text that could begin a
/// call is held back until it is known not to; whitespace before a call is
/// dropped, as the template writes it between the text and the call.
pub struct Scanner {
    text: String,
    done: usize,
    /// Where an unfinished call block starts.
    call: Option<usize>,
    /// Still inside the turn's reasoning, where no call counts.
    think: bool,
    pub calls: usize,
}

impl Scanner {
    pub fn new(thinking: bool) -> Self {
        Self {
            text: String::new(),
            done: 0,
            call: None,
            think: thinking,
            calls: 0,
        }
    }

    /// Adds released text; returns the pieces now known.
    pub fn feed(&mut self, s: &str, tools: &Tools) -> Vec<Piece> {
        self.text.push_str(s);
        self.scan(tools, false)
    }

    /// The remaining pieces at the end of the output.
    pub fn finish(&mut self, tools: &Tools) -> Vec<Piece> {
        self.scan(tools, true)
    }

    fn scan(&mut self, tools: &Tools, end: bool) -> Vec<Piece> {
        let mut out = Vec::new();
        let text = |out: &mut Vec<Piece>, s: &str| {
            if !s.is_empty() {
                out.push(Piece::Text(s.to_string()));
            }
        };
        loop {
            let rest = &self.text[self.done..];
            if self.think {
                if let Some(i) = rest.find(THINK_END) {
                    text(&mut out, &rest[..i + THINK_END.len()]);
                    self.done += i + THINK_END.len();
                    self.think = false;
                    continue;
                }
                let keep = if end {
                    0
                } else {
                    partial_suffix(rest, THINK_END)
                };
                text(&mut out, &rest[..rest.len() - keep]);
                self.done += rest.len() - keep;
                return out;
            }
            if let Some(start) = self.call {
                let block = &self.text[start + OPEN.len()..];
                let (inner, used) = match block.find(CLOSE) {
                    Some(j) => (&block[..j], j + CLOSE.len()),
                    None if end => (block, block.len()),
                    None => return out,
                };
                match parse_call(inner, tools) {
                    Some(c) => {
                        self.calls += 1;
                        out.push(Piece::Call(c));
                    }
                    None => text(&mut out, &self.text[start..start + OPEN.len() + used]),
                }
                self.done = start + OPEN.len() + used;
                self.call = None;
                continue;
            }
            if let Some(i) = rest.find(OPEN) {
                text(&mut out, rest[..i].trim_end());
                self.call = Some(self.done + i);
                self.done += i;
                continue;
            }
            // Hold back what may begin a call, and whitespace that may come
            // before one; at the end, whitespace after the calls goes too.
            let keep = if end {
                if self.calls > 0 {
                    rest.len() - rest.trim_end().len()
                } else {
                    0
                }
            } else {
                let p = partial_suffix(rest, OPEN);
                let body = &rest[..rest.len() - p];
                p + body.len() - body.trim_end().len()
            };
            text(&mut out, &rest[..rest.len() - keep]);
            self.done += rest.len() - keep;
            if end {
                self.done = self.text.len();
            }
            return out;
        }
    }
}

/// The length of the longest end of `s` that begins `tag`.
fn partial_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len().min(s.len() + 1))
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

/// Splits a whole output into its content (None when only calls remain) and
/// its calls.
pub fn parse(text: &str, tools: &Tools, thinking: bool) -> (Option<String>, Vec<Call>) {
    let mut s = Scanner::new(thinking);
    let mut pieces = s.feed(text, tools);
    pieces.extend(s.finish(tools));
    let mut content = String::new();
    let mut calls = Vec::new();
    for p in pieces {
        match p {
            Piece::Text(t) => content.push_str(&t),
            Piece::Call(c) => calls.push(c),
        }
    }
    let content = (!(content.is_empty() || (!calls.is_empty() && content.trim().is_empty())))
        .then_some(content);
    (content, calls)
}

/// A fresh id for a call.
pub fn call_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    );
    format!("call_{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> Tools {
        Tools::from_request(
            Some(&[
                json!({ "type": "function", "function": { "name": "get_weather",
                "parameters": { "type": "object", "required": ["city"], "properties": {
                    "city": { "type": "string" }, "days": { "type": "integer" },
                    "exact": { "type": "boolean" }, "filters": { "type": "object" } } } } }),
            ]),
            None,
            None,
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn python_style_json() {
        assert_eq!(
            py_json(&json!({ "b": [1, "é\n"], "a": { "c": true } })),
            r#"{"b": [1, "é\n"], "a": {"c": true}}"#
        );
    }

    #[test]
    fn parses_xml_calls_by_schema() {
        let t = weather();
        let out = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=exact>\nTrue\n</parameter>\n<parameter=filters>\n{\"a\": [1]}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLima\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse(out, &t, false);
        assert_eq!(content.as_deref(), Some("Let me check."));
        assert_eq!(
            calls,
            vec![
                Call {
                    name: "get_weather".into(),
                    arguments:
                        r#"{"city": "New\nYork", "days": 3, "exact": true, "filters": {"a": [1]}}"#
                            .into()
                },
                Call {
                    name: "get_weather".into(),
                    arguments: r#"{"city": "Lima"}"#.into()
                }
            ]
        );
    }

    #[test]
    fn parses_hermes_calls_and_leaves_broken_ones_as_text() {
        let t = weather();
        let (content, calls) = parse(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>",
            &t,
            false,
        );
        assert_eq!(content, None);
        assert_eq!(calls[0].arguments, r#"{"city": "Oslo"}"#);
        let (content, calls) = parse("Sure <tool_call>{not json</tool_call>", &t, false);
        assert_eq!(
            content.as_deref(),
            Some("Sure<tool_call>{not json</tool_call>")
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn calls_inside_reasoning_do_not_count() {
        let t = weather();
        let (content, calls) = parse(
            "maybe <tool_call>x</tool_call></think>\n\nNo call needed.",
            &t,
            true,
        );
        assert!(calls.is_empty());
        assert_eq!(
            content.as_deref(),
            Some("maybe <tool_call>x</tool_call></think>\n\nNo call needed.")
        );
    }

    #[test]
    fn streaming_gives_the_same_pieces_at_any_split() {
        let t = weather();
        let out = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>\n";
        let (whole, calls) = parse(out, &t, false);
        for step in 1..8 {
            let mut s = Scanner::new(false);
            let mut text = String::new();
            let mut got = Vec::new();
            let chars: Vec<char> = out.chars().collect();
            let mut pieces = Vec::new();
            for c in chars.chunks(step) {
                pieces.extend(s.feed(&c.iter().collect::<String>(), &t));
            }
            pieces.extend(s.finish(&t));
            for p in pieces {
                match p {
                    Piece::Text(x) => text.push_str(&x),
                    Piece::Call(c) => got.push(c),
                }
            }
            assert_eq!(Some(text), whole, "step {step}");
            assert_eq!(got, calls, "step {step}");
        }
    }

    #[test]
    fn tool_choice_is_validated() {
        let tools = [json!({ "type": "function", "function": { "name": "f" } })];
        let get = |c: Value| Tools::from_request(Some(&tools), Some(&c), None);
        assert!(get(json!("none")).unwrap().is_none());
        assert_eq!(get(json!("required")).unwrap().unwrap().forced, Some(None));
        assert_eq!(
            get(json!({ "type": "function", "function": { "name": "f" } }))
                .unwrap()
                .unwrap()
                .forced,
            Some(Some("f".into()))
        );
        assert!(get(json!({ "type": "function", "function": { "name": "g" } })).is_err());
        assert!(Tools::from_request(None, Some(&json!("required")), None).is_err());
        assert!(Tools::from_request(Some(&[json!({ "type": "function" })]), None, None).is_err());
    }

    /// Whether a grammar accepts `text` whole, a byte per token.
    fn accepts(grammar: &str, text: &str) -> bool {
        let env = llguidance::toktrie::ApproximateTokEnv::single_byte_env();
        let mut f = llguidance::ParserFactory::new_simple(&env).unwrap();
        f.quiet();
        let g = llguidance::api::TopLevelGrammar::from_lark(grammar.to_string());
        let mut m = llguidance::Matcher::new(f.create_parser(g));
        assert!(m.get_error().is_none(), "{:?}\n{grammar}", m.get_error());
        let toks: Vec<u32> = text.bytes().map(u32::from).collect();
        m.consume_tokens(&toks).is_ok() && m.is_accepting().unwrap_or(false)
    }

    #[test]
    fn forced_grammars_accept_calls_and_nothing_else() {
        let none = |_: &str| None;
        let mut t = weather();
        t.forced = Some(None);
        let one = "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let g = t.grammar(Format::Xml, false, &none).unwrap();
        assert!(accepts(&g, one));
        assert!(accepts(&g, &format!("{one}\n{one}")));
        assert!(!accepts(&g, "Sure."));
        // The required parameter is missing; days is not an integer.
        assert!(!accepts(
            &g,
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        ));
        assert!(!accepts(&g, &one.replace("\n3\n", "\nthree\n")));
        let g = t.grammar(Format::Xml, true, &none).unwrap();
        assert!(accepts(&g, &format!("I should look.\n</think>\n\n{one}")));
        t.forced = Some(Some("get_weather".into()));
        t.parallel = true;
        let g = t.grammar(Format::Xml, false, &none).unwrap();
        assert!(!accepts(&g, &format!("{one}\n{one}")));
        let g = t.grammar(Format::Hermes, false, &none).unwrap();
        assert!(accepts(
            &g,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>"
        ));
        assert!(!accepts(
            &g,
            "<tool_call>\n{\"name\": \"other\", \"arguments\": {}}\n</tool_call>"
        ));
        t.forced = None;
        assert!(t.grammar(Format::Xml, false, &none).is_none());
    }

    /// With the tags as tokens, as in Qwen's vocabulary (here the test
    /// vocabulary's first two special tokens).
    #[test]
    fn forced_grammars_name_tag_tokens() {
        let mut t = weather();
        t.forced = Some(None);
        let ids = |s: &str| match s {
            OPEN => Some(256),
            CLOSE => Some(257),
            THINK_END => Some(258),
            _ => None,
        };
        let g = t.grammar(Format::Xml, true, &ids).unwrap();
        let env = llguidance::toktrie::ApproximateTokEnv::single_byte_env();
        let mut f = llguidance::ParserFactory::new_simple(&env).unwrap();
        f.quiet();
        let mut m = llguidance::Matcher::new(
            f.create_parser(llguidance::api::TopLevelGrammar::from_lark(g.clone())),
        );
        assert!(m.get_error().is_none(), "{:?}\n{g}", m.get_error());
        let mut toks: Vec<u32> = "Oslo, then.".bytes().map(u32::from).collect();
        toks.push(258);
        toks.extend("\n\n".bytes().map(u32::from));
        toks.push(256);
        toks.extend(
            "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n"
                .bytes()
                .map(u32::from),
        );
        toks.push(257);
        m.consume_tokens(&toks).unwrap();
        assert!(m.is_accepting().unwrap());
    }
}
