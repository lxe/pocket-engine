//! Continuous batching, without the GPU. Each step feeds every running
//! sequence its next token and fills the remaining token budget with prompt
//! chunks, so new requests start while others decode. Admission reserves KV
//! blocks for the whole request, so nothing is ever preempted.

use crate::image::Image;
use crate::model::{BLOCK_SIZE, Meta};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

#[derive(Debug)]
pub enum Event {
    Token(u32),
    /// Log-probabilities of the token the next `Token` carries, for
    /// requests that ask for them.
    Logprobs(TokenLogprobs),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// A token's log-probability and the most likely tokens at its position.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

/// How the next token is picked. Temperature 0 is greedy. Penalties and
/// biases adjust the logits first, as OpenAI and vLLM define them: presence
/// and frequency count generated tokens, repetition covers prompt and output.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// 0 keeps every token.
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

impl Sampling {
    fn adjusts_logits(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
            || !self.logit_bias.is_empty()
    }
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub sampling: Sampling,
    pub tx: UnboundedSender<Event>,
    /// Images whose placeholders the prompt holds, one per embedding.
    pub images: Vec<Image>,
    /// Rotary positions of the prompt's tokens and the offset after it, when
    /// they differ from the token index (prompts with images).
    pub rope: Option<crate::image::Rope>,
    /// The LoRA adapter to run with (an id from `Loras`), or 0.
    pub adapter: u32,
    /// Log-probabilities to return with each token: the picked one's and
    /// this many most likely.
    pub logprobs: Option<usize>,
    /// A grammar the output must follow (structured output).
    pub constraint: Option<Box<llguidance::Matcher>>,
}

impl Request {
    pub fn text(
        prompt: Vec<u32>,
        max_new: usize,
        sampling: Sampling,
        tx: UnboundedSender<Event>,
    ) -> Self {
        Self {
            prompt,
            max_new,
            sampling,
            tx,
            images: Vec::new(),
            rope: None,
            adapter: 0,
            logprobs: None,
            constraint: None,
        }
    }
}

/// Images to encode into pool rows `row..` before this step runs.
pub struct VisionJob {
    pub images: Vec<Image>,
    pub row: usize,
}

/// One teacher-forced position for accuracy checks: the model's pick and its
/// top log-probabilities before the forced token is fed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Scored {
    pub argmax: u32,
    pub top: Vec<(u32, f32)>,
}

pub struct Seq {
    pub req: Request,
    slot: usize,
    /// The GPU slot holding its adapter, or 0.
    lora: usize,
    /// The grammar's allowed tokens for this step's pick, when constrained.
    mask: Option<llguidance::toktrie::SimpleVob>,
    /// Draft tokens to verify next step (multi-token prediction).
    drafts: Vec<u32>,
    /// While verifying drafts: the slot to keep the state in after each fed
    /// token, free slots borrowed for the step and its own slot last.
    outs: Vec<usize>,
    blocks: Vec<u32>,
    /// Tokens of prompt ++ generated whose state is computed.
    fed: usize,
    pub generated: Vec<u32>,
    /// Teacher forcing: the continuation to feed instead of the model's picks.
    pub forced: Option<Vec<u32>>,
    pub scored: Vec<Scored>,
    admitted: Instant,
    /// Generated-token counts, kept only when the sampling adjusts logits.
    counts: HashMap<u32, u32>,
    prompt_tokens: HashSet<u32>,
    /// Rows of the image embedding pool this sequence holds until its prompt
    /// is fed, and each prompt position's row offset (-1 for text).
    pool: Option<(usize, usize)>,
    image_rows: Vec<i32>,
    vision_pending: bool,
    /// Prefix-cache boundaries to snapshot this sequence's state at, in
    /// increasing order.
    snap_at: Vec<usize>,
}

impl Seq {
    fn token(&self, i: usize) -> u32 {
        let p = self.req.prompt.len();
        if i < p {
            self.req.prompt[i]
        } else {
            self.generated[i - p]
        }
    }
    fn total(&self) -> usize {
        self.req.prompt.len() + self.generated.len()
    }
    fn in_prompt(&self) -> bool {
        self.fed < self.req.prompt.len()
    }
    /// Whether a row feeding `n` tokens verifies drafts.
    fn verifies(&self, n: usize) -> bool {
        !self.in_prompt() && n > 1
    }
}

/// Staged logit adjustments for a step: entry i sets
/// logit[tok] = (logit > 0 ? logit / rep : logit * rep) + add.
/// `add` and `rep` hold f32 bits.
pub struct Pen<'a> {
    pub tok: &'a mut [i32],
    pub add: &'a mut [i32],
    pub rep: &'a mut [i32],
}

impl<'a> Pen<'a> {
    /// Splits a staging area of three equal regions.
    pub fn split(words: &'a mut [i32]) -> Self {
        let cap = words.len() / 3;
        let (tok, rest) = words.split_at_mut(cap);
        let (add, rep) = rest.split_at_mut(cap);
        Self {
            tok,
            add,
            rep: &mut rep[..cap],
        }
    }

    /// Appends a sequence's adjustments from entry `e`; returns the new end.
    fn push_seq(&mut self, mut e: usize, s: &Seq) -> usize {
        let sp = &s.req.sampling;
        let mut adj: HashMap<u32, (f32, f32)> = HashMap::new();
        for (&t, &c) in &s.counts {
            let a = adj.entry(t).or_insert((0.0, 1.0));
            a.0 -= sp.presence_penalty + sp.frequency_penalty * c as f32;
        }
        if sp.repetition_penalty != 1.0 {
            for &t in s.prompt_tokens.iter().chain(s.counts.keys()) {
                adj.entry(t).or_insert((0.0, 1.0)).1 = sp.repetition_penalty;
            }
        }
        for &(t, b) in &sp.logit_bias {
            adj.entry(t).or_insert((0.0, 1.0)).0 += b;
        }
        for (t, (add, rep)) in adj {
            if e == self.tok.len() {
                break; // capacity covers a full context plus a bias list
            }
            self.tok[e] = t as i32;
            self.add[e] = add.to_bits() as i32;
            self.rep[e] = rep.to_bits() as i32;
            e += 1;
        }
        e
    }
}

/// The work of one step: `rows[i]` feeds `rows[i].1` tokens of running
/// sequence `rows[i].0`, whose output rows start at `outs_at[i]`.
#[derive(Debug, Default)]
pub struct Step {
    rows: Vec<(usize, usize)>,
    outs_at: Vec<usize>,
    pub tokens: usize,
    /// Output rows: one per sequence, or one per token of a sequence
    /// verifying drafts.
    pub outs: usize,
    /// State slot copies (from, to) to make, in order, before the step runs:
    /// prefix-cache snapshots and restores.
    pub copies: Vec<(usize, usize)>,
    /// Prefix-cache snapshots the step writes itself: (row, prefix length,
    /// state slot). The recurrence keeps its state at the prefix's end, which
    /// falls on its chunk grid, so the prompt need not stop there.
    snaps: Vec<(usize, usize, usize)>,
    /// Adapters to copy into GPU slots before the step: (slot, adapter).
    pub lora_loads: Vec<(usize, u32)>,
    /// Kernel tiles of the step's tokens that have adapters.
    pub lora_tiles: usize,
}

impl Step {
    pub fn seqs(&self) -> usize {
        self.rows.len()
    }
    pub fn decode_only(&self) -> bool {
        self.tokens == self.rows.len()
    }
    /// Whether every row verifies `spec - 1` drafts (a step a captured
    /// verification graph can replay).
    pub fn verify_only(&self, spec: usize) -> bool {
        spec > 1 && self.tokens == self.rows.len() * spec && self.outs == self.tokens
    }
    /// The output row of sequence row `si` (its first, when it verifies).
    pub fn out_row(&self, si: usize) -> usize {
        self.outs_at[si]
    }
}

#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Counters {
    pub steps: u64,
    pub decode_tokens: u64,
    pub prompt_tokens: u64,
    /// Prompt tokens served from the prefix cache instead of computed.
    pub cached_tokens: u64,
    /// Drafts verified, and of those accepted.
    pub drafted: u64,
    pub accepted: u64,
}

/// Most tokens a step verifying drafts may feed by default (see
/// [`Scheduler::with_verify_rows`]). Past this, verification turns the step's
/// matmuls compute-bound and costs more than the accepted drafts save
/// (Qwen3.5-9B on an RTX 5090: three drafts each for 32 sequences, 128 rows,
/// ran 6% slower than plain decoding).
pub const VERIFY_ROWS: usize = 96;

/// The recurrence's chunk length (GC in qwen35.cu): it can keep its state at
/// any multiple of it inside a step.
const SNAP_GRID: usize = 32;

/// Prompt tokens per prefix-cache unit: entries cover whole units, a
/// multiple of the KV block.
pub const CACHE_UNIT: usize = 64;

/// A cached prompt prefix: its tokens, the KV blocks that hold their keys
/// and values, and a state slot holding the recurrent state after them.
struct Entry {
    adapter: u32,
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    snap: usize,
    used: u64,
}

/// Prefix cache: entries by the chained hash of their units.
#[derive(Default)]
struct Prefixes {
    entries: HashMap<u64, Entry>,
    free_snaps: Vec<usize>,
    clock: u64,
    /// The token that opens a chat message.
    turn: u32,
}

/// Chained hashes of a prompt's whole units: element k covers tokens
/// [0, (k + 1) * CACHE_UNIT). Prompts for different adapters never share.
fn unit_hashes(adapter: u32, tokens: &[u32]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut out = Vec::with_capacity(tokens.len() / CACHE_UNIT);
    let mut h = adapter as u64;
    for unit in tokens.as_chunks::<CACHE_UNIT>().0 {
        let mut st = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut st);
        unit.hash(&mut st);
        h = st.finish();
        out.push(h);
    }
    out
}

/// Which adapter each GPU slot holds (index 0 unused), how many running
/// sequences use it, and when it was last placed.
#[derive(Default)]
struct LoraSlots {
    held: Vec<Option<u32>>,
    refs: Vec<usize>,
    used: Vec<u64>,
    clock: u64,
    /// Adapters to copy into slots before the next step.
    loads: Vec<(usize, u32)>,
}

impl LoraSlots {
    fn holding(&self, a: u32) -> Option<usize> {
        (1..self.held.len()).find(|&i| self.held[i] == Some(a))
    }
    fn can_place(&self, a: u32) -> bool {
        self.holding(a).is_some() || (1..self.held.len()).any(|i| self.refs[i] == 0)
    }
    /// The slot for adapter `a`: the one holding it, or a free one (empty
    /// first, then least recently used), queued for upload.
    fn place(&mut self, a: u32) -> usize {
        self.clock += 1;
        let i = match self.holding(a) {
            Some(i) => i,
            None => {
                let i = (1..self.held.len())
                    .filter(|&i| self.refs[i] == 0)
                    .min_by_key(|&i| (self.held[i].is_some(), self.used[i]))
                    .expect("checked by can_place");
                self.held[i] = Some(a);
                self.loads.push((i, a));
                i
            }
        };
        self.refs[i] += 1;
        self.used[i] = self.clock;
        i
    }
}

pub struct Scheduler {
    meta: Meta,
    lora: LoraSlots,
    image_token: u32,
    pool_free: Vec<(usize, usize)>,
    pool_rows: usize,
    max_context: usize,
    /// Free state slots: one per sequence, and lent for a step to sequences
    /// verifying drafts.
    free_slots: Vec<usize>,
    free_blocks: Vec<u32>,
    /// References to each KV block, from sequences and cache entries; a block
    /// is free at zero.
    block_refs: Vec<u16>,
    total_blocks: usize,
    prefixes: Prefixes,
    /// State copies for the next step (see Step::copies).
    pending_copies: Vec<(usize, usize)>,
    /// Most tokens a step verifying drafts may feed.
    verify_rows: usize,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    eos: Vec<u32>,
    pub counters: Counters,
}

/// Capacity, as reported by /health.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Load {
    pub running: usize,
    pub waiting: usize,
    pub free_kv_blocks: usize,
    pub kv_blocks: usize,
    pub max_seqs: usize,
}

impl Scheduler {
    pub fn new(meta: Meta, max_context: usize, blocks: usize, eos: Vec<u32>) -> Self {
        Self::with_images(meta, max_context, blocks, eos, 0, 0)
    }

    /// A scheduler whose requests may carry images: `pool_rows` image
    /// embeddings can wait at once for their prompt steps.
    pub fn with_images(
        meta: Meta,
        max_context: usize,
        blocks: usize,
        eos: Vec<u32>,
        image_token: u32,
        pool_rows: usize,
    ) -> Self {
        Self {
            meta,
            image_token,
            pool_free: if pool_rows > 0 {
                vec![(0, pool_rows)]
            } else {
                Vec::new()
            },
            pool_rows,
            max_context,
            free_slots: (0..meta.max_seqs).rev().collect(),
            free_blocks: (0..blocks as u32).rev().collect(),
            block_refs: vec![0; blocks],
            total_blocks: blocks,
            prefixes: Prefixes::default(),
            lora: LoraSlots::default(),
            pending_copies: Vec::new(),
            verify_rows: VERIFY_ROWS,
            waiting: VecDeque::new(),
            running: Vec::new(),
            eos,
            counters: Counters::default(),
        }
    }

    /// Keeps prompt prefixes for reuse, with state snapshots in `slots`
    /// (state slots no sequence uses). A prompt leaves up to two: at its last
    /// whole unit, for a conversation's next turn, and at the unit before its
    /// last message (opened by `turn`), for prompts that share everything
    /// before it, such as a system prompt.
    /// Lets steps verifying drafts feed up to `rows` tokens: more on GPUs
    /// with more compute per byte of memory bandwidth.
    pub fn with_verify_rows(mut self, rows: usize) -> Self {
        self.verify_rows = rows;
        self
    }

    /// Lets requests run adapters, `slots` of them on the GPU at once.
    pub fn with_lora_slots(mut self, slots: usize) -> Self {
        self.lora = LoraSlots {
            held: vec![None; slots + 1],
            refs: vec![0; slots + 1],
            used: vec![0; slots + 1],
            ..Default::default()
        };
        self
    }

    /// Whether a running or waiting request uses adapter `a`.
    pub fn uses_adapter(&self, a: u32) -> bool {
        self.running
            .iter()
            .chain(&self.waiting)
            .any(|s| s.req.adapter == a)
    }

    /// Forgets adapter `a`'s slot and cache entries once it is unloaded.
    pub fn forget_adapter(&mut self, a: u32) {
        if let Some(i) = self.lora.holding(a) {
            self.lora.held[i] = None;
        }
        let gone: Vec<u64> = self
            .prefixes
            .entries
            .iter()
            .filter(|(_, e)| e.adapter == a)
            .map(|(&h, _)| h)
            .collect();
        for h in gone {
            let e = self.prefixes.entries.remove(&h).expect("present");
            self.unref_blocks(&e.blocks);
            self.prefixes.free_snaps.push(e.snap);
        }
    }

    pub fn with_prefix_cache(mut self, slots: std::ops::Range<usize>, turn: u32) -> Self {
        self.prefixes.free_snaps = slots.rev().collect();
        self.prefixes.turn = turn;
        self
    }

    pub fn load(&self) -> Load {
        Load {
            running: self.running.len(),
            waiting: self.waiting.len(),
            free_kv_blocks: self.free_blocks.len(),
            kv_blocks: self.total_blocks,
            max_seqs: self.meta.max_seqs,
        }
    }

    /// The state slot that padding rows of a graph replay write to.
    pub fn scratch_slot(&self) -> usize {
        self.meta.max_seqs
    }

    pub fn idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    pub fn submit(&mut self, req: Request) {
        self.enqueue(req, None);
    }

    pub fn submit_forced(&mut self, req: Request, forced: Vec<u32>) {
        self.enqueue(req, Some(forced));
    }

    fn enqueue(&mut self, mut req: Request, forced: Option<Vec<u32>>) {
        let limit = self.max_context;
        if req.prompt.is_empty() || req.prompt.len() >= limit {
            let _ = req.tx.send(Event::Error(format!(
                "prompt of {} tokens does not fit the context of {limit}",
                req.prompt.len()
            )));
            return;
        }
        let image_tokens: usize = req.images.iter().map(Image::tokens).sum();
        if image_tokens > self.pool_rows {
            let _ = req.tx.send(Event::Error(if self.pool_rows == 0 {
                "this server does not accept images".into()
            } else {
                format!(
                    "images need {image_tokens} embeddings; the server holds {}",
                    self.pool_rows
                )
            }));
            return;
        }
        req.max_new = match &forced {
            Some(f) => f.len().min(limit - req.prompt.len()),
            None => req.max_new.clamp(1, limit - req.prompt.len()),
        };
        self.waiting.push_back(Seq {
            req,
            slot: 0,
            lora: 0,
            mask: None,
            drafts: Vec::new(),
            outs: Vec::new(),
            blocks: Vec::new(),
            fed: 0,
            generated: Vec::new(),
            forced,
            scored: Vec::new(),
            admitted: Instant::now(),
            counts: HashMap::new(),
            prompt_tokens: HashSet::new(),
            pool: None,
            image_rows: Vec::new(),
            vision_pending: false,
            snap_at: Vec::new(),
        });
    }

    fn admit(&mut self) {
        // Requests whose adapter cannot get a GPU slot yet wait; those after
        // them may still start.
        let mut at = 0;
        while self.running.len() < self.meta.max_seqs {
            let Some(seq) = self.waiting.get(at) else {
                break;
            };
            if seq.req.adapter != 0 && !self.lora.can_place(seq.req.adapter) {
                at += 1;
                continue;
            }
            let total_blocks = (seq.req.prompt.len() + seq.req.max_new).div_ceil(BLOCK_SIZE);
            let rows: usize = seq.req.images.iter().map(Image::tokens).sum();
            // Text prompts may reuse and leave cached prefixes; prompts with
            // images do not.
            let cacheable = rows == 0
                && (!self.prefixes.free_snaps.is_empty() || !self.prefixes.entries.is_empty());
            let hit = if cacheable {
                self.lookup(seq.req.adapter, &seq.req.prompt)
            } else {
                None
            };
            let reused = hit.map_or(0, |(_, len)| len / BLOCK_SIZE);
            let need = total_blocks - reused;
            // Cache entries give way to new requests, oldest first (but not
            // the one this request is about to use).
            while self.free_blocks.len() < need
                && self.prefixes.entries.len() > hit.is_some() as usize
            {
                let keep = hit.map(|(h, _)| h);
                let Some((&h, _)) = self
                    .prefixes
                    .entries
                    .iter()
                    .filter(|(h, _)| Some(**h) != keep)
                    .min_by_key(|(_, e)| e.used)
                else {
                    break;
                };
                let e = self.prefixes.entries.remove(&h).expect("present");
                self.unref_blocks(&e.blocks);
                self.prefixes.free_snaps.push(e.snap);
            }
            let pool_at = if rows > 0 {
                self.pool_free.iter().position(|r| r.1 >= rows)
            } else {
                None
            };
            if self.free_slots.is_empty()
                || self.free_blocks.len() < need
                || (rows > 0 && pool_at.is_none())
            {
                break;
            }
            let mut seq = self.waiting.remove(at).expect("checked above");
            if seq.req.adapter != 0 {
                seq.lora = self.lora.place(seq.req.adapter);
            }
            if let Some(i) = pool_at {
                let (start, len) = self.pool_free[i];
                if len == rows {
                    self.pool_free.remove(i);
                } else {
                    self.pool_free[i] = (start + rows, len - rows);
                }
                seq.pool = Some((start, rows));
                seq.vision_pending = true;
                let mut k = 0;
                seq.image_rows = seq
                    .req
                    .prompt
                    .iter()
                    .map(|&t| {
                        if t == self.image_token {
                            k += 1;
                            k - 1
                        } else {
                            -1
                        }
                    })
                    .collect();
            }
            seq.slot = self.free_slots.pop().expect("slot available");
            seq.blocks = Vec::with_capacity(total_blocks);
            if let Some((h, len)) = hit {
                self.prefixes.clock += 1;
                let e = self.prefixes.entries.get_mut(&h).expect("looked up");
                e.used = self.prefixes.clock;
                for &b in &e.blocks {
                    self.block_refs[b as usize] += 1;
                }
                seq.blocks.extend_from_slice(&e.blocks);
                self.pending_copies.push((e.snap, seq.slot));
                seq.fed = len;
                self.counters.cached_tokens += len as u64;
            }
            let own = self.take_blocks(need);
            seq.blocks.extend(own);
            if cacheable {
                // Leave prefixes for later requests: before the last message
                // and the longest whole-unit prefix short of the last token.
                let p = &seq.req.prompt;
                let last = (p.len() - 1) / CACHE_UNIT * CACHE_UNIT;
                let turn = p
                    .iter()
                    .rposition(|&t| t == self.prefixes.turn)
                    .map_or(0, |i| i / CACHE_UNIT * CACHE_UNIT);
                for at in [turn, last] {
                    if at > seq.fed && seq.snap_at.last().is_none_or(|&l| l < at) {
                        seq.snap_at.push(at);
                    }
                }
            }
            seq.admitted = Instant::now();
            if seq.req.sampling.repetition_penalty != 1.0 {
                seq.prompt_tokens = seq.req.prompt.iter().copied().collect();
            }
            self.running.push(seq);
        }
    }

    fn take_blocks(&mut self, n: usize) -> Vec<u32> {
        let blocks = self.free_blocks.split_off(self.free_blocks.len() - n);
        for &b in &blocks {
            self.block_refs[b as usize] = 1;
        }
        blocks
    }

    fn unref_blocks(&mut self, blocks: &[u32]) {
        for &b in blocks {
            let r = &mut self.block_refs[b as usize];
            *r -= 1;
            if *r == 0 {
                self.free_blocks.push(b);
            }
        }
    }

    /// Frees the least recently used cache entry; false when there is none.
    fn evict_one(&mut self) -> bool {
        let Some((&h, _)) = self.prefixes.entries.iter().min_by_key(|(_, e)| e.used) else {
            return false;
        };
        let e = self.prefixes.entries.remove(&h).expect("present");
        self.unref_blocks(&e.blocks);
        self.prefixes.free_snaps.push(e.snap);
        true
    }

    /// The longest cached prefix of `prompt` that leaves at least one token to
    /// compute: its hash and length.
    fn lookup(&self, adapter: u32, prompt: &[u32]) -> Option<(u64, usize)> {
        let hashes = unit_hashes(adapter, &prompt[..prompt.len() - 1]);
        hashes.iter().enumerate().rev().find_map(|(k, h)| {
            let len = (k + 1) * CACHE_UNIT;
            self.prefixes
                .entries
                .get(h)
                .filter(|e| e.adapter == adapter && e.tokens[..] == prompt[..len])
                .map(|_| (*h, len))
        })
    }

    fn release(&mut self, mut seq: Seq) -> Seq {
        self.free_slots.push(seq.slot);
        if seq.lora != 0 {
            self.lora.refs[seq.lora] -= 1;
            seq.lora = 0;
        }
        let blocks = std::mem::take(&mut seq.blocks);
        self.unref_blocks(&blocks);
        self.release_pool(&mut seq);
        seq
    }

    fn release_pool(&mut self, seq: &mut Seq) {
        if let Some(range) = seq.pool.take() {
            seq.image_rows = Vec::new();
            self.free_pool(range);
        }
    }

    fn free_pool(&mut self, range: (usize, usize)) {
        self.pool_free.push(range);
        self.pool_free.sort_unstable();
        // Merge adjacent free ranges.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.pool_free.len());
        for &(s, l) in &self.pool_free {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == s => last.1 += l,
                _ => merged.push((s, l)),
            }
        }
        self.pool_free = merged;
    }

    /// Images of newly admitted sequences, to encode before the next step.
    pub fn take_vision(&mut self) -> Vec<VisionJob> {
        let mut jobs = Vec::new();
        for s in &mut self.running {
            if s.vision_pending {
                s.vision_pending = false;
                let row = s.pool.map_or(0, |p| p.0);
                jobs.push(VisionJob {
                    images: std::mem::take(&mut s.req.images),
                    row,
                });
            }
        }
        jobs
    }

    /// Drops requests whose clients left, admits waiting ones and plans the
    /// next step: decode rows first, then prompt chunks up to the budget.
    pub fn plan(&mut self) -> Option<Step> {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].forced.is_none() && self.running[i].req.tx.is_closed() {
                let s = self.running.swap_remove(i);
                self.release(s);
            } else {
                i += 1;
            }
        }
        self.waiting
            .retain(|s| s.forced.is_some() || !s.req.tx.is_closed());
        self.admit();
        if self.running.is_empty() {
            return None;
        }
        self.running.sort_by_key(|s| (s.in_prompt(), s.admitted));
        // Sequences verify drafts when free slots can hold the states after
        // their drafts and the step stays small enough that its matmuls are
        // still bound by reading the weights: all of them, so the step can
        // replay a graph, or none.
        let drafting = self.running.iter().filter(|s| !s.drafts.is_empty()).count();
        let verify = drafting > 0
            && self.free_slots.len() >= drafting * (self.meta.spec - 1)
            && drafting * self.meta.spec <= self.verify_rows;
        let mut budget = self.meta.max_tokens;
        let mut step = Step::default();
        for (i, s) in self.running.iter_mut().enumerate() {
            // A decoding sequence feeds its last token and its drafts, short
            // of positions past what it may generate.
            let ahead = if s.in_prompt() || !verify {
                0
            } else {
                s.drafts.len().min(s.req.max_new - s.generated.len() - 1)
            };
            let mut n = (s.total() - s.fed + ahead).min(budget);
            // Cache boundaries inside the chunk: the step keeps the state at
            // those on the recurrence's 32-token chunk grid (or at the
            // chunk's end); the chunk stops at the first other one, whose
            // state is copied after the step.
            for &at in &s.snap_at {
                if at > s.fed + n {
                    break;
                }
                let rel = at - s.fed;
                if n < 2 || (!rel.is_multiple_of(SNAP_GRID) && rel != n) {
                    n = n.min(rel);
                    break;
                }
            }
            if n == 0 {
                break;
            }
            budget -= n;
            step.tokens += n;
            step.rows.push((i, n));
            step.outs_at.push(step.outs);
            step.outs += if s.verifies(n) { n } else { 1 };
            if s.verifies(n) {
                s.outs = (1..n)
                    .map(|_| self.free_slots.pop().expect("counted above"))
                    .chain([s.slot])
                    .collect();
            }
        }
        // Reserve the in-step snapshots' slots.
        for si in 0..step.rows.len() {
            let (ri, n) = step.rows[si];
            let fed = self.running[ri].fed;
            while let Some(&at) = self.running[ri].snap_at.first() {
                let rel = at - fed;
                if at > fed + n || n < 2 || (!rel.is_multiple_of(SNAP_GRID) && rel != n) {
                    break;
                }
                self.running[ri].snap_at.remove(0);
                let prefix = self.running[ri].req.prompt[..at].to_vec();
                if let Some(slot) = self.reserve_snap(self.running[ri].req.adapter, &prefix) {
                    step.snaps.push((si, at, slot));
                }
            }
        }
        // Grammar masks for constrained rows whose pick is a token, on all
        // cores (each takes tens of microseconds). On a grammar error `fill`
        // allows nothing and the token's check in `finish` reports it.
        let picks: HashSet<usize> = step
            .rows
            .iter()
            .filter(|&&(ri, n)| {
                let s = &self.running[ri];
                s.req.constraint.is_some() && s.fed + n >= s.total()
            })
            .map(|&(ri, _)| ri)
            .collect();
        if !picks.is_empty() {
            use rayon::prelude::*;
            let mut todo: Vec<&mut Seq> = self
                .running
                .iter_mut()
                .enumerate()
                .filter(|(ri, _)| picks.contains(ri))
                .map(|(_, s)| s)
                .collect();
            let mask = |s: &mut &mut Seq| {
                s.mask = s
                    .req
                    .constraint
                    .as_mut()
                    .and_then(|m| m.compute_mask().ok());
            };
            if todo.len() > 2 {
                todo.par_iter_mut().for_each(mask);
            } else {
                todo.iter_mut().for_each(mask);
            }
        }
        step.copies = std::mem::take(&mut self.pending_copies);
        step.lora_loads = std::mem::take(&mut self.lora.loads);
        step.lora_tiles = step
            .rows
            .iter()
            .filter(|&&(ri, _)| self.running[ri].lora != 0)
            .map(|&(_, n)| n.div_ceil(crate::lora::TILE))
            .sum();
        Some(step)
    }

    /// A free snapshot slot for a new cache entry of `prefix`, evicting the
    /// least recently used entry if needed; None if the entry exists.
    fn reserve_snap(&mut self, adapter: u32, prefix: &[u32]) -> Option<usize> {
        let h = *unit_hashes(adapter, prefix).last()?;
        if self.prefixes.entries.contains_key(&h) {
            return None;
        }
        if self.prefixes.free_snaps.is_empty() && !self.evict_one() {
            return None;
        }
        self.prefixes.free_snaps.pop()
    }

    /// Makes the first `len` prompt tokens of running sequence `ri` a cache
    /// entry whose state is (or will be, before the next step) in `snap`.
    fn add_entry(&mut self, ri: usize, len: usize, snap: usize) {
        let s = &self.running[ri];
        let adapter = s.req.adapter;
        let Some(&h) = unit_hashes(adapter, &s.req.prompt[..len]).last() else {
            self.prefixes.free_snaps.push(snap);
            return;
        };
        let blocks = s.blocks[..len / BLOCK_SIZE].to_vec();
        let tokens = s.req.prompt[..len].to_vec();
        for &b in &blocks {
            self.block_refs[b as usize] += 1;
        }
        self.prefixes.clock += 1;
        self.prefixes.entries.insert(
            h,
            Entry {
                adapter,
                tokens,
                blocks,
                snap,
                used: self.prefixes.clock,
            },
        );
    }

    /// Writes the step's batch metadata (see [`Meta`]) into `m`, and the
    /// rows' logit adjustments into `pen`. Returns the number of adjustments.
    pub fn fill(
        &self,
        step: &Step,
        m: &mut [i32],
        pen: &mut Pen,
        masks: &mut [u32],
    ) -> (usize, usize) {
        let me = &self.meta;
        let mut t = 0;
        let mut e = 0;
        let mut tiles = 0;
        let mut nmask = 0;
        let words = masks.len() / me.max_seqs.max(1);
        m[me.cu()] = 0;
        m[me.pen_cu()] = 0;
        m[me.ocu()] = 0;
        for (si, &(ri, n)) in step.rows.iter().enumerate() {
            let s = &self.running[ri];
            let verifies = s.verifies(n);
            let o0 = step.outs_at[si];
            let plen = s.req.prompt.len();
            let image_row = |p: usize| match (s.pool, s.image_rows.get(p)) {
                (Some((start, _)), Some(&r)) if r >= 0 => start as i32 + r,
                _ => -1,
            };
            if s.lora != 0 {
                for k in (0..n).step_by(crate::lora::TILE) {
                    let at = me.lora() + 1 + 3 * tiles;
                    m[at] = (t + k) as i32;
                    m[at + 1] = (n - k).min(crate::lora::TILE) as i32;
                    m[at + 2] = s.lora as i32;
                    tiles += 1;
                }
            }
            for k in 0..n {
                let p = s.fed + k;
                m[me.ids() + t] = if verifies && k > 0 {
                    s.drafts[k - 1]
                } else {
                    s.token(p)
                } as i32;
                m[me.pos() + t] = p as i32;
                m[me.slot_map() + t] =
                    (s.blocks[p / BLOCK_SIZE] as usize * BLOCK_SIZE + p % BLOCK_SIZE) as i32;
                m[me.tok_seq() + t] = si as i32;
                m[me.emb_src() + t] = image_row(p);
                // The draft layer reads the token after this one: a prompt
                // token, or the pick at this token's output row.
                (m[me.mtp_ids() + t], m[me.mtp_src() + t]) = if p + 1 < plen {
                    (s.req.prompt[p + 1] as i32, image_row(p + 1))
                } else {
                    (-((o0 + if verifies { k } else { 0 }) as i32) - 1, -1)
                };
                let rope = match &s.req.rope {
                    Some((pos, _)) if p < pos.len() => pos[p],
                    Some((_, delta)) => [p as i32 + delta; 3],
                    None => [p as i32; 3],
                };
                for (axis, r) in rope.into_iter().enumerate() {
                    m[me.rope() + axis * me.max_tokens + t] = r;
                }
                t += 1;
            }
            m[me.cu() + si + 1] = t as i32;
            m[me.slots() + si] = s.slot as i32;
            m[me.fresh() + si] = (s.fed == 0) as i32;
            let outs = me.outs() + si * me.spec;
            if verifies {
                for (k, &slot) in s.outs.iter().enumerate() {
                    m[outs + k] = slot as i32;
                }
            } else {
                m[outs] = -1;
            }
            let nout = if verifies { n } else { 1 };
            m[me.ocu() + si + 1] = (o0 + nout) as i32;
            for j in 0..nout {
                m[me.masks() + o0 + j] = -1;
            }
            // A constrained sequence never verifies drafts: one output row.
            if let Some(v) = &s.mask {
                masks[nmask * words..(nmask + 1) * words].copy_from_slice(&v.as_slice()[..words]);
                m[me.masks() + o0] = nmask as i32;
                nmask += 1;
            } else if s.req.constraint.is_some() && s.fed + n >= s.total() {
                // The grammar failed: allow nothing; `finish` reports it.
                masks[nmask * words..(nmask + 1) * words].fill(0);
                m[me.masks() + o0] = nmask as i32;
                nmask += 1;
            }
            let sp = &s.req.sampling;
            for j in 0..nout {
                let o = o0 + j;
                m[me.last() + o] = if verifies { t - n + j } else { t - 1 } as i32;
                m[me.temps() + o] = sp.temperature.max(0.0).to_bits() as i32;
                let g = (s.generated.len() + j) as u64;
                let seed = sp.seed ^ g.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                m[me.seeds() + o] = (seed ^ (seed >> 32)) as i32;
                m[me.top_ks() + o] = sp.top_k as i32;
                m[me.top_ps() + o] = sp.top_p.to_bits() as i32;
                m[me.min_ps() + o] = sp.min_p.to_bits() as i32;
                if sp.adjusts_logits() && s.fed + n >= s.total() {
                    e = pen.push_seq(e, s);
                }
                m[me.pen_cu() + o + 1] = e as i32;
            }
            m[me.limits() + si] = (s.blocks.len() * BLOCK_SIZE) as i32 - 1;
            let snaps = me.snaps() + 4 * si;
            m[snaps..snaps + 4].fill(-1);
            for (j, &(_, at, slot)) in step.snaps.iter().filter(|x| x.0 == si).enumerate() {
                m[snaps + 2 * j] = (at - s.fed) as i32;
                m[snaps + 2 * j + 1] = slot as i32;
            }
            let row = me.block_table() + si * me.max_blocks;
            for (dst, &b) in m[row..row + s.blocks.len()].iter_mut().zip(&s.blocks) {
                *dst = b as i32;
            }
        }
        m[me.lora()] = tiles as i32;
        (e, nmask)
    }

    /// Output rows of `step` whose requests want log-probabilities, and how
    /// many top tokens each wants.
    pub fn logprob_rows(&self, step: &Step) -> Vec<(usize, usize)> {
        step.rows
            .iter()
            .enumerate()
            .filter_map(|(si, &(ri, n))| {
                let s = &self.running[ri];
                let k = s.req.logprobs?;
                (s.fed + n >= s.total() && s.forced.is_none()).then_some((step.outs_at[si], k))
            })
            .collect()
    }

    /// Writes padding sequences [from, to) for a graph replay, of `n`
    /// tokens each (1, or `spec` for a verification graph): sequences on the
    /// scratch slot that write no KV and whose picks are ignored.
    pub fn pad(&self, m: &mut [i32], from: usize, to: usize, n: usize) {
        let me = &self.meta;
        let scratch = self.scratch_slot() as i32;
        if from == 0 {
            m[me.cu()] = 0;
            m[me.pen_cu()] = 0;
            m[me.ocu()] = 0;
            m[me.lora()] = 0;
        }
        for i in from..to {
            for k in 0..n {
                // With n > 1 every token is an output row, so row == token.
                let t = i * n + k;
                m[me.ids() + t] = 0;
                m[me.pos() + t] = k as i32;
                m[me.slot_map() + t] = -1;
                m[me.tok_seq() + t] = i as i32;
                m[me.emb_src() + t] = -1;
                for axis in 0..3 {
                    m[me.rope() + axis * me.max_tokens + t] = k as i32;
                }
                m[me.mtp_ids() + t] = -(t as i32) - 1;
                m[me.mtp_src() + t] = -1;
                m[me.last() + t] = t as i32;
                m[me.masks() + t] = -1;
                m[me.temps() + t] = 0;
                m[me.seeds() + t] = 0;
                m[me.top_ks() + t] = 0;
                m[me.top_ps() + t] = 1f32.to_bits() as i32;
                m[me.min_ps() + t] = 0;
                m[me.pen_cu() + t + 1] = m[me.pen_cu() + t];
                m[me.outs() + i * me.spec + k] = if n > 1 { scratch } else { -1 };
            }
            m[me.cu() + i + 1] = ((i + 1) * n) as i32;
            m[me.ocu() + i + 1] = ((i + 1) * n) as i32;
            m[me.slots() + i] = scratch;
            m[me.fresh() + i] = 1;
            m[me.limits() + i] = -1;
            m[me.snaps() + 4 * i..me.snaps() + 4 * i + 4].fill(-1);
            m[me.block_table() + i * me.max_blocks..me.block_table() + (i + 1) * me.max_blocks]
                .fill(0);
        }
    }

    /// Whether row `si` of the step yields a token (it is not mid-prompt).
    pub fn yields(&self, step: &Step, si: usize) -> bool {
        let (ri, n) = step.rows[si];
        let s = &self.running[ri];
        s.fed + n >= s.total()
    }

    pub fn is_forced(&self, step: &Step, si: usize) -> bool {
        self.running[step.rows[si].0].forced.is_some()
    }

    /// Applies the step's picks, per output row, and each sequence's next
    /// draft (`drafts`, with multi-token prediction); `scored[si]` carries
    /// the forced rows' scores. Streams tokens and returns the sequences that
    /// finished.
    pub fn finish(
        &mut self,
        step: Step,
        picked: &[u32],
        drafts: Option<&[i32]>,
        mut scored: Vec<Option<Scored>>,
        logprobs: &mut [Option<TokenLogprobs>],
    ) -> Vec<Seq> {
        self.counters.steps += 1;
        for &(si, len, snap) in &step.snaps {
            self.add_entry(step.rows[si].0, len, snap);
        }
        let mut done = Vec::new();
        for (si, &(ri, n)) in step.rows.iter().enumerate() {
            let o0 = step.outs_at[si];
            let s = &mut self.running[ri];
            let verifies = s.verifies(n);
            // Tokens whose state stands: all of a prompt chunk; the last
            // token and the accepted drafts of a verifying row.
            let (fed, toks) = if verifies {
                let mut j = 0;
                while j < n - 1 && picked[o0 + j] == s.drafts[j] {
                    j += 1;
                }
                self.counters.drafted += (n - 1) as u64;
                self.counters.accepted += j as u64;
                // Keep the slot holding the state after the last accepted
                // token; return the others.
                s.slot = s.outs[j];
                for (k, &slot) in s.outs.iter().enumerate() {
                    if k != j {
                        self.free_slots.push(slot);
                    }
                }
                s.outs.clear();
                (j + 1, picked[o0..=o0 + j].to_vec())
            } else {
                (n, vec![picked[o0]])
            };
            if s.in_prompt() {
                self.counters.prompt_tokens += n as u64;
            } else {
                self.counters.decode_tokens += fed as u64;
            }
            s.fed += fed;
            s.drafts.clear();
            if s.snap_at.first() == Some(&s.fed) {
                s.snap_at.remove(0);
                self.keep_prefix(ri);
            }
            let s = &mut self.running[ri];
            // Image embeddings are read only by the prompt's own steps.
            if s.fed >= s.req.prompt.len()
                && let Some(range) = s.pool.take()
            {
                s.image_rows = Vec::new();
                self.free_pool(range);
            }
            let s = &mut self.running[ri];
            if s.fed < s.total() {
                continue; // mid-prompt chunk: its pick is not a token yet
            }
            let mut finished = false;
            s.mask = None;
            for (j, pick) in toks.into_iter().enumerate() {
                let tok = match &s.forced {
                    Some(f) => {
                        if let Some(sc) = scored.get_mut(si).and_then(Option::take) {
                            s.scored.push(sc);
                        }
                        f[s.generated.len()]
                    }
                    None => pick,
                };
                let mut stop = s.forced.is_none() && self.eos.contains(&tok);
                if let Some(m) = s.req.constraint.as_mut() {
                    if stop {
                        // The end of text ends the grammar too; it must be complete.
                        if !m.is_accepting().unwrap_or(false) {
                            let _ = s.req.tx.send(Event::Error(
                                "the output does not complete the requested format".into(),
                            ));
                            finished = true;
                            break;
                        }
                    } else if m.consume_token(tok).is_err() {
                        let _ = s.req.tx.send(Event::Error(format!(
                            "structured output failed: {}",
                            m.get_error().unwrap_or_default()
                        )));
                        finished = true;
                        break;
                    }
                }
                if !stop {
                    s.generated.push(tok);
                    if s.req.sampling.adjusts_logits() {
                        *s.counts.entry(tok).or_default() += 1;
                    }
                    if let Some(lp) = logprobs.get_mut(o0 + j).and_then(Option::take) {
                        let _ = s.req.tx.send(Event::Logprobs(lp));
                    }
                    let _ = s.req.tx.send(Event::Token(tok));
                    // A grammar that admits nothing more ends the output.
                    stop = s.req.constraint.as_ref().is_some_and(|m| m.is_stopped());
                }
                if stop || s.generated.len() >= s.req.max_new {
                    let _ = s.req.tx.send(Event::Done {
                        finish: if stop { Finish::Stop } else { Finish::Length },
                        prompt_tokens: s.req.prompt.len(),
                        completion_tokens: s.generated.len(),
                    });
                    finished = true;
                    break;
                }
            }
            if finished {
                done.push(ri);
            } else if let Some(d) = drafts
                && self.meta.spec > 1
                && s.forced.is_none()
                && !s.req.sampling.adjusts_logits()
                && s.req.logprobs.is_none()
                && s.req.constraint.is_none()
            {
                let n = step.rows.len();
                s.drafts
                    .extend((0..self.meta.spec - 1).map(|k| d[k * n + si] as u32));
            }
        }
        done.sort_unstable_by(|a, b| b.cmp(a));
        done.into_iter()
            .map(|ri| {
                let s = self.running.swap_remove(ri);
                self.release(s)
            })
            .collect()
    }

    /// Makes running sequence `ri`'s fed prompt prefix a cache entry: its
    /// blocks gain a reference and its state is copied to a snapshot slot
    /// before the next step.
    fn keep_prefix(&mut self, ri: usize) {
        let len = self.running[ri].fed;
        let prefix = self.running[ri].req.prompt[..len].to_vec();
        if let Some(snap) = self.reserve_snap(self.running[ri].req.adapter, &prefix) {
            self.pending_copies.push((self.running[ri].slot, snap));
            self.add_entry(ri, len, snap);
        }
    }

    /// Fails every request, as when the GPU stops working.
    pub fn fail_all(&mut self, message: &str) {
        for s in self.running.drain(..).chain(self.waiting.drain(..)) {
            let _ = s.req.tx.send(Event::Error(message.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    const EOS: u32 = 99;

    fn sched(max_seqs: usize, max_tokens: usize, blocks: usize) -> Scheduler {
        let meta = Meta {
            max_tokens,
            max_seqs,
            max_blocks: 8,
            spec: 1,
        };
        Scheduler::new(meta, 8 * BLOCK_SIZE, blocks, vec![EOS])
    }

    fn request(prompt: usize, max_new: usize) -> (Request, UnboundedReceiver<Event>) {
        let (tx, rx) = unbounded_channel();
        let req = Request::text(
            (1..=prompt as u32).collect(),
            max_new,
            Sampling::default(),
            tx,
        );
        (req, rx)
    }

    fn tokens(rx: &mut UnboundedReceiver<Event>) -> (Vec<u32>, Option<Finish>) {
        let (mut out, mut finish) = (Vec::new(), None);
        while let Ok(ev) = rx.try_recv() {
            match ev {
                Event::Token(t) => out.push(t),
                Event::Logprobs(_) => {}
                Event::Done { finish: f, .. } => finish = Some(f),
                Event::Error(e) => panic!("{e}"),
            }
        }
        (out, finish)
    }

    /// Runs a step whose every pick is `pick`.
    fn step(s: &mut Scheduler, pick: u32) -> (usize, usize, Vec<Seq>) {
        let step = s.plan().expect("work");
        let mut m = vec![0; s.meta.words()];
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        let (t, n) = (step.tokens, step.seqs());
        let done = s.finish(step, &vec![pick; n], None, Vec::new(), &mut []);
        (t, n, done)
    }

    #[test]
    fn prompts_sharing_everything_before_the_last_message_reuse_it() {
        let meta = Meta {
            max_tokens: 1024,
            max_seqs: 4,
            max_blocks: 64,
            spec: 1,
        };
        let mut s =
            Scheduler::new(meta, 64 * BLOCK_SIZE, 64, vec![EOS]).with_prefix_cache(5..9, 999);
        // A 200-token system prompt, the turn token, then a question.
        let ask = |q: u32| -> Vec<u32> {
            let mut p: Vec<u32> = (1..=200).collect();
            p.push(999);
            p.extend(q..q + 100);
            p
        };
        let (tx, _ra) = unbounded_channel();
        s.submit(Request::text(ask(1000), 1, Sampling::default(), tx));
        // One step keeps the state at 192 (the unit before the last message)
        // and 256 (the last whole unit).
        let st = s.plan().expect("work");
        assert_eq!(st.tokens, 301);
        let at: Vec<usize> = st.snaps.iter().map(|x| x.1).collect();
        assert_eq!(at, vec![192, 256]);
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        assert_eq!(m[s.meta.snaps()], 192);
        assert_eq!(m[s.meta.snaps() + 2], 256);
        s.finish(st, &[7], None, Vec::new(), &mut []);
        let (tx, _rb) = unbounded_channel();
        s.submit(Request::text(ask(2000), 1, Sampling::default(), tx));
        let st = s.plan().expect("work");
        assert_eq!(s.counters.cached_tokens, 192, "the system prompt's units");
        assert_eq!(st.tokens, 301 - 192, "the rest in one step");
        let at: Vec<usize> = st.snaps.iter().map(|x| x.1).collect();
        assert_eq!(at, vec![256], "keeping its own last unit");
    }

    #[test]
    fn prefixes_are_kept_reused_and_evicted() {
        // 4 sequence slots, then snapshot slots 5 and 6.
        let meta = Meta {
            max_tokens: 512,
            max_seqs: 4,
            max_blocks: 64,
            spec: 1,
        };
        let mut s = Scheduler::new(meta, 64 * BLOCK_SIZE, 64, vec![EOS]).with_prefix_cache(5..7, 0);
        let base: Vec<u32> = (1..=150).collect();
        let (tx, mut ra) = unbounded_channel();
        s.submit(Request::text(base.clone(), 2, Sampling::default(), tx));
        // One step runs the whole prompt and keeps the state at the
        // 128-token boundary in snapshot slot 5.
        let st = s.plan().expect("work");
        assert_eq!(st.tokens, 150);
        assert_eq!(st.snaps, vec![(0, 128, 5)]);
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        assert_eq!(&m[s.meta.snaps()..s.meta.snaps() + 4], &[128, 5, -1, -1]);
        s.finish(st, &[7], None, Vec::new(), &mut []);
        step(&mut s, 7);
        assert_eq!(tokens(&mut ra).0, vec![7, 7]);
        // The finished request's blocks past the prefix are free again; the
        // prefix's 8 blocks stay with the cache.
        assert_eq!(s.load().free_kv_blocks, 64 - 8);
        // A longer prompt with the same start reuses the prefix: its state is
        // restored and only the rest is computed.
        let mut longer = base.clone();
        longer.extend(500..600);
        let (tx, mut rb) = unbounded_channel();
        s.submit(Request::text(longer, 1, Sampling::default(), tx));
        let st = s.plan().expect("work");
        assert_eq!(st.copies, vec![(5, 0)]);
        assert_eq!(st.tokens, 250 - 128, "the rest in one step");
        assert_eq!(st.snaps, vec![(0, 192, 6)], "keeping its own boundary");
        assert_eq!(s.counters.cached_tokens, 128);
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        // Positions continue after the prefix, in the cache's blocks; the
        // snapshot is 64 tokens into this row.
        assert_eq!(m[s.meta.pos()], 128);
        assert_eq!(m[s.meta.snaps()], 64);
        s.finish(st, &[3], None, Vec::new(), &mut []);
        assert_eq!(tokens(&mut rb).0, vec![3]);
        // Both entries hold blocks; the second shares the first's 8.
        assert_eq!(s.load().free_kv_blocks, 64 - 12);
        // A request needing every block evicts the entries.
        let (tx, _rc) = unbounded_channel();
        s.submit(Request::text(
            (900..1000).collect(),
            64 * 16 - 100,
            Sampling::default(),
            tx,
        ));
        let st = s.plan().expect("admitted after eviction");
        assert_eq!(st.seqs(), 1);
        assert_eq!(s.load().free_kv_blocks, 0);
    }

    #[test]
    fn prompts_are_chunked_and_decodes_join_the_next_step() {
        let mut s = sched(4, 8, 64);
        let (a, mut ra) = request(5, 3);
        let (b, mut rb) = request(6, 3);
        s.submit(a);
        s.submit(b);
        // 5 + 3 of 6 prompt tokens: the budget is 8.
        assert_eq!(step(&mut s, 7).0, 8);
        assert_eq!(tokens(&mut ra).0, vec![7]);
        assert!(tokens(&mut rb).0.is_empty(), "b is mid-prompt");
        // a decodes one token; b finishes its prompt with 3.
        assert_eq!(step(&mut s, 8).0, 4);
        assert_eq!(tokens(&mut ra).0, vec![8]);
        assert_eq!(tokens(&mut rb).0, vec![8]);
    }

    #[test]
    fn metadata_describes_paged_positions() {
        let mut s = sched(2, 16, 64);
        let (a, _ra) = request(18, 4);
        s.submit(a);
        let step = s.plan().unwrap();
        let mut m = vec![0; s.meta.words()];
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        let me = s.meta;
        assert_eq!(step.tokens, 16);
        assert_eq!(&m[me.pos()..me.pos() + 3], &[0, 1, 2]);
        assert_eq!(m[me.cu() + 1], 16);
        assert_eq!(m[me.last()], 15);
        assert_eq!(m[me.fresh()], 1);
        // Tokens 0..16 fill the sequence's first block exactly.
        let first = m[me.block_table()];
        assert_eq!(m[me.slot_map() + 15], first * BLOCK_SIZE as i32 + 15);
        s.finish(step, &[0], None, Vec::new(), &mut []);
        let step = s.plan().unwrap();
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        assert_eq!(m[me.fresh()], 0, "second chunk continues the state");
        assert_eq!(m[me.pos()], 16);
        assert_eq!(
            m[me.slot_map()],
            m[me.block_table() + 1] * BLOCK_SIZE as i32
        );
    }

    #[test]
    fn eos_stops_without_emitting_and_frees_capacity() {
        let mut s = sched(1, 64, 64);
        let (a, mut ra) = request(4, 10);
        let (b, mut rb) = request(4, 2);
        s.submit(a);
        s.submit(b);
        assert_eq!(step(&mut s, 5).1, 1, "one slot: b waits");
        let (_, _, done) = step(&mut s, EOS);
        assert_eq!(done.len(), 1);
        assert_eq!(tokens(&mut ra), (vec![5], Some(Finish::Stop)));
        assert_eq!(s.load().free_kv_blocks, 64);
        step(&mut s, 6);
        step(&mut s, 6);
        assert_eq!(tokens(&mut rb), (vec![6, 6], Some(Finish::Length)));
        assert!(s.idle());
    }

    #[test]
    fn admission_reserves_the_whole_request() {
        // 2 blocks: one request of 20 prompt + 12 output tokens needs both.
        let mut s = sched(4, 64, 2);
        let (a, _ra) = request(20, 12);
        let (b, _rb) = request(4, 4);
        s.submit(a);
        s.submit(b);
        assert_eq!(step(&mut s, 1).1, 1);
        assert_eq!(s.load().waiting, 1);
    }

    #[test]
    fn oversized_prompts_and_cancelled_clients_are_dropped() {
        let mut s = sched(2, 64, 64);
        let (a, mut ra) = request(8 * BLOCK_SIZE, 1);
        s.submit(a);
        assert!(matches!(ra.try_recv(), Ok(Event::Error(_))));
        let (b, rb) = request(4, 4);
        s.submit(b);
        drop(rb);
        assert!(s.plan().is_none());
        assert_eq!(s.load().free_kv_blocks, 64);
    }

    #[test]
    fn forced_rows_feed_the_continuation() {
        let mut s = sched(2, 64, 64);
        let (a, mut ra) = request(4, 0);
        s.submit_forced(a, vec![11, 12]);
        let step1 = s.plan().unwrap();
        let sc = Scored {
            argmax: 3,
            top: vec![],
        };
        s.finish(step1, &[3], None, vec![Some(sc)], &mut []);
        let step2 = s.plan().unwrap();
        let mut m = vec![0; s.meta.words()];
        s.fill(&step2, &mut m, &mut Pen::split(&mut [0; 3 * 64]), &mut []);
        assert_eq!(m[s.meta.ids()], 11, "the forced token, not the pick");
        let done = s.finish(step2, &[4], None, vec![None], &mut []);
        assert_eq!(done[0].scored.len(), 1);
        assert_eq!(tokens(&mut ra), (vec![11, 12], Some(Finish::Length)));
    }

    #[test]
    fn penalties_cover_generated_tokens_and_bias() {
        let mut s = sched(2, 64, 64);
        let (mut a, _ra) = request(3, 8);
        a.sampling = Sampling {
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            repetition_penalty: 2.0,
            logit_bias: vec![(42, 3.0)],
            ..Sampling::default()
        };
        s.submit(a);
        let mut words = [0; 3 * 64];
        for pick in [7, 7] {
            let step = s.plan().unwrap();
            let mut m = vec![0; s.meta.words()];
            s.fill(&step, &mut m, &mut Pen::split(&mut words), &mut []);
            s.finish(step, &[pick], None, Vec::new(), &mut []);
        }
        let step = s.plan().unwrap();
        let mut m = vec![0; s.meta.words()];
        let (n, _) = s.fill(&step, &mut m, &mut Pen::split(&mut words), &mut []);
        // Prompt tokens 1..=3, generated 7 (twice) and the biased 42.
        assert_eq!(n, 5);
        assert_eq!(m[s.meta.pen_cu() + 1], 5);
        let pen = Pen::split(&mut words);
        let get = |t: i32| {
            let i = pen.tok[..n].iter().position(|&x| x == t).unwrap();
            (
                f32::from_bits(pen.add[i] as u32),
                f32::from_bits(pen.rep[i] as u32),
            )
        };
        assert_eq!(get(7), (-(0.5 + 0.25 * 2.0), 2.0));
        assert_eq!(get(1), (0.0, 2.0));
        assert_eq!(get(42), (3.0, 1.0));
    }

    #[test]
    fn padding_rows_point_at_the_scratch_slot() {
        let s = sched(4, 64, 64);
        let mut m = vec![0; s.meta.words()];
        s.pad(&mut m, 0, 3, 1);
        let me = s.meta;
        assert_eq!(&m[me.cu()..me.cu() + 4], &[0, 1, 2, 3]);
        assert_eq!(m[me.slots() + 2], 4);
        assert_eq!(m[me.slot_map() + 1], -1);
    }
}
