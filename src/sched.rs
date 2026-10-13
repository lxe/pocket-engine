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
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
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
/// sequence `rows[i].0`.
#[derive(Debug, Default)]
pub struct Step {
    rows: Vec<(usize, usize)>,
    pub tokens: usize,
    /// State slot copies (from, to) to make, in order, before the step runs:
    /// prefix-cache snapshots and restores.
    pub copies: Vec<(usize, usize)>,
}

impl Step {
    pub fn seqs(&self) -> usize {
        self.rows.len()
    }
    pub fn decode_only(&self) -> bool {
        self.tokens == self.rows.len()
    }
}

#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Counters {
    pub steps: u64,
    pub decode_tokens: u64,
    pub prompt_tokens: u64,
    /// Prompt tokens served from the prefix cache instead of computed.
    pub cached_tokens: u64,
}

/// Prompt tokens per prefix-cache unit: entries cover whole units, a
/// multiple of the KV block.
pub const CACHE_UNIT: usize = 64;

/// A cached prompt prefix: its tokens, the KV blocks that hold their keys
/// and values, and a state slot holding the recurrent state after them.
struct Entry {
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
/// [0, (k + 1) * CACHE_UNIT).
fn unit_hashes(tokens: &[u32]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut out = Vec::with_capacity(tokens.len() / CACHE_UNIT);
    let mut h = 0u64;
    for unit in tokens.as_chunks::<CACHE_UNIT>().0 {
        let mut st = std::collections::hash_map::DefaultHasher::new();
        h.hash(&mut st);
        unit.hash(&mut st);
        h = st.finish();
        out.push(h);
    }
    out
}

pub struct Scheduler {
    meta: Meta,
    image_token: u32,
    pool_free: Vec<(usize, usize)>,
    pool_rows: usize,
    max_context: usize,
    free_slots: Vec<usize>,
    free_blocks: Vec<u32>,
    /// References to each KV block, from sequences and cache entries; a block
    /// is free at zero.
    block_refs: Vec<u16>,
    total_blocks: usize,
    prefixes: Prefixes,
    /// State copies for the next step (see Step::copies).
    pending_copies: Vec<(usize, usize)>,
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
            pending_copies: Vec::new(),
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
        while self.running.len() < self.meta.max_seqs {
            let Some(seq) = self.waiting.front() else {
                break;
            };
            let total_blocks = (seq.req.prompt.len() + seq.req.max_new).div_ceil(BLOCK_SIZE);
            let rows: usize = seq.req.images.iter().map(Image::tokens).sum();
            // Plain text prompts may reuse and leave cached prefixes; images
            // and teacher-forced scoring do not.
            let cacheable = rows == 0
                && seq.forced.is_none()
                && (!self.prefixes.free_snaps.is_empty() || !self.prefixes.entries.is_empty());
            let hit = if cacheable {
                self.lookup(&seq.req.prompt)
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
            let mut seq = self.waiting.pop_front().expect("front exists");
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
    fn lookup(&self, prompt: &[u32]) -> Option<(u64, usize)> {
        let hashes = unit_hashes(&prompt[..prompt.len() - 1]);
        hashes.iter().enumerate().rev().find_map(|(k, h)| {
            let len = (k + 1) * CACHE_UNIT;
            self.prefixes
                .entries
                .get(h)
                .filter(|e| e.tokens[..] == prompt[..len])
                .map(|_| (*h, len))
        })
    }

    fn release(&mut self, mut seq: Seq) -> Seq {
        self.free_slots.push(seq.slot);
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
        let mut budget = self.meta.max_tokens;
        let mut step = Step::default();
        for (i, s) in self.running.iter().enumerate() {
            let mut n = (s.total() - s.fed).min(budget);
            if let Some(&at) = s.snap_at.first() {
                // Stop the chunk at the boundary, so the state can be kept.
                n = n.min(at - s.fed);
            }
            if n == 0 {
                break;
            }
            budget -= n;
            step.tokens += n;
            step.rows.push((i, n));
        }
        step.copies = std::mem::take(&mut self.pending_copies);
        Some(step)
    }

    /// Writes the step's batch metadata (see [`Meta`]) into `m`, and the
    /// rows' logit adjustments into `pen`. Returns the number of adjustments.
    pub fn fill(&self, step: &Step, m: &mut [i32], pen: &mut Pen) -> usize {
        let me = &self.meta;
        let mut t = 0;
        let mut e = 0;
        m[me.cu()] = 0;
        m[me.pen_cu()] = 0;
        for (si, &(ri, n)) in step.rows.iter().enumerate() {
            let s = &self.running[ri];
            for k in 0..n {
                let p = s.fed + k;
                m[me.ids() + t] = s.token(p) as i32;
                m[me.pos() + t] = p as i32;
                m[me.slot_map() + t] =
                    (s.blocks[p / BLOCK_SIZE] as usize * BLOCK_SIZE + p % BLOCK_SIZE) as i32;
                m[me.tok_seq() + t] = si as i32;
                m[me.emb_src() + t] = match (s.pool, s.image_rows.get(p)) {
                    (Some((start, _)), Some(&r)) if r >= 0 => start as i32 + r,
                    _ => -1,
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
            m[me.last() + si] = t as i32 - 1;
            let sp = &s.req.sampling;
            m[me.temps() + si] = sp.temperature.max(0.0).to_bits() as i32;
            let seed = sp.seed ^ (s.generated.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            m[me.seeds() + si] = (seed ^ (seed >> 32)) as i32;
            m[me.top_ks() + si] = sp.top_k as i32;
            m[me.top_ps() + si] = sp.top_p.to_bits() as i32;
            m[me.min_ps() + si] = sp.min_p.to_bits() as i32;
            if sp.adjusts_logits() && s.fed + n >= s.total() {
                e = pen.push_seq(e, s);
            }
            m[me.pen_cu() + si + 1] = e as i32;
            let row = me.block_table() + si * me.max_blocks;
            for (dst, &b) in m[row..row + s.blocks.len()].iter_mut().zip(&s.blocks) {
                *dst = b as i32;
            }
        }
        e
    }

    /// Writes padding rows [from, to) for a graph replay: one-token sequences
    /// on the scratch slot that write no KV and whose picks are ignored.
    pub fn pad(&self, m: &mut [i32], from: usize, to: usize) {
        let me = &self.meta;
        let scratch = self.scratch_slot() as i32;
        if from == 0 {
            m[me.cu()] = 0;
            m[me.pen_cu()] = 0;
        }
        for i in from..to {
            m[me.ids() + i] = 0;
            m[me.pos() + i] = 0;
            m[me.slot_map() + i] = -1;
            m[me.tok_seq() + i] = i as i32;
            m[me.emb_src() + i] = -1;
            for axis in 0..3 {
                m[me.rope() + axis * me.max_tokens + i] = 0;
            }
            m[me.cu() + i + 1] = i as i32 + 1;
            m[me.slots() + i] = scratch;
            m[me.fresh() + i] = 1;
            m[me.last() + i] = i as i32;
            m[me.temps() + i] = 0;
            m[me.seeds() + i] = 0;
            m[me.top_ks() + i] = 0;
            m[me.top_ps() + i] = 1f32.to_bits() as i32;
            m[me.min_ps() + i] = 0;
            m[me.pen_cu() + i + 1] = m[me.pen_cu() + i];
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

    /// Applies the step's picks, one per row; `scored[si]` carries the forced
    /// rows' scores. Streams tokens and returns the sequences that finished.
    pub fn finish(
        &mut self,
        step: Step,
        picked: &[u32],
        mut scored: Vec<Option<Scored>>,
    ) -> Vec<Seq> {
        self.counters.steps += 1;
        let mut done = Vec::new();
        for (si, &(ri, n)) in step.rows.iter().enumerate() {
            let s = &mut self.running[ri];
            if s.in_prompt() {
                self.counters.prompt_tokens += n as u64;
            } else {
                self.counters.decode_tokens += n as u64;
            }
            s.fed += n;
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
            let tok = match &s.forced {
                Some(f) => {
                    if let Some(sc) = scored.get_mut(si).and_then(Option::take) {
                        s.scored.push(sc);
                    }
                    f[s.generated.len()]
                }
                None => picked[si],
            };
            let stop = s.forced.is_none() && self.eos.contains(&tok);
            if !stop {
                s.generated.push(tok);
                if s.req.sampling.adjusts_logits() {
                    *s.counts.entry(tok).or_default() += 1;
                }
                let _ = s.req.tx.send(Event::Token(tok));
            }
            if stop || s.generated.len() >= s.req.max_new {
                let _ = s.req.tx.send(Event::Done {
                    finish: if stop { Finish::Stop } else { Finish::Length },
                    prompt_tokens: s.req.prompt.len(),
                    completion_tokens: s.generated.len(),
                });
                done.push(ri);
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
        let Some(&h) = unit_hashes(&self.running[ri].req.prompt[..len]).last() else {
            return;
        };
        if self.prefixes.entries.contains_key(&h) {
            return;
        }
        if self.prefixes.free_snaps.is_empty() && !self.evict_one() {
            return;
        }
        let Some(snap) = self.prefixes.free_snaps.pop() else {
            return;
        };
        let s = &self.running[ri];
        let blocks = s.blocks[..len / BLOCK_SIZE].to_vec();
        let tokens = s.req.prompt[..len].to_vec();
        self.pending_copies.push((s.slot, snap));
        for &b in &blocks {
            self.block_refs[b as usize] += 1;
        }
        self.prefixes.clock += 1;
        self.prefixes.entries.insert(
            h,
            Entry {
                tokens,
                blocks,
                snap,
                used: self.prefixes.clock,
            },
        );
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
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        let (t, n) = (step.tokens, step.seqs());
        let done = s.finish(step, &vec![pick; n], Vec::new());
        (t, n, done)
    }

    #[test]
    fn prompts_sharing_everything_before_the_last_message_reuse_it() {
        let meta = Meta {
            max_tokens: 1024,
            max_seqs: 4,
            max_blocks: 64,
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
        // Chunks stop at 192 (the unit before the last message) and 256.
        assert_eq!(step(&mut s, 7).0, 192);
        assert_eq!(step(&mut s, 7).0, 64);
        step(&mut s, 7);
        let (tx, _rb) = unbounded_channel();
        s.submit(Request::text(ask(2000), 1, Sampling::default(), tx));
        let st = s.plan().expect("work");
        assert_eq!(s.counters.cached_tokens, 192, "the system prompt's units");
        assert_eq!(
            st.tokens,
            301 - 192 - 45,
            "then up to its own boundary at 256"
        );
    }

    #[test]
    fn prefixes_are_kept_reused_and_evicted() {
        // 4 sequence slots, then snapshot slots 5 and 6.
        let meta = Meta {
            max_tokens: 512,
            max_seqs: 4,
            max_blocks: 64,
        };
        let mut s = Scheduler::new(meta, 64 * BLOCK_SIZE, 64, vec![EOS]).with_prefix_cache(5..7, 0);
        let base: Vec<u32> = (1..=150).collect();
        let (tx, mut ra) = unbounded_channel();
        s.submit(Request::text(base.clone(), 2, Sampling::default(), tx));
        // The prompt chunk stops at the 128-token boundary for the snapshot.
        let (t, _, _) = step(&mut s, 7);
        assert_eq!(t, 128);
        let st = s.plan().expect("work");
        assert_eq!(st.copies, vec![(0, 5)], "state kept after the boundary");
        assert_eq!(st.tokens, 22);
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        s.finish(st, &[7], Vec::new());
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
        assert_eq!(st.tokens, 192 - 128, "stops at its own boundary, 192");
        assert_eq!(s.counters.cached_tokens, 128);
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        // Positions continue after the prefix, in the cache's blocks.
        assert_eq!(m[s.meta.pos()], 128);
        s.finish(st, &[3], Vec::new());
        let st = s.plan().expect("work");
        assert_eq!(st.copies, vec![(0, 6)], "second entry");
        let mut m = vec![0; s.meta.words()];
        s.fill(&st, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        s.finish(st, &[3], Vec::new());
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
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        let me = s.meta;
        assert_eq!(step.tokens, 16);
        assert_eq!(&m[me.pos()..me.pos() + 3], &[0, 1, 2]);
        assert_eq!(m[me.cu() + 1], 16);
        assert_eq!(m[me.last()], 15);
        assert_eq!(m[me.fresh()], 1);
        // Tokens 0..16 fill the sequence's first block exactly.
        let first = m[me.block_table()];
        assert_eq!(m[me.slot_map() + 15], first * BLOCK_SIZE as i32 + 15);
        s.finish(step, &[0], Vec::new());
        let step = s.plan().unwrap();
        s.fill(&step, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
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
        s.finish(step1, &[3], vec![Some(sc)]);
        let step2 = s.plan().unwrap();
        let mut m = vec![0; s.meta.words()];
        s.fill(&step2, &mut m, &mut Pen::split(&mut [0; 3 * 64]));
        assert_eq!(m[s.meta.ids()], 11, "the forced token, not the pick");
        let done = s.finish(step2, &[4], vec![None]);
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
            s.fill(&step, &mut m, &mut Pen::split(&mut words));
            s.finish(step, &[pick], Vec::new());
        }
        let step = s.plan().unwrap();
        let mut m = vec![0; s.meta.words()];
        let n = s.fill(&step, &mut m, &mut Pen::split(&mut words));
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
        s.pad(&mut m, 0, 3);
        let me = s.meta;
        assert_eq!(&m[me.cu()..me.cu() + 4], &[0, 1, 2, 3]);
        assert_eq!(m[me.slots() + 2], 4);
        assert_eq!(m[me.slot_map() + 1], -1);
    }
}
