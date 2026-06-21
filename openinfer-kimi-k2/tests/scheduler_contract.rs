//! CPU scheduler-contract tests for Kimi-K2 (issue #222).
//!
//! These run the real `KimiK2Scheduler` admission/decode loop with no GPUs,
//! weights, or collectives — the GPU forward is faked through the
//! `test-harness` seam (`drive_scheduler_batch`), which maps each row's input
//! token to the token the engine would produce next. This is the "scheduler /
//! contract robustness cases that run without the full 8-GPU model" half of
//! #222; `tests/e2e_scheduler.rs` is the on-hardware half.
//!
//! The fake engine uses `next = input + 1` so token chains are easy to read,
//! and the stop set is chosen against that chain to drive each finish path.
//!
//! Build/run: `cargo test -p openinfer-kimi-k2 --features kimi-k2,test-harness
//! --test scheduler_contract`.

use openinfer_core::engine::{
    FinishReason, GenerateRequest, TokenEvent, TokenSink, TokenStreamReceiver,
};
use openinfer_core::sampler::SamplingParams;

/// 1024-block pool: large enough that admission never defers in these cases.
const ROOMY_POOL: usize = 1024;
/// Per-request KV cap mirrored from `runner/worker.rs`.
const MAX_REQUEST_TOKENS: usize = 8192;

/// `next = input + 1`; used by every scenario that reaches a forward.
fn increment(input: u32) -> u32 {
    input + 1
}

/// Never reached (rejection/finish-before-forward scenarios).
fn unused(input: u32) -> u32 {
    input
}

fn request(
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
    echo: bool,
) -> (GenerateRequest, TokenStreamReceiver) {
    let (token_tx, rx) = TokenSink::standalone();
    let req = GenerateRequest {
        request_id: None,
        queued_at_unix_s: None,
        prompt_tokens,
        params,
        max_tokens,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo,
    };
    (req, rx)
}

fn greedy(prompt_tokens: Vec<u32>, max_tokens: usize) -> (GenerateRequest, TokenStreamReceiver) {
    request(prompt_tokens, max_tokens, SamplingParams::default(), false)
}

/// `drive_scheduler_batch` runs the whole prefill→decode loop synchronously, so
/// by the time it returns every event is queued; drain without blocking.
fn drain(rx: &mut TokenStreamReceiver) -> Vec<TokenEvent> {
    let mut events = Vec::new();
    while let Ok((_, event)) = rx.try_recv() {
        events.push(event);
    }
    events
}

fn emitted_tokens(events: &[TokenEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|event| match event {
            TokenEvent::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

fn finished(events: &[TokenEvent]) -> Option<(FinishReason, usize)> {
    match events.last() {
        Some(TokenEvent::Finished {
            finish_reason,
            completion_tokens,
            ..
        }) => Some((*finish_reason, *completion_tokens)),
        _ => None,
    }
}

fn rejection_message(events: &[TokenEvent]) -> &str {
    match events.last() {
        Some(TokenEvent::Rejected { message, .. }) => message,
        other => panic!("expected Rejected, got {other:?}"),
    }
}

#[test]
fn length_finish_streams_every_generated_token() {
    let (req, mut rx) = greedy(vec![10, 20], 2);
    let deferred =
        openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], increment);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    // prefill 20→21 (streamed), decode 21→22 (streamed), completion 2 == max.
    assert_eq!(emitted_tokens(&events), vec![21, 22]);
    assert_eq!(finished(&events), Some((FinishReason::Length, 2)));
}

#[test]
fn eos_at_prefill_finishes_stop_and_streams_nothing() {
    // Prefill yields 21; with 21 in the stop set the request finishes before any
    // token is streamed.
    let (req, mut rx) = greedy(vec![10, 20], 5);
    let deferred =
        openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![21], vec![req], increment);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert!(
        emitted_tokens(&events).is_empty(),
        "the stop token is never streamed: {events:?}"
    );
    assert_eq!(finished(&events), Some((FinishReason::Stop, 0)));
}

#[test]
fn eos_at_decode_finishes_stop_after_prior_tokens() {
    // Prefill streams 21; the first decode step yields 22, which is the stop
    // token — streamed tokens stop at 21, but completion counts the EOS step.
    let (req, mut rx) = greedy(vec![10, 20], 5);
    let deferred =
        openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![22], vec![req], increment);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert_eq!(emitted_tokens(&events), vec![21]);
    assert_eq!(finished(&events), Some((FinishReason::Stop, 2)));
}

#[test]
fn concurrent_requests_retire_at_their_own_length() {
    // Two prompts share one decode batch but stop at different lengths: A after
    // 2 tokens, B after 3. A retires first (swap_remove) while B keeps decoding
    // in a shrinking batch. Disjoint token chains (20.. vs 40..) keep them
    // independently checkable.
    let (req_a, mut rx_a) = greedy(vec![10, 20], 2);
    let (req_b, mut rx_b) = greedy(vec![30, 40], 3);
    let deferred =
        openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req_a, req_b], increment);

    assert!(deferred.is_empty());

    let events_a = drain(&mut rx_a);
    assert_eq!(emitted_tokens(&events_a), vec![21, 22]);
    assert_eq!(finished(&events_a), Some((FinishReason::Length, 2)));

    let events_b = drain(&mut rx_b);
    assert_eq!(emitted_tokens(&events_b), vec![41, 42, 43]);
    assert_eq!(finished(&events_b), Some((FinishReason::Length, 3)));
}

#[test]
fn sampling_request_is_rejected_before_forward() {
    let params = SamplingParams {
        temperature: 0.7,
        top_k: -1,
        top_p: 0.9,
        ignore_eos: false,
    };
    let (req, mut rx) = request(vec![10, 20], 5, params, false);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], unused);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert!(
        rejection_message(&events).contains("TP8"),
        "rejection names the serving path: {events:?}"
    );
}

#[test]
fn echo_request_is_rejected_before_forward() {
    let (req, mut rx) = request(vec![10, 20], 5, SamplingParams::default(), true);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], unused);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert!(
        rejection_message(&events).contains("echo"),
        "rejection names the unsupported field: {events:?}"
    );
}

#[test]
fn over_capacity_request_is_rejected_before_forward() {
    // prompt (MAX+1) + max_tokens 1 needs MAX+1 KV tokens > per-request cap.
    let (req, mut rx) = greedy(vec![1u32; MAX_REQUEST_TOKENS + 1], 1);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], unused);

    assert!(
        deferred.is_empty(),
        "over-cap is a rejection, not a deferral"
    );
    let events = drain(&mut rx);
    assert!(
        rejection_message(&events).contains("per-request capacity"),
        "rejection names the limit: {events:?}"
    );
}

#[test]
fn empty_prompt_is_rejected_before_forward() {
    let (req, mut rx) = greedy(Vec::new(), 4);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], unused);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert!(
        rejection_message(&events).contains("at least one prompt token"),
        "rejection names the cause: {events:?}"
    );
}

#[test]
fn zero_max_tokens_finishes_length_before_any_forward() {
    let (req, mut rx) = greedy(vec![10, 20], 0);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(ROOMY_POOL, vec![], vec![req], unused);

    assert!(deferred.is_empty());
    let events = drain(&mut rx);
    assert!(emitted_tokens(&events).is_empty());
    assert_eq!(finished(&events), Some((FinishReason::Length, 0)));
}

#[test]
fn over_budget_request_is_deferred_silently() {
    // 4-block pool, 1 reserved for padding → budget 3 blocks. A 33-token prompt
    // (+1 max) needs ceil(34/16) = 3 blocks and is admitted, draining the
    // budget; the 1-token follow-up needs 1 more and is deferred with no event.
    let (big, _big_rx) = greedy((0..33).collect(), 1);
    let (small, mut small_rx) = greedy(vec![5], 1);
    let deferred = openinfer_kimi_k2::drive_scheduler_batch(4, vec![], vec![big, small], increment);

    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].prompt_tokens, vec![5]);
    assert!(
        drain(&mut small_rx).is_empty(),
        "deferral is silent: the request just waits for the next wave"
    );
}
