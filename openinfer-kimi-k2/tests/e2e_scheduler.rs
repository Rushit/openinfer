//! On-hardware E2E scheduler contract tests for Kimi-K2 (issue #222).
//!
//! Drives the full EngineHandle → KimiK2Scheduler path on the default TP8/DP1
//! NCCL shape and asserts the serving contract through the *real* wiring that a
//! mock cannot: echo rejection, sampling honor-or-reject, admission capacity,
//! consumer-drop safety, and finish-reason behaviour. The CPU-runnable half of
//! the contract lives in `tests/scheduler_contract.rs`; this is the half that
//! needs real weights and GPUs.
//!
//! Requires 8 GPUs and Kimi-K2 weights; skips cleanly when either is absent.
//! Set OPENINFER_TEST_MODEL_PATH to the weight directory to run.

use std::path::Path;
use std::time::Duration;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, EpBackend, FinishReason, GenerateRequest, TokenEvent,
    TokenSink, TokenStreamReceiver,
};
use openinfer_core::sampler::SamplingParams;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2");

/// KV token cap mirrored from `runner/worker.rs`; the gate asserts this
/// boundary so a change to the constant shows up here as a test failure.
const MAX_REQUEST_TOKENS: usize = 8192;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DEFAULT_MODEL_PATH).join("config.json").exists() => {
            Some(DEFAULT_MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 e2e_scheduler: {DEFAULT_MODEL_PATH}/config.json not found; \
                 set OPENINFER_TEST_MODEL_PATH to run"
            );
            None
        }
    }
}

fn start_engine_or_skip(model_path: &str) -> Option<EngineHandle> {
    match openinfer_kimi_k2::start_engine(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: (0..8).collect(),
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
    ) {
        Ok(handle) => Some(handle),
        Err(err) => {
            eprintln!("skipping kimi-k2 e2e_scheduler: engine startup failed: {err:#}");
            None
        }
    }
}

fn submit(
    handle: &EngineHandle,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
    echo: bool,
) -> TokenStreamReceiver {
    let (token_tx, token_rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo,
        })
        .expect("submit failed");
    token_rx
}

/// Drains `rx` into a `Vec`; the last element is always a terminal event
/// (`Finished`, `Rejected`, or `Error`).
fn drain(rx: &mut TokenStreamReceiver) -> Vec<TokenEvent> {
    let mut events = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(event) => {
                let done = matches!(
                    event,
                    TokenEvent::Finished { .. }
                        | TokenEvent::Rejected { .. }
                        | TokenEvent::Error { .. }
                );
                events.push(event);
                if done {
                    return events;
                }
            }
            None => panic!("scheduler channel closed without terminal event"),
        }
    }
}

#[test]
fn test_kimi_k2_scheduler_contract() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let Some(handle) = start_engine_or_skip(&model_path) else {
        return;
    };

    // ── 1. Echo rejection ────────────────────────────────────────────────────
    eprintln!("=== Phase 1: echo rejection ===");
    {
        let mut rx = submit(&handle, vec![1, 2, 3], 5, SamplingParams::default(), true);
        let events = drain(&mut rx);
        let Some(TokenEvent::Rejected { message, .. }) = events
            .iter()
            .find(|e| matches!(e, TokenEvent::Rejected { .. }))
        else {
            panic!("expected Rejected, got {events:?}");
        };
        assert!(
            message.to_lowercase().contains("echo"),
            "rejection should name the unsupported field: {message}"
        );
        eprintln!("  PASS: echo=true → Rejected(\"{message}\")");
    }

    // ── 2. Non-greedy rejection on TP8 path ──────────────────────────────────
    // The TP8/DP1 path cannot sample the global distribution (per-rank vocab
    // shard, #226), so sampling is rejected here by design (#237). On TP1/DP8
    // sampling is honored instead — this assertion is specific to the TP8 shape
    // this test starts.
    eprintln!("=== Phase 2: non-greedy rejection (TP8) ===");
    {
        let mut rx = submit(
            &handle,
            vec![1, 2, 3],
            5,
            SamplingParams {
                temperature: 0.7,
                top_k: -1,
                top_p: 0.9,
                ignore_eos: false,
            },
            false,
        );
        let events = drain(&mut rx);
        let Some(TokenEvent::Rejected { message, .. }) = events
            .iter()
            .find(|e| matches!(e, TokenEvent::Rejected { .. }))
        else {
            panic!("expected Rejected, got {events:?}");
        };
        assert!(
            message.contains("TP8"),
            "rejection should name the serving path: {message}"
        );
        eprintln!("  PASS: temperature=0.7 on TP8 → Rejected(\"{message}\")");
    }

    // ── 3. Over-capacity rejection ───────────────────────────────────────────
    eprintln!("=== Phase 3: over-capacity rejection ===");
    {
        // MAX_REQUEST_TOKENS + 1 prompt tokens, max_tokens=1 → KV = MAX+1 > cap
        let mut rx = submit(
            &handle,
            vec![1u32; MAX_REQUEST_TOKENS + 1],
            1,
            SamplingParams::default(),
            false,
        );
        let events = drain(&mut rx);
        let Some(TokenEvent::Rejected { message, .. }) = events
            .iter()
            .find(|e| matches!(e, TokenEvent::Rejected { .. }))
        else {
            panic!("expected Rejected, got {events:?}");
        };
        assert!(
            message.contains("per-request capacity"),
            "rejection should name the limit: {message}"
        );
        eprintln!(
            "  PASS: {}-token prompt → Rejected(\"{message}\")",
            MAX_REQUEST_TOKENS + 1
        );
    }

    // ── 4. Consumer drop ─────────────────────────────────────────────────────
    eprintln!("=== Phase 4: consumer drop ===");
    {
        let (token_tx, rx) = TokenSink::standalone();
        drop(rx);
        handle
            .submit(GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                prompt_tokens: vec![1, 2, 3],
                params: SamplingParams::default(),
                max_tokens: 10,
                lora_adapter: None,
                token_tx,
                logprobs: 0,
                echo: false,
            })
            .expect("submit with dropped receiver failed");
        std::thread::sleep(Duration::from_millis(500));
        eprintln!("  PASS: scheduler handled dropped receiver without deadlock");
    }

    // Verify the engine still serves after the dropped-receiver request.
    {
        let mut rx = submit(&handle, vec![1, 2, 3], 1, SamplingParams::default(), false);
        let events = drain(&mut rx);
        assert!(
            matches!(events.last(), Some(TokenEvent::Finished { .. })),
            "engine must serve a request after consumer drop, got {events:?}"
        );
        eprintln!("  PASS: engine alive after consumer drop");
    }

    // ── 5. Length finish-reason ───────────────────────────────────────────────
    eprintln!("=== Phase 5: length termination ===");
    {
        const MAX_TOKENS: usize = 3;
        // ignore_eos so the model cannot terminate early via a stop token —
        // exactly MAX_TOKENS must be produced, making the Length finish
        // deterministic regardless of what the prompt decodes to.
        let mut rx = submit(
            &handle,
            vec![1, 2, 3, 4, 5],
            MAX_TOKENS,
            SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            false,
        );
        let events = drain(&mut rx);
        let token_count = events
            .iter()
            .filter(|e| matches!(e, TokenEvent::Token { .. }))
            .count();
        assert_eq!(
            token_count, MAX_TOKENS,
            "should emit exactly {MAX_TOKENS} tokens before Length termination"
        );
        match events.last() {
            Some(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                completion_tokens,
                ..
            }) => {
                assert_eq!(
                    *completion_tokens, MAX_TOKENS,
                    "completion_tokens should match max_tokens"
                );
            }
            other => panic!("expected Finished{{Length}}, got {other:?}"),
        }
        eprintln!("  PASS: max_tokens={MAX_TOKENS} → Finished{{Length}}");
    }

    // ── 6. Sequential requests (scheduler state reuse) ───────────────────────
    eprintln!("=== Phase 6: sequential requests ===");
    for i in 0..3usize {
        let mut rx = submit(
            &handle,
            vec![1u32 + i as u32, 2, 3],
            2,
            SamplingParams::default(),
            false,
        );
        let events = drain(&mut rx);
        assert!(
            matches!(events.last(), Some(TokenEvent::Finished { .. })),
            "sequential request {i} failed: {events:?}"
        );
        eprintln!("  PASS: sequential request {i}");
    }

    eprintln!("kimi-k2 e2e_scheduler: all contract tests passed");
}
