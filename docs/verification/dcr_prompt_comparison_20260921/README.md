# DCR smoke format comparison

Matched IDs 1, 2, 3 (15 turns); training combines original job 271289 with job 271383. All variants use identical base/adapter artifacts, per-turn reference histories and seeds, BF16, greedy decoding and a 1,024-new-token cap. Source hashes match saved manifests.

| Format | Nonempty replies | Empty replies | Cap hits | Replies containing extra role markers |
| --- | ---: | ---: | ---: | ---: |
| training | 15/15 | 0 | 14 | 11 |
| raw | 10/15 | 5 | 6 | 0 |
| zephyr | 15/15 | 0 | 8 | 4 |



Training-more job 271383 completed in 5m34s. Raw job 271384 completed all 15 attempts but exited 2 in 4m12s because five responses decoded to empty text. Zephyr job 271399 completed in 5m13s. The training and Zephyr output/GPU audits and judge dry-runs passed. No paid judging, retries or postprocessing were performed.

Raw-empty cases (one-based turns): dialogue 1 turn 2; dialogue 2 turns 2 and 3; dialogue 3 turns 2 and 3. The error handler discarded generation token IDs for these failures, so their exact token count/stop cause is unknown. Saved completion_tokens=0 is an error placeholder, not evidence that the model generated zero tokens.

The original training format shows persistent continuation/repetition beyond the first dialogue. Raw text does not reliably fix it. Zephyr-style text reduces cap hits and explicit extra-role continuations in this small sample but still permits both. Literal </s> is ordinary Qwen text, and at least one response emits it before continuing the conversation. No benchmark-wide quality or safety conclusion follows from these counts.

Full unmodified responses: [responses.md](responses.md). Exact metrics and file hashes: [comparison.json](comparison.json).
