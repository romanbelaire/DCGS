# Four-template DCR smoke comparison

Job **271466 completed 0:0 in 7m42s** on lagoon / NVIDIA L40. All 15
turns produced nonempty output. GPU/output audit and judge dry-run passed;
no paid judging was performed. Peak allocated GPU memory was 3,246,438,400
bytes (3.02 GiB); generation runtime including model load was 428.94 seconds.

The standard Qwen template did not fix the repetition. Every response hit
the 1,024-token cap, and multilingual repetitive text increased in this sample.

| Template | Nonempty replies | Empty failures | Cap hits | Replies with CJK characters | Replies with Arabic characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uploaded training | 15 | 0 | 14 | 1 | 0 |
| Raw text | 10 | 5 | 6 known | 0 | 0 |
| Literal Zephyr | 15 | 0 | 8 | 0 | 0 |
| Standard Qwen | 15 | 0 | 15 | 6 | 4 |

All rows cover the same 15 turns from dialogue IDs 1,2,3. Raw's five empty
replies lost their generation metadata in the earlier runner, so their cap
status is unknown. CJK/Arabic counts indicate character presence, not coherent
language use; some replies contain both scripts.

## Concrete observations

- Qwen dialogue 1 / turn 1 echoes the question, then repeats `话语权` more
  than 500 times. Dialogue 2 / turn 1 repeats `栒`; dialogue 3 / turn 1 repeats
  `猞猁`. All reach the cap.
- Qwen dialogue 1 / turn 3 loops on a follow-up question with Arabic characters;
  dialogue 3 / turns 4 and 5 also devolve into Arabic-character loops.
- The Qwen run generated zero `<|im_start|>` or `<|im_end|>` special tokens
  and zero endoftext EOS endings across all 15 completions. Adding im_end as
  a stop would not intercept any of these saved sequences. The zero standard
  role-marker count does not mean the replies are repetition-free.
- Literal Zephyr remains better on cap hits in this selected smoke (8 versus
  Qwen's 15), with seven verified EOS endings and no CJK/Arabic characters.
  It still has repetition, four replies with extra roles and factual errors;
  this is not a quality/safety ranking or endorsement for a full run.

## Comparison integrity

Verified identical base/adapter artifact hashes, dataset hash, per-turn gold
history, current user/reference messages, per-turn seeds, BF16, greedy decoding,
1,024-token limit, revision and model. All saved runtime source hashes match.
The 31 prior evidence artifact hashes still match the earlier comparison.
New raw generation records match all 15 saved completion token sequences.
No prompt truncation, empty outputs, OOM or runtime exceptions occurred.
The stderr contains only a torch_dtype deprecation warning.

The standard Qwen template adds its default helpful-assistant system message.
It retains base-Qwen endoftext stopping, so this tests the exact prompt template
under matched generation settings, not Qwen-Instruct weights or decoding.
No inference reruns, repairs, retries or scheduler mutations were performed.

- [Exact metrics and source hashes](comparison.json)
- [All 60 unchanged responses](responses.md)
- [Fresh GPU/output audit](qwen_audit.json)
- [Reproducible read-only analysis](analyze.py)
