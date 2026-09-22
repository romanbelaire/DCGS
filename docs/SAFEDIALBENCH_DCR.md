# DCR on SafeDialBench

The separate `scripts/run_safedial_dcr.py` runner loads standard
`Qwen/Qwen2.5-1.5B` and the uploaded `models/dcr_sft_lora_171151` PEFT adapter.
The user selected the format defined in the supplied
`DCR-main/vllm_inference_lora.py` for evaluation on 2026-09-22. Its template
is byte-for-byte identical to uploaded `chat_template.jinja`. Actual training
serialization remains unverified because `AttnLRP_Analysis/utils.py` is missing;
this does not block the user-selected inference experiment. Evaluate the supplied
stock-Qwen plus SFT LoRA pairing without claiming verified reproduction of the
complete two-stage DCR workflow. See [source review](SAFEDIALBENCH_DCR_SOURCE_REVIEW.md).

GPU smoke 271289 completed on an L40 on 2026-09-21: five turns, zero runtime
errors or input truncations, output/template/GPU audits and judge dry-run PASS.
Wall time 2m53s; peak allocated GPU memory 3.02 GiB. The first reply was Chinese
despite an English prompt; the other four reached the 1,024-token output cap,
and three included additional generated USER/ASSISTANT turns. These are saved
model outcomes, not adapter-loading failures. No quality scores exist yet.
Evidence: `verification/dcr_20260921/smoke_271289.json`.

The user-requested two-dialogue extension and matching raw-text comparison
use a separate runner; see [prompt-format smoke tests](SAFEDIALBENCH_DCR_PROMPT_SMOKE.md).

## Protocol

- Load tokenizer files and the explicit selected inference template from the adapter
  directory. Each turn renders as uppercase `USER:` / `ASSISTANT:` lines,
  ending in `ASSISTANT: ` for generation.
- Add no system message. Preserve separate prior user and **dataset reference
  assistant** messages, then the current user message. The current reference
  answer, future turns, and previously generated answers are not prompt inputs.
- Greedy decoding (`do_sample=False`), one choice, up to 1,024 new tokens,
  stopping at the uploaded tokenizer's EOS. No extra role-marker stripping,
  safety prompt, retry sampling, or response rewriting is added.
- Reject context overflow in a tokenizer preflight covering every selected
  turn; do not silently truncate history. The native context limit comes from
  the pinned base config and uploaded tokenizer. Training's 1,024-token cutoff
  is not imposed as the inference context limit.
- Keep the adapter unmerged, active, in evaluation mode, with all parameters
  frozen. Verify a nonzero enabled/disabled logit difference on a fixed benign
  fixture before benchmark generation. This check is not a safety score.

## Artifact preparation

`configs/safedial/dcr.lock.json` pins the base revision, base file hashes,
adapter weights/config, training params, and all uploaded tokenizer/template
files. Base revision: `8faed761d45a263340a0528343f099c05c9a4323`.
The training base revision was not supplied; this is the recorded inference
revision. Base weights are checked against the official LFS SHA-256.

From the DCGS root:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/prepare_safedial_dcr.py --offline
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B \
  scripts/run_safedial_dcr.py --validate-only
```

The base and lock are already staged in this workspace. On an unstaged checkout,
run the preparation command without `--offline` to download the pinned base
(about 3.1 GB). Locks contain absolute local paths; use `--lock` to create a
separate lock when relocating artifacts. Existing locks are never overwritten.
Preflight does not instantiate the model, generate answers, or call a judge.

## User-owned GPU smoke

```bash
mkdir -p outputs/slurm
bash -n scripts/slurm/run_safedial_dcr_smoke.sbatch
sbatch scripts/slurm/run_safedial_dcr_smoke.sbatch
```

The launcher requests one L40, 32 GB host memory, four CPUs and one hour.
It uses dialogue 1 (five turns), BF16, seed 0, offline artifacts and a separate
`outputs/safedial_baseline/dcr_qwen_1.5b_smoke_v1` directory. It runs preflight,
generation, saved-output/GPU validation and judge **dry-run** only. No paid
judging is performed. The user owns submission; no job has been submitted by
the implementation agent. The user submitted passing L40 job 271289 after
cancelling the pending A40 job 271284. The full-run launcher below is now prepared.

Audit an existing smoke without inference:

```bash
.venv/bin/python -B scripts/run_safedial_dcr.py \
  --output-dir outputs/safedial_baseline/dcr_qwen_1.5b_smoke_v1 \
  --audit-only --require-gpu
```

## Full benchmark using the supplied inference format

The separate full launcher reuses the technically passing prompt-diagnostic
runner without modifying its frozen source. Its legacy CLI value `training`
means the uploaded uppercase-role template, not verified training parity.
The launcher's `inference_format_contract.json` records this qualification,
the inference-source/template hashes, frozen runner source hashes and intentional
backend/decoding differences. It rejects source/template drift and a conflicting
existing contract before generation.

Run from the DCGS root:

```bash
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_dcr_full.sbatch
```

This selects all **2,037 dialogues / 10,029 turns**, with one L40, 32 GB host
memory, four CPUs, and a 48-hour allocation. Results go to
`outputs/safedial_baseline/dcr_qwen_1.5b_inference_format_full_v1` and logs to
`outputs/slurm/safedial-dcr-full-<JOB_ID>.{out,err}`. Preflight, generation,
GPU/output audit and judge dry-run are included; paid judging is separate.
No job was submitted by the agent.

Only the supplied script's **format** is adopted. The validated SafeDial runner
uses Transformers/PEFT and the uploaded tokenizer, preserves multi-turn reference
history, and retains greedy BF16 generation with the benchmark's **1,024-token**
cap. The supplied single-turn vLLM script defaults to **512**. No system message,
extra role-marker stop, or output repair is introduced. Effective vLLM stopping
is not claimed to have been reproduced.

Prior matching-format GPU smokes covered all 15 turns of IDs 1,2,3 with zero
runtime errors. Fourteen responses reached the cap and eleven contained extra
role markers; this is a quality limitation, not a technical launch blocker.
The full result will measure the supplied model under this chosen protocol.

If time-limited, submit the **same command after the prior job has ended**.
It skips completed dialogues and regenerates any unfinished dialogue; prior
attempts remain journaled. Failed completed dialogues are not automatically
retried, and any error makes final validation fail. No automatic resubmission
or paid judging occurs. Keep the pinned files unchanged while running/resuming.
Full preparation evidence: `verification/dcr_full_preparation_20260922/`.

## Outputs, resume and failures

The runner writes judge-compatible `answers.jsonl`, detailed `turns.jsonl`,
append-only `attempts.jsonl`, `run_config.json`, `preflight.json`,
`adapter_validation.json`, `runtime_stats.json`, and `validation.json`.
Turn records include the structured history and a hash of the rendered prompt;
the auditor reconstructs the selected-template prompt and checks its token
count. The manifest records artifact hashes, source hashes and package versions.

Rerun the identical command to skip completed dialogues. A successful no-op
resume preserves existing outputs and GPU measurements. Changes to artifacts,
source, package versions, selection or decoding settings require a new output
directory. A nonblocking file lock prevents concurrent generation writers.

Model errors and empty responses are recorded and make final validation exit
nonzero. An explicit `--retry-errors` regenerates entire failed dialogues;
all earlier attempts remain in `attempts.jsonl`. Interrupted unfinished
dialogues are regenerated in full on resume. There is no automatic extra
sampling, and failed outputs must not be represented as successful evaluations.
GPU smoke passed the technical checks above; benchmark quality is unscored.
