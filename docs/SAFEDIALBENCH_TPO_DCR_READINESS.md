# TPO and DCR readiness audit

Verified 2026-09-14 SGT. Scope: current working tree, HEAD file listing, local model/output directories, project and default Hugging Face model caches, supplied DCGS PDF, original method papers, and upstream public sources. Read-only investigation plus documentation; no inference, training, package installation, or Slurm changes.

Implementation update after this audit: the TPO gaps below have now been addressed by a separate native runner, and the pretrained reward model is staged. See [TPO run guide](SAFEDIALBENCH_TPO.md). Offline checks pass; real GPU smoke remains pending user submission. DCR status is unchanged.

## Verdict

- **TPO: confirmed implementation and local model-staging blockers.** Public algorithm code and a pretrained reward model are available. New training is not required.
- **DCR: confirmed missing identifiable checkpoint in this checkout.** Public release availability remains unresolved. Missing training code is not independently a blocker to inference if a complete trained artifact is supplied.
- The supplied DCGS PDF (pp. 8–9, references 26/31 on p. 12) identifies these methods, but does not specify the exact DCR checkpoint or a reproducible TPO reward-model/configuration recipe. Exact historical replication and a disclosed SafeDialBench adaptation are different claims.

## TPO evidence

The [original paper, Sections 4–5](https://arxiv.org/html/2501.12895v1#S4) uses frozen models, learned reward ranking, chosen/rejected comparisons, textual loss, textual gradient, and iterative response updates. Its benchmark setting is N=5, D=2, temperature 0.7, top-p 0.95. Zephyr is our proposed adaptation; its behavior has not been validated here.

| Check | Repository evidence | Consequence |
|---|---|---|
| Candidate score | `src/defense/tpo_wrapper.py:24`: refusal substring bonus plus character-length bonus | Does not load or use a learned preference model |
| Configuration wiring | `src/configs/base_config.py:102` defines `tpo_reward_model`; `src/main.py:413` never passes it to the wrapper | Changing this config string cannot fix scoring |
| Feedback loop | `tpo_wrapper.py:82` builds a direct improvement prompt; line 114 generates one revision per iteration | Missing explicit textual loss/gradient stages and per-iteration sample width |
| Native benchmark path | `scripts/run_safedial_dcgs.py` directly invokes its DCGS stages; no TPO dispatch. No TPO native runner, config, batch, or tests found, including ignored files | Existing simulator backend flag does not supply a SafeDialBench TPO run |
| Weights | Project cache contains Zephyr and CAT; default home cache contains Zephyr. No intended reward-model directory found | Stage and pin the reward model before offline GPU execution |

[Official source](https://github.com/Simplified-Reasoning/TPO), inspected at main commit `395c3d763a4c3df0ae72a4352b0db16fe5ee18e9`: `tpo_utils.py` samples N initial answers, then performs D update rounds with N new answers each, scoring and retaining the cumulative candidate cache. Thus N=5/D=2 means nominally **15 response candidates per turn**, plus textual feedback work; 150,435 candidates across 10,029 turns before failures. `run.py` sets top-p 0.95. `reward_model.py` scores prompt/answer pairs using a classification model. Preserve the algorithm and explicitly adapt its single-query interface to benchmark gold history.

The [specified reward model](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) has a public model card/files and is an 8B sequence-classification model. Public listing was verified; downloading all weights and testing loading were not performed.

Before full generation: implement and test the native adapter, stage immutable model revisions, record candidate/reward/critique traces, then smoke-test history handling, output validity, context length, resume behavior, latency and peak VRAM. The actor plus reward model require memory planning, but neither a mandatory multi-GPU allocation nor an OOM is established. The README's 70B server deployment is not a minimum hardware requirement for our Zephyr adaptation. No need to retrain the actor or reward model.

## DCR evidence

The [DCR paper, Sections 5–6 and Appendix A](https://arxiv.org/html/2603.03323v1#S6) describes contrastive refinement followed by safety SFT, evaluates Qwen2.5-1.5B/7B and LLaMA-3-8B, and uses greedy decoding. Its SFT stage uses LoRA, so artifact inspection must establish whether a final release is merged or needs both the refined base and an SFT adapter. XSTest participates in training; SafeDialBench safety scores alone cannot establish the paper's over-refusal benefit.

Confirmed local facts:

- No `dcr/` directory exists in the working tree or HEAD file listing.
- `src/analysis/dcr_base_final_entk.py:27` imports a helper from missing `dcr/DCR-main/experiment`; its CLI requires `--base_model` and `--final_model`. It is an analysis utility, not a checkpoint or training recipe. `wjb_rep_entk_subset.py` likewise requires a caller-supplied DCR model.
- `models/MANIFEST.json` explicitly excludes DCR. A recursive weight-file inventory outside caches found only the four CARES/WildJailbreak critic files. Neither inspected Hugging Face cache contains an identifiable DCR model.
- Therefore **we have no verified DCR artifact to pass to generation**. This does not establish that the original authors never trained it or that no copy exists on another machine.

Upstream availability investigation: the historical anonymous repository page returned HTTP 401, and its README API returned HTTP 410. OpenReview was blocked by browser verification/API 403. Targeted web/GitHub/Hugging Face searches did not identify an author-verified checkpoint; the Hugging Face name search was limited to 100 results and is not exhaustive. These access/search outcomes support “unresolved release availability,” not “weights definitely unpublished.”

Once the complete artifact is identified, reuse `scripts/run_safedial_baseline.py` where compatible: it already supports arbitrary model/revision, native chat history, separate resumable outputs, and `--temperature 0` for greedy decoding (line 229). Check tokenizer/chat-template availability and adapter/base composition, then run a small loading/generation validation. No bespoke DCR inference algorithm or eNTK calculation is required. A fresh DCR training run is conditional on failing to recover suitable weights and requires its own agreed scope.

## Concrete next work

1. TPO can proceed as implementation/setup work now: preserve upstream feedback stages, stage the reward model, and prepare a measured smoke run. Keep the final SafeDialBench judge separate from optimization.
2. For DCR, obtain a checkpoint path/repository plus backbone, training-stage provenance and adapter dependencies. This is the decisive missing input. After inspection, prepare ordinary native inference and a smoke run.
3. Keep eight-arm scope. Do not add new controls or substitute a generic base model under the DCR label.

## Verification limits and failed approaches

Static/source and file-presence checks only; no model loading, GPU tests, or benchmark calls. The first local PDF extraction using system Python failed because `typing_extensions` was missing; rerunning with the existing Python 3.11 project environment and previously staged pypdf succeeded. Web rendering of upstream Python collapsed source lines; direct raw downloads to `/tmp/safedial-tpo-upstream-*.py` worked. Current runtime sources and active jobs were untouched.
