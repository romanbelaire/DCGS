# Separate SafeDialBench VDCGS and RDCGS runs

> Script cleanup (2026-09-18): superseded launchers and one-time migration
> scripts referenced below are now in the [audit archive](../archive/scripts_cleanup_20260918/README.md).
> Use the [current script index](../scripts/README.md) for active entry points.
> The original text below is retained as experiment history.

Prepared 2026-09-17 at the user's request. Both use the pinned Zephyr actor,
August11 WildJailbreak high-level checkpoint, and downloaded WildJailbreak
harm/follow token critic in `configs/safedial/dcgs_two_stage_aug11.lock.json`.

| Run | Intent proposals | Intent selection | Response selection |
| --- | --- | --- | --- |
| VDCGS | 5 nominal | Softmax over Q | Highest token-critic score among 5 responses |
| RDCGS | 5 nominal + 5 adversarial | Softmax over 0.8Q − 0.2regret | Highest token-critic score among 5 responses |

VDCGS constructs and evaluates only Q plus the two token heads. RDCGS also
constructs/evaluates the regret head. Q comes from the same August11 checkpoint
for both methods; it is not independently value-trained. Different intent pools
make this a complete-method comparison, not an isolated regret-term ablation.
Both preserve indexed duplicate candidates, use seeded intent selection, and
resolve response-score ties by first index. Each turn receives gold prior history.

## A5000 alternative when A40s are unavailable

The runner now accepts `--ll-device cuda:1`: BF16 actor/Q/regret stay on GPU0;
the FP16 encoder and FP32 harm/follow heads use GPU1. Checkpoint files, candidate
counts, prompt templates, seeds and scoring formulas are unchanged. This does
not pool GPU memory or require NVLink. Each complete backbone fits on its own GPU;
the10240-token startup probe checks the actual allocation before generation.

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_vdcgs_a5000_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_a5000_full.sbatch
```

Each requests two A5000s on one node,32GB host memory,eightCPUs,48h/researchlong.
Both jobs together request four GPUs. New outputs are
`outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_aug11_a5000_full`.
Logs: `outputs/slurm/safedial-{vdcgs,rdcgs}-a5000-full-<jobid>.{out,err}`.
After a timeout rerun the identical command when the previous job has ended.

A5000 has24GB and native BF16 support. One A5000 cannot hold the measured
27.38GiB combined smoke footprint. V100/P100/2080/1080 do not supply native
BF16 support for the pinned actor path; they are not a GPU-name-only replacement.
NVIDIA sources: [A5000 specifications](https://www.nvidia.com/en-au/products/workstations/rtx-a5000/),
[BF16 architecture requirement](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
The runner rejects pre-Ampere actor GPUs rather than silently changing precision.

At preparation, scheduler counters showed2 unallocated A5000s on candle and2
on comet; torpedo had0. Reservations/PLANNED state and priority can still delay
allocation.32GB host RAM fits reported remaining node memory; smoke256826's
MaxRSS was2307540K(~2.2GiB). Full host-memory usage remains unmeasured.

All126 tests PASS, including separate encoder/head placement, token-input routing,
per-device synchronization and finite startup probing. Both new launchers pass
bash syntax. Native dialogue1 audits and byte-identical no-op resume pass for
both split-device manifests with injected outputs. Prior full-tokenizer preflight
is reused after exact comparison of its function, policy, native span routine,
and artifact lock. Evidence: `docs/verification/dcgs_a5000_20260917/`.
No actual two-GPU execution has occurred yet; the job records startup capacity
and per-GPU peaks. Top-level legacy peak fields describe the actor GPU; the
`gpu_devices` map is authoritative for both devices. No jobs were submitted.

## Original A40 launch

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_vdcgs_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_full.sbatch
```

Each requests one A40, 64GB host memory, eight CPUs, and 48 hours on
`researchlong`. The two jobs have separate output folders:

- `outputs/safedial_dcgs/zephyr_vdcgs_aug11_full`
- `outputs/safedial_dcgs/zephyr_rdcgs_aug11_full`

Each covers all 2,037 dialogues / 10,029 turns. Logs use
`outputs/slurm/safedial-{vdcgs,rdcgs}-full-<jobid>.{out,err}`.
After a time limit, rerun the same command once the previous job has ended.
Successful calls and turns resume, including calls within an interrupted turn.
A saved error requires inspection before an explicit retry; launchers do not
blindly retry failures. Exclusive locks prevent concurrent writers per output.

The jobs validate completed generation and perform only a judge dry run.
Paid safety judging is a separate step. No jobs were submitted during preparation.

## Context and GPU validation

The completed two-stage RDCGS smoke256826 passed output/runtime/memory review:
five turns, zero truncations, 385.99 seconds, peak allocated27.38GiB on A10040GB.
Its exact sources and output remain intact. Full methods have separate versioned
entrypoints and source hashes. A40 is selected from the user's available GPUs;
the two resident encoders exceeded24GiB even in the short smoke.

Full-dataset critic placeholders reach8,000 tokens, leaving insufficient room
under the old8,192 guard for generated intents and responses. Full runners use
the backbone's native32,768 context, retaining upstream token offsets, float16
LL encoding, float32 token heads, and exact Shapley scoring. No truncation occurs.
This extends supported input length; it does not establish critic quality at long
contexts. Dynamic inputs receive exact length checks before model execution.

At each model-loading invocation, a synthetic actor and LL capacity probe runs
with both backbones resident. Its length is the longest observed prefix plus
intent/response budgets and token-boundary slack, rounded up to512. Probe costs
are separated from benchmark call counts; memory/runtime include the probe.
An unsuccessful probe stops generation and records `context_probe.json`.
The new full-context A40 path has not yet run on a GPU.

## Verification

All126 repository tests passed, including real checkpoint head loading,
upstream/full token-score equality on the real tokenizer and heads with synthetic
residuals, no regret evaluation in VDCGS, both selection rules, error retention,
interrupted-call resume, no-op resume, and context overflow guards.
Both batch scripts pass `bash -n`.

Full-tokenizer preflight, native dialogue1 offline audits, and the preserved
GPU-smoke audit are recorded under `docs/verification/dcgs_methods_20260917/`.
Offline candidate outputs are fixtures under `/tmp`, not benchmark results.
