# Standard Qwen template smoke preparation

Status: CPU validation complete; user submission and GPU inference pending.

- `tests.log`: all 23 DCR tests pass, including exact default serialization,
  history isolation, template tamper rejection, paired seeds, generation/audit,
  no-op resume, and raw EOS-only evidence preservation.
- `preflight.json`: actual uploaded tokenizer, 3 dialogues / 15 turns,
  max 1,545 prompt tokens, no overflow; official config/template hashes.
- `prepared_manifest.json`: fixed decoding, source/artifact hashes and identity.
- `prompts.json`: all 15 exact rendered prompts and paired seeds.
- `previous_run_checks.json`: original/training-more/raw/Zephyr source hashes
  and seeds match, plus original answers hashes.
- `bash -n scripts/slurm/run_safedial_dcr_qwen_smoke.sbatch`: PASS.

Official pinned configuration fetched from the public Hugging Face resolve URL
under `Qwen/Qwen2.5-1.5B` revision
`8faed761d45a263340a0528343f099c05c9a4323`. Sandbox DNS failed; approved public
network fetch succeeded. No previous runtime or result files edited.

The standard template adds its default system message. Stopping remains
endoftext151643 for parity with prior arms; im_end151645 is not a stop.
This is the DCR-adapted base model, not the Qwen-Instruct checkpoint.
