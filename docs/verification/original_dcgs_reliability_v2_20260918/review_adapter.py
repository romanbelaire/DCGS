"""Offline real-tokenizer/native-data checks; never loads the actor weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_safedial_dcgs_wildjailbreak as runner
import safedial_dcgs_wildjailbreak as adapter
from validate_safedial_dcgs_wildjailbreak import validate
from src.utils.llm_utils import batch_generate, SuppressWordsLogitsProcessor


def main():
    lock = json.loads(runner.LOCK.read_text())
    tokenizer = runner.tokenizer_for(lock)
    rows = runner.load_jsonl(runner.DEFAULT_DATASET)[:1]
    report = {"model_inference_calls": 0, "methods": {}}
    for method in ("vdcgs", "rdcgs"):
        config = adapter.configuration(method)
        args = SimpleNamespace(method=method, device="cpu", seed=0, dataset=runner.DEFAULT_DATASET)
        manifest = json.loads(json.dumps(runner.manifest_for(args, rows, lock)))
        def backend(request):
            if request["kind"] == "score":
                return {"scores": [0.5] * len(request["observations"])}
            text = "\n".join(f"{i}. Synthetic instruction {i}." for i in range(1, 6))
            return {"texts": [text if request["do_sample"] else "[RESPONSE]Synthetic fixture answer.[/RESPONSE]"]}
        with tempfile.TemporaryDirectory(prefix="dcgs-original-fixture-") as directory:
            folder = Path(directory)
            runner.write_json(folder / "run_config.json", manifest)
            runner.run_selected(folder, rows, manifest, config, tokenizer, lambda: backend)
            result = validate(folder)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            def forbidden():
                raise AssertionError("No-op resume loaded a model")
            runner.run_selected(folder, rows, manifest, config, tokenizer, forbidden)
            assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
            assert result["critic_truncated_turn_keys"] == [[1, 4]]
            assert result["critic_full_context"] is False
            result["byte_identical_noop"] = True
            report["methods"][method] = result

    # Exercise LocalBackend's instrumentation around the REAL batch_generate;
    # only model.generate is a tiny deterministic token-returning stub.
    continuation = tokenizer("Synthetic answer.", add_special_tokens=False, return_tensors="pt")["input_ids"]
    class Model:
        name_or_path = "HuggingFaceH4/zephyr-7b-beta"
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=32768)
        def generate(self, **kwargs):
            return torch.cat([kwargs["input_ids"], continuation], dim=1)
    local = adapter.LocalBackend.__new__(adapter.LocalBackend)
    local.model = Model()
    local.torch = torch
    local.tokenizer = tokenizer
    local.config = adapter.configuration("vdcgs")
    processor = SuppressWordsLogitsProcessor(tokenizer, ["Example", "example", "Examples", "examples"])
    kwargs = dict(prompts=["A sample prompt."], max_new_tokens=128, temperature=0.7,
                  do_sample=False, prefill_suffix="[RESPONSE]", chunk_size=1, enable_thinking=False)
    expected = batch_generate(model=local.model, tokenizer=tokenizer, logits_processor=processor, **kwargs)
    actual = local({"kind": "generate", "suppressed_token_ids": sorted(processor.suppress_token_ids), **kwargs})
    assert actual["texts"] == expected
    assert actual["raw_generations"][0]["generated_token_ids"] == continuation.tolist()
    report["original_batch_generate_instrumentation"] = "PASS"
    # Compare diagnostics to tokens actually passed by the original critic
    # encoder, using a stub body/head so no backbone weights are loaded.
    import hashlib
    from src.value.value_function import ValueFunction
    observation = adapter.episode_for(rows[0], 4).current_obs_for_action
    request = {"observations": [observation], "high_level_actions": ["Offer safe alternatives."]}
    expected_context = adapter.critic_diagnostics(request, tokenizer)
    captured = []
    def body(encoded, requires_grad=False):
        captured.append(encoded["input_ids"].tolist())
        return SimpleNamespace(last_hidden_state=torch.zeros((*encoded["input_ids"].shape, 1))), False
    critic = SimpleNamespace(device="cpu", dtype=torch.float32, _forward_body=body,
                             _escape_inference_tensor=lambda tensor, required: tensor)
    ValueFunction.predict_sa_value(critic, request["observations"], request["high_level_actions"],
                                   tokenizer, lambda pooled: pooled, False)
    actual_hash = hashlib.sha256(json.dumps(captured[0][0]).encode()).hexdigest()
    assert expected_context["encoded_sha256"] == [actual_hash]
    assert expected_context["retained_tokens"] == [1500]
    assert expected_context["truncation_side"] == "left"
    report["original_critic_encoding_matches_diagnostics"] = "PASS"
    report["source_sha256"] = runner.source_hashes()
    destination = Path(__file__).with_name("adapter_review.json")
    runner.write_json(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
