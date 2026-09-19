import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import safedial_dcgs_wildjailbreak as adapter
import run_safedial_dcgs_wildjailbreak as runner


class Tokenizer:
    truncation_side = "left"
    def __call__(self, texts, **kwargs):
        def encode(text):
            ids = [ord(c) for c in text]
            if kwargs.get("add_special_tokens", True):
                ids = [1] + ids
            if kwargs.get("truncation"):
                ids = ids[-kwargs["max_length"]:]
            return ids
        result = {"input_ids": encode(texts) if isinstance(texts, str) else [encode(text) for text in texts]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(0, 0)] + [(i, i + 1) for i in range(len(texts))]
            result["attention_mask"] = [1] * len(result["input_ids"])
        return result


ROW = {"id": 1, "task": "Safety", "method": "test", "history": [
    {"user": "Please help me understand this problem.", "bot": "PRIVATE CURRENT REFERENCE"},
    {"user": "What should I do next?", "bot": "PRIVATE NEXT REFERENCE"}]}
BELIEFS = "1. Explain the issue.\n2. Ask for context.\n3. Explain the issue.\n4. Offer alternatives.\n5. Summarize the options."


REPLIES = ["Candidate one.", "A helpful response.", "Candidate one.", "Candidate four.", "Candidate five."]

def is_ll_generation(request):
    return request["kind"] == "generate" and (request["max_new_tokens"] == 640 or not request["do_sample"])


def execute(request):
    if request["kind"] == "score":
        return {"scores": [float(len(b) % 7) for b in request["high_level_actions"]]}
    if request["kind"] == "score_ll":
        return {"scores": {a: 10.0 if a == "A helpful response." else 0.0 for a in request["actions"]}, "objective": "shapley"}
    if request["max_new_tokens"] == 640:
        return {"texts": ["\n".join(f"{i}. [RESPONSE]{a}[/RESPONSE]" for i, a in enumerate(REPLIES, 1))]}
    return {"texts": [BELIEFS if request["do_sample"] else "[RESPONSE]A helpful response.[/RESPONSE]"]}


class OriginalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_direct_original_orchestration_parity(self):
        # Independent direct invocation of src orchestration, without the adapter's
        # generation/critic boundary. Compare actual kwargs and selected answer.
        import torch
        from transformers import set_seed
        for method in ("vdcgs", "rdcgs"):
            with self.subTest(method=method):
                config = adapter.configuration(method)
                wrapped = adapter.generate_turn(ROW, 1, 42, config, self.tokenizer, execute)
                episode = adapter.episode_for(ROW, 1)
                manager = adapter.PromptManager(str(adapter.ROOT / "src/prompts/templates.jsonl"),
                                                str(adapter.ROOT / "src/prompts/personas.jsonl"))
                hl = adapter.FreeformHighLevelAgent(None, self.tokenizer, manager,
                    template_name=config.belief_gen_template, enable_thinking=config.hl_enable_thinking)
                ll = adapter.LowLevelAgent(None, self.tokenizer, manager, enable_thinking=config.ll_enable_thinking)
                requests = []
                def generation(**kwargs):
                    request = {k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "logits_processor")}
                    request.update(kind="generate", suppressed_token_ids=sorted(kwargs["logits_processor"].suppress_token_ids))
                    requests.append(request)
                    return execute(request)["texts"]
                class Critic:
                    use_regret_critic = config.use_regret_critic
                    device = "cpu"
                    def __getattr__(self, name):
                        def score(**kwargs):
                            request = {k: v for k, v in kwargs.items() if k != "tokenizer"}
                            request.update(kind="score", function=name)
                            requests.append(request)
                            return torch.tensor(execute(request)["scores"])
                        return score
                class LLCritic:
                    objective = "shapley"
                    def score_actions(self, observation, selected_belief, actions):
                        request = {"kind": "score_ll", "observation": observation,
                                   "selected_belief": selected_belief, "actions": list(actions)}
                        requests.append(request)
                        return execute(request)["scores"]
                config._ll_token_critic = LLCritic()
                set_seed(42)
                with patch.object(adapter.hl_module, "batch_generate", generation), patch.object(adapter.ll_module, "batch_generate", generation):
                    adapter.original.batch_generate_beliefs_for_episodes([episode], hl, config)
                    adapter.original.batch_compute_q_values_for_episodes([episode], Critic(), self.tokenizer, config)
                    valid = [c.summary for c in episode.belief_state.candidates if c.summary != "[SKIP]"]
                    selected = adapter.original._select_high_level_belief(valid, episode._q_values, episode, config, 0)
                    answer = adapter.original._generate_or_select_ll_action(episode, selected,
                        episode.belief_state.history, ll, Critic(), self.tokenizer, config)
                self.assertEqual(requests, [e["request"] for e in wrapped["dcgs_original"]["events"]])
                self.assertEqual(selected, wrapped["dcgs_original"]["selected_belief"])
                self.assertEqual(answer, wrapped["message"])
                self.assertLess(len(episode._q_values), len(valid))  # Original string-keyed duplicate merging.
                self.assertEqual(requests[-1]["kind"], "score_ll")
                self.assertTrue(requests[-2]["do_sample"])
                self.assertEqual(requests[-2]["max_new_tokens"], 640)
                self.assertEqual(requests[-1]["actions"], REPLIES)
                self.assertEqual(answer, "A helpful response.")

    def test_original_all_skip_retries_and_exhaustion(self):
        config = adapter.configuration("vdcgs")
        seen = []
        def backend(request):
            seen.append(request)
            return {"texts": [""]}
        with self.assertRaises(adapter.PolicyFailure):
            adapter.generate_turn(ROW, 0, 0, config, self.tokenizer, backend)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(r["do_sample"] for r in seen))

    def test_original_belief_retry_recovers(self):
        attempts = 0
        def backend(request):
            nonlocal attempts
            if request["kind"] == "generate" and request["do_sample"] and not is_ll_generation(request):
                attempts += 1
                if attempts == 1:
                    return {"texts": [""]}
            return execute(request)
        result = adapter.generate_turn(ROW, 0, 0, adapter.configuration("vdcgs"), self.tokenizer, backend)
        self.assertEqual(attempts, 2)
        self.assertEqual(result["message"], "A helpful response.")

    def test_empty_ll_does_not_gain_custom_retry(self):
        calls = []
        def backend(request):
            calls.append(request)
            if is_ll_generation(request):
                return {"texts": ["\n" * 128]}
            return execute(request)
        with self.assertRaises(adapter.PolicyFailure) as failure:
            adapter.generate_turn(ROW, 0, 0, adapter.configuration("rdcgs"), self.tokenizer, backend)
        self.assertEqual(sum(is_ll_generation(r) for r in calls), 2)  # Pool plus upstream fallback, no additional retries.
        self.assertEqual(failure.exception.details["code"], "blank_ll_candidate")
        self.assertEqual(failure.exception.audit["events"][-1]["result"]["texts"], ["\n" * 128])

    def test_gold_history_and_reference_isolation(self):
        first = adapter.episode_for(ROW, 0)
        second = adapter.episode_for(ROW, 1)
        self.assertNotIn("PRIVATE CURRENT", first.current_obs_for_action)
        self.assertIn("PRIVATE CURRENT", second.current_obs_for_action)
        self.assertNotIn("PRIVATE NEXT", second.current_obs_for_action)

    def test_replay_rejects_tampering(self):
        config = adapter.configuration("rdcgs")
        result = adapter.generate_turn(ROW, 0, 0, config, self.tokenizer, execute)
        record = json.loads(json.dumps({"seed": 0, "turn_index": 0,
            "generated_response": result["message"], "dcgs_original": result["dcgs_original"]}))
        adapter.validate_turn(record, ROW, config, self.tokenizer)
        record["dcgs_original"]["events"][0]["request"]["max_new_tokens"] += 1
        with self.assertRaises((adapter.PolicyFailure, ValueError)):
            adapter.validate_turn(record, ROW, config, self.tokenizer)

    def test_resume_completed_turns_and_byte_identical_noop(self):
        config = adapter.configuration("vdcgs")
        manifest = {"seed": 0, "model_id": "test"}
        first_turn_calls = len(adapter.generate_turn(ROW, 0, 0, config, self.tokenizer, execute)["dcgs_original"]["events"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            count = 0
            def backend(request):
                nonlocal count
                count += 1
                # Interrupt after the first complete turn; second turn restarts.
                if count == first_turn_calls + 1:
                    raise KeyboardInterrupt()
                return execute(request)
            with self.assertRaises(KeyboardInterrupt):
                runner.run_selected(folder, [ROW], manifest, config, self.tokenizer, lambda: backend)
            saved = (folder / "turns.jsonl").read_bytes()
            runner.run_selected(folder, [ROW], manifest, config, self.tokenizer, lambda: execute)
            self.assertTrue((folder / "turns.jsonl").read_bytes().startswith(saved))
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            def forbidden():
                self.fail("No-op resume loaded model")
            runner.run_selected(folder, [ROW], manifest, config, self.tokenizer, forbidden)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_failed_policy_is_not_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = adapter.configuration("vdcgs")
            manifest = {"seed": 0, "model_id": "test"}
            with self.assertRaises(adapter.PolicyFailure):
                runner.run_selected(folder, [ROW], manifest, config, self.tokenizer, lambda: lambda r: {"texts": [""]})
            with self.assertRaisesRegex(ValueError, "policy failure"):
                runner.run_selected(folder, [ROW], manifest, config, self.tokenizer, lambda: self.fail("Retried failed policy"))


if __name__ == "__main__":
    unittest.main()
