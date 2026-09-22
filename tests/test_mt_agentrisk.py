import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".cache/mt_agentrisk_python")]
import mt_agentrisk_protocol as protocol
from mt_agentrisk_engines import run_dcgs, run_smooth, run_tpo
from serve_mt_agentrisk import Service


TOOLS = [{"type": "function", "function": {"name": "execute_bash", "description": "Execute command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                         "required": ["command"], "additionalProperties": False}}}]
ACTION = json.dumps({"type": "tool", "name": "execute_bash", "arguments": {"command": "printf OK"}})


def payload(method="vdcgs"):
    return {"model": "mt-" + method + "--test", "messages": [{"role": "system", "content": "Use tools"},
            {"role": "user", "content": "Print OK"}], "tools": copy.deepcopy(TOOLS)}


class ProtocolTests(unittest.TestCase):
    def test_duplicate_fields_and_nonfinite_numbers_are_not_silently_decoded(self):
        request = protocol.normalize(payload(), 'vdcgs')
        for raw in (
                '{"type":"tool","name":"finish","name":"execute_bash","arguments":{"command":"printf OK"}}',
                '{"type":"tool","name":"execute_bash","arguments":{"command":"a","command":"b"}}',
                '{"type":"message","content":"OK","unused":NaN}',
                '{"type":"message","content":"OK","unused":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(protocol.ProtocolError):
                protocol.decode_action(raw, request, 'a' * 64)
        self.assertEqual(protocol.strict_json_loads('{"value":1.5}'), {'value': 1.5})

    def test_candidate_wrappers_preserve_arrays_and_literal_tags(self):
        action = json.dumps({"type": "tool", "name": "demo", "arguments": {
            "ranges": [[11, 20], [], ["x"]], "literal": '[RESPONSE] "quoted" [/RESPONSE] [abc]'}})
        wrapped = "[RESPONSE]" + action + "[/RESPONSE]"
        self.assertEqual(protocol.parse_tool_response(None, wrapped), action)
        self.assertEqual(protocol.parse_tool_response(None, action), action)
        raw = "1. " + wrapped + "\n2. [RESPONSE]not JSON[/RESPONSE]"
        self.assertEqual(protocol.parse_tool_candidates(None, raw, 5, 'action_generation'), [action, 'not JSON'])
        self.assertEqual(protocol.parse_tool_candidates(None, '1. [RESPONSE]' + action, 5, 'action_generation'), [])
        self.assertEqual(protocol.parse_tool_response(None, '[RESPONSE]' + action), action)

    def test_tool_round_trip_preserves_transcript(self):
        p = payload()
        request = protocol.normalize(p, "vdcgs")
        message, reason = protocol.decode_action(ACTION, request, "a" * 64)
        self.assertEqual(reason, "tool_calls")
        p["messages"].extend([message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "OK"}])
        following = protocol.normalize(p, "vdcgs")
        self.assertEqual(following["messages"][-1]["content"], "OK")
        self.assertIn('"command":"printf OK"', message["tool_calls"][0]["function"]["arguments"])
        self.assertIn('"role":"tool"', protocol.render(following))
        self.assertNotEqual(protocol.digest(request), protocol.digest(following))

    def test_invalid_actions_are_not_repaired(self):
        request = protocol.normalize(payload(), "vdcgs")
        for raw in ('run printf OK', ACTION + ACTION,
                    json.dumps({"type": "tool", "name": "unknown", "arguments": {}}),
                    json.dumps({"type": "tool", "name": "execute_bash", "arguments": {"command": 42}}),
                    json.dumps({"type": "tool", "name": "execute_bash", "arguments": {"command": "x", "extra": 1}})):
            with self.subTest(raw=raw), self.assertRaises(protocol.ProtocolError):
                protocol.decode_action(raw, request, "a" * 64)

    def test_request_rejects_lost_results_images_and_external_schemas(self):
        bad = []
        p = payload(); p["messages"].append({"role": "tool", "content": "unexpected", "tool_call_id": "missing"}); bad.append(p)
        p = payload(); p["messages"][-1]["content"] = [{"type": "image_url", "image_url": {"url": "secret"}}]; bad.append(p)
        p = payload(); p["tools"][0]["function"]["parameters"] = {"$ref": "https://invalid/schema"}; bad.append(p)
        p = payload(); p["stream"] = True; bad.append(p)
        p = payload("rdcgs"); bad.append(p)
        for p in bad:
            with self.subTest(p=p), self.assertRaises(protocol.ProtocolError):
                protocol.normalize(p, "vdcgs")

    def test_tool_choice_and_text_refusal(self):
        request = protocol.normalize(payload(), "vdcgs")
        text = json.dumps({"type": "message", "content": "I cannot do that."})
        self.assertEqual(protocol.decode_action(text, request, "a" * 64)[1], "stop")
        request["tool_choice"] = "required"
        with self.assertRaises(protocol.ProtocolError): protocol.decode_action(text, request, "a" * 64)
        request["tool_choice"] = "none"
        with self.assertRaises(protocol.ProtocolError): protocol.decode_action(ACTION, request, "a" * 64)


class ServerTests(unittest.TestCase):
    def test_runner_failure_preserves_audit_and_is_distinct_from_protocol_error(self):
        calls = []
        def engine(request, event):
            calls.append(request)
            exc = ValueError('Original-policy selection/audit mismatch')
            exc.audit = {'selected_response': ACTION, 'events': []}
            raise exc
        with tempfile.TemporaryDirectory() as d:
            service = Service(engine, 'secret', d, 'vdcgs')
            first = service.complete('Bearer secret', payload())
            self.assertEqual(first[0], 500)
            self.assertEqual(first[1]['error']['category'], 'runner_error')
            self.assertEqual(service.complete('Bearer secret', payload()), first)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(next(Path(d).glob('*/failed_audit.json')).read_text())['selected_response'], ACTION)

    def test_auth_cache_failure_and_budget(self):
        calls = []
        def engine(request, event):
            calls.append(request)
            return {"message": ACTION, "prompt_tokens": 10, "completion_tokens": 8, "audit_summary": {}}
        with tempfile.TemporaryDirectory() as d:
            service = Service(engine, "secret", d, "vdcgs", 1)
            self.assertEqual(service.complete(None, payload())[0], 401)
            self.assertFalse(calls)
            first = service.complete("Bearer secret", payload())
            self.assertEqual(first[0], 200)
            self.assertEqual(service.complete("Bearer secret", payload()), first)
            self.assertEqual(len(calls), 1)
            other = payload(); other["model"] = "mt-vdcgs--independent_run"
            self.assertEqual(service.complete("Bearer secret", other)[0], 429)
            self.assertEqual(len(list(Path(d).glob('*/audit.json'))), 1)

    def test_invalid_selected_candidate_cached_without_fallback(self):
        calls = []
        def engine(request, event):
            calls.append(request)
            return {"message": "plain prose", "audit_summary": {}}
        with tempfile.TemporaryDirectory() as d:
            service = Service(engine, "secret", d, "vdcgs")
            first = service.complete("Bearer secret", payload())
            self.assertEqual(first[0], 422)
            self.assertEqual(service.complete("Bearer secret", payload()), first)
            self.assertEqual(len(calls), 1)
            audit = json.loads(next(Path(d).glob('*/audit.json')).read_text())
            self.assertEqual(audit["message"], "plain prose")


class PolicyTests(unittest.TestCase):
    def test_validation_failure_carries_selected_response_and_events(self):
        from test_safedial_dcgs_wildjailbreak import Tokenizer, execute
        from safedial_dcgs_wildjailbreak import configuration
        from unittest.mock import patch
        class Tok(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return str(messages)
        with patch('mt_agentrisk_engines.validate_tool_turn', side_effect=ValueError('audit mismatch')):
            with self.assertRaises(ValueError) as raised:
                run_dcgs(protocol.normalize(payload(), 'vdcgs'), configuration('vdcgs'), Tok(), execute)
        self.assertEqual(raised.exception.audit['message'], 'A helpful response.')
        self.assertTrue(raised.exception.audit['dcgs_original']['events'])

    def test_belief_retry_replay_preserves_evidence_and_rejects_tampering(self):
        for method in ('vdcgs', 'rdcgs'):
            with self.subTest(method=method):
                self.check_retry_replay(method)

    def check_retry_replay(self, method):
        from test_safedial_dcgs_wildjailbreak import Tokenizer, BELIEFS
        from safedial_dcgs_wildjailbreak import configuration
        from mt_agentrisk_engines import row_for, validate_tool_turn
        from src.agents.low_level_agent import LowLevelAgent
        from unittest.mock import patch
        class Tok(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return str(messages)
        belief_calls = 0
        initial_calls = 2 if method == 'rdcgs' else 1
        first = "1. Only one belief; original policy must retry."
        def backend(request):
            nonlocal belief_calls
            if request['kind'] == 'score':
                return {'scores': [1.0] * len(request['observations'])}
            if request['kind'] == 'score_ll':
                return {'scores': {a: 1.0 for a in request['actions']}, 'objective': 'shapley'}
            if request['max_new_tokens'] == 96:
                belief_calls += 1
                return {'texts': [first if belief_calls <= initial_calls else BELIEFS]}
            return {'texts': ['\n'.join(f'{i}. [RESPONSE]{ACTION}[/RESPONSE]' for i in range(1, 6))]}
        req, config, tok = protocol.normalize(payload(method), method), configuration(method), Tok()
        result = run_dcgs(req, config, tok, backend)
        self.assertEqual(belief_calls, 2 * initial_calls)
        self.assertEqual(result['message'], ACTION)
        self.assertEqual(result['dcgs_original']['events'][0]['result']['texts'], [first])
        record = {**result, 'turn_index': 0, 'seed': req['seed'], 'generated_response': result['message']}
        before = copy.deepcopy(record)
        with patch.object(LowLevelAgent, '_parse_response_from_tags', protocol.parse_tool_response), \
                patch.object(LowLevelAgent, '_parse_multi_candidate_responses', protocol.parse_tool_candidates):
            validate_tool_turn(record, row_for(req), config, tok)
            self.assertEqual(record, before)
            for field in ('generated_response', 'selected_belief', 'event_request'):
                changed = copy.deepcopy(record)
                if field == 'generated_response':
                    changed[field] = 'tampered'
                elif field == 'selected_belief':
                    changed['dcgs_original'][field] = 'tampered'
                else:
                    changed['dcgs_original']['events'][0]['request']['max_new_tokens'] += 1
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_tool_turn(changed, row_for(req), config, tok)

    def test_fallback_preserves_arrays_before_scoring_and_restores_parser(self):
        from test_safedial_dcgs_wildjailbreak import Tokenizer, BELIEFS
        from safedial_dcgs_wildjailbreak import configuration
        from src.agents.low_level_agent import LowLevelAgent
        original_parser = LowLevelAgent._parse_response_from_tags
        original_multi = LowLevelAgent._parse_multi_candidate_responses
        class Tok(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return str(messages)
        action = json.dumps({'type': 'tool', 'name': 'execute_bash',
                            'arguments': {'command': "printf '[11, 20]'", 'values': [11, 20]}})
        def backend(request):
            if request['kind'] == 'score':
                return {'scores': [1.0] * len(request['observations'])}
            if request['kind'] == 'score_ll':
                self.assertEqual(request['actions'], [action] * 5)
                return {'scores': {action: 1.0}, 'objective': 'shapley'}
            budget = request['max_new_tokens']
            return {'texts': [BELIEFS if budget == 96 else
                    '1. [RESPONSE]unclosed' if budget == 640 else '[RESPONSE]' + action + '[/RESPONSE]']}
        result = run_dcgs(protocol.normalize(payload(), 'vdcgs'), configuration('vdcgs'), Tok(), backend)
        self.assertEqual(result['message'], action)
        self.assertTrue(result['audit_summary']['policy_replay_passed'])
        self.assertEqual(result['audit_summary']['candidate_parser'], protocol.CANDIDATE_PARSER)
        self.assertIs(LowLevelAgent._parse_response_from_tags, original_parser)
        self.assertIs(LowLevelAgent._parse_multi_candidate_responses, original_multi)

    def test_dcgs_both_methods_prefix_ll_and_replay(self):
        from test_safedial_dcgs_wildjailbreak import Tokenizer, BELIEFS
        from safedial_dcgs_wildjailbreak import configuration
        class Tok(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return ''.join('<' + m['role'] + '>' + m['content'] + '</' + m['role'] + '>'
                               for m in messages) + ('<assistant>' if kwargs.get('add_generation_prompt') else '')
        for method in ('vdcgs', 'rdcgs'):
            seen = []
            def backend(request):
                seen.append(request)
                if request['kind'] == 'score':
                    return {'scores': [1.0] * len(request['observations'])}
                if request['kind'] == 'score_ll':
                    return {'scores': {a: 1.0 for a in request['actions']}, 'objective': 'shapley'}
                if request['max_new_tokens'] == 640:
                    return {'texts': ['\n'.join(f'{i}. [RESPONSE]{ACTION}[/RESPONSE]' for i in range(1, 6))]}
                return {'texts': [BELIEFS]}
            with self.subTest(method=method):
                result = run_dcgs(protocol.normalize(payload(method), method), configuration(method), Tok(), backend)
                self.assertEqual(result['message'], ACTION)
                self.assertTrue(result['audit_summary']['policy_replay_passed'])
                self.assertEqual(result['audit_summary']['regret_enabled'], method == 'rdcgs')
                self.assertEqual(result['audit_summary']['ll_candidates_scored'], 5)
                ll = next(r for r in seen if r['kind'] == 'generate' and r['max_new_tokens'] == 640)
                self.assertTrue(ll['prompts'][0].startswith('<system>'))
                self.assertIn('execute_bash', ll['prompts'][0])
                self.assertIn('</system><user>', ll['prompts'][0])
                self.assertTrue(ll['prompts'][0].endswith('</user><assistant>'))
                self.assertIn('Inside EVERY [RESPONSE]', ll['prompts'][0])
                self.assertEqual(result['audit_summary']['dcgs_prompt_version'], 'dcgs-tool-candidates-v2')
                event = next(e for e in result['dcgs_original']['events']
                             if e['request']['kind'] == 'generate' and e['request']['max_new_tokens'] == 640)
                self.assertNotIn('<system>', event['request']['prompts'][0])
                self.assertEqual(ll['prefill_suffix'], event['request']['prefill_suffix'])

    def test_smooth_perturbs_latest_user_only_even_after_tool(self):
        p = payload('smoothllm')
        p['messages'].extend([{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_x', 'type': 'function', 'function': {'name': 'execute_bash', 'arguments': '{"command":"printf OK"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_x', 'content': 'KEEP_TOOL_RESULT'}])
        p['messages'][1]['content'] = 'This long user instruction should be perturbed while schemas and observations remain unchanged.'
        prompts = []
        def generate(text, seed):
            prompts.append(text)
            return {'message': ACTION, 'input_truncated': False, 'prompt_tokens': 20,
                    'original_prompt_tokens': 20, 'completion_tokens': 10}
        result = run_smooth(protocol.normalize(p, 'smoothllm'), generate)
        self.assertEqual(len(prompts), 8)
        self.assertTrue(all('KEEP_TOOL_RESULT' in text and '"required":["command"]' in text for text in prompts))
        self.assertEqual(result['prompt_tokens'], 160)
        self.assertEqual(result['message'], ACTION)

    def test_tpo_original_policy_replay(self):
        from test_safedial_tpo import fake_execute
        result = run_tpo(protocol.normalize(payload('tpo'), 'tpo'), fake_execute)
        self.assertTrue(result['audit_summary']['policy_replay_passed'])
        self.assertEqual(result['audit_summary']['sample_size'], 5)
        self.assertEqual(len(result['tpo']['rounds']), 2)


if __name__ == '__main__':
    unittest.main()
