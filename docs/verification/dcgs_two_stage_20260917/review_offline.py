"""Real checkpoint/tokenizer inspection and fake-call smoke; no backbone inference."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import torch
from transformers import AutoTokenizer
import run_safedial_baseline as base
import run_safedial_dcgs_two_stage as runner
import safedial_dcgs_two_stage_runtime as runtime
from test_safedial_dcgs_two_stage import fake_execute
from validate_safedial_dcgs_two_stage import validate_dcgs

lock = json.loads((ROOT / 'configs/safedial/dcgs_two_stage.lock.json').read_text())
# Full artifact hashes are checked separately by the runner's real preflight.
heads, metadata = runtime.load_heads(lock)
tok = AutoTokenizer.from_pretrained(lock['actor']['path'], local_files_only=True, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
class SyntheticEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=torch.zeros(*input_ids.shape, 4096, dtype=torch.float16))
backend = object.__new__(runtime.LocalBackend)
backend.torch, backend.args, backend.tokenizer = torch, SimpleNamespace(device='cpu'), tok
backend.ll = runtime.LLTokenCritic(heads['harm_head'], heads['follow_head'], SyntheticEncoder(), tok,
                                    'cpu', metadata['objective'], int(metadata['k']))
request = {'kind': 'll_score', 'stage': 'response_score', 'slot': 0,
           'observation': 'Turn 1:\nAgent: [NO_AGENT_ACTION]\nUser: synthetic input',
           'belief': 'Insight: user asks for assistance. Instruction: respond clearly.',
           'action': 'A synthetic candidate response for the CPU plumbing check.'}
result = backend(request)
assert result['input_truncated'] is False and result['action_start'] < result['action_end']
with torch.inference_mode():
    expected = backend.ll._score_action_hiddens(torch.zeros(result['action_end'] - result['action_start'], 4096)).item()
assert result['score'] == expected
# Native dialogue 1 with injected outcomes is isolated from benchmark directories.
fixture = Path(tempfile.mkdtemp(prefix='dcgs-two-stage-offline-'))
args = runner.parse_args(['--ids','1','--output-dir',str(fixture),'--device','cpu'])
rows = base.select_dialogues(base.load_jsonl(args.dataset), args)
manifest = runner.manifest_for(args, rows, lock)
base.ensure_manifest(fixture / 'run_config.json', manifest)
calls = []
def execute(req):
    calls.append(copy.deepcopy(req))
    return fake_execute(req)
assert runner.generate_selected(args, manifest, rows, execute) == 0
assert len(calls) == 110
report = validate_dcgs(fixture)
assert report['dialogues'] == 1 and report['turns'] == 5
assert report['call_counts'] == {'generate':35,'hl_score':50,'ll_score':25}
before = {n:base.file_sha256(fixture/n) for n in ('events.jsonl','turns.jsonl','answers.jsonl')}
def no_calls(req):
    raise AssertionError('No-op resume called a model')
assert runner.generate_selected(args, manifest, rows, no_calls) == 0
assert before == {n:base.file_sha256(fixture/n) for n in before}
# Active experiment sources remain unchanged.
for name in ('tpo_zephyr_full_v7', 'smoothllm_zephyr_full_turn_resume'):
    active = json.loads((ROOT/'outputs/safedial_baseline'/name/'run_config.json').read_text())
    for source, digest in active['implementation_sha256'].items():
        assert base.file_sha256(ROOT/'scripts'/source) == digest, source
out = {'checkpoint_heads_loaded_strictly': list(heads),
       'all_heads_frozen': all(not p.requires_grad for h in heads.values() for p in h.parameters()),
       'real_tokenizer_action_span': [result['action_start'], result['action_end']],
       'real_token_heads_synthetic_residual_score_passed': True,
       'fixture_directory':str(fixture),'fixture_only_injected_calls':len(calls),
       'fixture_validation':report,'byte_identical_noop_resume':True,
       'active_experiment_sources_unchanged':True,'real_backbone_calls':0,'api_calls':0,
       'gpu_smoke_completed':False,'implementation_sha256':runner.implementation_hashes()}
(ROOT/'docs/verification/dcgs_two_stage_20260917/offline_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='implementation_sha256'},indent=2))
