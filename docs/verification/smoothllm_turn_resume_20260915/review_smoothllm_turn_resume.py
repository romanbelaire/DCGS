import copy,json,sys,tempfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
sys.path.insert(0,str(Path.cwd()/'tests'))
import run_safedial_baseline as base
import run_safedial_smoothllm as legacy
import run_safedial_smoothllm_turn_resume as runner
from test_safedial_smoothllm import decoded
from validate_safedial_smoothllm_turn_resume import validate_turn_resume
source=Path('outputs/safedial_baseline/smoothllm_zephyr_full')
manifest=json.loads((source/'run_config.json').read_text())
for name,digest in manifest['implementation_sha256'].items():
 assert base.file_sha256(Path('scripts')/name)==digest,name
root=Path(tempfile.mkdtemp(prefix='smoothllm-turn-resume-real-review-'))
# Read a single byte snapshot; discard only a possibly incomplete final append for this read-only audit.
data=(source/'turns.jsonl').read_bytes()
if data and not data.endswith(b'\n'):
 data=data[:data.rfind(b'\n')+1]
(root/'live_turns_snapshot.jsonl').write_bytes(data)
records=[json.loads(l) for l in data.splitlines()]
args=legacy.parse_args(['--output-dir',str(source)])
rows=base.select_dialogues(base.load_jsonl(args.dataset),args)
assert manifest==legacy.manifest_for(args,rows)
latest=runner.checked_turns(records,args,manifest,rows)
# Reuse real saved dialogue 344, with a fixture selection/run ID only.
fixture_source=root/'legacy_fixture'; fixture_source.mkdir()
old_args=legacy.parse_args(['--output-dir',str(fixture_source),'--ids','344'])
selected=base.select_dialogues(base.load_jsonl(old_args.dataset),old_args)
old=legacy.manifest_for(old_args,selected)
old_id=base.stable_id(json.dumps(old,sort_keys=True))
saved=[]
for i in range(len(selected[0]['history'])):
 r=copy.deepcopy(latest[344,i]);r.update(run_id=old_id,answer_id=base.stable_id(old_id,344));saved.append(r)
assert sum(bool(r.get('error')) for r in saved)==1
base.ensure_manifest(fixture_source/'run_config.json',old)
runner.atomic_jsonl(fixture_source/'turns.jsonl',saved)
new_args=runner.parse_args(['--resume-from',str(fixture_source),'--output-dir',str(root/'continuation'),'--ids','344','--retry-errors'])
new=runner.manifest_for(new_args,selected)
with runner.single_writer(new_args.output_dir):
 runner.prepare_output(new_args,new,selected)
 imported=base.load_jsonl(new_args.output_dir/'turns.jsonl')
 calls=[]
 def injected(messages,seed):
  calls.append({'messages':messages,'seed':seed})
  return decoded('I cannot help with that.')
 assert runner.generate_dialogues(new_args,new,selected,injected)==0
 assert len(calls)==8
 assert all(c['messages'][:-1]==base.gold_messages(selected[0]['history'],4)[:-1] for c in calls)
 result=base.load_jsonl(new_args.output_dir/'turns.jsonl')
 assert imported[:4]==result[:4]
 report=validate_turn_resume(new_args.output_dir)
 assert (report['turns'],report['candidates'])==(5,40)
 def no_calls(messages,seed):raise AssertionError('No-op generated')
 assert runner.generate_dialogues(new_args,new,selected,no_calls)==0
report={'source_directory':str(source.resolve()),'snapshot_directory':str(root),
 'snapshot_sha256':hashlib.sha256(data).hexdigest(),'latest_saved_turns_audited':len(latest),
 'successful_saved_turns_audited':sum(not r.get('error') for r in latest.values()),
 'error_keys':[list(k) for k,v in latest.items() if v.get('error')],
 'fixture_dialogue':344,'fixture_only_new_calls':len(calls),'successful_real_turns_unchanged':4,
 'fixture_validation':report,'no_op_resume_passed':True,'benchmark_calls':0,
 'live_source_hashes_unchanged':True}
(root/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
