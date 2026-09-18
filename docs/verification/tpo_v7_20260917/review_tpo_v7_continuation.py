import copy,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
sys.path.insert(0,str(Path.cwd()/'tests'))
import run_safedial_baseline as base
import run_safedial_tpo as runner
import run_safedial_tpo_full as full
import safedial_tpo as tpo
from test_safedial_tpo import fake_execute
from validate_safedial_tpo_full import validate_full
p=Path('outputs/safedial_baseline/tpo_zephyr_full_v7')
source=Path('outputs/safedial_baseline/tpo_zephyr_full_v6')
manifest=json.loads((p/'run_config.json').read_text())
continuation=json.loads((p/'continuation.json').read_text())
assert manifest['implementation_sha256']==full.implementation_hashes()
for name,digest in continuation['original_file_sha256'].items():
 assert base.file_sha256(source/name)==digest,name
 assert base.file_sha256(p/'provenance/254463'/name)==digest,name
for file in (source/'provenance').rglob('*'):
 if file.is_file(): assert base.file_sha256(file)==base.file_sha256(p/'provenance/254463/provenance'/file.relative_to(source/'provenance'))
a=base.load_jsonl(source/'events.jsonl'); b=base.load_jsonl(p/'events.jsonl')
assert len(a)==len(b)==4439
changed=[]
for old,new in zip(a,b):
 assert old['event']['request']==new['event']['request']
 assert old['event']['result']==new['event']['result']
 if old['event']!=new['event']: changed.append([new['dialogue_id'],new['turn_index'],new['event']['index']])
assert changed==[[26,3,18]]
old_turns=[x for x in base.load_jsonl(source/'turns.jsonl') if not x.get('error')]
new_turns=base.load_jsonl(p/'turns.jsonl')
assert len(old_turns)==len(new_turns)==130
for old,new in zip(old_turns,new_turns):
 old=copy.deepcopy(old); old.update(run_id=new['run_id'],answer_id=new['answer_id'])
 assert old==new
old_answers=base.load_jsonl(source/'answers.jsonl'); new_answers=base.load_jsonl(p/'answers.jsonl')
assert len(old_answers)==len(new_answers)==25
for old,new in zip(old_answers,new_answers):
 old=copy.deepcopy(old); old['answer_id']=new['answer_id']; assert old==new
fixture=Path(tempfile.mkdtemp(prefix='tpo-v7-offline-continuation-'))
args=full.parse_args(['--output-dir',str(fixture),'--ids',','.join(map(str,range(1,27)))])
rows=base.select_dialogues(base.load_jsonl(args.dataset),args)
m=full.manifest_for(args,rows,manifest['artifacts'])
run_id=base.stable_id(json.dumps(m,sort_keys=True))
base.ensure_manifest(fixture/'run_config.json',m)
for entry in b: entry['run_id']=run_id
for entry in new_turns: entry.update(run_id=run_id,answer_id=base.stable_id(run_id,entry['dialogue_id']))
runner.atomic_jsonl(fixture/'events.jsonl',b)
runner.atomic_jsonl(fixture/'turns.jsonl',new_turns)
calls=[]
def execute(req):
 calls.append(copy.deepcopy(req))
 return fake_execute(req)
assert runner.generate_selected(args,m,rows,execute)==0
assert {k:calls[0][k] for k in ('kind','stage','iteration','slot')}=={'kind':'reward','stage':'score','iteration':0,'slot':3}
body=tpo.extract_update(a[-1]['event']['result']['message'])
assert calls[0]['messages'][-1]['content']==body
report=validate_full(fixture)
assert report['optimizer_duplicate_block_warnings']==2
assert report['optimizer_first_improvement_warnings']==1
assert len(calls)==report['turns']*34-4439
before={n:base.file_sha256(fixture/n) for n in ['events.jsonl','turns.jsonl','answers.jsonl']}
def no_call(req): raise AssertionError('No-op resume attempted generation')
assert runner.generate_selected(args,m,rows,no_call)==0
assert before=={n:base.file_sha256(fixture/n) for n in before}
# Preserve a runnable copy of this review, with fixtures clearly separated from real outputs.
result={'original_source_and_archive_hashes_passed':True,'nested_provenance_hashes_passed':True,
 'all_4439_raw_requests_and_results_unchanged':True,'reclassified_event_keys':changed,
 'all_130_successful_turns_unchanged_except_run_identifiers':True,
 'all_25_answers_unchanged_except_run_identifiers':True,'fixture_directory':str(fixture),
 'fixture_only_injected_calls':len(calls),'fixture_validation':report,
 'fixture_no_op_resume_unchanged':True,'benchmark_model_calls':0,
 'next_real_call':'reward score recovered answer at dialogue 26/index 3 round 0 slot 3'}
(p/'offline_preparation_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
