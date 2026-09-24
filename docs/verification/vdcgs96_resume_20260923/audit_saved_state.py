import sys,json,contextlib,io,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import run_safedial_dcgs_wildjailbreak as runner
import safedial_dcgs_wildjailbreak as adapter
import safedial_dcgs_run_state as state
from safedial_dcgs_replay_isolation import isolated_replay
p=Path('outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3')
out=Path('docs/verification/vdcgs96_resume_20260923');out.mkdir(exist_ok=True)
files=['turns.jsonl','events.jsonl','attempts.jsonl','failures.jsonl','run_config.json','answers.jsonl']
def hashes(): return {name:runner.file_sha256(p/name) for name in files}
start=time.time(); before=hashes()
m=json.loads((p/'run_config.json').read_text())
assert m['source_sha256']==runner.source_hashes()
args=type('Args',(),dict(method='vdcgs',device='cuda:0',dataset=Path(m['dataset']),seed=0,on_turn_error='record-and-continue'))()
rows=runner.load_jsonl(Path(m['dataset'])); selected=[r for r in rows if r['id'] in m['selected_ids']]
lock=runner.load_artifact_lock(); prospective=json.loads(json.dumps(runner.manifest_for(args,selected,lock)))
assert prospective==m, 'Prospective full manifest differs'
config=adapter.configuration(m['method'],m['effective_config']['device'],m['artifacts'])
tok=runner.tokenizer_for(m['artifacts'])
old=state.validate_turn; count=0

def progress(*args):
 global count
 old(*args);count+=1
 if count%250==0: print('AUDITED',count,'seconds',round(time.time()-start),file=sys.stderr,flush=True)
state.validate_turn=progress
print('START full state audit',flush=True)
with isolated_replay(),contextlib.redirect_stdout(io.StringIO()):
 result=state.audit_state(p,selected,m,config,tok)
after=hashes();assert before==after
report={'source_match':True,'full_prospective_manifest_match':True,'saved_files_unchanged':True,'sha256':before,'coverage':state.coverage(result),'blocker':result['blocker'],'uncommitted_groups':len(result['uncommitted']),'unknown_cost_calls':result['unknown_cost_calls'],'elapsed_seconds':time.time()-start,'model_calls':0,'replay_fix_sha256':runner.file_sha256(Path('scripts/safedial_dcgs_replay_isolation.py'))}
report['native_export_matches']=state.read_records(p/'answers.jsonl')==runner.answers_for(selected,m,result['records'])
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
