import json,os,shlex,subprocess,sys,tempfile
from pathlib import Path
root=Path.cwd();original=(root/'scripts/slurm/run_safedial_assistance_api_full_suite.sbatch').read_text()
results=[]
for mode,expected_calls,expected_exit in [('success',6,0),('source_gap',6,0),('judge_error',1,2),('guard_gap',1,2),('runtime_error',1,1)]:
 with tempfile.TemporaryDirectory(prefix='assist-suite-test-') as tmp:
  temp=Path(tmp);stub=temp/'python_stub'
  stub.write_text('#!'+sys.executable+'\n'+'''import sys,json,os
from pathlib import Path
if "scripts/judge_safedial_assistance_api.py" not in sys.argv:
 os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
with Path('calls.txt').open('a') as f:f.write('call\\n')
out=Path(sys.argv[sys.argv.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
mode=os.environ['SUITE_TEST_MODE']
if mode=='runtime_error':sys.exit(1)
(out/'aggregate.json').write_text(json.dumps({'available_responses_judged':mode!='judge_error','turn_counts':{'judge_error':1} if mode=='judge_error' else {'NO':1,'generation_missing':1}}))
(out/'combined_aggregate.json').write_text(json.dumps({'incomplete_dialogues':1}))
(out/'combined_turn_scores.jsonl').write_text(json.dumps({'input_sha256':'available','complete':mode not in ['guard_gap','judge_error']})+'\\n'+json.dumps({'input_sha256':'missing','complete':False})+'\\n')
(out/'frozen').mkdir(exist_ok=True)
(out/'frozen/inputs.jsonl').write_text(json.dumps({'input_sha256':'available','generation_status':'available'})+'\\n'+json.dumps({'input_sha256':'missing','generation_status':'missing'})+'\\n')
sys.exit(0 if mode=='success' else 2)
''');stub.chmod(0o755)
  script=original.replace('source /etc/profile.d/z00_lmod.sh','true').replace('module load Python/3.11.11-GCCcore-13.3.0','true').replace('cd '+str(root),'cd '+shlex.quote(str(temp))).replace('.venv/bin/python',shlex.quote(str(stub)))
  path=temp/'suite.sh';path.write_text(script)
  run=subprocess.run(['bash',str(path)],capture_output=True,text=True,env={**os.environ,'SUITE_TEST_MODE':mode})
  count=len((temp/'calls.txt').read_text().splitlines())
  assert (count,run.returncode)==(expected_calls,expected_exit),(mode,count,run.returncode,run.stdout,run.stderr)
  results.append({'case':mode,'mock_runner_calls':count,'suite_exit':run.returncode,'passed':True})
print(json.dumps(results,indent=2))
p=root/'docs/verification/safedial_assistance_api_full_20260924';(p/'suite_checks.json').write_text(json.dumps(results,indent=2)+'\n');(p/'verify_suite.py').write_text(Path(__file__).read_text())
