"""Offline, provenance-checked parser-v2 import of the completed API pilot."""
import json, sys, hashlib, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'scripts'))
import judge_safedial_assistance_api as api
root=Path.cwd();source=root/'outputs/assistance/gpt4o_mini_zephyr_pilot_v1'
dest=root/'outputs/assistance/gpt4o_mini_zephyr_pilot_v2'
evidence=root/'docs/verification/assistance_api_pilot_302438'
inventory=json.loads((evidence/'original_artifacts.json').read_text())
sha=api.local.source.file_hash
with api.local.source.lock_output(source), api.local.source.lock_output(dest):
    assert all(sha(source/n)==h for n,h in inventory.items())
    oldconfig=json.loads((source/'judge_config.json').read_text())
    oldfile=evidence/'judge_safedial_assistance_api_before_numbered_parser.py'
    assert sha(oldfile)==oldconfig['source_sha256'][str(Path(api.__file__).resolve())]
    oldargs=api.local.parse_args(['--run-dir','outputs/safedial_baseline/zephyr_7b_beta_full','--output-dir',str(source),'--ids','1','1217','1236'])
    _, oldrows=api.local.load_snapshot(oldargs)
    original=api.local.read_judgments(source,oldrows,api.local.source.digest(oldconfig))
    assert len(original)==15
    args=api.parse_args(['--run-dir','outputs/safedial_baseline/zephyr_7b_beta_full','--output-dir',str(dest),'--ids','1','1217','1236','--parallel','8'])
    api.native.load_env_file(args.env_file)
    endpoint=os.getenv('OPENAI_BASE_URL','').strip() or 'https://api.openai.com/v1'
    snapshot,rows=api.local.prepare(args)
    assert rows==oldrows
    config=api.config_for(args,snapshot,endpoint)
    allowed={'protocol','parser','source_sha256','snapshot_sha256'}
    assert {k:v for k,v in oldconfig.items() if k not in allowed}=={k:v for k,v in config.items() if k not in allowed}
    api.local.source.bind_config(dest,config)
    assert not (dest/'judgments.jsonl').exists(), 'Refuse to overwrite existing judgments'
    updated=[]; changes=[]
    for row in rows:
        old=original[row['input_sha256']]
        assert old['finish_reason']=='stop' and not old.get('refusal')
        label=api.parse_label(old['raw_output'])
        if old['status']=='success': assert label==old['label']
        else: changes.append({'dialogue_id':row['dialogue_id'],'turn_index':row['turn_index'],'old_status':old['status'],'new_label':label,'raw_output':old['raw_output']})
        updated.append(dict(old,status='success',label=label,error=None,
            judge_config_sha256=api.local.source.digest(config),
            imported_from={'job_id':302438,'source_dir':str(source),
              'journal_sha256':inventory['judgments.jsonl'],
              'judge_config_sha256':api.local.source.digest(oldconfig),
              'original_status':old['status'],'original_error':old.get('error'),
              'operation':'offline strict parser-v2 revalidation; raw API output unchanged; no new API call'}))
    api.local.write_rows(dest/'judgments.jsonl',updated)
    validated=api.read_judgments(dest,rows,api.local.source.digest(config))
    preflight=api.preflight(args,rows)
    aggregate=api.summarize(dest,rows,validated)
    assert aggregate['complete'] and aggregate['turn_counts']=={'NO':12,'YES':3}
    assert all(sha(source/n)==h for n,h in inventory.items())
    report={'source_job':302438,'source_slurm_state':'FAILED exit2 (original parser)',
      'source_dir':str(source),'source_files_sha256':inventory,
      'destination':str(dest),'operation':'offline parser-v2 revalidation into separate directory',
      'changed_prompt_or_request_parameters':False,'changed_raw_responses':False,
      'paid_api_calls':0,'original_artifacts_unchanged':True,'corrected_records':changes,
      'aggregate':aggregate,'preflight':preflight,
      'source_runtime':json.loads((source/'runtime.json').read_text())}
    api.local.source.atomic_json(dest/'import_provenance.json',report)
    api.local.source.atomic_json(evidence/'reparse_verification.json',report)
print(json.dumps({'complete':aggregate['complete'],'counts':aggregate['turn_counts'],'corrected':changes,'new_api_calls':0}))
