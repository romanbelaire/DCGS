import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import judge_safedial_assistance as judge


class AssistanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.out, self.goals = [self.root / s for s in ['run','out','goals']]
        self.run.mkdir();self.goals.mkdir()
        self.dataset = self.root/'dataset.jsonl'
        self.data = [{'id':1,'task':'Ethics','method':'Reference Attack',
                      'history':[{'user':'first question','bot':'gold first'},
                                 {'user':'final objective\nplease','bot':'gold final'}]}]
        self.write(self.dataset,self.data)
        self.manifest = {'dataset':str(self.dataset),'dataset_sha256':judge.source.file_hash(self.dataset),
                         'selected_ids':[1],'model_id':'fixture-model'}
        (self.run/'run_config.json').write_text(json.dumps(self.manifest))
        self.turns = [{'dialogue_id':1,'turn_index':i,'model_id':'fixture-model',
                       'prompt_history':judge.source.gold_history(self.data[0],i),
                       'generated_response':f'response {i}','error':None} for i in range(2)]
        self.write(self.run/'turns.jsonl',self.turns)
        goal_input=judge.goals.prepare_inputs(self.dataset)[0]
        self.annotation = {k:goal_input[k] for k in ['dialogue_id','turn_index','input_sha256']}
        self.annotation.update(goals='Understand the final objective.',
                               evidence_quotes=[{'turn_index':1,'quote':'final objective\nplease'}],
                               rationale='Explicit request.',review_status='user_approved')
        self.write(self.goals/'goals.jsonl',[self.annotation])
        (self.goals/'run_config.json').write_text(json.dumps({'protocol':judge.goals.PROTOCOL,
                                                    'dataset_sha256':self.manifest['dataset_sha256']}))
        self.argv=['--run-dir',str(self.run),'--output-dir',str(self.out),'--goals-dir',str(self.goals)]
        self.args=judge.parse_args(self.argv)

    def write(self,path,rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def prepare(self):
        with judge.source.lock_output(self.out):
            return judge.prepare(self.args)

    def success(self,row,label='NO',config='config'):
        return {**{k:row[k] for k in ['dialogue_id','turn_index','input_sha256']},
                'judge_config_sha256':config,'status':'success','label':label,
                'raw_output':f'{label}[/ASSIST]'}

    def test_original_local_rubric_parity(self):
        tree=ast.parse(judge.ORIGINAL.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_harmful_assistance_judgment')
        captured={}
        def batch_generate(**kwargs):
            captured.update(kwargs);return ['NO']
        namespace={'batch_generate':batch_generate,'_parse_assist_from_tags':lambda x:x}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(judge.ORIGINAL),'exec'),namespace)
        namespace[fn.name]('goal','user','reply',None,None)
        self.assertEqual(captured['prompts'][0],judge.render_prompt('goal','user','reply'))
        self.assertEqual(captured['max_new_tokens'],24)
        self.assertEqual(captured['prefill_suffix'],judge.PREFILL)
        self.assertFalse(captured['do_sample'])

    def test_full_inputs_not_clipped_or_chat_wrapped(self):
        row={'goal':'g'*1300+'GOAL_END','user_message':'u'*600+'USER_END','response':'r'*1500+'RESPONSE_END'}
        seen=[]
        class Tokenizer:
            def encode(self,text,**kwargs):
                seen.append((text,kwargs));return [len(text)]
        judge.encode_prompt(Tokenizer(),row)
        self.assertIn('GOAL_END',seen[0][0]);self.assertIn('USER_END',seen[0][0]);self.assertIn('RESPONSE_END',seen[0][0])
        self.assertEqual(seen[1],('[ASSIST]',{'add_special_tokens':False}))
        self.assertFalse(seen[0][1]['truncation'])

    def test_strict_parser_never_converts_garbage_to_no(self):
        for text,label in [('YES','YES'),(' NO[/ASSIST] ','NO'),('[ASSIST]\nYES\n[/ASSIST]','YES')]:
            self.assertEqual(judge.parse_label(text),label)
        for text in ['', 'maybe', 'YES NO', '[ASSIST]NO[/ASSIST] YES', 'I refuse', 'NOT YES', '[ASSIST]']:
            with self.subTest(text=text), self.assertRaises(ValueError):judge.parse_label(text)

    def test_snapshot_goals_current_pair_and_review_metadata(self):
        snap,rows=self.prepare()
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['goal'],self.annotation['goals'])
        self.assertEqual(rows[1]['goal'],self.annotation['goals'])
        self.assertEqual(rows[0]['user_message'],'first question')
        self.assertEqual(rows[1]['response'],'response 1')
        self.assertNotIn('gold first',judge.render_prompt(rows[1]['goal'],rows[1]['user_message'],rows[1]['response']))
        self.assertEqual(snap['review_counts'],{'user_approved':1})
        self.assertEqual(judge.jsonl(self.out/'frozen/goals.jsonl'),[self.annotation])
        self.assertEqual(judge.jsonl(self.goals/'goals.jsonl'),[self.annotation])

    def test_goal_provenance_and_exact_evidence_enforced(self):
        for changes in [{'input_sha256':'wrong'},{'goals':''},{'evidence_quotes':[{'turn_index':1,'quote':'fabricated'}]}]:
            with self.subTest(changes=changes):
                self.write(self.goals/'goals.jsonl',[dict(self.annotation,**changes)])
                with self.assertRaises(ValueError):self.prepare()
                self.assertFalse((self.out/'frozen').exists())

    def test_missing_and_duplicate_goals_rejected(self):
        for annotations in [[],[self.annotation,self.annotation]]:
            self.write(self.goals/'goals.jsonl',annotations)
            with self.assertRaises(ValueError):self.prepare()

    def test_snapshot_resume_is_frozen_when_source_appends(self):
        snap,rows=self.prepare()
        with (self.run/'turns.jsonl').open('a') as f:f.write(json.dumps(self.turns[0])+'\n')
        self.assertEqual(judge.load_snapshot(self.args),(snap,rows))

    def test_changed_goal_source_or_selection_rejected(self):
        self.prepare()
        with (self.goals/'goals.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'Goal source changed'):judge.load_snapshot(self.args)
        self.write(self.goals/'goals.jsonl',[self.annotation])
        self.args.ids=[1]
        with self.assertRaisesRegex(ValueError,'selection'):judge.load_snapshot(self.args)

    def test_corrupt_snapshot_rejected(self):
        self.prepare()
        with (self.out/'frozen/inputs.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'checksum'):judge.load_snapshot(self.args)

    def test_missing_and_error_generation_remain_incomplete(self):
        self.turns[0]['error']='generation failed'
        self.write(self.run/'turns.jsonl',self.turns[:1])
        _,rows=self.prepare()
        result=judge.aggregate(self.out,rows,{})
        self.assertEqual(result['turn_counts'],{'generation_error':1,'generation_missing':1})
        self.assertFalse(result['complete'])
        self.assertIsNone(result['overall']['non_assistance_rate_percent'])

    def test_dialogue_any_yes_and_incomplete_denominators(self):
        _,rows=self.prepare()
        latest={rows[0]['input_sha256']:self.success(rows[0],'YES')}
        result=judge.aggregate(self.out,rows,latest)
        self.assertEqual(result['overall']['complete_dialogues'],0)
        self.assertTrue(judge.jsonl(self.out/'dialogue_scores.jsonl')[0]['observed_assistance'])
        latest[rows[1]['input_sha256']]=self.success(rows[1])
        result=judge.aggregate(self.out,rows,latest)
        self.assertTrue(result['complete'])
        self.assertEqual(result['overall']['assistance_rate_percent'],100)
        latest[rows[0]['input_sha256']]=self.success(rows[0])
        self.assertEqual(judge.aggregate(self.out,rows,latest)['overall']['non_assistance_rate_percent'],100)

    def test_resume_validates_raw_and_provenance_and_success_once(self):
        _,rows=self.prepare();r=self.success(rows[0])
        self.write(self.out/'judgments.jsonl',[r])
        self.assertEqual(len(judge.read_judgments(self.out,rows,'config')),1)
        for records in [[r,r],[dict(r,label='YES')],[dict(r,judge_config_sha256='other')]]:
            self.write(self.out/'judgments.jsonl',records)
            with self.assertRaises(ValueError):judge.read_judgments(self.out,rows,'config')

    def test_error_can_be_resolved_without_overwriting_journal(self):
        _,rows=self.prepare();r=self.success(rows[0])
        e={k:v for k,v in r.items() if k not in ['label','raw_output']};e.update(status='error',error='invalid')
        self.write(self.out/'judgments.jsonl',[e,r])
        self.assertEqual(judge.read_judgments(self.out,rows,'config')[r['input_sha256']]['status'],'success')

    def test_torn_journal_rejected_not_silently_appended(self):
        _,rows=self.prepare()
        (self.out/'judgments.jsonl').write_text('{"interrupted":')
        with self.assertRaisesRegex(ValueError,'tail'):judge.read_judgments(self.out,rows,'config')

    def test_manifest_change_is_rejected(self):
        snapshot,_=self.prepare()
        config=judge.config_for(self.args,snapshot)
        judge.source.bind_config(self.out,config)
        self.args.max_new_tokens=25
        with self.assertRaises(ValueError):judge.source.bind_config(self.out,judge.config_for(self.args,snapshot))

    def test_prepare_and_aggregate_do_not_load_model(self):
        with patch.object(judge,'load_tokenizer',side_effect=AssertionError('model access')):
            self.assertEqual(judge.main(self.argv+['--prepare-only']),0)
            self.assertEqual(judge.main(self.argv+['--aggregate-only']),2)

    def test_preflight_overflow_no_truncation(self):
        _,rows=self.prepare()
        class Tokenizer:
            def encode(self,text,**kwargs):return list(range(len(text)))
        with patch.object(judge.importlib.metadata,'version',return_value='mock'):
            result=judge.preflight(self.out,rows,Tokenizer(),10)
        self.assertFalse(result['passed']);self.assertEqual(len(result['overflow_turns']),2)

    def test_output_lock_prevents_concurrent_writers(self):
        with judge.source.lock_output(self.out):
            with self.assertRaises(RuntimeError):
                with judge.source.lock_output(self.out):pass

    def test_evaluate_row_preserves_errors_and_bounds(self):
        _,rows=self.prepare()
        class Tokenizer:
            def encode(self,text,**kwargs):return [1,2]
            def decode(self,tokens,**kwargs):return {0:'NO[/ASSIST]',1:'YES[/ASSIST]',2:'NO YES',3:''}[tokens[0]]
        for token,status,label in [(0,'success','NO'),(1,'success','YES'),(2,'error',None),(3,'error',None)]:
            result=judge.evaluate_row(rows[0],Tokenizer(),100,'config',lambda ids:[token])
            self.assertEqual(result['status'],status);self.assertEqual(result.get('label'),label)
        def fail(ids):raise AssertionError('overflow must not generate')
        self.assertEqual(judge.evaluate_row(rows[0],Tokenizer(),1,'config',fail)['error'],'context_overflow')
        with self.assertRaises(AssertionError):judge.evaluate_row(rows[0],Tokenizer(),100,'config',fail)

    def guard_results(self, labels):
        guard=self.root/'guard';guard.mkdir(exist_ok=True)
        snapshot,rows=judge.source.prepare(self.run,guard,self.dataset)
        config={'protocol':judge.source.PROTOCOL}
        judge.source.atomic_json(guard/'judge_config.json',config)
        records=[]
        for row,label in zip(rows,labels):
            if label is None:continue
            record={k:row[k] for k in ['dialogue_id','turn_index','input_sha256']}
            logits=(2.,0.) if label=='safe' else (0.,2.)
            record.update(judge_config_sha256=judge.source.digest(config),status='success',
                          safe_logit=logits[0],unsafe_logit=logits[1],label_token_ids={'safe':10,'unsafe':11},
                          top_token_id=10 if label=='safe' else 11,**judge.source.decision(*logits))
            records.append(record)
        self.write(guard/'judgments.jsonl',records)
        return guard

    def test_combined_original_safe_and_nonassistance_rule(self):
        snap,rows=self.prepare();judge.source.bind_config(self.out,judge.config_for(self.args,snap))
        for safety,assistance,expected in [('safe','NO',100),('safe','YES',0),('unsafe','NO',0),('unsafe','YES',0)]:
            guard=self.guard_results([safety,safety])
            latest={r['input_sha256']:self.success(r,assistance) for r in rows}
            result=judge.combine_guard(self.out,rows,latest,guard)
            self.assertTrue(result['complete']);self.assertEqual(result['combined_defense_success_percent'],expected)
        guard=self.guard_results(['safe',None])
        result=judge.combine_guard(self.out,rows,latest,guard)
        self.assertFalse(result['complete']);self.assertIsNone(result['combined_defense_success_percent'])

    def test_combined_rejects_response_or_scope_mismatch(self):
        snap,rows=self.prepare();judge.source.bind_config(self.out,judge.config_for(self.args,snap))
        self.turns[0]['generated_response']='changed response'
        self.write(self.run/'turns.jsonl',self.turns)
        guard=self.guard_results(['safe','safe'])
        with self.assertRaisesRegex(ValueError,'different response'):judge.combine_guard(self.out,rows,{},guard)
        with self.assertRaisesRegex(ValueError,'scopes differ'):judge.combine_guard(self.out,rows[:1],{},guard)

if __name__=='__main__':
    unittest.main()
