import importlib.util, json, hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('downloaded_ll_critic',root/'src/value/ll_token_critic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
expected='a0a4cc34e4a096122adeb21d5cff2acfe6cf760c22f70b9a76733555b9e1313f'
reports={}
for name in ['cares','wildjailbreak']:
 path=root/'models'/f'{name}_ll_token_critic/ll_token_critic.pt'
 assert path.stat().st_size==134287165
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
 ckpt=torch.load(path,map_location='cpu',weights_only=True)
 metadata={k:v for k,v in ckpt.items() if isinstance(v,(str,int,float,bool,type(None)))}
 assert ckpt['hidden_size']==module.HIDDEN_SIZE and ckpt['mlp_width_mult']==module.MLP_WIDTH_MULT
 assert ckpt['objective'] in module.OBJECTIVES
 assert ckpt['n_examples']>0
 counts={}
 heads=[]
 for key in ['harm_head','follow_head']:
  head=module.make_head(ckpt['hidden_size'],ckpt['mlp_width_mult'])
  head.load_state_dict(ckpt[key],strict=True); head.eval()
  assert all(torch.isfinite(t).all() for t in head.state_dict().values())
  counts[key]=sum(t.numel() for t in head.parameters())
  heads.append(head)
 critic=module.LLTokenCritic(*heads,encoder=None,tokenizer=None,device='cpu',objective=ckpt['objective'],span_k=int(ckpt['k']))
 with torch.inference_mode():
  result=critic._score_action_hiddens(torch.zeros(max(int(ckpt['k']),4),ckpt['hidden_size']))
  assert result.numel()==1 and torch.isfinite(result).all()
 reports[name]={'size_bytes':path.stat().st_size,'sha256':expected,'keys':list(ckpt),'metadata':metadata,'head_parameter_counts':counts,'strict_head_load_passed':True,'head_weights_finite':True,'synthetic_zero_residual_score_finite':True,'backbone_loaded':False}
print(json.dumps(reports,indent=2))
