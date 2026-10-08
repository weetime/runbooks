from pathlib import Path
import time
from types import SimpleNamespace
ROOT=Path("/workspace/decision-models-20261008/models")
def load(model,device="npu:0"):
 a=SimpleNamespace(model=model,device=device)
 result={}
 save=lambda:None
 import torch
 torch.set_num_threads(16)
 if a.device.startswith('npu'):
  import torch_npu
  torch.npu.set_device(a.device)
  # PyTorch's fused encoder fastpath falls back to CPU on torch_npu 2.10. Use supported eager primitives.
  torch.backends.mha.set_fastpath_enabled(False)
  result['mha_fastpath_enabled']=torch.backends.mha.get_fastpath_enabled()
  # Ascend FlashAttention requires explicit [B,1,Q,K] masks, unlike PyTorch's broadcast [B,1,1,K].
  import torch.nn.functional as functional
  original_sdpa=functional.scaled_dot_product_attention
  mask_expansions=[]
  def compatible_sdpa(query,key,value,*args,**kwargs):
   positional=list(args);mask=positional[0] if positional else kwargs.get('attn_mask')
   if mask is not None and query.device.type=='npu' and mask.ndim==4 and mask.shape[-2]==1 and query.shape[-2]>1:
    mask_expansions.append({'from':list(mask.shape),'query_tokens':query.shape[-2]})
    mask=mask.expand(*mask.shape[:-2],query.shape[-2],mask.shape[-1]).contiguous()
    if positional:positional[0]=mask
    else:kwargs['attn_mask']=mask
   return original_sdpa(query,key,value,*positional,**kwargs)
  functional.scaled_dot_product_attention=compatible_sdpa
  result['sdpa_mask_expansions']=mask_expansions
 t=time.monotonic()
 if a.model.startswith('decider'):
  from decider.infer import Decider
  path=ROOT/'Mapika'/('decider-4b' if a.model=='decider' else a.model)
  if a.model in ('decider-35b-a3b','decider-2b-vision','decider-chat-qwen3.6-27b','decider-chat-gemma4-31b'):
   import sys;sys.path.insert(0,str(Path(__file__).resolve().parent))
   from family_models import load_decider
   kind='moe' if a.model=='decider-35b-a3b' else ('vision' if a.model=='decider-2b-vision' else 'chat')
   agent=load_decider(path,kind,a.device)
   result['adapter']='two-NPU layer dispatch' if kind in ('moe','chat') else 'vision model text-only with reference Decider prompt/readout'
   result['parameter_devices']=sorted({str(p.device) for p in agent.m.parameters()})
  else:agent=Decider(str(path),device=a.device,use_graphs=False)
  infer=lambda state,questions:agent.system_one(state,questions)
  result['actual_device']=str(next(agent.m.parameters()).device)
 else:
  import laya
  path=ROOT/'convaiinnovations/laya'
  if a.model=='laya-ml':path=path/'multilingual'
  if a.model=='laya-typed':path=path/'typed-decisions'
  agent=laya.load(str(path),device=a.device,backend='eager')
  infer=lambda state,questions:agent.predict(state,questions)
  result['actual_device']=str(next(agent.model.parameters()).device)
 result['path']=str(path);result['load_seconds']=time.monotonic()-t
 print('LOADED',a.model,result['actual_device'],result['load_seconds'],flush=True);save()
 if a.device.startswith('npu') and not result['actual_device'].startswith('npu'):raise RuntimeError('model silently fell back from NPU')
 return infer,agent,result,torch
