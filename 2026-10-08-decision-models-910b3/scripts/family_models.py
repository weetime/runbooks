"""Explicit reference adapters for vision text-only scoring and two-NPU BF16 MoE."""
import torch
from decider.model import DecisionModel

class VisionTextModel(DecisionModel):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=False):
        torch.nn.Module.__init__(self)
        from decider.vision import VisionDecisionModel
        vision = VisionDecisionModel(name, dtype=dtype, grad_ckpt=False)
        self.vision = vision
        self.tok, self.lm = vision.tok, vision.lm
        self.register_buffer('letters', vision.letters, persistent=False)
        self.softcap = None

class ShardedMoeModel(DecisionModel):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=False):
        torch.nn.Module.__init__(self)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from decider.model import letter_ids
        self.tok = AutoTokenizer.from_pretrained(name)
        device_map = {'model.embed_tokens':'npu:0', 'model.rotary_emb':'npu:0',
                      'model.norm':'npu:1', 'lm_head':'npu:1'}
        device_map.update({f'model.layers.{i}':f'npu:{0 if i<20 else 1}' for i in range(40)})
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map,
                    attn_implementation='sdpa', experts_implementation='eager')
        self.lm.config.use_cache = False
        from npu_moe import install, grouped_experts
        from types import MethodType
        self.reference_experts_forward = install()
        self.expert_modules = [m for m in self.lm.modules() if type(m).__name__=="Qwen3_5MoeExperts"]
        for module in self.expert_modules:module.forward=MethodType(grouped_experts,module)
        self.device_map = device_map
        self.register_buffer('letters', torch.tensor(letter_ids(self.tok), device='npu:1'), persistent=False)
        self.softcap = None

    def to(self, *args, **kwargs):
        # Decider calls .to(input_device) after construction. Preserve dispatched layers.
        assert {str(p.device) for p in self.lm.parameters()} == {'npu:0','npu:1'}
        return self

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        dev = self.lm.lm_head.weight.device
        h = h.to(dev)
        hs = h[slot_batch.to(dev), slot_idx.to(dev)]
        W = self.lm.lm_head.weight[self.letters.to(dev)]
        logits = torch.nn.functional.linear(hs,W).float()
        return logits.masked_fill(torch.arange(W.shape[0],device=dev)[None,:] >= nopts.to(dev)[:,None],float('-inf'))

class ShardedChatModel(ShardedMoeModel):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=False):
        torch.nn.Module.__init__(self)
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from decider.model import letter_ids, softcap_value
        cfg=AutoConfig.from_pretrained(name);text=cfg.get_text_config()
        self.tok=AutoTokenizer.from_pretrained(name)
        n=text.num_hidden_layers
        # Keep tied Gemma embeddings and LM head together; Qwen's head is untied.
        head='npu:0' if cfg.tie_word_embeddings else 'npu:1'
        # Qwen AutoModelForCausalLM specializes the multimodal config into a text-only model.
        language_prefix='model' if cfg.model_type=='qwen3_5' else 'model.language_model'
        # Avoid a broad root entry: Transformers' per-module loading hooks can then
        # align child linear inputs to the root device despite a layer override.
        with torch.device('meta'):
            template=AutoModelForCausalLM.from_config(cfg)
        layer_prefix=language_prefix+'.layers.'
        device_map={}
        for key,parameter in template.named_parameters(remove_duplicate=False):
            if key.startswith(layer_prefix):continue
            module=key.rsplit('.',1)[0]
            device_map[module]=head if module=='lm_head' else ('npu:1' if module==language_prefix+'.norm' else 'npu:0')
        for key,buffer in template.named_buffers():
            if not key.startswith(layer_prefix):device_map.setdefault(key.rsplit('.',1)[0],'npu:0')
        del template
        device_map.update({f'{language_prefix}.layers.{i}':f'npu:{0 if i<n//2 else 1}' for i in range(n)})
        self.lm=AutoModelForCausalLM.from_pretrained(name,dtype=dtype,device_map=device_map,attn_implementation='sdpa')
        self.lm.config.use_cache=False;self.device_map=device_map
        modules=dict(self.lm.named_modules())
        assert all(k in modules for k in device_map), [k for k in device_map if k not in modules]
        for i in range(n):
            assert {str(p.device) for p in modules[f'{language_prefix}.layers.{i}'].parameters()}=={f'npu:{0 if i<n//2 else 1}'}
        self.register_buffer('letters',torch.tensor(letter_ids(self.tok),device=head),persistent=False)
        self.softcap=softcap_value(self.lm.config)

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        h=self.lm.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
        dev=self.lm.lm_head.weight.device;h=h.to(dev)
        hs=h[slot_batch.to(dev),slot_idx.to(dev)]
        W=self.lm.lm_head.weight[self.letters.to(dev)]
        logits=self.cap(torch.nn.functional.linear(hs,W).float())
        return logits.masked_fill(torch.arange(W.shape[0],device=dev)[None,:]>=nopts.to(dev)[:,None],float('-inf'))

def load_decider(path, kind, device):
    import decider.infer as inference
    original = inference.DecisionModel
    inference.DecisionModel = {'moe':ShardedMoeModel,'vision':VisionTextModel,'chat':ShardedChatModel}[kind]
    try:
        model = inference.Decider(str(path),device=device,dtype=torch.bfloat16,use_graphs=False)
    finally:
        inference.DecisionModel = original
    return model
