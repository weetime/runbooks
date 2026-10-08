"""Ascend grouped matmul expert path. The router and expert weights stay unchanged."""
import torch
import torch_npu

def grouped_experts(self, hidden_states, top_k_index, top_k_weights):
    k = top_k_index.shape[1]
    order = torch.argsort(top_k_index.flatten().float(), stable=True)
    token = torch.div(order, k, rounding_mode='floor')
    counts = torch.bincount(top_k_index.flatten(), minlength=self.num_experts)
    groups = counts.cumsum(0).to(torch.int64)
    selected = hidden_states[token]
    gate_up = torch_npu.npu_grouped_matmul(
        [selected], [self.gate_up_proj.transpose(1, 2)],
        group_list=groups, split_item=3, group_type=0)[0]
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = self.act_fn(gate) * up
    output = torch_npu.npu_grouped_matmul(
        [hidden], [self.down_proj.transpose(1, 2)],
        group_list=groups, split_item=3, group_type=0)[0]
    output = output * top_k_weights.flatten()[order, None]
    result = torch.zeros_like(hidden_states)
    result.index_add_(0, token, output.to(result.dtype))
    return result

def install():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
    reference = Qwen3_5MoeExperts.forward
    Qwen3_5MoeExperts.forward = grouped_experts
    return reference

if __name__ == '__main__':
    import json
    from types import SimpleNamespace
    torch.manual_seed(20261008)
    for experts, tokens in [(8, 32), (256, 24), (256, 384)]:
        h, intermediate, k = 128, 64, 4
        obj = SimpleNamespace(num_experts=experts,
            gate_up_proj=(torch.randn(experts,2*intermediate,h)*.04).to('npu:0',torch.bfloat16),
            down_proj=(torch.randn(experts,h,intermediate)*.04).to('npu:0',torch.bfloat16),
            act_fn=torch.nn.functional.silu)
        x=torch.randn(tokens,h).to('npu:0',torch.bfloat16)
        idx=torch.stack([torch.randperm(experts)[:k] for _ in range(tokens)]).to('npu:0')
        weights=torch.softmax(torch.randn(tokens,k),-1).to('npu:0',torch.bfloat16)
        expected=torch.zeros_like(x)
        for e in range(experts):
            rows, positions=torch.where(idx==e)
            if not rows.numel():continue
            a,b=torch.nn.functional.linear(x[rows],obj.gate_up_proj[e]).chunk(2,-1)
            y=torch.nn.functional.linear(obj.act_fn(a)*b,obj.down_proj[e])
            expected.index_add_(0,rows,y*weights[rows,positions,None])
        actual=grouped_experts(obj,x,idx,weights)
        torch.npu.synchronize()
        delta=(actual-expected).float().abs()
        torch.testing.assert_close(actual.float(),expected.float(),rtol=.05,atol=.004)
        print(json.dumps({'experts':experts,'tokens':tokens,'max_abs_error':delta.max().item(),'mean_abs_error':delta.mean().item(),'status':'passed'}),flush=True)
