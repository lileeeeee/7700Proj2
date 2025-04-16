# utils.py
import torch

def nucleus_sample(next_logits, top_p=0.9):
    """
    对给定形状为 [1, vocab_size] 的 logits 进行 nucleus (top-p) sampling。
    1. 计算 softmax 得到概率分布；
    2. 将概率排序、累计求和；
    3. 截断累计概率超过 top_p 的 token（注意至少保留第一个 token）；
    4. 对截断后的分布归一化后进行采样，返回采样得到的 token ID（int）。
    """
    probs = torch.softmax(next_logits, dim=-1)  # shape: [1, vocab_size]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # mask: 超过 top_p 的位置设为 True；保证第一个 token 始终保留
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    sorted_probs[sorted_mask] = 0
    normalized_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    next_token = torch.multinomial(normalized_probs, num_samples=1)
    next_token_id = sorted_indices.gather(-1, next_token)
    return int(next_token_id.item())