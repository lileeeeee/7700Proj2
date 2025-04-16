# utils.py
import torch

def nucleus_sample(next_logits, top_p=0.9):
    """
    Nucleus sampling:
    - next_logits: [1, vocab_size]
    - top_p: cumulative probability threshold
    - next_token: sampled token id
    """
    probs = torch.softmax(next_logits, dim=-1)  # shape: [1, vocab_size]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find the index where cumulative probability exceeds top_p
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    sorted_probs[sorted_mask] = 0
    normalized_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    next_token = torch.multinomial(normalized_probs, num_samples=1)
    next_token_id = sorted_indices.gather(-1, next_token)
    return int(next_token_id.item())