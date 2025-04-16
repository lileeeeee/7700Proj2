# models/transformer_model.py
import torch
import torch.nn as nn
from utils import nucleus_sample

# class TransformerLM(nn.Module):
#     def __init__(self, vocab_size: int, embedding_dim: int, nhead: int, num_layers: int, hidden_dim: int,
#                  max_seq_length: int = 512, eos_token: int = 2):
#         super(TransformerLM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(embedding_dim, vocab_size)
#         self.max_seq_length = max_seq_length
#         self.eos_token = eos_token
#
#     def forward(self, x):
#         batch_size, seq_len = x.size()
#         position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
#         x = self.embedding(x) + self.positional_embedding(position_ids)
#         x = x.transpose(0, 1)  # (seq_len, batch, embedding_dim)
#         x = self.transformer(x)
#         x = x.transpose(0, 1)  # (batch, seq_len, embedding_dim)
#         logits = self.fc(x)
#         return logits
#
#     def generate(self, tokenizer, prompt: str, max_seq_length: int = 100, top_p: float = 0.9, temperature: float = 1.0, device: str = 'cpu'):
#         self.eval()
#         token_ids = tokenizer.encode(prompt)
#         generated = token_ids.copy()
#         tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
#         for _ in range(max_seq_length - len(token_ids)):
#             logits = self.forward(tokens)
#             next_logits = logits[:, -1, :] / temperature
#             next_token = nucleus_sample(next_logits, top_p=top_p)
#             generated.append(next_token)
#             if next_token == self.eos_token:
#                 break
#             tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
#         # filtered_generated = [t for t in generated if t not in [1, self.eos_token]]
#         return tokenizer.decode(generated)

class TransformerLM(nn.Module):
    """
    Transformer 语言模型示例：Embedding -> (Positional Embedding) -> TransformerEncoder -> Linear
    """

    def __init__(self, vocab_size=10000, embedding_dim=256, nhead=8, num_layers=2, hidden_dim=1024,
                 dropout=0.1, max_seq_length=512, eos_token=2):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
        # 构造 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                                   dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        返回: logits, shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        # embedding + positional embedding
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x_embed = self.embedding(x) + self.positional_embedding(pos_ids)
        # TransformerEncoder 默认输入 [seq_len, batch_size, embed_dim] 如果 batch_first=False
        x_embed = x_embed.transpose(0, 1)  # shape [seq_len, batch_size, embed_dim]
        out = self.transformer(x_embed)  # [seq_len, batch_size, embed_dim]
        out = out.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        logits = self.fc(out)
        return logits

    def generate(self, tokenizer, prompt, max_seq_length=100, top_p=0.9, temperature=1.0, device='cpu'):
        self.eval()
        token_ids = tokenizer.encode(prompt)
        generated = token_ids.copy()
        tokens = torch.tensor([generated], dtype=torch.long).to(device)

        for _ in range(max_seq_length - len(token_ids)):
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :] / temperature  # 取最后一个时间步，做温度缩放
            next_token = nucleus_sample(next_logits, top_p=top_p)
            generated.append(next_token)
            if next_token == self.eos_token:
                break
            tokens = torch.tensor([generated], dtype=torch.long).to(device)
        return tokenizer.decode(generated)