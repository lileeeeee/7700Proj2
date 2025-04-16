# models/lstm_model.py
import torch
import torch.nn as nn
from torch.nn.functional import dropout

from utils import nucleus_sample

class LSTMLM(nn.Module):
    """
    LSTM language model example: Embedding -> LSTM -> Linear
    """
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, num_layers=2,
                 dropout=0.3, eos_token=2):
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # If num_layers>1, the built-in dropout will take effect between layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.eos_token = eos_token

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, tokenizer, prompt, max_seq_length=100, top_p=0.9, temperature=1.0, device='cpu'):
        self.eval()
        token_ids = tokenizer.encode(prompt)
        generated = token_ids.copy()
        tokens = torch.tensor([generated], dtype=torch.long).to(device)
        hidden = None

        for _ in range(max_seq_length - len(token_ids)):
            logits, hidden = self.forward(tokens, hidden)
            next_logits = logits[:, -1, :] / temperature
            next_token = nucleus_sample(next_logits, top_p=top_p)
            generated.append(next_token)
            if next_token == self.eos_token:
                break
            tokens = torch.tensor([[next_token]], dtype=torch.long).to(device)
        return tokenizer.decode(generated)