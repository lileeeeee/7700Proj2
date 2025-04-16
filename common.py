# common.py
import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu

def load_raw_texts(raw_dir="data/raw"):
    """
    Load all raw text files from the specified directory and return a single string.
    """
    texts = []
    for filename in os.listdir(raw_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)

def load_jsonl_file(file_path):
    """
    Load a JSONL file and return a list of texts.
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "prompt" in data and "completion" in data:
                    sample = data["prompt"] + " " + data["completion"]
                    texts.append(sample)
                elif "text" in data:
                    texts.append(data["text"])
                else:
                    print(f"Line {i+1} missing expected fields: {line}")
            except json.JSONDecodeError as e:
                print(f"Line {i+1} JSON decode error: {e}")
    return texts

def train_tokenizer_from_texts(texts, model_prefix="spm_model", vocab_size=10000):
    """
    Train a SentencePiece tokenizer from the given texts.
    """
    temp_file = "temp_corpus.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    cmd = (
        f"--input={temp_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--unk_id=0 "  
        f"--bos_id=1 "  
        f"--eos_id=2 "
    ).strip()
    spm.SentencePieceTrainer.Train(cmd)
    os.remove(temp_file)
    print(f"SentencePiece model saved as {model_prefix}.model")
    tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    return tokenizer

def chunk_tokens(token_ids, chunk_size=512):
    """
    divide token_ids into chunks of size chunk_size.
    """
    chunks = []
    for i in range(0, len(token_ids) - chunk_size + 1, chunk_size):
        chunks.append(token_ids[i: i + chunk_size])
    return chunks

class LMDataset(Dataset):
    """
    dataset for language modeling:
        - chunks: list of token sequences
    """
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        seq = self.chunks[idx]
        inp = torch.tensor(seq[:-1], dtype=torch.long)
        target = torch.tensor(seq[1:], dtype=torch.long)
        return inp, target

def evaluate_perplexity(model, data_loader, device):
    """
    calculate the perplexity of the model on the given data loader.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 对 RNN/LSTM 模型有 rnn 或 lstm 属性
            if hasattr(model, 'rnn') or hasattr(model, 'lstm'):
                logits, _ = model(inputs)
            else:
                logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def evaluate_bleu(model, test_texts, tokenizer, top_p=0.9, temperature=1.0, prompt_token_count=20, device='cpu', sample_size=100):
    """
    Evaluate the model using BLEU score:
      - test_texts: list of texts for evaluation
      - tokenizer: SentencePiece tokenizer
      - top_p: nucleus sampling parameter
      - temperature: temperature parameter for generation
      - prompt_token_count: number of tokens to use as prompt
      - device: device to use (e.g., 'cpu', 'cuda')
      - sample_size: number of samples to evaluate
    """
    if len(test_texts) > sample_size:
        print(len(test_texts))
        test_texts = random.sample(test_texts, sample_size)

    references = []
    hypotheses = []
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        prompt_ids = token_ids[:prompt_token_count] if len(token_ids) >= prompt_token_count else token_ids
        prompt = tokenizer.decode(prompt_ids)
        generated = model.generate(tokenizer, prompt, max_seq_length=len(token_ids), top_p=top_p, temperature=temperature, device=device)
        references.append([text.split()])
        hypotheses.append(generated.split())
    return corpus_bleu(references, hypotheses)

def train_model(model, train_loader, val_loader, device, num_epochs=30, learning_rate=1e-3, patience=5):
    """
    training loop for the model:
        - train_loader: training data loader
        - val_loader: validation data loader
        - device: device to use (e.g., 'cpu', 'cuda')
        - num_epochs: number of epochs to train
        - learning_rate: learning rate for optimizer
        - patience: early stopping patience
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []  # record training loss
    val_losses = []    # record validation loss

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if hasattr(model, 'rnn') or hasattr(model, 'lstm'):
                logits, _ = model(inputs)
            else:
                logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if hasattr(model, 'rnn') or hasattr(model, 'lstm'):
                    logits, _ = model(inputs)
                else:
                    logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    # plot the training and validation loss
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()