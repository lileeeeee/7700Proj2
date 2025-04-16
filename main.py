import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import random
from nltk.translate.bleu_score import corpus_bleu

###############################################
# 0. 辅助函数：Nucleus (Top-p) Sampling
###############################################
def nucleus_sample(next_logits, top_p=0.9):
    """
    对给定的 logits（形状 [1, vocab_size]）进行 nucleus sampling。
    1. 计算 softmax 得到概率分布。
    2. 按概率从大到小排序，并计算累计概率。
    3. 截断累计概率超过 top_p 的 token（保证至少保留一个 token）。
    4. 对截断后的分布进行归一化，然后进行多项式采样。
    返回采样得到的 token ID（int）。
    """
    # 计算概率分布
    probs = torch.softmax(next_logits, dim=-1)  # shape: [1, vocab_size]
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 创建截断 mask：累计概率大于 top_p 的 token 设为 True
    # 注意：为了保证至少保留一个 token，始终保留第一个 token
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = 0
    # 将 mask 对应位置的概率设为 0
    sorted_probs[sorted_mask] = 0
    # 对截断后的概率重新归一化
    normalized_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    # 从归一化后的分布中进行采样
    next_token = torch.multinomial(normalized_probs, num_samples=1)
    # 将采样得到的索引映射回原来的 token ID
    next_token_id = sorted_indices.gather(-1, next_token)
    return int(next_token_id.item())

###############################################
# 1. 数据加载与预处理
###############################################

def load_jsonl_file(file_path):
    """
    读取 .jsonl 文件，逐行解析 JSON，并返回所有文本组成的列表。
    根据数据格式：
      - 如果 JSON 包含 "prompt" 和 "completion" 两个字段，则将两者拼接成一个完整文本；
      - 如果只包含 "text" 字段，则直接提取；
    对于调试，每行格式有问题时会打印提示信息。
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
                    print(f"第 {i + 1} 行缺少预期字段: {line}")
            except json.JSONDecodeError as e:
                print(f"第 {i + 1} 行 JSON 解析错误: {e}")
    return texts

###############################################
# 2. SentencePiece 分词器训练或加载
###############################################

def train_tokenizer_from_texts(texts, model_prefix="spm_model", vocab_size=10000):
    """
    将所有文本写入临时文件，然后使用 SentencePiece 训练 BPE 分词器。
    训练完成后，会生成 model_prefix.model 和 model_prefix.vocab 文件。
    注意：移除了 --pad_id 参数，并在构造参数字符串后调用 strip()，以避免因末尾额外空格导致的错误。
    """
    temp_file = "temp_corpus.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    # 构造参数字符串，同时注意末尾不要有多余的空格
    cmd = (
        f"--input={temp_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--unk_id=0 "  # 未知字符 ID
        f"--bos_id=1 "  # 开始标记
        f"--eos_id=2 "  # 结束标记
    ).strip()

    spm.SentencePieceTrainer.Train(cmd)

    # 删除临时文件
    os.remove(temp_file)
    print(f"SentencePiece 模型已保存为 {model_prefix}.model")
    tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    return tokenizer

###############################################
# 3. 构造 Dataset（以语言模型为例）
###############################################

def chunk_tokens(token_ids, chunk_size=512):
    """
    按照不重叠方式，将 token 序列切分为固定长度块。
    若剩余长度不足 chunk_size，则舍弃（你也可以选择保留剩余部分）。
    """
    chunks = []
    for i in range(0, len(token_ids) - chunk_size + 1, chunk_size):
        chunks.append(token_ids[i: i + chunk_size])
    return chunks

class LMDataset(Dataset):
    """
    语言模型数据集，每个样本为一个 token 序列块，
    输入为该序列除最后一个 token，目标为该序列除第一个 token（即右移1位）。
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

###############################################
# 4. 模型定义
###############################################

class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, eos_token: int = 2):
        """
        eos_token 默认设置为 2，与 SentencePiece 训练时 eos_id 一致
        """
        super(VanillaRNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.eos_token = eos_token

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.rnn(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, tokenizer, prompt: str, max_seq_length: int = 100, top_p: float = 0.9,
                 device: str = 'cpu'):
        self.eval()
        token_ids = tokenizer.encode(prompt)
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        hidden = None
        generated = token_ids.copy()

        for _ in range(max_seq_length - len(token_ids)):
            logits, hidden = self.forward(tokens, hidden)
            next_logits = logits[:, -1, :]  # shape: [1, vocab_size]
            # 使用 nucleus (top-p) sampling 得到下一个 token
            next_token = nucleus_sample(next_logits, top_p=top_p)
            generated.append(next_token)
            if next_token == self.eos_token:
                break
            tokens = torch.tensor([[next_token]], dtype=torch.long).to(device)

        return tokenizer.decode(generated)


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, eos_token: int = 2):
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.eos_token = eos_token

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, tokenizer, prompt: str, max_seq_length: int = 100, top_p: float = 0.9,
                 device: str = 'cpu'):
        self.eval()
        token_ids = tokenizer.encode(prompt)
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        hidden = None
        generated = token_ids.copy()

        for _ in range(max_seq_length - len(token_ids)):
            logits, hidden = self.forward(tokens, hidden)
            next_logits = logits[:, -1, :]
            next_token = nucleus_sample(next_logits, top_p=top_p)
            generated.append(next_token)
            if next_token == self.eos_token:
                break
            tokens = torch.tensor([[next_token]], dtype=torch.long).to(device)

        return tokenizer.decode(generated)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, nhead: int, num_layers: int, hidden_dim: int,
                 max_seq_length: int = 512, eos_token: int = 2):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置嵌入（learnable embedding）
        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token

    def forward(self, x):
        """
        输入 x 的形状 [batch, seq_len]
        """
        batch_size, seq_len = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(x) + self.positional_embedding(position_ids)
        # Transformer 要求输入形状为 (seq_len, batch, embedding_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # 恢复为 (batch, seq_len, embedding_dim)
        logits = self.fc(x)
        return logits

    def generate(self, tokenizer, prompt: str, max_seq_length: int = 100, top_p: float = 0.9,
                 device: str = 'cpu'):
        self.eval()
        token_ids = tokenizer.encode(prompt)
        generated = token_ids.copy()
        tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_seq_length - len(token_ids)):
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :]
            next_token = nucleus_sample(next_logits, top_p=top_p)
            generated.append(next_token)
            if next_token == self.eos_token:
                break
            tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

        return tokenizer.decode(generated)

###############################################
# 5. 评价指标函数
###############################################

def evaluate_perplexity(model, data_loader, device):
    """
    计算模型在 data_loader 上的困惑度（Perplexity）。
    计算方式为 exp(average cross entropy loss)。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(model, TransformerLM):
                logits = model(inputs)
            else:
                logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()


def evaluate_bleu(model, test_texts, tokenizer, top_p=0.9, prompt_token_count=20, device='cpu'):
    """
    采用 corpus_bleu 计算 BLEU 分数：
    对于 test_texts 中的每个样本，取前 prompt_token_count 个 token 作为提示，
    生成完整文本，然后将生成结果（hypothesis）与原始文本（reference）进行比较。
    """
    references = []
    hypotheses = []
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < prompt_token_count:
            prompt_ids = token_ids
        else:
            prompt_ids = token_ids[:prompt_token_count]
        prompt = tokenizer.decode(prompt_ids)
        generated = model.generate(tokenizer, prompt, max_seq_length=len(token_ids), top_p=top_p, device=device)
        # 简单采用空格分词
        ref_tokens = text.split()
        hyp_tokens = generated.split()
        references.append([ref_tokens])
        hypotheses.append(hyp_tokens)
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

###############################################
# 6. 训练与验证函数
###############################################

def train_model(model, train_loader, val_loader, device, num_epochs=30, learning_rate=1e-3, patience=5):
    """
    训练模型，并在验证集上监控损失，实现早停策略。
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if isinstance(model, TransformerLM):
                logits = model(inputs)
            else:
                logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if isinstance(model, TransformerLM):
                    logits = model(inputs)
                else:
                    logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

###############################################
# 7. 主函数：数据加载、训练、评价、生成文本
###############################################

def main():
    # 超参数设置
    vocab_size = 10000
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    nhead = 8
    max_seq_length = 512   # 序列切分的长度
    batch_size = 128
    num_epochs = 30
    learning_rate = 1e-3
    device = torch.device("mps")  # 若使用 macOS MPS，请确保设备支持；否则改为 "cuda" 或 "cpu"

    # 1. 从 data 文件夹加载数据（请确保 data/train.jsonl 与 data/test.jsonl 存在且格式正确）
    train_file = os.path.join("data", "train.jsonl")
    test_file = os.path.join("data", "test.jsonl")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError("请检查 data/train.jsonl 和 data/test.jsonl 文件是否存在！")

    train_texts = load_jsonl_file(train_file)
    test_texts = load_jsonl_file(test_file)
    print(f"加载到 {len(train_texts)} 条训练文本 和 {len(test_texts)} 条测试文本")

    if len(train_texts) == 0:
        raise ValueError("训练文本为空，请检查 train.jsonl 文件格式是否正确。")

    # 2. 分词器：检查是否存在已训练的 SentencePiece 模型
    spm_model_file = "spm_model.model"
    if not os.path.exists(spm_model_file):
        print("未检测到已有 SentencePiece 模型，开始训练分词器……")
        tokenizer = train_tokenizer_from_texts(train_texts, model_prefix="spm_model", vocab_size=vocab_size)
    else:
        print("加载已有的 SentencePiece 模型")
        tokenizer = spm.SentencePieceProcessor(model_file=spm_model_file)

    # 3. 数据预处理：将所有文本拼接后分词，并按固定长度切分
    big_train_text = "\n".join(train_texts)
    big_test_text = "\n".join(test_texts)

    train_token_ids = tokenizer.encode(big_train_text)
    test_token_ids = tokenizer.encode(big_test_text)

    train_chunks = chunk_tokens(train_token_ids, chunk_size=max_seq_length)
    test_chunks = chunk_tokens(test_token_ids, chunk_size=max_seq_length)
    print(f"训练集共 {len(train_chunks)} 个序列块，测试集共 {len(test_chunks)} 个序列块")

    # 4. 构造 Dataset 和 DataLoader（将训练集再划分为训练集和验证集，前 90% 用于训练，后 10% 用于验证）
    split_idx = int(0.9 * len(train_chunks))
    train_dataset = LMDataset(train_chunks[:split_idx])
    val_dataset = LMDataset(train_chunks[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 5. 初始化并训练模型
    # 训练 Vanilla RNN 模型
    print("----- Training Vanilla RNN Model -----")
    rnn_model = VanillaRNNLM(vocab_size, embedding_dim, hidden_dim, num_layers)
    train_model(rnn_model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate)

    # 训练 LSTM 模型
    print("----- Training LSTM Model -----")
    lstm_model = LSTMLM(vocab_size, embedding_dim, hidden_dim, num_layers)
    train_model(lstm_model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate)

    # 训练 Transformer 模型
    print("----- Training Transformer Model -----")
    transformer_model = TransformerLM(vocab_size, embedding_dim, nhead, num_layers, hidden_dim, max_seq_length)
    train_model(transformer_model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=learning_rate)

    # 6. 生成文本示例
    prompt = "Which do you prefer? Dogs or cats?"
    print("\n----- 生成文本示例 -----")
    rnn_generated = rnn_model.generate(tokenizer, prompt, max_seq_length=100, top_p=0.9, device=device)
    print("RNN Generation:", rnn_generated)

    lstm_generated = lstm_model.generate(tokenizer, prompt, max_seq_length=100, top_p=0.9, device=device)
    print("LSTM Generation:", lstm_generated)

    transformer_generated = transformer_model.generate(tokenizer, prompt, max_seq_length=100, top_p=0.9, device=device)
    print("Transformer Generation:", transformer_generated)

    # 7. 评价指标计算
    # 构造测试集 DataLoader 用于计算困惑度
    test_dataset = LMDataset(test_chunks)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    rnn_ppl = evaluate_perplexity(rnn_model, test_loader, device)
    lstm_ppl = evaluate_perplexity(lstm_model, test_loader, device)
    transformer_ppl = evaluate_perplexity(transformer_model, test_loader, device)
    print("\n----- Evaluation Metrics -----")
    print("RNN Test Perplexity:", rnn_ppl)
    print("LSTM Test Perplexity:", lstm_ppl)
    print("Transformer Test Perplexity:", transformer_ppl)

    # 计算 BLEU 分数：对测试文本中每个样本，用前 20 个 token 作为 prompt，生成输出后与原文本比较
    rnn_bleu = evaluate_bleu(rnn_model, test_texts, tokenizer, top_p=0.9, prompt_token_count=20, device=device)
    lstm_bleu = evaluate_bleu(lstm_model, test_texts, tokenizer, top_p=0.9, prompt_token_count=20, device=device)
    transformer_bleu = evaluate_bleu(transformer_model, test_texts, tokenizer, top_p=0.9, prompt_token_count=20, device=device)
    print("RNN BLEU Score:", rnn_bleu)
    print("LSTM BLEU Score:", lstm_bleu)
    print("Transformer BLEU Score:", transformer_bleu)


if __name__ == "__main__":
    main()