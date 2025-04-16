# main_lstm.py
import os
import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
from models.lstm_model import LSTMLM
from common import (load_jsonl_file, train_tokenizer_from_texts, chunk_tokens, LMDataset,
                    train_model, evaluate_perplexity, evaluate_bleu, load_raw_texts)


def main():
    # Define hyperparameters
    vocab_size = 10000
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    max_seq_length = 512
    batch_size = 128
    num_epochs = 30
    learning_rate = 1e-3
    device = torch.device("mps")

    # Check if the data files exist
    train_file = os.path.join("data", "train.jsonl")
    test_file = os.path.join("data", "test.jsonl")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError("Please check that data/train.jsonl and data/test.jsonl exist!")

    # Load training and testing data
    train_texts = load_jsonl_file(train_file)
    test_texts = load_jsonl_file(test_file)
    print(f"Loaded {len(train_texts)} train texts and {len(test_texts)} test texts")
    if not train_texts:
        raise ValueError("No training texts loaded.")

    # Load or train the SentencePiece tokenizer
    spm_model_file = "spm_model.model"
    if not os.path.exists(spm_model_file):
        print("Training SentencePiece tokenizer using raw texts from data/raw/...")
        raw_text = load_raw_texts("data/raw")
        tokenizer = train_tokenizer_from_texts([raw_text], model_prefix="spm_model", vocab_size=vocab_size)
    else:
        print("Loading SentencePiece tokenizer...")
        tokenizer = spm.SentencePieceProcessor(model_file=spm_model_file)

    # Tokenize and chunk the training and testing data
    big_train_text = "\n".join(train_texts)
    big_test_text = "\n".join(test_texts)
    train_token_ids = tokenizer.encode(big_train_text)
    test_token_ids = tokenizer.encode(big_test_text)
    train_chunks = chunk_tokens(train_token_ids, chunk_size=max_seq_length)
    test_chunks = chunk_tokens(test_token_ids, chunk_size=max_seq_length)
    print(f"Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")

    # Split the training data into training and validation sets
    split_idx = int(0.9 * len(train_chunks))
    train_dataset = LMDataset(train_chunks[:split_idx])
    val_dataset = LMDataset(train_chunks[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = LSTMLM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    print("----- Training LSTM Model -----")
    train_model(model, train_loader, val_loader, device, num_epochs, learning_rate)
    #
    # # Save the model
    model_save_path = "lstm_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the model for evaluation
    model.load_state_dict(torch.load(model_save_path))

    prompt = "What is your name?"
    print("----- Generating Text (LSTM) -----")
    generated = model.generate(tokenizer, prompt, max_seq_length=100, top_p=0.9, device=device, temperature=1.0)
    print("Generated:", generated)

    test_dataset_full = LMDataset(test_chunks)
    test_loader = DataLoader(test_dataset_full, batch_size=batch_size)
    ppl = evaluate_perplexity(model, test_loader, device)
    print("Test Perplexity:", ppl)
    bleu = evaluate_bleu(model, test_texts, tokenizer, top_p=0.9, prompt_token_count=20, device=device, sample_size=200, temperature=1.0)
    print("Test BLEU:", bleu)


if __name__ == "__main__":
    main()