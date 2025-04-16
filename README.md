This project implements three language models – Vanilla RNN, LSTM, and Transformer – in PyTorch, focusing on text generation from a public-domain literature dataset. The dataset has been pre-processed with SentencePiece for subword tokenization (vocab size = 10,000). The goal is to compare these models based on Perplexity and BLEU scores, as well as examine their generated samples.
# Usage
Train & Evaluate Model

    python main_modelname.py

This script will:

•	Load or train the SentencePiece tokenizer.
 
•	Load the data, build the dataset & dataloader.
 
•	Train the RNN model, save checkpoints, and do evaluation (perplexity & BLEU).
 
•	Print sample generations to the console.
