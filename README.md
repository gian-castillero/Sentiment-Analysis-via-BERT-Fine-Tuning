# Sentiment Analysis via BERT Fine-Tuning

Fine-tuning a pretrained BERT encoder model for binary sentiment classification on movie reviews, with analysis of model errors and review ambiguity.

## Overview

This project fine-tunes Google's `bert-base-uncased` — a bidirectional encoder-only transformer pretrained on masked language modeling — for sentiment analysis on the Rotten Tomatoes movie review dataset. Unlike decoder-only GPT-style models, BERT attends to both preceding and subsequent tokens, making it well-suited for tasks that require understanding an entire sequence rather than generating new text.

## Model Architecture

A custom `SentimentBert` module wraps the pretrained BERT backbone with a single binary classification head:

```
BERT encoder (bert-base-uncased)
→ [CLS] token hidden state (dim 768)
→ Linear(768 → 1)
→ Sigmoid → probability of positive sentiment
```

The `[CLS]` token's final hidden state serves as an aggregate representation of the entire input sequence — BERT is trained to encode sequence-level meaning into this position, making it a natural input to a classification head.

## Dataset

[Rotten Tomatoes](https://huggingface.co/datasets/rotten_tomatoes) movie review dataset:
- **Training:** ~8,530 reviews
- **Validation:** ~1,066 reviews
- Labels: `1` = positive sentiment, `0` = negative sentiment

Reviews vary in length, so per-batch dynamic padding is used via a custom `collate_fn` passed to the PyTorch `DataLoader`. This avoids padding the entire dataset to the maximum sequence length, reducing wasted computation.

## Training

**Optimizer:** SGD, `lr=0.001`  
**Loss:** Binary Cross-Entropy (`BCELoss`)  
**Epochs:** 3  
**Batch size:** 8

Training loss is logged every 100 batches. Validation loss and accuracy are evaluated at the end of each epoch. The `attention_mask` is passed to BERT at every step to prevent the model from attending to padding tokens within a batch.

**Result: >80% validation accuracy**

## Error Analysis

Five misclassified validation examples are retrieved and examined. The dominant pattern: most errors occur on reviews with **genuinely ambiguous sentiment** — texts that use typically negative framing to describe positively received films (e.g. praising a horror film for being "dark and disturbing," or calling a plot twist "shocking" approvingly). The model's confusion on these examples reflects a real linguistic challenge rather than a simple model failure.

## Key Findings

- A pretrained BERT model with a lightweight classification head achieves >80% validation accuracy with only 3 epochs of fine-tuning on ~8,500 examples — demonstrating the power of transfer learning from large-scale pretraining.
- Sentiment classification is not always well-defined: domain-specific language (film criticism) uses conventional "negative" vocabulary positively, creating genuine ambiguity.

## Tech Stack

- Python 3
- PyTorch (`torch.nn`, `torch.optim`, `torch.utils.data`)
- Hugging Face `transformers` (`BertTokenizer`, `BertModel`)
- Hugging Face `datasets` (`rotten_tomatoes`)

## How to Run

```bash
pip install torch transformers datasets jupyter
jupyter notebook bert-sentiment.ipynb
```

GPU acceleration (CUDA) is recommended — a single epoch takes several minutes on CPU. The dataset and pretrained model download automatically on first run.
