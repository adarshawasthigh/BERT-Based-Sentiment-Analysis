# BERT Sentiment Analysis on IMDB Reviews

## Problem
Binary sentiment classification (Positive / Negative) on IMDB movie reviews.

## Model
- BERT Base (bert-base-uncased)
- Fine-tuned using HuggingFace Trainer API

## Dataset
- IMDB Reviews Dataset (50k samples)

## Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Features
- Early stopping
- Precision–Recall curve
- Confusion matrix
- GPU/CPU compatible

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/inference.py
