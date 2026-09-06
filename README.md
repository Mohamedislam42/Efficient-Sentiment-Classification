# Efficient Sentiment Classification — LoRA & Knowledge Distillation

A parameter-efficient NLP pipeline that combines **LoRA (Low-Rank Adaptation)**
with **teacher-student knowledge distillation** to build a compact sentiment
classifier — cutting trainable parameters by **97.85%** while retaining
**96.9%** of a full fine-tuned teacher model's Macro F1 score.

## Overview

Full fine-tuning of a language model for a downstream task is expensive and
unnecessary for most of the model's parameters. This project explores how far
a small, LoRA-adapted student model can go when it's trained not just on
labels, but on the *soft knowledge* of a larger teacher model.

Three models are trained and compared on each dataset:

1. **Frozen Baseline** — pretrained encoder, no fine-tuning, classifier head only.
2. **Teacher** — GPT-2, fully fine-tuned (all parameters trainable).
3. **LoRA Student** — DistilGPT-2 with LoRA adapters, trained via knowledge
   distillation from the Teacher.

## Datasets

- **SST-5** — 5-class sentiment (Very Negative → Very Positive)
- **IMDB** — binary sentiment (Positive / Negative)

## Architecture

- Base encoder: GPT-2 (teacher) / DistilGPT-2 (student), from Hugging Face
  Transformers.
- **LoRA adapters** applied to `c_attn` and `c_proj` layers (`r=16`,
  `alpha=32`, dropout `0.1`) via the PEFT library.
- **Attention pooling** layer that learns a weighted combination of token
  hidden states, masked for variable-length sequences.
- Final representation blends **80% attention-pooled + 20% mean-pooled**
  hidden states, followed by LayerNorm, dropout, and an MLP classification
  head.

## Results

| Model | Trainable Params | Macro F1 (relative to Teacher) |
|---|---|---|
| Frozen Baseline | Lowest | Lowest |
| GPT-2 Teacher (full fine-tune) | 100% | 100% (reference) |
| LoRA Student (distilled) | **2.15%** | **96.9%** |

The LoRA student achieves near-teacher performance while training a fraction
of the parameters, cutting compute and memory requirements substantially.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), scikit-learn,
pandas, Weighted Random Sampling for class imbalance.

## Running the Notebook

```bash
pip install -r requirements.txt
```

Open `Project_Code.ipynb` in Jupyter or Google Colab and run the cells in
order. The notebook covers:

- Data loading and tokenization (SST-5 / IMDB)
- Model architecture definition
- Baseline, teacher, and LoRA student training loops
- Evaluation, per-class reports, and error analysis
- Inference demo on custom sentences

## Live Demo

Try the trained model directly: *(link to Hugging Face Space)*

## Author

Mohamed Islam Elshourbagy — [GitHub](https://github.com/Mohamedislam42) · [LinkedIn](https://www.linkedin.com/in/mohamed-islam-292269241/)
