# Transformer-Based Neural Machine Translation (English → German)

A complete, from-scratch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for English-to-German translation. **No `nn.Transformer`, no pretrained models, no HuggingFace checkpoints.**

Trained on the [Multi30k](https://github.com/multi30k/dataset) dataset (~29k training pairs).

---

## Project Structure

```
├── model/
│   ├── attention.py        # Scaled dot-product & multi-head attention
│   ├── layers.py           # Positional encoding, encoder/decoder layers
│   └── transformer.py      # Full encoder-decoder Transformer
├── data/
│   ├── download.py         # Multi30k download script
│   ├── tokenizer.py        # Word-level & BPE tokenizers
│   └── dataset.py          # PyTorch Dataset & DataLoader
├── utils/
│   ├── decoding.py         # Greedy & beam search decoding
│   └── training.py         # Label smoothing, LR warmup, checkpointing
├── baseline/
│   ├── seq2seq.py          # LSTM Seq2Seq with Bahdanau attention
│   └── train_baseline.py   # Baseline training script
├── scripts/
│   └── ablation.py         # Automated ablation study
├── train.py                # Main training script (toggleable flags)
├── evaluate.py             # BLEU evaluation + example translations
├── app.py                  # Streamlit interactive demo
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Transformer

```bash
# Baseline (word-level tokenization, standard cross-entropy)
python train.py --run_name baseline_word --epochs 30

# With BPE tokenization
python train.py --run_name bpe --use_bpe --epochs 30

# With all features (BPE + label smoothing + warmup)
python train.py --run_name all_features --use_bpe --use_label_smoothing --use_warmup --epochs 30
```

### 3. Evaluate

```bash
# Greedy decoding
python evaluate.py --checkpoint checkpoints/all_features/best_model.pt

# Beam search
python evaluate.py --checkpoint checkpoints/all_features/best_model.pt --use_beam --beam_width 5
```

### 4. Train LSTM Baseline (for comparison)

```bash
python -m baseline.train_baseline --epochs 30
```

### 5. Run Ablation Study

```bash
python -m scripts.ablation
```

### 6. Launch Interactive Demo

```bash
streamlit run app.py
```

## Toggleable Features (Phase 3)

Each feature is a command-line flag, tested in isolation via the ablation script:

| Flag | Description |
|------|-------------|
| `--use_bpe` | BPE subword tokenization (trained from scratch via `tokenizers` library) |
| `--use_label_smoothing` | Label smoothing (ε=0.1, from the original paper) |
| `--use_warmup` | Linear warmup + inverse-sqrt decay LR schedule |

## BLEU Results

*Run `python -m scripts.ablation` to populate this table with your results.*

| Configuration | Greedy BLEU | Beam (k=5) BLEU |
|---------------|-------------|------------------|
| Word-level baseline | — | — |
| + BPE | — | — |
| + Label smoothing | — | — |
| + LR warmup | — | — |
| All features | — | — |
| LSTM baseline | — | — |

## Example Translations

*After training, run `python evaluate.py --checkpoint <path> --n_examples 5` to see examples like:*

```
SRC: A dog is running through the grass.
REF: Ein Hund rennt durch das Gras.
HYP: Ein Hund läuft durch das Gras.
```

## Design Decisions

- **Checkpoint config**: Every checkpoint saves a `model_config` dict so any loading code (including the Streamlit demo) reconstructs the exact architecture — no hardcoded defaults
- **Multi30k's own splits**: Uses the dataset's pre-defined train/val/test files — no custom splitting
- **Ablation methodology**: Each feature toggled independently against the same baseline
- **Custom modules**: All attention, encoder, decoder layers built from scratch using only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, etc.

## Interactive Demo

The Streamlit app (`app.py`) reads all settings from the checkpoint:
- Model architecture (d_model, n_heads, layers, vocab sizes)
- Tokenizer type and vocabulary
- No independent hardcoded UI defaults

Features:
- Type any English sentence → see German translation
- Cross-attention heatmap showing which source words the model attended to
- Toggle between greedy and beam search decoding
- Select from any trained checkpoint
