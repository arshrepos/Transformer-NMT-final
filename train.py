"""Main training script with toggleable flags for all components."""

import argparse
import os
import time
import torch
import torch.nn as nn

from data.download import download_multi30k
from data.dataset import load_multi30k_split, TranslationDataset, get_dataloader
from data.tokenizer import WordTokenizer, BPETokenizer, PAD_IDX
from model.transformer import Transformer
from utils.training import (
    LabelSmoothingLoss, WarmupDecayScheduler, save_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer NMT (EN→DE)")
    # Data
    p.add_argument("--data_dir", default="data/multi30k")
    # Architecture
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_enc_layers", type=int, default=3)
    p.add_argument("--n_dec_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    # Toggleable features (Phase 3)
    p.add_argument("--use_bpe", action="store_true", help="Use BPE tokenization instead of word-level")
    p.add_argument("--bpe_vocab_size", type=int, default=8000)
    p.add_argument("--use_label_smoothing", action="store_true", help="Use label smoothing loss")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_warmup", action="store_true", help="Use warmup+decay LR schedule")
    p.add_argument("--warmup_steps", type=int, default=4000)
    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--run_name", default="transformer_base")
    return p.parse_args()


def build_tokenizers(args, train_src, train_tgt):
    tok_dir = os.path.join(args.checkpoint_dir, args.run_name, "tokenizers")
    os.makedirs(tok_dir, exist_ok=True)

    if args.use_bpe:
        src_tok = BPETokenizer(vocab_size=args.bpe_vocab_size)
        tgt_tok = BPETokenizer(vocab_size=args.bpe_vocab_size)
        src_path = os.path.join(tok_dir, "src_bpe.json")
        tgt_path = os.path.join(tok_dir, "tgt_bpe.json")
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            src_tok.load(src_path)
            tgt_tok.load(tgt_path)
            print(f"  Loaded BPE tokenizers from {tok_dir}")
        else:
            print("  Training BPE tokenizers...")
            src_tok.train(train_src, save_path=src_path)
            tgt_tok.train(train_tgt, save_path=tgt_path)
    else:
        src_tok = WordTokenizer(min_freq=2)
        tgt_tok = WordTokenizer(min_freq=2)
        src_path = os.path.join(tok_dir, "src_word.json")
        tgt_path = os.path.join(tok_dir, "tgt_word.json")
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            src_tok.load(src_path)
            tgt_tok.load(tgt_path)
            print(f"  Loaded word tokenizers from {tok_dir}")
        else:
            print("  Building word vocabularies...")
            src_tok.build_vocab(train_src)
            tgt_tok.build_vocab(train_tgt)
            src_tok.save(src_path)
            tgt_tok.save(tgt_path)

    print(f"  Source vocab: {src_tok.vocab_size}, Target vocab: {tgt_tok.vocab_size}")
    return src_tok, tgt_tok


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, clip_grad, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits, _ = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if scheduler is not None:
            scheduler.step()

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits, _ = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Download data
    print("Downloading Multi30k...")
    download_multi30k(args.data_dir)

    # Load splits
    print("Loading data splits...")
    train_src, train_tgt = load_multi30k_split(args.data_dir, "train")
    val_src, val_tgt = load_multi30k_split(args.data_dir, "val")
    print(f"  Train: {len(train_src)}, Val: {len(val_src)}")

    # Build tokenizers
    print("Building tokenizers...")
    src_tok, tgt_tok = build_tokenizers(args, train_src, train_tgt)

    # Datasets & dataloaders
    train_ds = TranslationDataset(train_src, train_tgt, src_tok, tgt_tok)
    val_ds = TranslationDataset(val_src, val_tgt, src_tok, tgt_tok)
    train_dl = get_dataloader(train_ds, args.batch_size, shuffle=True)
    val_dl = get_dataloader(val_ds, args.batch_size, shuffle=False)

    # Model
    model = Transformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    if args.use_label_smoothing:
        criterion = LabelSmoothingLoss(tgt_tok.vocab_size, PAD_IDX, args.label_smoothing)
        print(f"  Using label smoothing (ε={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        print("  Using standard cross-entropy loss")

    # Optimizer & scheduler
    initial_lr = 0.0 if args.use_warmup else args.lr

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = None
    if args.use_warmup:
        scheduler = WarmupDecayScheduler(optimizer, args.d_model, args.warmup_steps)
        print(f"  Using warmup+decay schedule (warmup_steps={args.warmup_steps})")
    else:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        print("  Using ReduceLROnPlateau scheduler")

    # Train config for checkpoint
    train_config = {
        "model_type": "transformer",
        "use_bpe": args.use_bpe,
        "bpe_vocab_size": args.bpe_vocab_size,
        "use_label_smoothing": args.use_label_smoothing,
        "label_smoothing": args.label_smoothing,
        "use_warmup": args.use_warmup,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "tokenizer_type": "bpe" if args.use_bpe else "word",
    }

    # Training loop
    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training: {args.run_name}")
    print(f"  BPE={args.use_bpe}, LabelSmoothing={args.use_label_smoothing}, Warmup={args.use_warmup}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scheduler, args.clip_grad, device)
        val_loss = evaluate(model, val_dl, criterion, device)
        elapsed = time.time() - t0

        lr = scheduler.current_lr if scheduler else optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_config,
                os.path.join(run_dir, "best_model.pt"),
                scheduler=scheduler,
            )

        # Plateau scheduler step (only if not using warmup)
        if not args.use_warmup:
            plateau_scheduler.step(val_loss)

    # Save final
    save_checkpoint(
        model, optimizer, args.epochs, train_config,
        os.path.join(run_dir, "final_model.pt"),
        scheduler=scheduler,
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
