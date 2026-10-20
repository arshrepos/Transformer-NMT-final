"""Train the LSTM Seq2Seq baseline for comparison."""

import argparse
import os
import time
import torch
import torch.nn as nn

from data.download import download_multi30k
from data.dataset import load_multi30k_split, TranslationDataset, get_dataloader
from data.tokenizer import WordTokenizer, PAD_IDX, SOS_IDX, EOS_IDX
from baseline.seq2seq import Seq2Seq
from utils.training import save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/multi30k")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--run_name", default="lstm_baseline")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    download_multi30k(args.data_dir)
    train_src, train_tgt = load_multi30k_split(args.data_dir, "train")
    val_src, val_tgt = load_multi30k_split(args.data_dir, "val")

    # Word tokenizers
    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    tok_dir = os.path.join(run_dir, "tokenizers")
    os.makedirs(tok_dir, exist_ok=True)

    src_tok = WordTokenizer(min_freq=2)
    tgt_tok = WordTokenizer(min_freq=2)
    src_tok.build_vocab(train_src)
    tgt_tok.build_vocab(train_tgt)
    src_tok.save(os.path.join(tok_dir, "src_word.json"))
    tgt_tok.save(os.path.join(tok_dir, "tgt_word.json"))
    print(f"Vocab: src={src_tok.vocab_size}, tgt={tgt_tok.vocab_size}")

    train_ds = TranslationDataset(train_src, train_tgt, src_tok, tgt_tok)
    val_ds = TranslationDataset(val_src, val_tgt, src_tok, tgt_tok)
    train_dl = get_dataloader(train_ds, args.batch_size, shuffle=True)
    val_dl = get_dataloader(val_ds, args.batch_size, shuffle=False)

    model = Seq2Seq(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        pad_idx=PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTM Baseline parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    train_config = {
        "tokenizer_type": "word",
        "model_type": "seq2seq_lstm",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss, n = 0, 0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / n

        # Val
        model.eval()
        total_loss, n = 0, 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                total_loss += loss.item()
                n += 1
        val_loss = total_loss / n

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save with model_config for consistent loading
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "model_config": model.get_config(),
                "train_config": train_config,
            }
            path = os.path.join(run_dir, "best_model.pt")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(checkpoint, path)
            print(f"  Saved best: {path}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
