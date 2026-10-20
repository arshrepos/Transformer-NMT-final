"""Evaluate the LSTM Seq2Seq baseline on Multi30k."""

import argparse
import os
import torch
import sacrebleu

from data.download import download_multi30k
from data.dataset import load_multi30k_split
from data.tokenizer import WordTokenizer, SOS_IDX, EOS_IDX
from baseline.seq2seq import Seq2Seq


def load_model_and_tokenizers(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = Seq2Seq.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    run_dir = os.path.dirname(checkpoint_path)
    tok_dir = os.path.join(run_dir, "tokenizers")

    src_tok = WordTokenizer()
    tgt_tok = WordTokenizer()

    src_tok.load(os.path.join(tok_dir, "src_word.json"))
    tgt_tok.load(os.path.join(tok_dir, "tgt_word.json"))

    return model, src_tok, tgt_tok


def greedy_decode(model, src_ids, max_len=50):
    device = next(model.parameters()).device

    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        src_mask = src != model.pad_idx
        enc_outputs, h, c = model.encoder(src)

        current_token = torch.tensor(
            [[SOS_IDX]],
            dtype=torch.long,
            device=device
        )

        generated = [SOS_IDX]

        for _ in range(max_len):
            prediction, h, c, _ = model.decoder(
                current_token,
                h,
                c,
                enc_outputs,
                src_mask
            )

            next_token = prediction.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == EOS_IDX:
                break

            current_token = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=device
            )

    return generated


def beam_search_decode(model, src_ids, beam_width=5, max_len=50):
    device = next(model.parameters()).device

    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        src_mask = src != model.pad_idx
        enc_outputs, h0, c0 = model.encoder(src)

        # score, sequence, hidden, cell
        beams = [(0.0, [SOS_IDX], h0, c0)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for score, seq, h, c in beams:

                if seq[-1] == EOS_IDX:
                    completed.append((score, seq))
                    continue

                current_token = torch.tensor(
                    [[seq[-1]]],
                    dtype=torch.long,
                    device=device
                )

                prediction, new_h, new_c, _ = model.decoder(
                    current_token,
                    h,
                    c,
                    enc_outputs,
                    src_mask
                )

                log_probs = torch.log_softmax(
                    prediction,
                    dim=-1
                ).squeeze(0)

                topk_probs, topk_ids = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token = topk_ids[i].item()
                    token_score = topk_probs[i].item()

                    candidates.append((
                        score + token_score,
                        seq + [token],
                        new_h.clone(),
                        new_c.clone()
                    ))

            if not candidates:
                break

            candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            beams = candidates[:beam_width]

            if all(seq[-1] == EOS_IDX for _, seq, _, _ in beams):
                completed.extend(
                    (score, seq)
                    for score, seq, _, _ in beams
                )
                break

        completed.extend(
            (score, seq)
            for score, seq, _, _ in beams
        )

        completed.sort(
            key=lambda x: x[0] / len(x[1]),
            reverse=True
        )

        return completed[0][1]


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("Loading model...")
    model, src_tok, tgt_tok = load_model_and_tokenizers(
        args.checkpoint,
        device
    )

    print("Loading test data...")
    download_multi30k(args.data_dir)

    test_src, test_tgt = load_multi30k_split(
        args.data_dir,
        "test"
    )

    print(f"  {len(test_src)} sentence pairs")

    hypotheses = []
    references = []

    print()
    print(
        f"Computing BLEU ({args.decoding}"
        + (
            f", k={args.beam_width}"
            if args.decoding == "beam"
            else ""
        )
        + ")..."
    )

    for i, (src_sentence, tgt_sentence) in enumerate(
        zip(test_src, test_tgt)
    ):
        src_ids = src_tok.encode(src_sentence)

        if args.decoding == "greedy":
            output_ids = greedy_decode(
                model,
                src_ids,
                args.max_len
            )
        else:
            output_ids = beam_search_decode(
                model,
                src_ids,
                args.beam_width,
                args.max_len
            )

        translation = tgt_tok.decode(output_ids)

        hypotheses.append(translation)
        references.append(tgt_sentence)

        if (i + 1) % 100 == 0:
            print(
                f"  Translated {i + 1}/{len(test_src)}..."
            )

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        lowercase=True
    )

    print()
    print("=" * 60)
    print(f"BLEU Score: {bleu.score:.2f}")
    print("=" * 60)

    print()
    print("Example translations (10):")
    print("-" * 60)

    for i in range(min(10, len(test_src))):
        print(f"  SRC: {test_src[i]}")
        print(f"  REF: {references[i]}")
        print(f"  HYP: {hypotheses[i]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lstm_baseline/best_model.pt"
    )

    parser.add_argument(
        "--data_dir",
        default="data/multi30k"
    )

    parser.add_argument(
        "--decoding",
        choices=["greedy", "beam"],
        default="greedy"
    )

    parser.add_argument(
        "--beam_width",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=50
    )

    args = parser.parse_args()
    evaluate(args)