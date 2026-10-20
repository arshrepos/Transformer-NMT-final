"""Evaluate a trained model: compute BLEU score and show example translations."""

import argparse
import os
import torch

from data.dataset import load_multi30k_split
from data.tokenizer import WordTokenizer, BPETokenizer, PAD_IDX, SOS_IDX
from model.transformer import Transformer
from utils.training import load_checkpoint
from utils.decoding import greedy_decode, beam_search_decode


def load_model_and_tokenizers(checkpoint_path, device="cpu"):
    """Load model from checkpoint, reconstructing architecture from saved config."""
    ckpt = load_checkpoint(checkpoint_path, device)
    model_config = ckpt["model_config"]
    train_config = ckpt["train_config"]

    model = Transformer.from_config(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizers from the same run directory
    run_dir = os.path.dirname(checkpoint_path)
    tok_dir = os.path.join(run_dir, "tokenizers")

    if train_config.get("tokenizer_type") == "bpe":
        src_tok = BPETokenizer()
        tgt_tok = BPETokenizer()
        src_tok.load(os.path.join(tok_dir, "src_bpe.json"))
        tgt_tok.load(os.path.join(tok_dir, "tgt_bpe.json"))
    else:
        src_tok = WordTokenizer()
        tgt_tok = WordTokenizer()
        src_tok.load(os.path.join(tok_dir, "src_word.json"))
        tgt_tok.load(os.path.join(tok_dir, "tgt_word.json"))

    return model, src_tok, tgt_tok, train_config


def translate_sentence(model, sentence, src_tok, tgt_tok, use_beam=False, beam_width=5, device="cpu"):
    """Translate a single sentence."""
    src_ids = src_tok.encode(sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if use_beam:
            output, attn = beam_search_decode(model, src_tensor, beam_width=beam_width, device=device)
        else:
            output, attn = greedy_decode(model, src_tensor, device=device)

    translated = tgt_tok.decode(output[0].tolist())
    return translated, attn


def compute_bleu(model, src_sentences, tgt_sentences, src_tok, tgt_tok,
                 use_beam=False, beam_width=5, device="cpu"):
    """Compute corpus-level BLEU using sacrebleu."""
    import sacrebleu

    hypotheses = []
    references = []

    for i, (src, ref) in enumerate(zip(src_sentences, tgt_sentences)):
        translated, _ = translate_sentence(model, src, src_tok, tgt_tok, use_beam, beam_width, device)
        hypotheses.append(translated)
        references.append(ref)

        if (i + 1) % 100 == 0:
            print(f"  Translated {i+1}/{len(src_sentences)}...")

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        lowercase=True,
    )
    return bleu, hypotheses, references


def main():
    p = argparse.ArgumentParser(description="Evaluate NMT model")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--data_dir", default="data/multi30k")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--use_beam", action="store_true")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--n_examples", type=int, default=10, help="Number of example translations to show")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, src_tok, tgt_tok, train_config = load_model_and_tokenizers(args.checkpoint, device)
    print(f"  Tokenizer: {train_config.get('tokenizer_type', 'word')}")
    print(f"  Decoding: {'beam (k=' + str(args.beam_width) + ')' if args.use_beam else 'greedy'}")

    print(f"\nLoading {args.split} data...")
    src_sentences, tgt_sentences = load_multi30k_split(args.data_dir, args.split)
    print(f"  {len(src_sentences)} sentence pairs")

    print("\nComputing BLEU...")
    bleu, hypotheses, references = compute_bleu(
        model, src_sentences, tgt_sentences, src_tok, tgt_tok,
        args.use_beam, args.beam_width, device
    )
    print(f"\n{'='*60}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"{'='*60}")

    print(f"\nExample translations ({args.n_examples}):")
    print("-" * 60)
    for i in range(min(args.n_examples, len(src_sentences))):
        print(f"  SRC: {src_sentences[i]}")
        print(f"  REF: {references[i]}")
        print(f"  HYP: {hypotheses[i]}")
        print()


if __name__ == "__main__":
    main()
