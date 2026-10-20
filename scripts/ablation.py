"""Aggregate completed NMT experiments into ablation_results.json.

This script does NOT train or evaluate models.
It records results from experiments that have already been completed.
"""

import json


RESULTS = {
    "baseline_word": {
        "description": "Word-level Transformer",
        "greedy_bleu": 20.96,
        "beam_bleu": 23.79,
        "status": "completed",
    },

    "bpe": {
        "description": "Transformer + BPE",
        "greedy_bleu": 24.40,
        "beam_bleu": 25.93,
        "status": "completed",
    },

    "label_smoothing": {
        "description": "Transformer + Label Smoothing",
        "greedy_bleu": 21.59,
        "beam_bleu": 24.60,
        "status": "completed",
    },

    "warmup": {
        "description": "Transformer + Warmup/Decay",
        "greedy_bleu": 19.70,
        "beam_bleu": 22.92,
        "status": "completed",
    },

    "all_features": {
        "description": "Transformer + BPE + Label Smoothing + Warmup",
        "greedy_bleu": 24.20,
        "beam_bleu": 26.65,
        "status": "completed",
    },

    "lstm_baseline": {
        "description": "LSTM Seq2Seq + Bahdanau Attention",
        "greedy_bleu": 22.99,
        "beam_bleu": 25.70,
        "status": "completed",
    },

    "bpe_label_smoothing": {
        "description": "Transformer + BPE + Label Smoothing",
        "greedy_bleu": None,
        "beam_bleu": None,
        "status": "not_run",
    },
}


def main():
    output_file = "ablation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(f"{'Configuration':<25} {'Greedy':>12} {'Beam':>12}")
    print("-" * 55)

    for name, result in RESULTS.items():
        greedy = (
            f"{result['greedy_bleu']:.2f}"
            if result["greedy_bleu"] is not None
            else "N/A"
        )

        beam = (
            f"{result['beam_bleu']:.2f}"
            if result["beam_bleu"] is not None
            else "N/A"
        )

        print(f"{name:<25} {greedy:>12} {beam:>12}")

    print("-" * 55)
    print(f"\nSaved results to: {output_file}")
    print("No training or model evaluation was performed.")


if __name__ == "__main__":
    main()