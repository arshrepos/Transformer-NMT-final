"""Streamlit demo: translate English → German with attention heatmap."""

import os
import streamlit as st
import torch
import numpy as np

st.set_page_config(page_title="EN→DE Translator", layout="wide")


@st.cache_resource
def load_model(checkpoint_path):
    """Load model and tokenizers from checkpoint config (no hardcoded defaults)."""
    from utils.training import load_checkpoint
    from model.transformer import Transformer
    from data.tokenizer import WordTokenizer, BPETokenizer

    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model_config = ckpt["model_config"]
    train_config = ckpt["train_config"]

    model = Transformer.from_config(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    return model, src_tok, tgt_tok, model_config, train_config

def group_bpe_tokens(tokens, attention, axis):
    """
    Group BPE pieces ending in </w> into complete words and
    average their attention.

    axis=0 -> group target rows
    axis=1 -> group source columns
    """
    words = []
    groups = []

    current_word = []
    current_indices = []

    for i, token in enumerate(tokens):
        # Remove end-of-word marker
        clean_token = token.replace("</w>", "")

        current_word.append(clean_token)
        current_indices.append(i)

        # </w> means this is the end of the word
        if token.endswith("</w>"):
            words.append("".join(current_word))
            groups.append(current_indices)
            current_word = []
            current_indices = []

    # Handle any remaining pieces
    if current_word:
        words.append("".join(current_word))
        groups.append(current_indices)

    if axis == 0:
        # Group rows
        grouped_attention = np.stack(
            [attention[group, :].mean(axis=0) for group in groups],
            axis=0
        )
    else:
        # Group columns
        grouped_attention = np.stack(
            [attention[:, group].mean(axis=1) for group in groups],
            axis=1
        )

    return words, grouped_attention


def get_bpe_tokens_from_ids(tokenizer, ids):
    """Convert BPE token IDs into their raw BPE token strings."""
    return [
        tokenizer.tokenizer.id_to_token(int(idx))
        for idx in ids
    ]


def translate(model, sentence, src_tok, tgt_tok, use_beam, beam_width):
    from utils.decoding import greedy_decode, beam_search_decode
    from data.tokenizer import SOS_IDX, EOS_IDX, PAD_IDX

    src_ids = src_tok.encode(sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long)

    with torch.no_grad():
        if use_beam:
            output, attn = beam_search_decode(
                model,
                src_tensor,
                beam_width=beam_width
            )
        else:
            output, attn = greedy_decode(
                model,
                src_tensor
            )

    translated = tgt_tok.decode(output[0].tolist())

    # ---------------------------------------------------------
    # Build attention heatmap
    # ---------------------------------------------------------

    if attn is not None:

        # Average over attention heads
        attn_matrix = attn[0].mean(dim=0).cpu().numpy()

        # -----------------------------------------------------
        # SOURCE TOKENS
        # -----------------------------------------------------

        src_content_indices = [
            i for i, idx in enumerate(src_ids)
            if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
        ]

        if hasattr(src_tok, "tokenizer") and src_tok.tokenizer is not None:
            src_raw_tokens = get_bpe_tokens_from_ids(
                src_tok,
                src_ids
            )

            src_raw_tokens = [
                src_raw_tokens[i]
                for i in src_content_indices
            ]

            src_attention = attn_matrix[
                :,
                src_content_indices
            ]

            src_tokens, attn_matrix = group_bpe_tokens(
                src_raw_tokens,
                src_attention,
                axis=1
            )

        else:
            src_tokens = sentence.lower().split()

            src_attention = attn_matrix[
                :,
                src_content_indices
            ]

            # Make sure lengths match
            src_len = min(
                len(src_tokens),
                src_attention.shape[1]
            )

            src_tokens = src_tokens[:src_len]
            attn_matrix = src_attention[:, :src_len]

        # -----------------------------------------------------
        # TARGET TOKENS
        # -----------------------------------------------------

        output_ids = output[0].tolist()

        tgt_content_indices = [
            i for i, idx in enumerate(output_ids)
            if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
        ]

        if hasattr(tgt_tok, "tokenizer") and tgt_tok.tokenizer is not None:
            tgt_raw_tokens = get_bpe_tokens_from_ids(
                tgt_tok,
                output_ids
            )

            tgt_raw_tokens = [
                tgt_raw_tokens[i]
                for i in tgt_content_indices
            ]

            tgt_attention = attn_matrix[
                tgt_content_indices,
                :
            ]

            tgt_tokens, attn_matrix = group_bpe_tokens(
                tgt_raw_tokens,
                tgt_attention,
                axis=0
            )

        else:
            tgt_tokens = translated.split()

            tgt_attention = attn_matrix[
                tgt_content_indices,
                :
            ]

            tgt_len = min(
                len(tgt_tokens),
                tgt_attention.shape[0]
            )

            tgt_tokens = tgt_tokens[:tgt_len]
            attn_matrix = tgt_attention[:tgt_len, :]

    else:
        attn_matrix = None
        src_tokens = sentence.lower().split()
        tgt_tokens = translated.split()

    return translated, attn_matrix, src_tokens, tgt_tokens


def plot_attention(attn_matrix, src_tokens, tgt_tokens):
    """Create attention heatmap using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.8), max(4, len(tgt_tokens) * 0.6)))
    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=11)

    ax.set_xlabel("Source (English)", fontsize=12)
    ax.set_ylabel("Target (German)", fontsize=12)
    ax.set_title("Cross-Attention Heatmap", fontsize=14)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def main():
    st.title("🌐 English → German Translator")
    st.markdown("*Transformer built from scratch — no pretrained models*")
    # Find available Transformer checkpoints
    ckpt_dir = "checkpoints"
    available = []

    if os.path.exists(ckpt_dir):
        for run in os.listdir(ckpt_dir):
            best = os.path.join(ckpt_dir, run, "best_model.pt")

            if not os.path.exists(best):
                continue

            try:
                from utils.training import load_checkpoint

                ckpt = load_checkpoint(best, device="cpu")
                train_config = ckpt.get("train_config", {})

                # New Transformer checkpoints explicitly contain model_type.
                # Older Transformer checkpoints did not, so we treat a missing
                # model_type as Transformer for backward compatibility.
                model_type = train_config.get("model_type", "transformer")

                if model_type == "transformer":
                    available.append(best)

            except Exception as e:
                print(f"Skipping checkpoint {best}: {e}")


    if not available:
        st.error("No trained checkpoints found in `checkpoints/`. Train a model first with `python train.py`.")
        return

    # Sidebar: settings loaded from checkpoint
    st.sidebar.header("⚙️ Settings")
    checkpoint_path = st.sidebar.selectbox("Checkpoint", available)

    model, src_tok, tgt_tok, model_config, train_config = load_model(checkpoint_path)

    # Display model config from checkpoint
    st.sidebar.subheader("Model Config (from checkpoint)")
    st.sidebar.json(model_config)
    st.sidebar.subheader("Training Config")
    st.sidebar.json(train_config)

    use_beam = st.sidebar.checkbox("Use Beam Search", value=False)
    beam_width = st.sidebar.slider("Beam Width", 2, 10, 5) if use_beam else 1

    # Main area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("English Input")
        input_text = st.text_area(
            "Type an English sentence:",
            value="A dog is running through the grass.",
            height=100,
        )

    if st.button("🔄 Translate", type="primary"):
        if input_text.strip():
            with st.spinner("Translating..."):
                translated, attn_matrix, src_tokens, tgt_tokens = translate(
                    model, input_text.strip(), src_tok, tgt_tok, use_beam, beam_width
                )

            with col2:
                st.subheader("German Translation")
                st.success(translated)

            # Attention heatmap
            if attn_matrix is not None and attn_matrix.size > 0:
                st.subheader("Attention Heatmap")
                st.markdown("*Shows which source words the model attended to for each output word*")
                fig = plot_attention(attn_matrix, src_tokens, tgt_tokens)
                st.pyplot(fig)
        else:
            st.warning("Please enter a sentence to translate.")


if __name__ == "__main__":
    main()
