"""Greedy and beam search decoding for inference."""

import torch
from data.tokenizer import SOS_IDX, EOS_IDX, PAD_IDX


def greedy_decode(model, src, max_len=50, device="cpu"):
    """Greedy decoding: pick the most likely token at each step."""
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    enc_output = model.encoder(src, src_mask)

    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        dec_output, attn_weights = model.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = model.fc_out(dec_output[:, -1:])
        next_token = logits.argmax(dim=-1)
        tgt = torch.cat([tgt, next_token], dim=1)

        if (next_token == EOS_IDX).all():
            break

    return tgt, attn_weights


def beam_search_decode(model, src, beam_width=5, max_len=50, device="cpu"):
    """Beam search decoding for a single source sentence."""
    model.eval()
    src = src.to(device)  # (1, src_len)
    src_mask = model.make_src_mask(src)
    enc_output = model.encoder(src, src_mask)

    # Each beam: (log_prob, token_sequence)
    beams = [(0.0, [SOS_IDX])]
    complete = []

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == EOS_IDX:
                complete.append((score, seq))
                continue

            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = model.make_tgt_mask(tgt)
            dec_output, attn_weights = model.decoder(tgt, enc_output, src_mask, tgt_mask)
            logits = model.fc_out(dec_output[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_probs, topk_ids = log_probs.topk(beam_width)
            for i in range(beam_width):
                new_score = score + topk_probs[i].item()
                new_seq = seq + [topk_ids[i].item()]
                candidates.append((new_score, new_seq))

        if not candidates:
            break

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # Early stop if all beams ended
        if all(seq[-1] == EOS_IDX for _, seq in beams):
            complete.extend(beams)
            break

    # Add remaining beams
    complete.extend(beams)
    complete.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # length-normalized

    best_seq = complete[0][1]
    best_tgt = torch.tensor([best_seq], dtype=torch.long, device=device)

    # Get attention weights for the best sequence
    tgt_mask = model.make_tgt_mask(best_tgt)
    _, attn_weights = model.decoder(best_tgt, enc_output, src_mask, tgt_mask)

    return best_tgt, attn_weights
