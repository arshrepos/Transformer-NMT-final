"""Word-level and BPE tokenizers with vocabulary management."""

import json
import os
from collections import Counter


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class WordTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):
        counter = Counter()
        for sent in sentences:
            counter.update(sent.lower().split())

        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):
        tokens = [SOS_IDX]
        for word in sentence.lower().split():
            tokens.append(self.word2idx.get(word, UNK_IDX))
        tokens.append(EOS_IDX)
        return tokens

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.min_freq = data.get("min_freq", 2)
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)


class BPETokenizer:
    """BPE tokenizer with explicit word-boundary markers."""

    END_OF_WORD = "</w>"

    def __init__(self, vocab_size=8000):
        self._vocab_size = vocab_size
        self.tokenizer = None

    @property
    def vocab_size(self):
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        return self._vocab_size

    def train(self, sentences, save_path=None):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.decoders import BPEDecoder

        # IMPORTANT:
        # The model's encode() method lowercases text,
        # so the tokenizer must also be trained on lowercase text.
        normalized_sentences = [sentence.lower() for sentence in sentences]

        # Create BPE model with an explicit marker for the end of a word.
        self.tokenizer = Tokenizer(
            BPE(
                unk_token="<unk>",
                end_of_word_suffix=self.END_OF_WORD,
            )
        )

        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            end_of_word_suffix=self.END_OF_WORD,
        )

        # Train on the same lowercase form used during encode().
        self.tokenizer.train_from_iterator(
            normalized_sentences,
            trainer=trainer,
        )

        # This converts:
        #   "run</w> ning</w>"
        # into normal text with spaces.
        self.tokenizer.decoder = BPEDecoder(
            suffix=self.END_OF_WORD
        )

        if save_path:
            self.tokenizer.save(save_path)

    def encode(self, sentence):
        ids = self.tokenizer.encode(sentence.lower()).ids
        return [SOS_IDX] + ids + [EOS_IDX]

    def decode(self, indices):
        import re

        filtered = []

        for idx in indices:
            idx = int(idx)

            if idx == EOS_IDX:
                break

            if idx in (PAD_IDX, SOS_IDX):
                continue

            filtered.append(idx)

        # BPEDecoder reconstructs word boundaries.
        text = self.tokenizer.decode(filtered)

        # Remove spaces before punctuation.
        text = re.sub(r"\s+([.,!?;:%])", r"\1", text)

        # Remove spaces just inside brackets.
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)

        # Normalize repeated whitespace.
        return " ".join(text.split())

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(path)
        self._vocab_size = self.tokenizer.get_vocab_size()

        # The saved tokenizer must contain the decoder.
        # If it doesn't, it is an OLD tokenizer file.
        if self.tokenizer.decoder is None:
            raise ValueError(
                f"BPE tokenizer at '{path}' has no decoder. "
                "This is likely an old tokenizer file. "
                "Retrain the BPE tokenizer."
            )