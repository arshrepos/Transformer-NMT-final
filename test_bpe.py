from data.tokenizer import BPETokenizer


def main():
    # Small test dataset
    sentences = [
        "A man is wearing a hat.",
        "Two dogs are running in the park.",
        "The woman is riding a bicycle.",
        "A little girl is playing with a ball.",
    ]

    print("=" * 60)
    print("BPE TOKENIZER TEST")
    print("=" * 60)

    # Create tokenizer
    tokenizer = BPETokenizer(vocab_size=8000)

    # Train a temporary tokenizer
    print("\nTraining temporary BPE tokenizer...")
    tokenizer.train(sentences)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test each sentence
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)

        print("\nOriginal:")
        print(sentence)

        print("\nEncoded IDs:")
        print(encoded)

        print("\nDecoded:")
        print(decoded)

        print("-" * 60)


if __name__ == "__main__":
    main()