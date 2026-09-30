"""Command-line interface for the Text-to-Gloss translator."""

import sys
from .text_to_gloss.translator import DictionaryGlossTranslator


def run_cli():
    translator = DictionaryGlossTranslator(fingerspell_unknown=True)

    # If sentence was passed via command line arguments
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        result = translator.translate(sentence)
        print(f"\n[Original]: {result.original_text}")
        print(f"[Glosses] : {result.glosses}")
        if result.unmapped_words:
            print(f"[Unknown] : {result.unmapped_words} (fingerspelled)")
        return

    print("=" * 60)
    print("SLSL Avatar — Text-to-Gloss Quick Start (Day 1 Baseline)")
    print("Type a sentence to translate to sign glosses, or 'q' to quit.")
    print("=" * 60)

    # Demo quick-start sentences
    demo_sentences = [
        "I am going to the store",
        "Hello, how are you?",
        "I went to school yesterday",
        "What is your name?",
        "Thank you my friend",
    ]

    print("\nPre-tested examples:")
    for s in demo_sentences:
        res = translator.translate(s)
        print(f"  • \"{s}\" → {res.glosses}")

    print("\n" + "-" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter sentence > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting translator.")
                break

            result = translator.translate(user_input)
            print(f"  → Glosses: {result.glosses}")
            if result.unmapped_words:
                print(f"    (Fingerspelled / unseen: {result.unmapped_words})")
            if result.reordered:
                print("    (Grammar: Time/Wh-question reordering applied)")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting translator.")
            break


if __name__ == "__main__":
    run_cli()
