"""
main.py — Command-line entry point for the from-scratch RAG system.

Usage:
    python main.py                       # interactive: ask questions in a loop
    python main.py "who created Rust?"   # one-shot: answer a single question
    python main.py --k 6 "..."           # retrieve more chunks per question

The first run downloads the embedding model (~80 MB) once, then caches it.
Answer generation needs OPENAI_API_KEY (put it in a .env file); without it, you
still get the retrieved passages so you can see retrieval working on its own.
"""

import sys
import argparse

from rag import RAG

CORPUS_DIR = "corpus"


def print_result(result: dict) -> None:
    print()
    if result["answer"]:
        print("Answer:")
        print(" ", result["answer"].replace("\n", "\n  "))
    else:
        print("(No OPENAI_API_KEY set, so showing retrieved passages only —")
        print(" this is the retrieval half of RAG working without an LLM.)")

    print("\nRetrieved passages (most relevant first):")
    for rank, (chunk, score) in enumerate(result["retrieved"], 1):
        preview = chunk.text[:160] + ("..." if len(chunk.text) > 160 else "")
        print(f"  {rank}. [{score:.3f}] {chunk.source} (chunk {chunk.index})")
        print(f"     {preview}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal from-scratch RAG.")
    parser.add_argument("question", nargs="*", help="A question to answer. Omit for interactive mode.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default 4).")
    args = parser.parse_args()

    print(f"Building index from '{CORPUS_DIR}/' ...")
    rag = RAG(CORPUS_DIR)
    n = rag.build_index()
    print(f"Indexed {n} chunks.\n")

    if args.question:
        print_result(rag.query(" ".join(args.question), k=args.k))
        return

    print("Interactive mode. Ask a question, or type 'quit' to exit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        print_result(rag.query(question, k=args.k))


if __name__ == "__main__":
    main()
