"""
main.py — Command-line entry point for the from-scratch RAG system.

Usage:
    python main.py                          # interactive: ask questions in a loop
    python main.py "who created Rust?"      # one-shot: answer a single question
    python main.py --k 6 "..."              # retrieve more chunks per question
    python main.py --chunker recursive "..."   # pick a chunking strategy
    python main.py --store faiss "..."         # pick a vector-store backend
    python main.py --chunker semantic --store chroma "..."

Chunking strategies (see chunkers.py). All sizes are measured in TOKENS:
    fixed       fixed token windows + overlap         (the original baseline)
    recursive   respect paragraph/line/sentence boundaries, recursively
    sentence    whole sentences grouped to a token budget
    semantic    cut where the topic actually shifts (uses the embedding model)

Vector-store backends (see stores.py). The retrieved passages should be nearly
identical across all three on a small corpus — the difference is scale/speed:
    numpy       one numpy matrix, exact brute-force search   (the baseline)
    faiss       Meta's FAISS library: a fast in-process index (pip install faiss-cpu)
    chroma      an embedded vector database, SQLite-style     (pip install chromadb)

The first run downloads the embedding model (~80 MB) once, then caches it, and
tiktoken downloads its vocab once. Answer generation needs OPENAI_API_KEY (put
it in a .env file); without it, you still get the retrieved passages.
"""

import argparse

from rag import RAG, Embedder
from chunkers import (Tokenizer, FixedTokenChunker, RecursiveChunker,
                      SentenceChunker, SemanticChunker)
from stores import NumpyStore, FaissStore, ChromaStore

CORPUS_DIR = "corpus"


def build_chunker(name: str, embedder: Embedder, tokenizer: Tokenizer):
    """Construct the requested chunker. Everything shares one `tokenizer` (so
    sizes are comparable and tiktoken loads once); the semantic chunker also
    reuses `embedder` so the embedding model isn't loaded twice."""
    if name == "fixed":
        return FixedTokenChunker(chunk_size=200, overlap=40, tokenizer=tokenizer)
    if name == "recursive":
        return RecursiveChunker(chunk_size=200, overlap=40, tokenizer=tokenizer)
    if name == "sentence":
        return SentenceChunker(max_tokens=200, overlap_sentences=1, tokenizer=tokenizer)
    if name == "semantic":
        return SemanticChunker(embedder, percentile=90, max_tokens=256, tokenizer=tokenizer)
    raise ValueError(f"unknown chunker: {name}")


def build_store(name: str):
    """Construct the requested vector-store backend. They share one interface
    (.add / .search), so the rest of the pipeline doesn't care which you pick."""
    if name == "numpy":
        return NumpyStore()
    if name == "faiss":
        return FaissStore()              # exact IndexFlatIP; pass index_type="hnsw" for ANN
    if name == "chroma":
        return ChromaStore()             # in-memory; pass persist_dir=... to keep on disk
    raise ValueError(f"unknown store: {name}")


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
    parser.add_argument("--chunker", default="recursive",
                        choices=["fixed", "recursive", "sentence", "semantic"],
                        help="Chunking strategy (default: recursive).")
    parser.add_argument("--store", default="numpy",
                        choices=["numpy", "faiss", "chroma"],
                        help="Vector-store backend (default: numpy).")
    args = parser.parse_args()

    # Build the embedder and tokenizer once, then share them with everything.
    embedder = Embedder()
    tokenizer = Tokenizer()
    chunker = build_chunker(args.chunker, embedder, tokenizer)
    store = build_store(args.store)

    print(f"Building index from '{CORPUS_DIR}/' using '{args.chunker}' chunking "
          f"and the '{args.store}' store ...")
    rag = RAG(CORPUS_DIR, chunker=chunker, embedder=embedder, store=store)
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
