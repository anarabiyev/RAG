"""
main.py — Command-line entry point for the from-scratch RAG system.

Usage:
    python main.py                          # interactive: ask questions in a loop
    python main.py "who created Rust?"      # one-shot: answer a single question
    python main.py --k 6 "..."              # retrieve more chunks per question
    python main.py --chunker recursive "..."   # pick a chunking strategy
    python main.py --store faiss "..."         # pick a vector-store backend
    python main.py --retriever hybrid "..."    # BM25 + dense, fused with RRF
    python main.py --retriever hybrid --rerank "..."   # + cross-encoder rerank
    python main.py --retriever bm25 "..."      # keyword-only (no embeddings)

Retrieval strategies (see retrievers.py):
    dense       embeddings + vector store (meaning-based; the original path)
    bm25        from-scratch keyword ranking (exact terms embeddings miss)
    hybrid      run both, fuse rankings with Reciprocal Rank Fusion  (default)

Reranking (see rerankers.py): pass --rerank to retrieve a wide ~30-chunk pool
and re-score it with a local cross-encoder down to the --k best. Off by default.

Chunking strategies (see chunkers.py). All sizes are measured in TOKENS:
    fixed       fixed token windows + overlap         (the original baseline)
    recursive   respect paragraph/line/sentence boundaries, recursively
    sentence    whole sentences grouped to a token budget
    semantic    cut where the topic actually shifts (uses the embedding model)

Vector-store backends (see stores.py). The retrieved passages should be nearly
identical across all backends on a small corpus — the difference is scale/speed:
    numpy       one numpy matrix, exact brute-force search   (the baseline)
    faiss       Meta's FAISS library: a fast in-process index (pip install faiss-cpu)
    chroma      an embedded vector database, SQLite-style     (pip install chromadb)
    pinecone    a remote, managed vector database             (pip install pinecone)

The first run downloads the embedding model (~80 MB) once, then caches it, and
tiktoken downloads its vocab once. Answer generation needs OPENAI_API_KEY (put
it in a .env file); without it, you still get the retrieved passages. The
pinecone store also needs PINECONE_API_KEY in that same .env file.
"""

import argparse

from rag import RAG, Embedder
from chunkers import (Tokenizer, FixedTokenChunker, RecursiveChunker,
                      SentenceChunker, SemanticChunker)
from stores import NumpyStore, FaissStore, ChromaStore, PineconeStore
from retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from rerankers import NoOpReranker, CrossEncoderReranker

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
    if name == "pinecone":
        return PineconeStore()           # remote; reads PINECONE_API_KEY from .env
    raise ValueError(f"unknown store: {name}")


def build_retriever(name: str, embedder: Embedder, store):
    """Construct the requested retriever. Dense uses the embedder + the chosen
    store; bm25 needs neither (pure keyword); hybrid fuses the two with RRF."""
    dense = DenseRetriever(embedder, store)
    if name == "dense":
        return dense
    if name == "bm25":
        return BM25Retriever()           # from-scratch keyword ranking, no store
    if name == "hybrid":
        return HybridRetriever(dense, BM25Retriever())   # both, fused via RRF
    raise ValueError(f"unknown retriever: {name}")


def build_reranker(enabled: bool):
    """A local cross-encoder when --rerank is set, otherwise a pass-through."""
    return CrossEncoderReranker() if enabled else NoOpReranker()


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
                        choices=["numpy", "faiss", "chroma", "pinecone"],
                        help="Vector-store backend for the dense half (default: numpy).")
    parser.add_argument("--retriever", default="hybrid",
                        choices=["dense", "bm25", "hybrid"],
                        help="Retrieval strategy (default: hybrid).")
    parser.add_argument("--rerank", action="store_true",
                        help="Rerank a wide candidate pool with a local cross-encoder.")
    parser.add_argument("--candidates", type=int, default=30,
                        help="Pool size retrieved before reranking (default: 30).")
    args = parser.parse_args()

    # Build the embedder and tokenizer once, then share them with everything.
    embedder = Embedder()
    tokenizer = Tokenizer()
    chunker = build_chunker(args.chunker, embedder, tokenizer)
    store = build_store(args.store)
    retriever = build_retriever(args.retriever, embedder, store)
    reranker = build_reranker(args.rerank)

    store_note = f" (dense store: {args.store})" if args.retriever != "bm25" else ""
    rerank_note = " + cross-encoder rerank" if args.rerank else ""
    print(f"Building index from '{CORPUS_DIR}/' using '{args.chunker}' chunking, "
          f"'{args.retriever}' retrieval{store_note}{rerank_note} ...")
    rag = RAG(CORPUS_DIR, chunker=chunker, embedder=embedder,
              retriever=retriever, reranker=reranker, candidates=args.candidates)
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