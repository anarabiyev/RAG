"""
evaluate.py — Measure retrieval quality objectively, instead of eyeballing it.

Everywhere else in this project you compare chunkers and retrievers by *reading*
the passages and judging. That doesn't scale and it isn't reproducible. This
script does it with numbers: a small hand-labeled set of (question -> which
corpus file should answer it) pairs, run through the pipeline, scored on two
standard retrieval metrics.

    Hit@k   fraction of questions where a chunk from the correct file appears
            in the top k. "Did we retrieve the right thing at all?"
    MRR     Mean Reciprocal Rank: average of 1/(rank of the first correct chunk).
            Rewards putting the right chunk HIGH, not just somewhere in k.
            (1.0 = always rank 1; 0.5 = typically rank 2; 0.0 = never found.)

We label by SOURCE FILE, not by exact chunk, because file-level truth is cheap
to write by hand and stays valid no matter which chunker you run (chunk indices
change with the strategy; the answer's file does not). That makes this the right
tool for the job the README asks for: comparing chunkers head-to-head.

`ragas` is the standard library for RAG evaluation and goes further — it uses an
LLM to judge answer faithfulness and relevancy, not just retrieval. This harness
is the from-scratch, no-API-key version that measures the retrieval half you can
check with pure bookkeeping. Reach for ragas once you're grading generated
answers.

Usage:
    python evaluate.py                                  # default: hybrid retrieval
    python evaluate.py --retriever hybrid --rerank      # add the cross-encoder
    python evaluate.py --retriever bm25 --chunker recursive
    python evaluate.py --compare-chunkers               # all 4 chunkers, one table
    python evaluate.py --compare-chunkers --retriever hybrid --rerank -k 5
"""

from __future__ import annotations

import argparse

from rag import RAG, Embedder
from chunkers import Tokenizer
from main import build_chunker, build_store, build_retriever, build_reranker

CORPUS_DIR = "corpus"

# ---------------------------------------------------------------------------
# The labeled set. Each pair is (question, the corpus file that answers it).
# Hand-written against the five sample docs; extend it with your own corpus.
# Keep questions that exercise BOTH kinds of retrieval: some are semantic
# ("automatic memory management"), some hinge on exact tokens ("Rust 1.0",
# "pgvector", "gofmt") that keyword search catches and embeddings often miss.
# ---------------------------------------------------------------------------
QA_PAIRS: list[tuple[str, str]] = [
    ("who created Rust?", "rust.md"),
    ("what is the borrow checker?", "rust.md"),
    ("when did Rust reach its first stable release?", "rust.md"),
    ("what is Cargo?", "rust.md"),
    ("does Go have a garbage collector?", "go.md"),
    ("what are goroutines?", "go.md"),
    ("what tool enforces Go's formatting?", "go.md"),          # exact token: gofmt
    ("which language was created by Brendan Eich?", "javascript.md"),
    ("what runs natively in every web browser?", "javascript.md"),
    ("what is the event loop?", "javascript.md"),
    ("what is Node.js?", "javascript.md"),
    ("who created Python?", "python.md"),
    ("what is the GIL?", "python.md"),                          # exact token: GIL
    ("what does significant indentation mean in Python?", "python.md"),
    ("who designed the relational model?", "sql.md"),
    ("what does declarative querying mean?", "sql.md"),
    ("how do vector databases relate to SQL?", "sql.md"),
    ("what is pgvector?", "sql.md"),                            # exact token: pgvector
]


def evaluate_config(chunker, embedder, retriever, reranker,
                    k: int, candidates: int) -> dict:
    """Index the corpus with one configuration and score it on QA_PAIRS."""
    rag = RAG(CORPUS_DIR, chunker=chunker, embedder=embedder,
              retriever=retriever, reranker=reranker, candidates=candidates)
    n_chunks = rag.build_index()

    hits = 0
    rr_sum = 0.0
    rows = []
    for question, expected in QA_PAIRS:
        retrieved = rag.retrieve(question, k)
        sources = [chunk.source for chunk, _ in retrieved]
        # 1-based rank of the first retrieved chunk from the expected file.
        rank = next((i + 1 for i, s in enumerate(sources) if s == expected), None)
        hits += 1 if rank is not None else 0
        rr_sum += (1.0 / rank) if rank is not None else 0.0
        rows.append((question, expected, rank, sources))

    m = len(QA_PAIRS)
    return {
        "n_chunks": n_chunks,
        "hit_at_k": hits / m,
        "mrr": rr_sum / m,
        "rows": rows,
    }


def print_detail(result: dict, k: int) -> None:
    print(f"\nPer-question (rank of first correct chunk within top {k}; '-' = missed):\n")
    for question, expected, rank, _sources in result["rows"]:
        marker = str(rank) if rank is not None else "-"
        flag = " " if rank == 1 else ("." if rank is not None else "x")
        print(f"  [{flag}] rank {marker:>2}  {expected:<14} {question}")
    print(f"\n  Hit@{k}: {result['hit_at_k']:.3f}    MRR: {result['mrr']:.3f}    "
          f"({result['n_chunks']} chunks)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-quality evaluation.")
    parser.add_argument("-k", "--k", type=int, default=5,
                        help="Top-k retrieved chunks to score (default 5).")
    parser.add_argument("--chunker", default="recursive",
                        choices=["fixed", "recursive", "sentence", "semantic"])
    parser.add_argument("--store", default="numpy",
                        choices=["numpy", "faiss", "chroma", "pinecone"])
    parser.add_argument("--retriever", default="hybrid",
                        choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--rerank", action="store_true",
                        help="Rerank with a local cross-encoder before scoring.")
    parser.add_argument("--candidates", type=int, default=30,
                        help="Pool size retrieved before reranking (default 30).")
    parser.add_argument("--compare-chunkers", action="store_true",
                        help="Score all four chunkers under the same retriever.")
    args = parser.parse_args()

    # Shared across every configuration so the 80 MB model loads at most once.
    embedder = Embedder()
    tokenizer = Tokenizer()

    def make_retriever():
        # Fresh store + retriever per config so nothing leaks between runs.
        return build_retriever(args.retriever, embedder, build_store(args.store))

    reranker = build_reranker(args.rerank)
    rr = " + rerank" if args.rerank else ""

    if args.compare_chunkers:
        print(f"Comparing chunkers | retriever='{args.retriever}'{rr} | k={args.k}\n")
        print(f"  {'chunker':<11} {'chunks':>6}  {'Hit@'+str(args.k):>7}  {'MRR':>6}")
        print("  " + "-" * 34)
        for name in ["fixed", "recursive", "sentence", "semantic"]:
            chunker = build_chunker(name, embedder, tokenizer)
            res = evaluate_config(chunker, embedder, make_retriever(),
                                  reranker, args.k, args.candidates)
            print(f"  {name:<11} {res['n_chunks']:>6}  "
                  f"{res['hit_at_k']:>7.3f}  {res['mrr']:>6.3f}")
        print("\nHigher is better on both. Hit@k = found at all; MRR = found high.\n")
        return

    print(f"Config: chunker='{args.chunker}' retriever='{args.retriever}'{rr} k={args.k}")
    chunker = build_chunker(args.chunker, embedder, tokenizer)
    res = evaluate_config(chunker, embedder, make_retriever(),
                          reranker, args.k, args.candidates)
    print_detail(res, args.k)


if __name__ == "__main__":
    main()
