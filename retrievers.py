"""
retrievers.py — Pluggable retrieval strategies for the RAG pipeline.

Until now retrieval was a single step: embed the question, ask the vector store
for the nearest chunk vectors. That's *dense* retrieval, and it's great at
meaning ("garbage collector" ~ "automatic memory management") but weak at exact
tokens it never learned to associate — names, versions, error codes, rare
identifiers. Keyword search (BM25) has the opposite bias: it nails exact terms
and is blind to meaning. Hybrid search runs BOTH and fuses their rankings, so
you get each one's strengths.

Everything here lives behind one small interface, the same shape as chunkers.py
and stores.py:

    .index(chunks)                     # ingest the corpus once
    .retrieve(query, k) -> [(Chunk, score), ...]   # k best, score desc

Three retrievers, mirroring the diagram (Query -> BM25 + Dense -> RRF):

    DenseRetriever    embeddings + a vector store   (what the project already did)
    BM25Retriever     classic keyword ranking       (from scratch, no dependencies)
    HybridRetriever   fuse the two with Reciprocal Rank Fusion

ONE THING TO NOTE ABOUT SCORES: DenseRetriever returns cosine similarity (the
store's convention, ~1 = great match). BM25 scores are unbounded relevance
weights. HybridRetriever returns RRF scores (small numbers ~1/60). They are NOT
comparable across retrievers — each is only meaningful for ranking *within* one
retriever's output. The reranker downstream (rerankers.py) is what puts a single
trustworthy score on the final shortlist.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from chunkers import Chunk


class Retriever:
    """Base class. A retriever ingests chunks, then returns the k best for a
    query as (Chunk, score) pairs sorted by descending score."""

    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1 — DenseRetriever  (embeddings + vector store — the original retrieval path)
# ---------------------------------------------------------------------------
# This just wraps the two pieces the project already had: an Embedder (turns
# text into vectors) and a VectorStore (indexes them and finds nearest ones).
# Pulling it behind the Retriever interface is what lets the hybrid retriever
# treat "dense" as one interchangeable half of the pipeline.

class DenseRetriever(Retriever):
    def __init__(self, embedder, store):
        self.embedder = embedder          # rag.Embedder
        self.store = store                # stores.VectorStore

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.encode([c.text for c in chunks])
        self.store.add(chunks, vectors)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        query_vector = self.embedder.encode([query])[0]
        return self.store.search(query_vector, k)


# ---------------------------------------------------------------------------
# 2 — BM25Retriever  (classic keyword ranking, built from scratch)
# ---------------------------------------------------------------------------
# BM25 ("Best Matching 25") is the workhorse of keyword search — the ranking
# function behind Elasticsearch and Lucene for decades. It scores a document
# for a query by summing, over the query's terms:
#
#       idf(term) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl/avgdl))
#
#   f      = how often the term appears in this document (rewards matches)
#   idf    = how rare the term is across the corpus (rare terms count more)
#   dl/avgdl = this doc's length vs the average (long docs don't win by bulk)
#   k1, b  = the two standard knobs (term-frequency saturation; length penalty)
#
# It's pure counting — no model, no embeddings, no API. That's the point: it
# catches the exact tokens dense retrieval smooths over. The from-scratch build
# here plays the same role NumpyStore does for the vector store: the honest
# baseline. To graduate to a fast, tested implementation, `pip install bm25s`
# and swap this class's internals — the interface stays identical.

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics. Deliberately simple; a real
    system would add stemming and stopword removal (what bm25s/Lucene do)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1                      # term-frequency saturation
        self.b = b                        # document-length normalization
        self.chunks: list[Chunk] = []
        self.doc_freqs: list[Counter] = []   # per-doc term counts
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self.chunks]
        self.doc_freqs = [Counter(toks) for toks in tokenized]
        self.doc_len = [len(toks) for toks in tokenized]
        n = len(self.chunks)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

        # Document frequency: in how many chunks does each term appear?
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # Standard BM25 idf with the +1 smoothing that keeps it non-negative.
        self.idf = {
            term: math.log(1.0 + (n - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

    def _score(self, query_terms: list[str], i: int) -> float:
        freqs = self.doc_freqs[i]
        dl = self.doc_len[i]
        score = 0.0
        for term in query_terms:
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1.0)) / denom
        return score

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = _tokenize(query)
        scored = [(self.chunks[i], self._score(q, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        # Keep only chunks that matched at least one term; if nothing matched at
        # all, fall back to returning the top-k zeros so the pipeline still runs.
        hits = [pair for pair in scored if pair[1] > 0.0]
        return (hits or scored)[:k]


# ---------------------------------------------------------------------------
# 3 — HybridRetriever  (fuse dense + BM25 with Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
# Now the interesting part. Dense and BM25 produce two ranked lists whose scores
# live on totally different scales (cosine ~0.6 vs BM25 ~8.3), so you can't just
# add the numbers. Reciprocal Rank Fusion sidesteps that entirely: it throws
# away the raw scores and fuses by RANK. A chunk sitting at rank r in a list
# contributes 1/(rrf_k + r); a chunk's final score is the sum of those
# contributions across both lists. Appearing high in either list helps;
# appearing high in both wins. rrf_k (60 is the value from the original paper)
# damps the influence of the very top ranks so one list can't dominate.
#
# We identify a chunk by (source, index) so the same passage retrieved by both
# halves fuses into one entry — even when the dense store reconstructs Chunk
# objects (Chroma/Pinecone) rather than handing back the originals.

class HybridRetriever(Retriever):
    def __init__(self, dense: Retriever, bm25: Retriever,
                 rrf_k: int = 60, pool: int = 50):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k                 # RRF damping constant
        self.pool = pool                   # how many to pull from each half before fusing

    def index(self, chunks: list[Chunk]) -> None:
        self.dense.index(chunks)
        self.bm25.index(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        pool = max(k, self.pool)
        dense_hits = self.dense.retrieve(query, pool)
        bm25_hits = self.bm25.retrieve(query, pool)

        fused: dict[tuple[str, int], float] = {}
        chunk_by_key: dict[tuple[str, int], Chunk] = {}
        for hits in (dense_hits, bm25_hits):
            for rank, (chunk, _score) in enumerate(hits):
                key = (chunk.source, chunk.index)
                chunk_by_key.setdefault(key, chunk)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [(chunk_by_key[key], score) for key, score in ranked[:k]]
