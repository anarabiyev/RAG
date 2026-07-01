"""
rerankers.py — Rerank a retrieved shortlist with a cross-encoder.

Retrieval (dense, BM25, or hybrid) is fast but shallow: it compares the query
and each chunk *independently* — the query became one vector, each chunk became
one vector, and it never looks at them together. A cross-encoder does the
opposite: it feeds (query, chunk) as a single pair through a transformer, so
every query word can attend to every chunk word. That's far more accurate and
far more expensive — which is exactly why it goes LAST, on a shortlist:

    retrieve ~30 cheap candidates  ->  cross-encoder re-scores them  ->  keep top few

This is the "retrieve wide, rerank narrow" pattern. Retrieval only has to get
the right chunk *somewhere* in the top 30; the reranker's job is to float it to
the top.

One interface, same shape as everything else:

    .rerank(query, candidates, top_k) -> [(Chunk, score), ...]   # best first

Two implementations ship here:

    NoOpReranker           pass the candidates straight through (rerank disabled)
    CrossEncoderReranker   a local cross-encoder via sentence-transformers (default)

The diagram's `rerank-v4.0-fast` box is a *hosted* rerank API (Cohere, Jina,
Voyage, mixedbread, ...). That's the same idea over the network instead of a
local model; a thin sketch of how to wire one in is at the bottom
(ApiReranker) — left as a template because it needs a provider + key you choose.
"""

from __future__ import annotations

from chunkers import Chunk


class Reranker:
    """Base class. Given a query and a candidate shortlist, return the best
    `top_k` as (Chunk, score) pairs, highest score first."""

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]],
               top_k: int = 4) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NoOpReranker — the "reranking off" switch
# ---------------------------------------------------------------------------
# Keeps whatever order retrieval produced and just trims to top_k. Handy as the
# default when you haven't installed a cross-encoder, and as the honest baseline
# to measure the real reranker against.

class NoOpReranker(Reranker):
    def rerank(self, query: str, candidates: list[tuple[Chunk, float]],
               top_k: int = 4) -> list[tuple[Chunk, float]]:
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# CrossEncoderReranker — a local cross-encoder (the default real reranker)
# ---------------------------------------------------------------------------
# `cross-encoder/ms-marco-MiniLM-L-6-v2` is the standard small reranker: trained
# on MS MARCO to score how well a passage answers a query. It runs locally (CPU
# is fine for a 30-item shortlist) and needs no API key — the same "learn with
# the free local model, graduate to a hosted API" shape the rest of the project
# uses. Its output is an unbounded relevance logit (higher = more relevant), not
# a probability or a cosine similarity, so read the numbers as a ranking, not a
# calibrated score.

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Imported lazily so this module imports even before sentence-
        # transformers is installed. `pip install sentence-transformers`.
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]],
               top_k: int = 4) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        chunks = [chunk for chunk, _ in candidates]
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)   # one relevance score per pair
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


# ---------------------------------------------------------------------------
# ApiReranker — template for a hosted rerank API (the diagram's box)
# ---------------------------------------------------------------------------
# A hosted reranker is the same cross-encoder idea, run on the provider's
# hardware and reached over the network with an API key. Providers differ only
# in SDK details; the shape is always "send the query + candidate texts, get
# back scored/ordered indices." Fill in one provider's call and this drops into
# the pipeline exactly like CrossEncoderReranker. Kept as a template (not wired
# to a specific vendor) so you can pick the one you have a key for.
#
# Example shape (Cohere-style; adapt to your provider):
#
#   class ApiReranker(Reranker):
#       def __init__(self, model="rerank-...", api_key=None):
#           import os, cohere
#           self.client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
#           self.model = model
#       def rerank(self, query, candidates, top_k=4):
#           docs = [c.text for c, _ in candidates]
#           res = self.client.rerank(model=self.model, query=query,
#                                    documents=docs, top_n=top_k)
#           return [(candidates[r.index][0], float(r.relevance_score))
#                   for r in res.results]
