"""
rag.py — A minimal Retrieval-Augmented Generation system, built from scratch.

No LangChain, no LlamaIndex. The whole pipeline is here so you can read it
top to bottom and understand every step:

    documents -> chunks -> embeddings -> vector store
                                              |
   question -> [ BM25 keyword  +  dense embedding ] -> RRF fuse -> ~30 candidates
                                              |
                            cross-encoder rerank -> top-k chunks
                                              |
                          chunks + question -> LLM -> grounded answer

The only third-party pieces are:
  - sentence-transformers : turns text into embedding vectors (runs locally, free)
  - numpy                 : does the similarity math
  - openai (optional)     : writes the final answer (skip it to run retrieval-only)
  - python-dotenv         : loads your OPENAI_API_KEY from a local .env file

Each pipeline step now lives in its own module, behind one interface:
  - CHUNKING       -> chunkers.py    (fixed / recursive / sentence / semantic)
  - the VECTOR STORE -> stores.py    (numpy / faiss / chroma / pinecone)
  - RETRIEVAL      -> retrievers.py  (dense / bm25 / hybrid via RRF)
  - RERANKING      -> rerankers.py   (none / local cross-encoder)
Read the four sections below in order, then read those modules.
"""

from __future__ import annotations

import os

import numpy as np

# Chunking, vector stores, retrieval, and reranking each live in their own module.
from chunkers import Chunk, Chunker, FixedTokenChunker
from stores import VectorStore, NumpyStore
from retrievers import Retriever, DenseRetriever
from rerankers import Reranker, NoOpReranker

# Load environment variables from a local .env file if python-dotenv is present.
# This is what makes OPENAI_API_KEY available without exporting it by hand.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Loading and chunking
# ---------------------------------------------------------------------------
# Loading reads files off disk. Chunking — splitting each document into
# focused, retrievable passages — is now pluggable: see chunkers.py for the
# strategies (fixed window, recursive, sentence, semantic).

import glob


def load_documents(corpus_dir: str) -> list[tuple[str, str]]:
    """Read every .md/.txt file in a directory.

    Returns a list of (filename, full_text) pairs.
    """
    paths = sorted(
        glob.glob(os.path.join(corpus_dir, "*.md"))
        + glob.glob(os.path.join(corpus_dir, "*.txt"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No .md or .txt files found in '{corpus_dir}'. "
            "Add some documents to the corpus folder."
        )
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def build_chunks(corpus_dir: str, chunker: Chunker) -> list[Chunk]:
    """Load every document and run the chosen chunker over all of them."""
    chunks: list[Chunk] = []
    for source, text in load_documents(corpus_dir):
        chunks.extend(chunker.split(text, source))
    return chunks


# ---------------------------------------------------------------------------
# SECTION 2 — Embeddings
# ---------------------------------------------------------------------------
# An embedding model maps a piece of text to a vector (a list of numbers) such
# that texts with similar meaning land close together. We use the same model
# for documents and for the question so they live in the same space.

class Embedder:
    """Thin wrapper around a sentence-transformers model.

    'all-MiniLM-L6-v2' is the standard starting point: small, fast, and good
    enough to learn with. We L2-normalise every vector so that cosine
    similarity becomes a plain dot product (see the vector stores in stores.py).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily so the rest of the file works even before you've
        # installed sentence-transformers.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # avoid division by zero
    return vectors / norms


# ---------------------------------------------------------------------------
# SECTION 3 — Retrieval (vector store, keyword search, fusion, reranking)
# ---------------------------------------------------------------------------
# Retrieval is now a small pipeline of its own, and each piece is pluggable:
#
#   the VECTOR STORE (stores.py)   holds the dense vectors and finds the nearest
#       NumpyStore / FaissStore / ChromaStore / PineconeStore
#
#   the RETRIEVER (retrievers.py)  decides HOW candidates are found
#       DenseRetriever   embeddings + a vector store         (meaning)
#       BM25Retriever    from-scratch keyword ranking        (exact terms)
#       HybridRetriever  fuse both with Reciprocal Rank Fusion
#
#   the RERANKER (rerankers.py)    re-scores the shortlist with a cross-encoder
#       NoOpReranker / CrossEncoderReranker
#
# The flow when everything is on: retrieve a WIDE candidate pool (~30) with the
# hybrid retriever, then let the cross-encoder rerank it down to the top-k the
# LLM actually sees. Retrieval only has to get the right chunk *somewhere* in
# the pool; the reranker floats it to the top. See RAG.retrieve below.


# ---------------------------------------------------------------------------
# SECTION 4 — Generation (the "G" in RAG)
# ---------------------------------------------------------------------------
# Retrieval found the relevant passages. Now we paste them into a prompt and
# ask an LLM to answer *using only that text*. The instruction to stick to the
# provided context — and to say "I don't know" otherwise — is what keeps the
# model grounded instead of falling back on its own memory.

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context passages. "
    "If the answer is not in the context, say you don't know rather than guessing. "
    "Be concise and cite the source filename(s) you used."
)


def build_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(f"[source: {chunk.source}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def generate_answer(question: str, retrieved: list[tuple[Chunk, float]], model: str = DEFAULT_MODEL) -> str:
    """Call the OpenAI API to write a grounded answer.

    If the 'openai' package isn't installed or no API key is set, we return
    None so the caller can fall back to showing the retrieved chunks only.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI()  # reads OPENAI_API_KEY from the environment (loaded from .env)
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, retrieved)},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

class RAG:
    """Ties the pipeline into one object: build an index, then query it.

    You can hand it any chunker (chunkers.py), any retriever (retrievers.py),
    and any reranker (rerankers.py). You can also hand it an already-constructed
    Embedder — useful because the SemanticChunker and the DenseRetriever both
    need one, and sharing a single instance means the 80 MB model loads once.

    `candidates` is the width of the shortlist retrieval hands to the reranker
    (the "retrieve ~30, rerank to k" pattern). With no reranker it's ignored and
    retrieval returns k directly.
    """

    def __init__(self, corpus_dir: str, chunker: Chunker | None = None,
                 embedder: Embedder | None = None,
                 retriever: Retriever | None = None,
                 reranker: Reranker | None = None,
                 candidates: int = 30,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.corpus_dir = corpus_dir
        self.chunker = chunker or FixedTokenChunker()
        # Embeddings now live in the retriever, so we only need an Embedder if
        # we're actually building the default dense retriever below. BM25-only
        # setups pay nothing — no model download, no 80 MB load.
        if retriever is None:
            embedder = embedder or Embedder(model_name)
            retriever = DenseRetriever(embedder, NumpyStore())
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker or NoOpReranker()
        self.candidates = candidates

    def build_index(self) -> int:
        """Load, chunk, and index everything. Returns the chunk count.

        Note the embedding step moved *into* the retriever: DenseRetriever (and
        the dense half of HybridRetriever) embeds and fills its store here, while
        BM25Retriever builds its keyword index. BM25-only needs no embeddings.
        """
        chunks = build_chunks(self.corpus_dir, self.chunker)
        self.retriever.index(chunks)
        return len(chunks)

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Retrieve a wide candidate pool, then rerank it down to k.

        With a NoOpReranker this is just "retrieve k". With a real reranker we
        pull `max(k, candidates)` candidates first so the cross-encoder has a
        deep enough pool to rescue a good chunk that ranked, say, 18th.
        """
        pool = max(k, self.candidates) if not isinstance(self.reranker, NoOpReranker) else k
        candidates = self.retriever.retrieve(question, pool)
        return self.reranker.rerank(question, candidates, top_k=k)

    def query(self, question: str, k: int = 4) -> dict:
        """Full RAG: retrieve, then generate. Returns answer + the sources used."""
        retrieved = self.retrieve(question, k)
        answer = generate_answer(question, retrieved)
        return {
            "question": question,
            "answer": answer,                # None if no API key / package
            "retrieved": retrieved,          # always present, so retrieval is inspectable
        }
