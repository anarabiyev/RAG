"""
rag.py — A minimal Retrieval-Augmented Generation system, built from scratch.

No LangChain, no LlamaIndex. The whole pipeline is here so you can read it
top to bottom and understand every step:

    documents -> chunks -> embeddings -> vector store
                                              |
                       question -> embedding -> similarity search -> top-k chunks
                                              |
                          chunks + question -> LLM -> grounded answer

The only third-party pieces are:
  - sentence-transformers : turns text into embedding vectors (runs locally, free)
  - numpy                 : does the similarity math
  - openai (optional)     : writes the final answer (skip it to run retrieval-only)
  - python-dotenv         : loads your OPENAI_API_KEY from a local .env file

Two pipeline steps now live in their own modules, each behind one interface:
  - CHUNKING       -> chunkers.py  (fixed / recursive / sentence / semantic)
  - the VECTOR STORE -> stores.py  (numpy / faiss / chroma)
Read the four sections below in order, then read chunkers.py and stores.py.
"""

from __future__ import annotations

import os

import numpy as np

# Chunking strategies and vector-store backends live in their own modules now.
from chunkers import Chunk, Chunker, FixedTokenChunker
from stores import VectorStore, NumpyStore

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
# SECTION 3 — The vector store + retrieval
# ---------------------------------------------------------------------------
# This is the "database" at the heart of RAG: it holds one vector per chunk and,
# given a query vector, returns the chunks whose vectors are closest. Like
# chunking, it's now pluggable — the implementations live in stores.py behind a
# single interface (.add and .search):
#
#   NumpyStore   one numpy matrix, exact brute-force search   (the baseline)
#   FaissStore   Meta's FAISS library — a fast in-process index, not a server
#   ChromaStore  an embedded vector database (SQLite-for-vectors)
#
# A real large-scale system reaches for a managed/dedicated vector database so
# it persists to disk and scales to billions of vectors. For a few thousand
# chunks all three backends here behave identically; switch with `--store` and
# run the same question through each to feel the difference.


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
    """Ties the four sections into one object: build an index, then query it.

    You can hand it any chunker from chunkers.py and any store from stores.py.
    You can also hand it an already-constructed Embedder — useful because the
    SemanticChunker needs an embedder too, and sharing one instance means the
    80 MB model loads once.
    """

    def __init__(self, corpus_dir: str, chunker: Chunker | None = None,
                 embedder: Embedder | None = None,
                 store: VectorStore | None = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.corpus_dir = corpus_dir
        self.embedder = embedder or Embedder(model_name)
        self.chunker = chunker or FixedTokenChunker()
        self.store = store or NumpyStore()

    def build_index(self) -> int:
        """Load, chunk, embed, and store everything. Returns the chunk count."""
        chunks = build_chunks(self.corpus_dir, self.chunker)
        vectors = self.embedder.encode([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        query_vector = self.embedder.encode([question])[0]
        return self.store.search(query_vector, k)

    def query(self, question: str, k: int = 4) -> dict:
        """Full RAG: retrieve, then generate. Returns answer + the sources used."""
        retrieved = self.retrieve(question, k)
        answer = generate_answer(question, retrieved)
        return {
            "question": question,
            "answer": answer,                # None if no API key / package
            "retrieved": retrieved,          # always present, so retrieval is inspectable
        }
