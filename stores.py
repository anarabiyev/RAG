"""
stores.py — Pluggable vector-store backends for the RAG pipeline.

The vector store is the "database" at the heart of RAG: it holds one vector per
chunk and, given a query vector, returns the chunks whose vectors are closest.
SECTION 3 of rag.py used to define a single NumPy store inline. That store still
lives here (NumpyStore), but it now sits behind one interface alongside two real
libraries, so you can switch backends with `--store` and watch the *same* query
run through three different engines.

    NumpyStore    brute-force cosine over one matrix      (what you already built)
    FaissStore    Meta's similarity-search library         (a library, not a server)
    ChromaStore   an embedded vector *database*            (SQLite-for-vectors)

Every store implements the same two methods, exactly the interface rag.py's RAG
class already calls:

    .add(chunks, vectors)              # index a batch of chunks + their vectors
    .search(query_vector, k) -> [(Chunk, score), ...]   # k nearest, score desc

ONE CONVENTION TO HOLD ONTO: every store returns a *similarity* score where
higher == more relevant (cosine similarity, in [-1, 1], ~1 for a great match).
The Embedder L2-normalizes its vectors, which is what lets all three agree:
for unit vectors, cosine similarity == dot product, and "cosine distance" is
just 1 - similarity. FAISS and Chroma natively speak distance/inner-product;
each store does the small conversion so the number you print means the same
thing everywhere and the backends are genuinely comparable.

Heavy third-party imports (faiss, chromadb) are done lazily inside each class,
so this file imports fine even before you've installed them:

    pip install faiss-cpu      # for FaissStore
    pip install chromadb       # for ChromaStore
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chunkers import Chunk


class VectorStore:
    """Base class. A vector store indexes chunk vectors and finds the nearest
    ones to a query vector. Subclasses implement `add` and `search`."""

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1 — NumpyStore  (the from-scratch baseline — your original VectorStore)
# ---------------------------------------------------------------------------
# Chunk vectors live in one NumPy matrix. Searching is a single matrix multiply:
# because every vector is normalized, (matrix @ query) is the cosine similarity
# of the query to EVERY chunk at once. We then take the highest scores. This is
# exact (it really does compare against every chunk) and dead simple. It's also
# O(n) per query — perfect up to a few thousand chunks, and the thing the next
# two backends improve on as you scale.

@dataclass
class NumpyStore(VectorStore):
    chunks: list[Chunk] = field(default_factory=list)
    matrix: np.ndarray | None = None  # shape (n_chunks, embedding_dim)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.matrix is None:
            return []
        scores = self.matrix @ query_vector            # cosine similarity to every chunk
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]      # k highest (unsorted)
        top = top[np.argsort(-scores[top])]            # then sort those k
        return [(self.chunks[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# 2 — FaissStore  (a LIBRARY, not a database)
# ---------------------------------------------------------------------------
# FAISS (Facebook AI Similarity Search) is the standard fast nearest-neighbour
# library. The crucial mental model: it is NOT a server or a database. It is a
# component you embed in your process, exactly like the NumPy matrix above —
# just far faster and with real ANN indexes. It stores ONLY the vectors; it has
# no idea what a Chunk is. So, just like your original store kept `self.chunks`
# alongside the matrix, we keep our own `self.chunks` list and map FAISS's
# integer row ids back to Chunk objects ourselves. That bookkeeping IS the
# lesson: a library gives you the index and nothing else.
#
# Two index types, to make last session's theory concrete:
#   "flat" (default) — IndexFlatIP: exact inner-product search. Because our
#       vectors are normalized, inner product == cosine similarity, so this
#       returns byte-for-byte what NumpyStore does, just via FAISS's optimized C++.
#   "hnsw"          — IndexHNSWFlat: an approximate graph index (the HNSW we
#       discussed). Sub-linear search; may occasionally miss the true nearest
#       neighbour (that's "recall"). Pointless on 5 docs, transformative on 5M.

class FaissStore(VectorStore):
    def __init__(self, index_type: str = "flat", hnsw_neighbors: int = 32):
        self.index_type = index_type        # "flat" (exact) or "hnsw" (approximate)
        self.hnsw_neighbors = hnsw_neighbors  # graph connectivity (FAISS's M)
        self.index = None                   # the FAISS index, built on first add
        self.chunks: list[Chunk] = []       # our own id -> Chunk mapping

    def _new_index(self, dim: int):
        import faiss
        if self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(dim, self.hnsw_neighbors, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexFlatIP(dim)       # exact; IP == cosine for unit vectors

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        # FAISS wants a contiguous float32 array.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.index is None:
            self.index = self._new_index(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(q, k)   # both shape (1, k); scores are inner products
        out: list[tuple[Chunk, float]] = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:                         # FAISS pads with -1 if it found < k
                continue
            out.append((self.chunks[i], float(score)))
        return out


# ---------------------------------------------------------------------------
# 3 — ChromaStore  (an embedded vector DATABASE)
# ---------------------------------------------------------------------------
# Chroma is the smallest jump from "library" to "database". Its signature trick
# is running embedded, in-process, like SQLite — no server, no Docker, no API
# key. Unlike FAISS, it IS a database: it stores your text and metadata next to
# the vectors and hands them back on query, so we DON'T keep our own chunk list.
#
# We feed Chroma the embeddings we already computed (via `embeddings=`/
# `query_embeddings=`) instead of letting it embed for us — that keeps the
# query and documents in the SAME vector space the rest of the RAG uses, which
# is the one rule you must never break. We create the collection with cosine
# space, so Chroma returns cosine DISTANCE (1 - similarity); we convert back to
# a similarity so the score matches the other two stores.
#
# Default client is in-memory (fresh each run, matching main.py's rebuild-each-
# run model). Pass persist_dir to keep the index on disk between runs instead.

class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "rag", persist_dir: str | None = None):
        import chromadb
        self.client = (chromadb.PersistentClient(path=persist_dir)
                       if persist_dir else chromadb.Client())  # in-memory if None
        # hnsw:space=cosine -> distances are cosine distance (1 - cosine similarity)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        self._next_id = 0   # Chroma needs a unique string id per item

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        ids = [str(self._next_id + i) for i in range(len(chunks))]
        self._next_id += len(chunks)
        self.collection.add(
            ids=ids,
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)
        res = self.collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=k,
        )
        # Chroma returns one list-per-query; we sent one query, so take index 0.
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        out: list[tuple[Chunk, float]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            chunk = Chunk(text=doc, source=str(meta["source"]), index=int(meta["index"]))
            out.append((chunk, 1.0 - float(dist)))   # distance -> similarity
        return out
