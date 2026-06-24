"""
chunkers.py — Pluggable text-chunking strategies for the RAG pipeline.

Chunking decides *what a retrievable unit is*. Get it wrong and even a perfect
embedding model retrieves the wrong thing cleanly. This module collects four
strategies, simplest to most sophisticated, behind one interface so you can
switch between them and watch the retrieved passages change.

    FixedTokenChunker   fixed token windows + overlap          (the baseline)
    RecursiveChunker    respect paragraph/line/sentence         (the workhorse)
                        boundaries, recursively
    SentenceChunker     whole sentences grouped to a budget
    SemanticChunker     cut where the topic actually shifts     (embedding-based)

Every chunker implements .split(text, source) -> list[Chunk].

EVERY size is now measured in TOKENS — the unit the LLM actually charges for
and that fills its context window. All four chunkers share one Tokenizer, so a
`chunk_size` of 200 means the same thing everywhere and the strategies are
finally comparable head-to-head.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    """One retrievable passage of text, plus where it came from."""
    text: str
    source: str          # filename the chunk came from
    index: int           # position of this chunk within that file


# ---------------------------------------------------------------------------
# Tokenizer — the shared unit of measurement
# ---------------------------------------------------------------------------
# "Token" is the natural unit for chunk sizing: it's what the LLM is billed in
# and what its context window is measured in. We use tiktoken (OpenAI's BPE
# tokenizer). One token is roughly 3/4 of an English word.
#
# A subtlety worth holding onto: there are really TWO tokenizers in this
# project. This one (the LLM's) governs chunk-size budgets. The EMBEDDING model
# (all-MiniLM-L6-v2) has its own tokenizer with a hard input limit of ~256
# tokens — anything past that is silently truncated before embedding. So if you
# set a chunk size above ~256, the tail of each chunk never makes it into the
# vector. Keep sizes at or under that and the two tokenizers stay in agreement.

class Tokenizer:
    """Counts and splits text in tokens, backed by tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        # Imported lazily so the rest of the file imports even before tiktoken
        # is installed. `pip install tiktoken` to enable it.
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


# One shared instance, created on first use, so we don't re-init the tokenizer
# for every chunker. Chunkers default to this unless you inject your own.
_DEFAULT_TOKENIZER: Tokenizer | None = None


def default_tokenizer() -> Tokenizer:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = Tokenizer()
    return _DEFAULT_TOKENIZER


# A cheap, dependency-free sentence splitter: break after . ! or ? followed by
# whitespace. It is *good enough to learn with* but naive — it splits on "Dr."
# and "3.14". The standard upgrade is nltk's punkt or a spaCy sentencizer.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class Chunker:
    """Base class. A chunker turns one document's text into a list of Chunks."""

    def split(self, text: str, source: str) -> list[Chunk]:
        raise NotImplementedError

    @staticmethod
    def _wrap(pieces: list[str], source: str) -> list[Chunk]:
        """Turn raw strings into numbered Chunk objects, dropping blanks."""
        chunks: list[Chunk] = []
        for piece in pieces:
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source, index=len(chunks)))
        return chunks


# ---------------------------------------------------------------------------
# 1 — Fixed token window  (the OG baseline)
# ---------------------------------------------------------------------------
# Encode the whole document to token IDs, then slide a window of chunk_size
# tokens with overlap and decode each window back to text. This is a *true*
# token splitter (it's what LangChain's TokenTextSplitter does). One honest
# side effect: because tokens are sub-word pieces, a window can begin or end
# mid-word — decode stitches the bytes back together regardless. Still no idea
# of sentences or meaning; overlap is the band-aid against cutting a thought.

class FixedTokenChunker(Chunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 40,
                 tokenizer: Tokenizer | None = None):
        self.chunk_size = chunk_size      # in TOKENS
        self.overlap = overlap            # in TOKENS
        self.tokenizer = tokenizer or default_tokenizer()

    def split(self, text: str, source: str) -> list[Chunk]:
        ids = self.tokenizer.encode(text)
        if not ids:
            return []
        step = max(1, self.chunk_size - self.overlap)
        pieces: list[str] = []
        for start in range(0, len(ids), step):
            window = ids[start:start + self.chunk_size]
            if not window:
                break
            pieces.append(self.tokenizer.decode(window))
            if start + self.chunk_size >= len(ids):
                break  # we've covered the whole document
        return self._wrap(pieces, source)


# ---------------------------------------------------------------------------
# 2 — Recursive splitter  (the production workhorse)
# ---------------------------------------------------------------------------
# The idea that powers LangChain's default splitter. You give it a PRIORITY
# LIST of separators, coarsest first: paragraph, then line, then sentence, then
# word, then bare character. It splits on the coarsest separator that keeps
# pieces under the token budget; any piece still too big is split again on the
# next separator down — hence "recursive". Then small adjacent pieces are
# merged back up toward the budget, with overlap. Cuts land on natural
# boundaries without any model.
#
# All length checks below are TOKEN counts via self._len. (Small caveat: BPE
# isn't additive across boundaries, so summing per-piece token counts during
# the merge is a hair approximate — exactly the same approximation LangChain
# makes with its length_function. Close enough, and far cheaper than
# re-tokenizing every candidate join.)

class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 40,
                 separators: list[str] | None = None,
                 tokenizer: Tokenizer | None = None):
        self.chunk_size = chunk_size      # in TOKENS
        self.overlap = overlap            # in TOKENS
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.tokenizer = tokenizer or default_tokenizer()

    def _len(self, s: str) -> int:
        return self.tokenizer.count(s)

    def split(self, text: str, source: str) -> list[Chunk]:
        return self._wrap(self._split(text, self.separators), source)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Pick the first separator in the list that actually occurs in `text`.
        # Everything after it becomes the fallback set for oversized pieces.
        sep = separators[-1]
        remaining: list[str] = []
        for i, candidate in enumerate(separators):
            if candidate == "":
                sep = ""
                break
            if candidate in text:
                sep = candidate
                remaining = separators[i + 1:]
                break

        splits = list(text) if sep == "" else text.split(sep)

        final: list[str] = []
        good: list[str] = []          # small pieces waiting to be merged
        for piece in splits:
            if self._len(piece) <= self.chunk_size:
                good.append(piece)
            else:
                # flush what we've accumulated, then recurse into the big piece
                if good:
                    final.extend(self._merge(good, sep))
                    good = []
                if remaining:
                    final.extend(self._split(piece, remaining))
                else:
                    final.append(piece)   # no finer separator left; keep as-is
        if good:
            final.extend(self._merge(good, sep))
        return final

    def _merge(self, splits: list[str], sep: str) -> list[str]:
        """Greedily pack small splits into chunks up to chunk_size tokens,
        keeping a tail of `overlap` tokens at the start of the next chunk."""
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        sep_len = self._len(sep)
        for s in splits:
            s_len = self._len(s)
            addition = s_len + (sep_len if current else 0)
            if length + addition > self.chunk_size and current:
                chunks.append(sep.join(current))
                # slide the window: drop from the front until the carried-over
                # tail is no larger than `overlap`
                while length > self.overlap and current:
                    dropped = current.pop(0)
                    length -= self._len(dropped) + (sep_len if current else 0)
            current.append(s)
            length += s_len + (sep_len if len(current) > 1 else 0)
        if current:
            chunks.append(sep.join(current))
        return chunks


# ---------------------------------------------------------------------------
# 3 — Sentence chunker  (structure-aware, still no model)
# ---------------------------------------------------------------------------
# Split into whole sentences, then greedily group consecutive sentences until
# the next would blow the token budget. Every chunk is made of complete
# sentences — never a fragment. Overlap is expressed in *sentences*: carry the
# last N sentences of one chunk into the start of the next so a thought that
# spans the boundary survives in both.

class SentenceChunker(Chunker):
    def __init__(self, max_tokens: int = 200, overlap_sentences: int = 1,
                 tokenizer: Tokenizer | None = None):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.tokenizer = tokenizer or default_tokenizer()

    def split(self, text: str, source: str) -> list[Chunk]:
        # Precompute each sentence's token count once instead of re-counting.
        sentences = split_sentences(text)
        counts = {s: self.tokenizer.count(s) for s in set(sentences)}

        pieces: list[str] = []
        current: list[str] = []
        length = 0
        for sent in sentences:
            t = counts[sent]
            if length + t > self.max_tokens and current:
                pieces.append(" ".join(current))
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                length = sum(counts[s] for s in current)
            current.append(sent)
            length += t
        if current:
            pieces.append(" ".join(current))
        return self._wrap(pieces, source)


# ---------------------------------------------------------------------------
# 4 — Semantic chunker  (the emerging idea: cut where MEANING shifts)
# ---------------------------------------------------------------------------
# Every method above cuts on length or punctuation — blind to what the text is
# about. Semantic chunking instead asks the embedding model where the topic
# changes. Procedure:
#   1. split into sentences
#   2. embed each sentence
#   3. measure how different each sentence is from the next (cosine distance)
#   4. wherever that distance spikes above a threshold, declare a boundary
# The threshold is a PERCENTILE of the distances in this document, so it adapts:
# a high percentile (95) means "only cut at the sharpest topic jumps" -> few,
# large chunks; a lower percentile cuts more eagerly -> more, smaller chunks.
#
# Needs an embedder (we reuse the one the RAG already loaded — no second model).

class SemanticChunker(Chunker):
    def __init__(self, embedder, percentile: float = 90.0,
                 max_tokens: int = 256, buffer: int = 2,
                 tokenizer: Tokenizer | None = None):
        self.embedder = embedder          # any object with .encode(list[str]) -> np.ndarray
        self.percentile = percentile      # 0-100; higher = fewer, bigger chunks
        self.max_tokens = max_tokens      # hard cap so one topic can't run forever
        self.buffer = buffer              # neighbour sentences mixed in before embedding
        self.tokenizer = tokenizer or default_tokenizer()

    def split(self, text: str, source: str) -> list[Chunk]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return self._wrap(sentences, source)

        # Embed each sentence with a little neighbour context, so a single odd
        # short sentence doesn't fake a topic change. The embeddings come back
        # L2-normalized from our Embedder, so dot product == cosine similarity.
        windows = self._with_buffer(sentences)
        vectors = self.embedder.encode(windows)

        sims = (vectors[:-1] * vectors[1:]).sum(axis=1)   # consecutive cosine sims
        distances = 1.0 - sims                            # big distance = topic shift
        threshold = np.percentile(distances, self.percentile)

        pieces: list[str] = []
        start = 0
        for i, dist in enumerate(distances):
            if dist > threshold:
                pieces.append(" ".join(sentences[start:i + 1]))
                start = i + 1
        pieces.append(" ".join(sentences[start:]))

        return self._wrap(self._enforce_max(pieces), source)

    def _with_buffer(self, sentences: list[str]) -> list[str]:
        if self.buffer <= 0:
            return sentences
        windows = []
        for i in range(len(sentences)):
            lo = max(0, i - self.buffer)
            hi = min(len(sentences), i + self.buffer + 1)
            windows.append(" ".join(sentences[lo:hi]))
        return windows

    def _enforce_max(self, pieces: list[str]) -> list[str]:
        """A semantic group can still be huge. Fall back to sentence-grouping
        any piece that exceeds max_tokens, so nothing blows the budget (or the
        embedder's 256-token input limit)."""
        out: list[str] = []
        for piece in pieces:
            if self.tokenizer.count(piece) <= self.max_tokens:
                out.append(piece)
                continue
            current, length = [], 0
            for sent in split_sentences(piece):
                t = self.tokenizer.count(sent)
                if length + t > self.max_tokens and current:
                    out.append(" ".join(current))
                    current, length = [], 0
                current.append(sent)
                length += t
            if current:
                out.append(" ".join(current))
        return out
