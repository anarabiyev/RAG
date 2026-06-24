"""
from chunkers import Tokenizer, FixedTokenChunker, RecursiveChunker, SentenceChunker

tok = Tokenizer()
text = open("corpus/go.md").read()
for ch in [FixedTokenChunker(tokenizer=tok), RecursiveChunker(tokenizer=tok), SentenceChunker(tokenizer=tok)]:
    cs = ch.split(text, "go.md")
    print(f"\n{type(ch).__name__}: {len(cs)} chunks, sizes={[tok.count(c.text) for c in cs]}")
    print("  first chunk starts:", repr(cs[0].text[:60]))

"""

from chunkers import (Tokenizer, FixedTokenChunker, RecursiveChunker,
                      SentenceChunker, SemanticChunker)
from rag import Embedder

tok = Tokenizer()
emb = Embedder()   # loads the ~80 MB model once

text = open("corpus/go.md").read()

chunkers = [
    FixedTokenChunker(tokenizer=tok),
    RecursiveChunker(tokenizer=tok),
    SentenceChunker(tokenizer=tok),
    SemanticChunker(emb, tokenizer=tok),   # note: embedder is the first arg
]

for ch in chunkers:
    cs = ch.split(text, "go.md")
    print(f"\n{type(ch).__name__}: {len(cs)} chunks, sizes={[tok.count(c.text) for c in cs]}")
    print("  first chunk starts:", repr(cs[0].text[:60]))