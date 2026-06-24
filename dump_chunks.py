from chunkers import (Tokenizer, FixedTokenChunker, RecursiveChunker,
                      SentenceChunker, SemanticChunker)
from rag import Embedder

tok = Tokenizer()
emb = Embedder()   # loads the ~80 MB model once (needed for semantic)

DOC = "corpus/go.md"          # change to any file in corpus/
text = open(DOC, encoding="utf-8").read()

chunkers = [
    FixedTokenChunker(tokenizer=tok),
    RecursiveChunker(tokenizer=tok),
    SentenceChunker(tokenizer=tok),
    SemanticChunker(emb, tokenizer=tok),
]

for ch in chunkers:
    name = type(ch).__name__
    cs = ch.split(text, DOC)
    print("\n" + "=" * 70)
    print(f"{name}  —  {len(cs)} chunks from {DOC}")
    print("=" * 70)
    for c in cs:
        print(f"\n--- chunk {c.index}  ({tok.count(c.text)} tokens) " + "-" * 30)
        print(c.text)