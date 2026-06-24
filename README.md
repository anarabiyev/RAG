# RAG from scratch

A minimal **Retrieval-Augmented Generation** system in ~250 lines of Python, with
no RAG frameworks (no LangChain, no LlamaIndex). The point is to make every step
of the pipeline visible so you understand what the frameworks do for you before
you reach for them.

```
documents ──▶ chunks ──▶ embeddings ──▶ vector store
                                             │
   question ──▶ embedding ──▶ similarity search ──▶ top-k chunks
                                             │
                      chunks + question ──▶ LLM ──▶ grounded answer
```

The retrieval half (finding the right passages) is pure math and runs with no
API key. The generation half (writing the answer) calls an LLM. You can run the
project with just retrieval to see that half working on its own.

## Quickstart

```bash
pip install -r requirements.txt

# Optional: add your OpenAI key to get generated answers. Without it you
# still see the retrieved passages, which is the retrieval half of RAG.
cp .env.example .env      # then edit .env and paste in your key

python main.py                          # interactive question loop
python main.py "who created Rust?"      # one-shot question
```

The key is read from `.env` automatically (via python-dotenv), so you never
have to export it by hand. `.env` is gitignored, so your key is never committed.

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~80 MB) once.

Try questions like:

- `who created Rust?`
- `which language is best for systems programming?`
- `what runs natively in the browser?`
- `which language is declarative?`

Watch the retrieved passages printed under each answer — they show *which*
chunks the system pulled and how similar each was to your question. That
transparency is the whole reason to build this by hand.

## How it works

The code is in two files. `rag.py` is the library, organised into four sections
that match the diagram above; read them in order:

1. **Loading & chunking** — documents are split into overlapping word-windows.
   Overlap stops a sentence on a chunk boundary from losing its meaning.
2. **Embeddings** — `sentence-transformers` maps each chunk (and later each
   question) to a vector. Similar meaning → nearby vectors. Vectors are
   normalised so cosine similarity becomes a plain dot product.
3. **Vector store & retrieval** — chunk vectors live in one NumPy matrix.
   Searching is a single matrix multiply that scores the query against every
   chunk at once; we keep the top *k*. This is exactly what a vector database
   (Chroma, FAISS, Qdrant, …) does, minus the scaling and persistence.
4. **Generation** — the retrieved chunks are pasted into a prompt that tells the
   LLM to answer *only* from that context and to say "I don't know" otherwise.
   That instruction is what keeps answers grounded instead of hallucinated.

`main.py` is just the command-line wrapper around `RAG.build_index()` and
`RAG.query()`.

## Using your own data

Drop any `.md` or `.txt` files into `corpus/` and re-run. The sample corpus is
five short docs about programming languages; swap in your own and the system
indexes them automatically. Good starter corpora: a documentation set, a folder
of articles, or papers in a field you care about.

## Ways to extend this (good next commits)

Each of these is a self-contained improvement and makes the repo more
interesting to a reviewer:

- **Smarter chunking** — split on sentence or paragraph boundaries instead of a
  fixed word window; experiment with chunk size and overlap and observe the
  effect on retrieval.
- **A real vector DB** — swap the NumPy `VectorStore` for Chroma or FAISS so it
  persists to disk and scales past a few thousand chunks.
- **Hybrid search** — combine the embedding similarity with keyword (BM25)
  matching, which catches exact terms (names, error codes) that embeddings miss.
- **Reranking** — retrieve the top ~30 chunks, then use a cross-encoder or a
  rerank API to re-score them down to the best few.
- **Evaluation** — add a small set of question/expected-answer pairs and measure
  retrieval quality (did the right chunk appear in the top *k*?). The `ragas`
  library is the standard tool for this.
- **PDF ingestion** — parse PDFs into text so you can point it at real documents.

## Files

```
rag-from-scratch/
├── rag.py            # the whole pipeline (read this)
├── main.py           # command-line interface
├── corpus/           # sample documents — replace with your own
├── requirements.txt
├── .env.example      # template — copy to .env and add your OpenAI key
└── README.md
```
