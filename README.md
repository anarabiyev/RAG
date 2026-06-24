# RAG from scratch

A minimal **Retrieval-Augmented Generation** system in a few hundred lines of
Python, with no RAG frameworks (no LangChain, no LlamaIndex). The point is to
make every step of the pipeline visible so you understand what the frameworks
do for you before you reach for them.

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

The very first step — **chunking** — is now pluggable: four strategies live in
`chunkers.py`, and you choose one with `--chunker`. Chunking quietly sets a
ceiling on everything downstream (you can only retrieve chunks you created), so
it's worth understanding deeply, which is why this project makes the choice
explicit rather than hiding it.

## Quickstart

```bash
pip install -r requirements.txt

# Optional: add your OpenAI key to get generated answers. Without it you
# still see the retrieved passages, which is the retrieval half of RAG.
cp .env.example .env      # then edit .env and paste in your key

python main.py                              # interactive question loop
python main.py "who created Rust?"          # one-shot question
python main.py --chunker semantic "..."     # pick a chunking strategy
python main.py --chunker fixed --k 6 "..."  # and how many chunks to retrieve
```

The key is read from `.env` automatically (via python-dotenv), so you never
have to export it by hand. `.env` is gitignored, so your key is never committed.

The first run downloads two things once and caches them: the embedding model
(`all-MiniLM-L6-v2`, ~80 MB) and `tiktoken`'s tokenizer vocabulary (small).
Neither needs an API key.

Try questions like:

- `who created Rust?`
- `does Go have a garbage collector like Rust?`
- `what runs natively in the browser?`
- `how do vector databases relate to SQL?`

Watch the retrieved passages printed under each answer — they show *which*
chunks the system pulled and how similar each was to your question. Run the same
question under different `--chunker` values and watch the passages change; that
comparison is the whole point.

## How it works

The code is in three files. `rag.py` is the core pipeline; `chunkers.py` holds
the chunking strategies; `main.py` is the command-line wrapper. Read `rag.py`'s
four sections in order — they match the diagram above:

1. **Loading & chunking** — documents are loaded from `corpus/` and split into
   passages by a *chunker* (see the next section). The default is the recursive
   chunker. Overlap between chunks stops a thought on a boundary from being lost.
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

## Chunking strategies

All four live in `chunkers.py` behind one interface (`.split(text, source)`),
and every size is measured in **tokens** via a shared `tiktoken` tokenizer — so
`chunk_size=200` means the same thing across strategies and they're directly
comparable. Pick one with `--chunker`:

- **`fixed`** — `FixedTokenChunker`. Slides a fixed token-sized window with
  overlap. The OG baseline: dead simple, but blind to sentences and meaning, so
  it happily cuts mid-thought (and, because tokens are sub-word, sometimes
  mid-word).
- **`recursive`** *(default)* — `RecursiveChunker`. The production workhorse
  (the idea behind LangChain's default splitter). It splits on the coarsest of a
  priority list of separators (paragraph → line → sentence → word → character)
  that keeps pieces under the budget, recursing into anything still too big, then
  merges small pieces back up with overlap. Cuts land on natural boundaries, no
  model required.
- **`sentence`** — `SentenceChunker`. Groups whole sentences up to the token
  budget, so a chunk is never a sentence fragment. Overlap is measured in
  sentences.
- **`semantic`** — `SemanticChunker`. The emerging idea: embed each sentence,
  measure the cosine distance between consecutive sentences, and cut wherever
  that distance spikes — i.e. where the topic actually shifts. The threshold is a
  *percentile* of the distances in the document, so it adapts. Reuses the
  embedding model the RAG already loaded.

A subtlety baked into the code: there are **two** tokenizers in play. `tiktoken`
(the LLM's) governs chunk-size budgets, but the embedding model has its *own*
tokenizer with a hard ~256-token input limit and silently truncates longer
inputs before embedding. So a chunk larger than ~256 tokens has its tail dropped
from the vector (invisible to retrieval) even though the LLM still sees it. Keep
chunk sizes at or under ~256 and the two stay in agreement; this is why the
semantic chunker caps chunks at 256 tokens.

## Inspecting the chunkers

Two small helper scripts let you *see* how the strategies differ on a document,
without running retrieval and without an API key:

- **`inspect_chunks.py`** — an at-a-glance overview: for each strategy, how many
  chunks it makes, the token size of each, and a preview of the first chunk.
  Best for comparing the *shape* of the strategies quickly.
- **`dump_chunks.py`** — prints every chunk in full for each strategy. Best for
  reading exactly where boundaries fall and spotting the overlap between
  consecutive chunks (the fixed/recursive/sentence chunkers repeat content
  across a boundary; the semantic chunker doesn't, by design).

```bash
python inspect_chunks.py                 # defaults to corpus/go.md
python dump_chunks.py corpus/sql.md      # or pass any file in corpus/
```

`sql.md` is a good one to dump under `--chunker semantic`: it ends with a
section tying SQL to RAG and pgvector, a sharp topic shift the semantic chunker
should cut cleanly on.

## Using your own data

Drop any `.md` or `.txt` files into `corpus/` and re-run. The sample corpus is
five docs about programming languages; swap in your own and the system indexes
them automatically. Good starter corpora: a documentation set, a folder of
articles, or papers in a field you care about.

## Ways to extend this (good next commits)

- **Smarter chunking** — *done.* Four strategies now ship in `chunkers.py`
  (fixed, recursive, sentence, semantic). Natural follow-ups: markdown
  *header-aware* splitting that keeps each `##` section together and prepends its
  header trail to every chunk; **late chunking** (embed the whole document with a
  long-context model, then pool into chunk vectors so each chunk keeps document
  context); or **contextual retrieval** (have an LLM prepend a one-line situating
  blurb to each chunk before embedding).
- **A real vector DB** — swap the NumPy `VectorStore` for Chroma or FAISS so it
  persists to disk and scales past a few thousand chunks.
- **Hybrid search** — combine the embedding similarity with keyword (BM25)
  matching, which catches exact terms (names, error codes) that embeddings miss.
- **Reranking** — retrieve the top ~30 chunks, then use a cross-encoder or a
  rerank API to re-score them down to the best few.
- **Evaluation** — add a small set of question/expected-answer pairs and measure
  retrieval quality (did the right chunk appear in the top *k*?), then use it to
  compare the chunkers objectively instead of by eye. The `ragas` library is the
  standard tool for this.
- **PDF ingestion** — parse PDFs into text so you can point it at real documents.

## Files

```
rag-from-scratch/
├── rag.py              # core pipeline: loading, embeddings, vector store, generation
├── chunkers.py         # the four pluggable chunking strategies + the tokenizer
├── main.py             # command-line interface (--chunker, --k)
├── inspect_chunks.py   # helper: chunk counts + sizes per strategy (quick overview)
├── dump_chunks.py      # helper: every chunk printed in full per strategy
├── corpus/             # sample documents — replace with your own
├── requirements.txt    # includes tiktoken (token counting) and sentence-transformers
├── .env.example        # template — copy to .env and add your OpenAI key
└── README.md
```
