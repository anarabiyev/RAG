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
API key (the one exception is the `pinecone` store, a remote service that needs
its own `PINECONE_API_KEY`). The generation half (writing the answer) calls an
LLM. You can run the project with just retrieval to see that half working on its
own.

The very first step — **chunking** — is now pluggable: four strategies live in
`chunkers.py`, and you choose one with `--chunker`. Chunking quietly sets a
ceiling on everything downstream (you can only retrieve chunks you created), so
it's worth understanding deeply, which is why this project makes the choice
explicit rather than hiding it.

The **vector store** — where the embeddings live and the nearest-neighbour
search runs — is pluggable too: four backends live in `stores.py`, and you
choose one with `--store`. The default NumPy store is the from-scratch baseline;
FAISS and Chroma are the real tools you'd graduate to on your own machine; and
Pinecone is the step off your machine entirely — a remote, managed database you
talk to over the network.

## Quickstart

```bash
pip install -r requirements.txt

# Optional: add your OpenAI key to get generated answers. Without it you
# still see the retrieved passages, which is the retrieval half of RAG.
# The `pinecone` store also needs a PINECONE_API_KEY in the same .env file.
cp .env.example .env      # then edit .env and paste in your key(s)

python main.py                              # interactive question loop
python main.py "who created Rust?"          # one-shot question
python main.py --chunker semantic "..."     # pick a chunking strategy
python main.py --store faiss "..."          # pick a vector-store backend
python main.py --store pinecone "..."       # use a remote, managed vector DB
python main.py --chunker fixed --k 6 "..."  # and how many chunks to retrieve
```

The key is read from `.env` automatically (via python-dotenv), so you never
have to export it by hand. `.env` is gitignored, so your keys are never
committed.

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

The code is in four files. `rag.py` is the core pipeline; `chunkers.py` holds
the chunking strategies; `stores.py` holds the vector-store backends; `main.py`
is the command-line wrapper. Read `rag.py`'s four sections in order — they match
the diagram above:

1. **Loading & chunking** — documents are loaded from `corpus/` and split into
   passages by a *chunker* (see the next section). The default is the recursive
   chunker. Overlap between chunks stops a thought on a boundary from being lost.
2. **Embeddings** — `sentence-transformers` maps each chunk (and later each
   question) to a vector. Similar meaning → nearby vectors. Vectors are
   normalised so cosine similarity becomes a plain dot product.
3. **Vector store & retrieval** — chunk vectors are indexed by a *store* (see
   the Vector stores section). The default NumPy store holds them in one matrix
   and searches with a single matrix multiply that scores the query against
   every chunk at once; we keep the top *k*. That's exactly what a vector
   database does, minus the scaling and persistence — so `stores.py` also lets
   you swap in the real thing (FAISS, Chroma, or a remote Pinecone database)
   behind the same interface.
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

## Vector stores

The vector store is the "database" at the heart of RAG: it holds one vector per
chunk and, given the question's vector, returns the chunks whose vectors are
closest. All four backends live in `stores.py` behind one interface (`.add`
and `.search`) — the same pattern as the chunkers — and every store returns a
*similarity* score where higher means more relevant, so the retrieved passages
are directly comparable across backends. Pick one with `--store`:

- **`numpy`** *(default)* — `NumpyStore`. The from-scratch baseline: every
  vector in one NumPy matrix, and search is a single matrix multiply against all
  of them. Exact and exhaustive — it really checks every chunk. Fine up to a few
  thousand chunks, O(n) per query, stores nothing to disk.
- **`faiss`** — `FaissStore`. Meta's FAISS, the standard fast nearest-neighbour
  **library** — and the word matters: it's a library you embed in your process,
  *not* a server. It keeps the vectors in its own C++ memory and knows nothing
  about your text, so the store keeps its own `index → Chunk` mapping by hand
  (that bookkeeping is the lesson). The default `IndexFlatIP` is *exact* — the
  same exhaustive search as NumPy, just in optimised native code. Pass
  `index_type="hnsw"` to switch to an approximate HNSW graph index: sub-linear
  search that may miss the odd true neighbour (that's "recall"), pointless on
  five docs but transformative at millions. `pip install faiss-cpu`.
- **`chroma`** — `ChromaStore`. The smallest step from library to **database**:
  Chroma runs *embedded, in-process, like SQLite for vectors* — no server, no
  Docker, no API key. Unlike FAISS it stores your text and metadata next to the
  vectors and hands them back on query, so the store doesn't keep its own list.
  Its engine uses HNSW, so search is approximate. By default it's in-memory and
  vanishes on exit; pass `persist_dir=...` and it writes to disk. `pip install
  chromadb`.
- **`pinecone`** — `PineconeStore`. The step off your machine entirely: a
  **remote, managed** vector database you reach over the network with an API key
  (`PINECONE_API_KEY`, read from `.env` like your OpenAI key). Unlike the three
  above, the vectors don't live in your process at all — Pinecone stores them on
  its servers and runs the nearest-neighbour search *there*, so a million-vector
  corpus never has to fit in your RAM. Like Chroma it keeps your text and
  metadata beside each vector and hands them back on query, so the store rebuilds
  each `Chunk` from the match's metadata rather than keeping its own list; the
  index is created with the cosine metric, whose score is already a similarity,
  so (unlike Chroma's distance) no conversion is needed. It uses stable
  `source-index` ids so re-runs overwrite instead of duplicating, and — because
  `main.py` rebuilds the index every run while a real database *persists* — it
  wipes its namespace on each run by default (`wipe=True`) to match the in-memory
  stores; flip `wipe=False` and the index survives between runs, which is the
  whole reason you'd reach for a database in the first place. The free tier is
  serverless on AWS `us-east-1`, one index, 2 GB. `pip install pinecone`.

A subtlety worth holding onto, parallel to the two-tokenizers note above: the
backends differ on **exact vs approximate** search and on **where the vectors
live**. NumPy and FAISS-`flat` are exact (they compare against every vector);
FAISS-`hnsw`, Chroma, and Pinecone are approximate (they navigate a graph and
skip most). On the five-doc corpus all four return essentially identical
passages — the approximate ones still find the true neighbours on data this
small — so the point of switching `--store` here is to *confirm they agree*, not
to change the results; the speedup only shows up with far more chunks. As for
where the vectors live: the NumPy matrix, FAISS's vectors, and an in-memory
Chroma all sit in your process's RAM and vanish when it exits; Chroma with
`persist_dir` writes them to local disk (a `chroma.sqlite3` of text + metadata
beside a binary HNSW index) so they survive a restart; and Pinecone keeps them
off your machine entirely, on its own servers, so they survive not just a
restart but a move to a different computer — the price being a network hop and an
API key on every call.

## Hybrid search, reranking & evaluation

Retrieval used to be one step — embed the question, ask the store for the
nearest vectors. It's now a short pipeline, mirroring the diagram
*Query → (BM25 + Dense) → RRF → cross-encoder → top-k*, with each stage
pluggable like the chunkers and stores.

**Retrieval** (`retrievers.py`, choose with `--retriever`):

- **`dense`** — embeddings + a vector store. Great at *meaning* ("garbage
  collector" ≈ "automatic memory management"), weak at exact tokens.
- **`bm25`** — a from-scratch Okapi BM25 keyword ranker (the honest baseline,
  like `NumpyStore` for stores — no dependencies; `pip install bm25s` is the
  fast graduate option). Nails exact terms embeddings miss: `GIL`, `pgvector`,
  `gofmt`, version numbers.
- **`hybrid`** *(default)* — run both and fuse with **Reciprocal Rank Fusion**:
  ignore the two incomparable score scales, combine by *rank* (`1/(60+rank)`).
  High in either list helps; high in both wins.

**Reranking** (`rerankers.py`, enable with `--rerank`): the "retrieve wide,
rerank narrow" pattern. Retrieval pulls a wide pool (`--candidates`, default 30),
then a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads each
`(question, chunk)` pair *together* and rescores to the `--k` best. Retrieval
only has to get the right chunk *somewhere* in the pool; the reranker floats it
up. The diagram's hosted `rerank-*` box is the same idea over an API — there's a
drop-in template at the bottom of `rerankers.py`.

```bash
python main.py --retriever hybrid "does Go have a garbage collector?"
python main.py --retriever hybrid --rerank "what is pgvector?"
python main.py --retriever bm25 "what enforces Go's formatting?"   # no embeddings
```

**Evaluation** (`evaluate.py`): stop judging by eye. A small labeled set of
(question → which file answers it) pairs is scored on **Hit@k** (did a chunk
from the right file land in the top k?) and **MRR** (how high?). Use it to
compare chunkers/retrievers objectively:

```bash
python evaluate.py --retriever hybrid --rerank          # score one config
python evaluate.py --compare-chunkers --retriever hybrid  # all 4 chunkers, one table
```

Labels are by *source file* (cheap to hand-write, stable across chunkers, no API
key). `ragas` is the standard for grading generated answers; this is the
from-scratch retrieval-only version.

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
- **A real vector DB** — *done.* `stores.py` now ships four backends (numpy,
  faiss, chroma, pinecone) behind one interface, chosen with `--store`, and
  Pinecone is the networked database that moves storage and search off your
  machine onto a server you query. Natural follow-ups: **split ingestion from
  querying** so you build the index once and then answer questions against it
  without re-embedding and re-upserting the whole corpus every run (the current
  rebuild-each-run model is fine for five docs but wrong at scale); turn on
  Pinecone **namespaces** to keep multiple corpora (or tenants) in one index and
  isolate query cost; turn on Chroma's `persist_dir` so its index survives
  between runs; or try other networked DBs (Qdrant, Weaviate, or `pgvector` on
  Postgres).
- **Hybrid search** — *done.* `retrievers.py` adds a from-scratch BM25 keyword
  retriever and fuses it with dense embeddings via Reciprocal Rank Fusion
  (`--retriever hybrid`). See "Hybrid search, reranking & evaluation" above.
- **Reranking** — *done.* `rerankers.py` adds a local cross-encoder that
  re-scores a wide candidate pool down to the best few (`--rerank`).
- **Evaluation** — *done.* `evaluate.py` ships a labeled question set and scores
  Hit@k and MRR, with `--compare-chunkers` to rank the strategies by number.
  `ragas` remains the tool to reach for once you're grading *generated answers*.
- **PDF ingestion** — parse PDFs into text so you can point it at real documents.

## Files

```
rag-from-scratch/
├── rag.py              # core pipeline: loading, chunking, retrieval, generation
├── chunkers.py         # the four pluggable chunking strategies + the tokenizer
├── stores.py           # the four pluggable vector-store backends (numpy/faiss/chroma/pinecone)
├── retrievers.py       # retrieval strategies: dense / bm25 / hybrid (RRF fusion)
├── rerankers.py        # cross-encoder reranking (none / local cross-encoder)
├── evaluate.py         # retrieval-quality harness: labeled Q&A, Hit@k + MRR
├── main.py             # command-line interface (--chunker, --store, --retriever, --rerank, --k)
├── inspect_chunks.py   # helper: chunk counts + sizes per strategy (quick overview)
├── dump_chunks.py      # helper: every chunk printed in full per strategy
├── corpus/             # sample documents — replace with your own
├── requirements.txt    # tiktoken + sentence-transformers; faiss-cpu/chromadb/pinecone optional
├── .env.example        # template — copy to .env and add your OpenAI key (and Pinecone key)
└── README.md
```
