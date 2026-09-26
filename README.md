# Simple RAG Starter

A minimal, from-scratch Retrieval-Augmented Generation pipeline, built to
understand each moving part before adding anything more advanced.

## How it works

1. **`documents/`** — plain `.txt` files you want to be able to ask questions about.
2. **`ingest.py`** — reads those files, splits them into overlapping chunks,
   converts each chunk into an embedding (a vector of numbers representing
   its meaning), and stores everything in a local vector database (Chroma).
3. **`query.py`** — takes your question, embeds it the same way, finds the
   most similar stored chunks, and (optionally) sends them + your question
   to an LLM to generate a final answer.

## Setup

```bash
pip install -r requirements.txt
```

The first time you run `ingest.py`, Chroma will download a small embedding
model (all-MiniLM-L6-v2, ~90MB) automatically. This needs an internet
connection but only happens once.

## Usage

```bash
# 1. Build the vector store from your documents
python ingest.py

# 2. Ask questions (retrieval only, no LLM key needed)
python query.py "What is an embedding?"

# 3. For a full generated answer, set an OpenAI API key first
export OPENAI_API_KEY="sk-..."
python query.py "What is an embedding?"
```

## Next steps once this makes sense

- Swap in your own documents (drop `.txt` files into `documents/`, or
  extend `ingest.py` to handle PDFs with a library like `pypdf`).
- Try different `CHUNK_SIZE` / `CHUNK_OVERLAP` values in `ingest.py` and
  see how retrieval quality changes.
- Print the similarity scores in `query.py` and experiment with a
  threshold — this is your first step toward an "abstain if nothing is
  relevant enough" behavior.
- Look at swapping Chroma's default embedding model for a stronger one,
  or trying a different vector store (FAISS) to compare.
