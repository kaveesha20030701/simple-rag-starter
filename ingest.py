"""
STEP 1 + 2 + 3 of the RAG pipeline: Ingestion, Embedding, Storage.

Run this once (or whenever your documents change) to build the vector store.
Usage: python ingest.py
"""

import os
import glob
import chromadb

DOCS_FOLDER = "documents"
DB_FOLDER = "chroma_db"          # where the vector store saves to disk
COLLECTION_NAME = "my_documents"

CHUNK_SIZE = 120       # words per chunk
CHUNK_OVERLAP = 20     # words shared between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping word-based chunks.

    Overlap matters: if a sentence gets cut exactly at a chunk boundary,
    the overlap means the next chunk still contains that sentence's full
    context, so we don't lose meaning at the seams.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_and_chunk_documents():
    all_chunks = []
    all_ids = []
    all_metadata = []

    filepaths = glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))
    print(f"Found {len(filepaths)} document(s) in '{DOCS_FOLDER}/'")

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"  {filename} -> {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}-chunk-{i}")
            all_metadata.append({"source": filename, "chunk_index": i})

    return all_chunks, all_ids, all_metadata


def main():
    chunks, ids, metadata = load_and_chunk_documents()

    if not chunks:
        print("No documents found. Add .txt files to the documents/ folder first.")
        return

    # A persistent client saves the vector store to disk in DB_FOLDER,
    # so you don't have to re-embed everything every time you run a query.
    client = chromadb.PersistentClient(path=DB_FOLDER)

    # If we've run this before, start fresh so re-running ingest.py
    # doesn't create duplicate entries.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Chroma's default embedding function (all-MiniLM-L6-v2, ONNX version)
    # is used automatically here — we don't have to call an embedding
    # model ourselves. Under the hood, when we call .add(), Chroma
    # converts each chunk's text into a vector for us.
    collection = client.create_collection(COLLECTION_NAME)

    print("\nEmbedding and storing chunks... (first run downloads a small model, be patient)")
    collection.add(documents=chunks, ids=ids, metadatas=metadata)

    print(f"\nDone. Stored {collection.count()} chunks in '{DB_FOLDER}/'.")


if __name__ == "__main__":
    main()
