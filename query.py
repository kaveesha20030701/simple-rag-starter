"""
STEP 4 + 5 of the RAG pipeline: Retrieval and Generation.

Usage: python query.py "your question here"

If you set an OPENAI_API_KEY environment variable, this will generate a
real answer using the retrieved chunks. Without a key, it just shows you
which chunks were retrieved, so you can still see retrieval working.
"""

import os
import sys
import chromadb

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"
TOP_K = 3  # how many chunks to retrieve


def retrieve(question, top_k=TOP_K):
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection(COLLECTION_NAME)

    # Chroma embeds the question with the same model used during ingest,
    # then returns the chunks whose stored embeddings are closest to it.
    results = collection.query(query_texts=[question], n_results=top_k)

    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return retrieved_chunks, sources


def generate_answer(question, chunks):
    """Calls an LLM with the retrieved chunks as context. Requires OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI()

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question"')
        return

    question = sys.argv[1]
    print(f"Question: {question}\n")

    chunks, sources = retrieve(question)

    print("Retrieved chunks:")
    for chunk, source in zip(chunks, sources):
        print(f"  [{source}] {chunk[:120]}...")

    if os.environ.get("OPENAI_API_KEY"):
        print("\nGenerating answer...\n")
        answer = generate_answer(question, chunks)
        print(f"Answer: {answer}")
    else:
        print("\n(Set OPENAI_API_KEY to also generate a full answer from these chunks.)")


if __name__ == "__main__":
    main()
