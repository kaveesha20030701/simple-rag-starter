"""
Simple web frontend for the RAG pipeline, built with Streamlit.

Run with:  streamlit run app.py
"""

import os
import chromadb
import streamlit as st
from ingest import chunk_text

DOCS_FOLDER = "documents"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"

st.set_page_config(page_title="Simple RAG", page_icon="📚")
st.title("📚 Simple RAG App")

# --- Sidebar: optional API key, entered here instead of the terminal ---
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API Key (optional)", type="password")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    st.caption("Leave blank to only see retrieved chunks, without a generated answer.")

# --- Section 1: upload and ingest documents ---
st.header("1. Add documents")

uploaded_files = st.file_uploader(
    "Upload .txt files", type="txt", accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join(DOCS_FOLDER, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to '{DOCS_FOLDER}/'.")

existing_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")] if os.path.exists(DOCS_FOLDER) else []
if existing_files:
    st.caption(f"Currently in documents/: {', '.join(existing_files)}")

if st.button("🔨 Build / Update Knowledge Base"):
    if not existing_files:
        st.error("No .txt files found in documents/ yet — upload some first.")
    else:
        with st.spinner("Chunking and embedding documents..."):
            chunks, ids, metadata = [], [], []
            for filename in existing_files:
                filepath = os.path.join(DOCS_FOLDER, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                for i, chunk in enumerate(chunk_text(text)):
                    chunks.append(chunk)
                    ids.append(f"{filename}-chunk-{i}")
                    metadata.append({"source": filename, "chunk_index": i})

            client = chromadb.PersistentClient(path=DB_FOLDER)
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            collection = client.create_collection(COLLECTION_NAME)
            collection.add(documents=chunks, ids=ids, metadatas=metadata)

        st.success(f"Knowledge base built: {len(chunks)} chunks from {len(existing_files)} file(s).")

st.divider()

# --- Section 2: ask questions ---
st.header("2. Ask a question")

question = st.text_input("Your question")
top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)

if st.button("🔍 Get Answer") and question:
    try:
        client = chromadb.PersistentClient(path=DB_FOLDER)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        st.error("No knowledge base found yet. Upload documents and click 'Build / Update Knowledge Base' first.")
        st.stop()

    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    st.subheader("Retrieved chunks")
    st.caption("Lower distance = closer match.")
    for chunk, source, distance in zip(retrieved_chunks, sources, distances):
        with st.expander(f"{source}  (distance: {distance:.3f})"):
            st.write(chunk)

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        llm_client = OpenAI()
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""
        with st.spinner("Generating answer..."):
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
        st.subheader("Answer")
        st.write(response.choices[0].message.content)
    else:
        st.info("Add your OpenAI API key in the sidebar to also get a generated answer.")
