# rag_core.py

import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import hashlib
import google.generativeai as genai

embedding_model = None
faiss_indexes = {}
vector_id_to_text_map = {}
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_DIM = 384


def initialize_rag_system():
    """
    Loads the embedding model into memory and prepares the vector store directory.
    """
    global embedding_model
    print("🧠 RAG Core: Initializing...")
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        print(f"✅ Created directory: {VECTOR_STORE_PATH}")
    print("🧠 RAG Core: Loading embedding model (this may take a moment on first run)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ RAG Core: Embedding model loaded successfully.")


def _get_user_specific_paths(user_api_key, mode):
    """
    Creates a unique, safe filename prefix for a user based on a hash of their API key.
    """
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    index_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_{mode}_index.faiss')
    map_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_{mode}_map.json')
    return index_path, map_path


def _load_user_data(user_api_key, mode):
    """
    Loads a user's FAISS index and text map from disk into memory for a specific mode.
    """
    if mode not in faiss_indexes:
        faiss_indexes[mode] = {}
    if mode not in vector_id_to_text_map:
        vector_id_to_text_map[mode] = {}

    if user_api_key in faiss_indexes[mode]: # Already loaded for this mode
        return
        
    index_path, map_path = _get_user_specific_paths(user_api_key, mode)

    if os.path.exists(index_path):
        print(f"🧠 RAG Core: Loading FAISS index for user in '{mode}' mode from {index_path}")
        faiss_indexes[mode][user_api_key] = faiss.read_index(index_path)
    else:
        print(f"🧠 RAG Core: No index found. Creating new FAISS index for user in '{mode}' mode.")
        faiss_indexes[mode][user_api_key] = faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            vector_id_to_text_map[mode][user_api_key] = {int(k): v for k, v in json.load(f).items()}
    else:
        vector_id_to_text_map[mode][user_api_key] = {}


def _save_user_data(user_api_key, mode):
    """
    Saves a user's FAISS index and text map from memory to disk for a specific mode.
    """
    index_path, map_path = _get_user_specific_paths(user_api_key, mode)
    if mode in faiss_indexes and user_api_key in faiss_indexes[mode]:
        faiss.write_index(faiss_indexes[mode][user_api_key], index_path)
    if mode in vector_id_to_text_map and user_api_key in vector_id_to_text_map[mode]:
        with open(map_path, 'w') as f:
            json.dump(vector_id_to_text_map[mode][user_api_key], f)


def _chunk_text(text, chunk_size=350, chunk_overlap=50):
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    """
    Processes a document's text and adds it to the correct knowledge base based on the mode.
    """
    _load_user_data(user_api_key, mode)
    print(f"🧠 RAG Core: Adding/Updating document '{document_id}' in '{mode}' knowledge base...")
    chunks = _chunk_text(document_text)
    if not chunks:
        print("⚠️ RAG Core: Document contains no text to add.")
        return

    chunk_embeddings = embedding_model.encode(chunks)
    index = faiss_indexes[mode][user_api_key]
    text_map = vector_id_to_text_map[mode][user_api_key]
    start_index = index.ntotal
    new_vector_ids = list(range(start_index, start_index + len(chunks)))
    index.add(np.array(chunk_embeddings, dtype=np.float32))

    for i, chunk in enumerate(chunks):
        vector_id = new_vector_ids[i]
        text_map[vector_id] = {"text": chunk, "source_doc": document_id}

    print(f"✅ RAG Core: Added {len(chunks)} chunks from '{document_id}'. Total vectors in '{mode}' mode: {index.ntotal}")
    _save_user_data(user_api_key, mode)
    
def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    """
    Finds and removes all vector chunks associated with a specific document ID.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map:
        print(f"⚠️ RAG Core: No knowledge base found for user in '{mode}' mode.")
        return

    ids_to_remove = [
        vector_id for vector_id, meta in text_map.items() 
        if meta.get("source_doc") == document_id
    ]

    if not ids_to_remove:
        print(f"⚠️ RAG Core: No vectors found for document '{document_id}' to remove.")
        return

    index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
    
    for vector_id in ids_to_remove:
        del text_map[vector_id]
        
    print(f"✅ RAG Core: Removed {len(ids_to_remove)} vectors for document '{document_id}'.")
    _save_user_data(user_api_key, mode)


def query_knowledge_base(user_api_key, query_text, mode, history=[]):
    """
    Searches the knowledge base, considering chat history, and generates a contextual answer.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return f"The {mode} knowledge base is empty. Please upload some documents first."

    print(f"🧠 RAG Core: Received query for '{mode}' mode: '{query_text}'")
    query_embedding = embedding_model.encode([query_text])
    k = min(5, index.ntotal) # Retrieve more chunks for better context
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_chunks = [text_map[i]["text"] for i in indices[0] if i in text_map]
    if not retrieved_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join(retrieved_chunks)
    print(f"🧠 RAG Core: Found {len(retrieved_chunks)} relevant chunks.")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Start a chat session with the existing history
        chat = model.start_chat(history=history)
        
        # The prompt now includes the chat history implicitly
        prompt = f"""
        You are an expert analyst. Your task is to provide a detailed and proper answer to the user's question based ONLY on the provided text snippets and the previous conversation turns.

        Follow these steps:
        1.  First, consider the ongoing conversation history to understand the full context of the user's latest question.
        2.  Carefully read the new context snippets provided below.
        3.  Think step-by-step about how the snippets and the conversation history can be combined to answer the latest question.
        4.  Formulate a comprehensive answer. If the information is not in the context or history, you must explicitly state that the information is not available in the documents.

        CONTEXT SNIPPETS:
        {context}

        LATEST QUESTION:
        {query_text}

        FINAL ANSWER:
        """
        response = chat.send_message(prompt)
        print("✅ RAG Core: Generated final answer with Gemini, considering history.")
        return response.text
    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call for chat: {e}")
        return f"An error occurred while trying to generate an answer: {e}"
