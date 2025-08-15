# rag_core.py

import hashlib
import google.generativeai as genai
from supabase import create_client, Client
import time

# --- Supabase Configuration ---
# IMPORTANT: Replace these with your actual Supabase URL and Public Anon Key
SUPABASE_URL = "https://kpiiqhpuwlncztkczjzx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtwaWlxaHB1d2xuY3p0a2N6anp4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTI5NTIsImV4cCI6MjA3MDg2ODk1Mn0.M4zfqNMr1Zwgil1i7Y06tvW6FjDkAetnV8eU6eIdNm4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Functions ---

def _get_user_hash(user_api_key):
    """Creates a unique, safe hash for a user based on their API key."""
    return hashlib.sha256(user_api_key.encode()).hexdigest()[:16]

def _chunk_text(text, chunk_size=350, chunk_overlap=50):
    """Splits text into manageable chunks for embedding."""
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# --- Core RAG Functions (Now using Supabase pgvector) ---

def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    """Creates embeddings for a document and stores them in the Supabase database."""
    print(f"🧠 RAG Core: Adding/Updating document '{document_id}' in Supabase for mode '{mode}'...")
    user_hash = _get_user_hash(user_api_key)
    
    try:
        genai.configure(api_key=user_api_key)
    except Exception as e:
        print(f"❌ RAG Core: Invalid Google API Key provided. Error: {e}")
        return

    chunks = _chunk_text(document_text)
    if not chunks:
        print("⚠️ RAG Core: Document contains no text to add.")
        return

    try:
        # The free tier of the Gemini API has a limit of 100 chunks per request.
        # We process in batches to stay within this limit.
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            print(f"🧠 RAG Core: Creating embeddings for batch {i//batch_size + 1}...")
            
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=batch_chunks,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = response['embedding']

            documents_to_insert = []
            for j, chunk in enumerate(batch_chunks):
                documents_to_insert.append({
                    "user_hash": user_hash,
                    "document_id": document_id,
                    "mode": mode,
                    "content": chunk,
                    "embedding": embeddings[j]
                })

            supabase.table('documents').insert(documents_to_insert).execute()
            print(f"✅ RAG Core: Inserted batch of {len(batch_chunks)} chunks into Supabase.")
            time.sleep(1) # Add a small delay to respect potential rate limits

    except Exception as e:
        print(f"❌ RAG Core: Failed to create embeddings or insert into Supabase. Error: {e}")

def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    """Removes all vectors associated with a specific document from the database."""
    print(f"🧠 RAG Core: Removing document '{document_id}' from Supabase for mode '{mode}'...")
    user_hash = _get_user_hash(user_api_key)
    try:
        (
            supabase.table('documents')
            .delete()
            .match({'user_hash': user_hash, 'document_id': document_id, 'mode': mode})
            .execute()
        )
        print(f"✅ RAG Core: Removal complete for document '{document_id}'.")
    except Exception as e:
        print(f"❌ RAG Core: Failed to remove document from Supabase. Error: {e}")

def query_knowledge_base(user_api_key, query_text, mode, history=[]):
    """Queries the knowledge base by creating a query embedding and searching in Supabase."""
    print(f"🧠 RAG Core: Received query for '{mode}' mode: '{query_text}'")
    user_hash = _get_user_hash(user_api_key)
    
    try:
        genai.configure(api_key=user_api_key)
    except Exception as e:
        return f"Invalid Google API Key provided. Error: {e}"

    try:
        # 1. Create an embedding for the user's query
        query_embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = query_embedding_response['embedding']

        # 2. Call the Supabase database function to find similar documents
        matches = supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'p_user_hash': user_hash,
            'p_mode': mode,
            'match_threshold': 0.7,  # How similar documents must be (0 to 1)
            'match_count': 5         # How many documents to return
        }).execute()

        if not matches.data:
            return f"I couldn't find any relevant information in your {mode} documents to answer that question."

        context = "\n\n---\n\n".join([item['content'] for item in matches.data])
        print(f"🧠 RAG Core: Found {len(matches.data)} relevant chunks from Supabase.")

        # 3. Use the retrieved context to generate a final answer with Gemini
        chat_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        chat = chat_model.start_chat(history=history)
        
        prompt = f"""
        You are an expert analyst. Your task is to provide a detailed and proper answer to the user's question based ONLY on the provided text snippets and the previous conversation turns.

        Follow these steps:
        1. First, consider the ongoing conversation history to understand the full context of the user's latest question.
        2. Carefully read the new context snippets provided below.
        3. Think step-by-step about how the snippets and the conversation history can be combined to answer the latest question.
        4. Formulate a comprehensive answer. If the information is not in the context or history, you must explicitly state that the information is not available in the documents.

        CONTEXT SNIPPETS:
        {context}

        LATEST QUESTION:
        {query_text}

        FINAL ANSWER:
        """
        response = chat.send_message(prompt)
        print("✅ RAG Core: Generated final answer with Gemini.")
        return response.text

    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call or Supabase query: {e}")
        return f"An error occurred while trying to generate an answer: {e}"

