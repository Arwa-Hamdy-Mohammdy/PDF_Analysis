import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import GEMINI_API_KEY


def get_embedding(text: str) -> list[float]:
    """
    Generates embedding vector for a given text using Google Gemini API.
    Uses gemini-embedding-2 (verified HTTP 200).
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in Secrets or .env file.")

    models_to_try = [
        "gemini-embedding-2",
        "text-embedding-004"
    ]

    last_error = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={GEMINI_API_KEY}"
        body = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            response = requests.post(url, json=body, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["embedding"]["values"]
            else:
                last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Failed to generate embedding from Gemini API. Error: {last_error}")


def embed_chunks(chunks: list[str], progress_callback=None) -> list[dict]:
    """
    Processes a list of text chunks and generates embeddings for each.
    Allows passing a progress callback for Streamlit UI updates.
    """
    embedded_docs = []
    total = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embedded_docs.append({
            "text": chunk,
            "embedding": embedding
        })
        if progress_callback:
            progress_callback(idx + 1, total)
            
    return embedded_docs


def chat_completion(prompt: str, system_instruction: str = None) -> str:
    """
    Calls Gemini Chat Completion model using verified working endpoint (gemini-flash-latest).
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in Secrets or .env file.")

    models_to_try = [
        "gemini-flash-latest",
        "gemini-2.0-flash"
    ]

    last_error = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        
        body = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        try:
            response = requests.post(url, json=body, timeout=60)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Failed to generate response from Gemini API. Error: {last_error}")


def ask_pdf(question: str, embedded_docs: list[dict], top_k: int = 3) -> tuple[str, list[dict]]:
    """
    Performs RAG search: Embeds query, ranks document chunks using cosine similarity,
    builds grounded context, and fetches answer from Gemini.
    Returns (answer, top_relevant_chunks).
    """
    if not embedded_docs:
        return "No content indexed from PDF.", []

    query_embedding = get_embedding(question)

    scores = []
    for doc in embedded_docs:
        score = cosine_similarity(
            [query_embedding],
            [doc["embedding"]]
        )[0][0]
        
        scores.append({
            "text": doc["text"],
            "score": float(score)
        })

    # Sort descending by similarity score
    scores.sort(key=lambda x: x["score"], reverse=True)
    top_docs = scores[:top_k]

    # Combine top context chunks
    context_blocks = []
    for i, doc in enumerate(top_docs, start=1):
        context_blocks.append(f"[Excerpt {i}]:\n{doc['text']}")
        
    context = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an expert AI assistant that answers questions about uploaded PDF documents.\n"
        "Rules:\n"
        "1. Base your answer strictly on the provided Context excerpts.\n"
        "2. If the context does not contain enough information to answer, state clearly: "
        "'I couldn't find the answer in the uploaded PDF document.'\n"
        "3. Provide clear, concise, and structured formatting (use bullet points or markdown where applicable).\n"
        "4. Respond in the same language as the user's question."
    )

    prompt = f"""Context from PDF:
---
{context}
---

User Question: {question}

Answer:"""

    answer = chat_completion(prompt, system_instruction=system_instruction)

    return answer, top_docs