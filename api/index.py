import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load env variables for local testing (Vercel will inject these automatically online)
load_dotenv()

app = Flask(__name__)

CHUNK_SIZE = 512  
OVERLAP = 100
OVERLAP_RATIO = OVERLAP / CHUNK_SIZE
TOP_K = 20

# Initialize Models and DB (Doing this globally keeps the app fast)
emb = AzureOpenAIEmbeddings(
    azure_endpoint="https://api.llmod.ai",
    api_key=os.environ.get("LLMOD_API_KEY"),
    api_version="2024-02-01",
    model="4UHRUIN-text-embedding-3-small",
    chunk_size=256 
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

llm = AzureChatOpenAI(
    azure_endpoint="https://api.llmod.ai",
    api_key=os.environ.get("LLMOD_API_KEY"),
    api_version="2024-02-01",
    azure_deployment="4UHRUIN-gpt-5-mini",
    reasoning_effort="low"
)


@app.route('/api/stats', methods=['GET'])
def stats():
    """Endpoint 2: Returns the configuration of the RAG system."""
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    })


@app.route('/api/prompt', methods=['POST'])
def prompt():
    """Endpoint 1: Receives a question and returns the agent's answer + context."""
    data = request.get_json()
    
    # Safety check in case the user sends a bad request
    if not data or "question" not in data:
        return jsonify({"error": "Please provide a JSON with a 'question' key."}), 400
        
    query_text = data["question"]
    
    # Embed the question and query Pinecone
    query_embedding = emb.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    context_chunks = []
    context_text_for_llm = ""
    
    # Format the retrieved context
    for match in results.get('matches', []):
        meta = match.get('metadata', {})
        
        chunk_data = {
            "article_id": str(match.get('id', 'N/A')),
            "title": str(meta.get('title', 'N/A')),
            "chunk": str(meta.get('chunk', '')),
            "score": float(match['score'])
        }
        context_chunks.append(chunk_data)
        
        context_text_for_llm += (
            f"--- SOURCE ---\n"
            f"Title: {chunk_data['title']}\n"
            f"Content: {chunk_data['chunk']}\n\n"
        )

    system_prompt = (
    "You are a Medium-article assistant that answers questions strictly and only based on the "
    "Medium articles dataset context provided to you (metadata and article passages). "
    "Rules:"
    "1. You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
    "2. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.” ONLY and ADD NOTHING ELSE. "
    "3. If the user explicitly requests a specific format, stick to his instruction and output EXACTLY the requested format, adding no additional content. "
    "4. When finding a match, always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful. "
    "5. DO NOT BREAK THE FOURTH WALL. Speak directly to the user and answer the question natively."
)
    
    user_prompt = f"Context:\n{context_text_for_llm}\nUser query: {query_text}"

    # Call the LLM and return final output
    chat_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    final_output = {
        "response": chat_response.content,
        "context": context_chunks,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }
    
    return jsonify(final_output)