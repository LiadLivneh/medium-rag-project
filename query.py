import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

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
    azure_deployment="4UHRUIN-gpt-5-mini"
    )

def run_rag_pipeline(query_text, top_k=10):
    print(f"Searching for: '{query_text}'...")
    
    
    query_embedding = emb.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    context_chunks = []
    context_text_for_llm = ""
    
    for match in results['matches']:
        chunk_data = {
            "article_id": str(match['metadata'].get('article_id', match['id'])),
            "title": str(match['metadata']['title']),
            "chunk": str(match['metadata']['chunk']),
            "score": float(match['score'])
        }
        context_chunks.append(chunk_data)
        context_text_for_llm += f"Title: {chunk_data['title']}\nContent: {chunk_data['chunk']}\n\n"

    system_prompt = (
        "You are a Medium-article assistant that answers questions strictly and only based on the "
        "Medium articles dataset context provided to you (metadata and article passages). You must not "
        "use any external knowledge, the open internet, or information that is not explicitly contained "
        "in the retrieved context. "
        "If the answer cannot be determined from the provided context, respond EXACTLY and ONLY with: "
        "“I don’t know based on the provided Medium articles data.” Do not add any further explanation. "
        "However, unless the user explicitly requests a specific format, if the answer CAN be determined, always explain your answer using the given context, "
        "quoting or paraphrasing the relevant article passage or metadata when helpful."
    )    
    user_prompt = f"Context:\n{context_text_for_llm}\nQuestion: {query_text}"

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
    
    return final_output


if __name__ == "__main__":
    test_question = "I want to treat my writing more like a business and focus on marketing and 'money-content', but I'm not sure where to start. Which article would you recommend I read, and why?"
    output = run_rag_pipeline(test_question)
    
    # Clean, readable terminal output for testing
    print("\n" + "="*60)
    print("🤖 MODEL RESPONSE:")
    print("-" * 60)
    print(output["response"])
    print("\n" + "="*60)
    
    print("📚 RETRIEVED CONTEXT (Top 10 Matches):")
    print("-" * 60)
    for i, chunk in enumerate(output["context"]):
        print(f"Match {i+1} | Score: {chunk['score']:.4f} | Title: {chunk['title']}")
        # Print just the first 150 characters of the chunk so it doesn't flood the terminal
        print(f"Preview: {chunk['chunk']}...\n")
    print("="*60 + "\n")