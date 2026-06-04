import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

client = AzureOpenAI(
    azure_endpoint="https://api.llmod.ai", 
    api_key=os.environ.get("LLMOD_API_KEY"),
    api_version="2024-02-01" 
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

def chunk_text(text, max_words=600, overlap=150):
    """
    Basic word-based chunking to approximate the token limits.
    Max tokens is 1024, max overlap is 30%.
    """
    words = str(text).split()
    chunks = []
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        # Stop if we reached the end of the text
        if i + max_words >= len(words):
            break
            
    return chunks

def main():
    print("Loading CSV...")
    df = pd.read_csv("medium-english-50mb.csv", nrows=5)
    
    vectors_to_upsert = []
    
    for idx, row in df.iterrows():
        article_id = str(idx)
        text_content = row['text']
        title = row['title']
        
        chunks = chunk_text(text_content)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Generate embedding vector
            response = client.embeddings.create(
                model="4UHRUIN-text-embedding-3-small",
                input=chunk
            )
            embedding = response.data[0].embedding
            
            vector_id = f"{article_id}_chunk_{chunk_idx}"
            
            metadata = {
                "article_id": article_id,
                "title": title,
                "chunk": chunk
            }
            
            vectors_to_upsert.append((vector_id, embedding, metadata))
            
    if vectors_to_upsert:
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        index.upsert(vectors=vectors_to_upsert)
        print("Upsert complete.")

if __name__ == "__main__":
    main()