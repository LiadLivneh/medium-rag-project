import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import langchain

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

def main():
    parser = argparse.ArgumentParser(description="Process and embed Medium articles into Pinecone.")
    parser.add_argument(
        "--nrows",
        type=str,
        default="5",
        help="Number of rows to process from the CSV, or 'all' to process the entire file"
    )
    args = parser.parse_args()
    if args.nrows.lower() == 'all':
        print("Loading ALL rows from CSV...")
        df = pd.read_csv("medium-english-50mb.csv")
    else:
        num_rows = int(args.nrows)
        print(f"Loading {num_rows} rows from CSV...")
        df = pd.read_csv("medium-english-50mb.csv", nrows=num_rows)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=512,
        chunk_overlap=100,
    )
    processed_data = []
    #Process the dataframe and extract chunks and metadata
    for idx, row in df.iterrows():
        article_id = str(idx)
        text_content = str(row['text'])
        title = row['title']
        
        chunks = text_splitter.split_text(text_content)
        
        for chunk_idx, chunk in enumerate(chunks):
            vector_id = f"{article_id}_chunk_{chunk_idx}"
            metadata = {
                "article_id": article_id,
                "title": title,
                "chunk": chunk
            }
            processed_data.append({
                "id": vector_id,
                "chunk": chunk,
                "metadata": metadata
            })
     #Prepare lists for batch embedding and upsert       
    all_texts = [str(item['chunk']) for item in processed_data]
    all_metadata = [item['metadata'] for item in processed_data]
    all_ids = [str(item['id']) for item in processed_data]
    
    print(f"Generating embeddings for {len(all_texts)} chunks...")
    all_embeddings = emb.embed_documents(all_texts)
    
    # Delete all existing vectors in the default namespace
    index.delete(delete_all=True)
    
    print(f"Upserting vectors to Pinecone...")
    vectors_to_upsert = list(zip(all_ids, all_embeddings, all_metadata))
    batch_size = 200
    for i in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[i : i + batch_size])
        
    print("Upsert complete.")
if __name__ == "__main__":
    main()