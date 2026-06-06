import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

def main():
    print("Loading ALL rows from CSV...")
    df = pd.read_csv("medium-english-50mb.csv")
    df = df.fillna("Unknown")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=512,
        chunk_overlap=100,
    )

    updates_count = 0

    for idx, row in df.iterrows():
        article_id = str(idx)
        text_content = str(row['text'])

        # Split text only to recreate the exact number of chunk IDs
        chunks = text_splitter.split_text(text_content)

        for chunk_idx, chunk in enumerate(chunks):
            vector_id = f"{article_id}_chunk_{chunk_idx}"
            
            new_metadata = {
                "article_id": article_id,
                "title": str(row['title']),
                "author": str(row['authors']),
                "url": str(row['url']),
                "timestamp": str(row['timestamp']),
                "tags": str(row['tags']),
                "chunk": chunk
            }

            try:
                # Update metadata without modifying the vector embeddings
                index.update(
                    id=vector_id,
                    set_metadata=new_metadata
                )
                updates_count += 1
                if updates_count % 1000 == 0:
                    print(f"Updated metadata for {updates_count} chunks...")
            except Exception as e:
                print(f"Error updating {vector_id}: {e}")

    print(f"Update complete. Total chunks updated: {updates_count}")

if __name__ == "__main__":
    main()