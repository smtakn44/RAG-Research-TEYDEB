'''
PDF data , RecursiveCharacterTextSplitter, Ollama Embedding, LateChunking

'''

import argparse
import os
import shutil
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
import numpy as np
from langchain_community.embeddings.ollama import OllamaEmbeddings 

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    embeddings = embed_documents(documents)
    chunks = late_chunking(documents, embeddings)
    add_to_chroma(chunks)


def load_documents(data_path=DATA_PATH):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def get_embedding_function():  
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

def embed_documents(documents: list[Document]):
    embedding_function = get_embedding_function()
    return [embedding_function.embed_query(doc.page_content) for doc in documents]

def late_chunking(documents: list[Document], embeddings: list):
    chunk_size = 800
    chunk_overlap = 100
    chunks = []
    
    for doc, embedding in zip(documents, embeddings):
        text = doc.page_content
        total_chars = len(text)
        
        for i in range(0, total_chars, chunk_size - chunk_overlap):
            chunk_text = text[i:i+chunk_size]
            chunk_embedding = embedding[i:i+chunk_size]
            
            if len(chunk_embedding) < chunk_size:
                # Pad the embedding if it's shorter than chunk_size
                chunk_embedding = np.pad(chunk_embedding, (0, chunk_size - len(chunk_embedding)))
            
            chunk = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "start_index": i,
                    "end_index": i + len(chunk_text)
                }
            )
            chunks.append((chunk, chunk_embedding))
    
    return chunks

def add_to_chroma(chunks: list[tuple[Document, list]]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids([chunk for chunk, _ in chunks])
    
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    new_embeddings = []
    for (chunk, embedding) in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_embeddings.append(embedding)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, embeddings=new_embeddings, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()