'''
PDF data , SoruCevap Şeklinde TextSplitter, Turkish Embedding

'''

import argparse
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

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
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents(data_path=DATA_PATH):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_documents(documents):
    # Convert all document content to a single string
    all_text = " ".join([doc.page_content for doc in documents])

    # Split by "Soru:" and check for corresponding "Cevap:"
    chunks = all_text.split("Soru:")
    paragraphs = []
    for chunk in chunks:
        if "Cevap:" in chunk:
            paragraphs.append("Soru:" + chunk.strip())

    return paragraphs

def add_to_chroma(chunks):
    # Initialize the embedding model
    model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf)

    # Calculate chunk IDs
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Get existing items
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Find new chunks
    new_chunks = []
    new_ids = []
    for i, chunk in enumerate(chunks):
        if chunk_ids[i] not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_ids[i])

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_texts(texts=new_chunks, ids=new_ids)
    else:
        print("✅ No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()