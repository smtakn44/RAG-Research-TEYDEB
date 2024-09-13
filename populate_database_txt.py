'''
TXT data , RecursiveCharacterTextSplitter, Ollama Embedding

'''

import argparse
import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader  # txt dosyalarını yüklemek için
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Veritabanını sıfırlama kontrolü (reset bayrağı ile).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Veri deposunu oluşturma veya güncelleme.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents(data_path=DATA_PATH):
    documents = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):  # Sadece .txt dosyalarını yükleyelim
            try:
                loader = TextLoader(file_path, encoding="utf-8")  # UTF-8 ile dene
                documents.extend(loader.load())
            except UnicodeDecodeError:
                print(f"UTF-8 ile okunamadı: {file_path}, farklı bir kodlama deneniyor...")
                # Eğer UTF-8 hatası alırsak, windows-1254 kodlamasını deniyoruz.
                loader = TextLoader(file_path, encoding="windows-1254")
                documents.extend(loader.load())
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

def add_to_chroma(chunks: list[Document]):
    # Mevcut veritabanını yükle
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Parça ID'lerini hesapla
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Dokümanları ekle veya güncelle
    existing_items = db.get(include=[])  # Varsayılan olarak ID'ler her zaman dahil edilir
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Sadece veritabanında olmayan dokümanları ekle
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # Bu ID'ler "data/monopoly.txt:6:2" gibi oluşturulacak
    # Kaynak : Sayfa Numarası : Parça İndeksi

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Eğer sayfa ID'si öncekiyle aynıysa, indeksi artır
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Parça ID'sini hesapla
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Meta veriye ID'yi ekle
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
