'''
database_agent
1.PDF data , RecursiveCharacterTextSplitter, Turkish Embedding (kodun şu anki haliyle aktif)
1.PDF data , RecursiveCharacterTextSplitter, Ollama Embedding (kodun şu anki haliyle inaktif)

'''
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings #1.
# from langchain_community.embeddings.ollama import OllamaEmbeddings #2.  (ollama embedding için)

CHROMA_PATH = "chroma"
MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

def get_embedding_function(): #1
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# def get_embedding_function():   #2.      (ollama embedding için)
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#     return embeddings

def database_agent(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    return context_text

if __name__ == "__main__":
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score("BiGG Yatırım Programı Çağrıları kapsamında proje desteği üst limiti nedir? Ne kadarlık hisse karşılığında yatırım yapılacaktır?", k=4)

    # Combine context with scores.
    context_chunks = []
    for doc, score in results:
        chunk_with_score = f"Chunk:\n\n{doc.page_content}\nSimilarity Score: {score}" # treshold için if score < 250:
        context_chunks.append(chunk_with_score)

    context_text = "\n\n---\n\n".join(context_chunks)

    print(context_text)