from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def chroma_vector_store(chroma_path):
    vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function= OllamaEmbeddings(model = 'llama3.2'),
    persist_directory="./rag_demo_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store


# def faiss_vector_store():
