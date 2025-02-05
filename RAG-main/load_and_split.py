# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_content(root_path):
    loader = PyPDFLoader(root_path)
    data = loader.load()
    return data

def split_content(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 800, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

