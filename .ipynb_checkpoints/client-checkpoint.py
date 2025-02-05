# UI for RAG application 
import streamlit as st 
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv


# logging for debugging
import logging 
logging.basicConfig(
    filename='pdf_loader.log',  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filemode='w'  
)

load_dotenv()

st.title("EdTech RAG Application")
st.subheader('Please upload your research pdf')


# # # # # # # # # # # # # # # 
# Functions for loading pdf content and splits
def load_content(root_path):
    loader = PyPDFLoader(root_path)
    data = loader.load()
    return data

def split_content(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 800, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

# File Upload
uploaded_file = st.file_uploader("Upload your pdf", type = ('pdf', 'text', 'md'))


from PyPDF2 import PdfReader

if uploaded_file:
    reader = PdfReader(uploaded_file)  # Use the in-memory file directly
    text = ""
    for page in reader.pages:
        text += page.extract_text()    

    st.info("PDF Ingestion successfully completed!")


# Initializing Vector Store

def create_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_text(text)
    return chunks 

# # # # # # # # #

pdf_chunks = create_chunks(text)
logging.info('Chunks Created Successfully')

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(pdf_chunks))]

# # # # # # # # #


def chroma_vector_store(persistent_directory = 'chroma-db'):
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function = OllamaEmbeddings(model = 'llama3.2'),
        persist_directory= persistent_directory,
        )
    return vector_store


# # # # # # # # #
vector_store = chroma_vector_store()
vector_store.from_documents(documents = pdf_chunks, ids = uuids)
logging.info('Elements added to the vector store')


retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)

# # # # # # # # #
# RAG Fusion Implementation
# # # # # # # # #

from langchain.prompts import ChatPromptTemplate 

template = """
You are a helpful assistant that generate multiple search queries based on a single search input query. \n 
generate multiple search queries related to: {question} \n
Output (4 queries) : 
"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# Generation of Perspective Questions
model_name = "llama3.2"
ollama_url = "http://localhost:11434/"

llm = OllamaLLM(model = model_name, 
                    base_url = ollama_url,
                    temperature = 0.2,
                )

# Perspective Chain
generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# Reciprocal Rank fusion -> retrieves top documents for the input query
from langchain.load import loads, dumps

def reciprocal_rank_fusion(Documents : list[list], k = 60):
    fused_scores = {}

    for docs in Documents:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                # Assign Initial Rank -> 0 
                fused_scores[doc_str] = 0
            # Else if already exists apply the rank 
            fused_scores[doc_str] += 1 / (rank + k) 
    
    # Sort in descending order to 
    reranked_scores = [
       (loads(doc), scores) for doc, scores in sorted(fused_scores.items() , key = lambda x: x[1], reverse = True)  
    ]

    return fused_scores 


# # # # # # # # #

question = st.text_input(
    "Ask something related to your research paper!",
    placeholder = "Can you give me a short summary?"
)

rag_fusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

docs = rag_fusion_chain.invoke({'question': question})

print(len(docs))

# # # # # # # # # # # # # # # # # # # # # 
# Final RAG Chain 
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

template = """ 
Answer the following question based on the context provided:
{context}

Question: {question}
"""

final_prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {
        "context" : rag_fusion_chain,
        "question" : itemgetter("question")
    }
    | final_prompt
    | llm 
    | StrOutputParser()
)


# # # # # # # # #