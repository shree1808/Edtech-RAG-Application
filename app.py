# UI for RAG application 
import streamlit as st 
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# Streamlit page config
st.set_page_config(page_title = "🧠 RAG-powered Chatbot", layout = 'wide')

load_dotenv()

st.title("EdTech RAG Application")
st.subheader('Please upload your research pdf')

# # # # # # # # # # # # # # # 

# Function to extract text from PDFs using PyPDF2
def extract_text_from_pdf(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        documents.append(Document(page_content=text))
    return documents

# # # # # # # # # # # # # # # 

uploaded_files = st.file_uploader("Upload your pdf", type = ['pdf'], accept_multiple_files = True)
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose Local Ollama Model", ["llama3.2", "mistral"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
# For RAG Fusion
top_k = st.sidebar.slider("Top K documents", 1,5,10)

ollama_url = "http://localhost:11434/"
llm = OllamaLLM(model = model_name, 
                    base_url = ollama_url,
                    temperature = temperature,
                )

if uploaded_files:
    st.sidebar.info("Processing uploaded files...")
    documents = extract_text_from_pdf(uploaded_files)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    from uuid import uuid4
    uuids = [str(uuid4()) for _ in range(len(split_docs))]

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding= OllamaEmbeddings(model = 'llama3.2'), 
        collection_name="rag_collection",
        persist_directory="chroma-db",

    )

    retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )

    st.sidebar.success("PDF Indexed successfully completed!")


# # # # # # # # #
# RAG Fusion Implementation
# # # # # # # # #

# a) Generation of Perspective Questions
from langchain.prompts import ChatPromptTemplate 
template = """
You are a helpful assistant that generate multiple search queries based on a single search input query. \n 
generate multiple search queries related to: {question} \n
Output (4 queries) : 
"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)


# Reciprocal Rank fusion -> retrieves top documents for the input query
from langchain.load import loads, dumps

def reciprocal_rank_fusion(Documents : list[list], k = 5):
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

# RAG Fusion Chain combined with final response generation
def rag_fusion_chain_response(question, llm, retriever, top_k=5):
    # Perspective Chain
    generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
    )
    retrieved_documents = [retriever.get_relevant_documents(q, top_k=top_k) for q in generate_queries.invoke({'question': question})]

    fused_docs = reciprocal_rank_fusion(retrieved_documents, k=top_k)

    # Prepare context for the final prompt
    context = "\n".join([doc[0] for doc in fused_docs])

    final_prompt = ChatPromptTemplate.from_template(template)
    final_chain_input = {
        "context" : context,
        "question" : itemgetter("question")
    }
    # Generate the final response
    final_prompt_filled = final_prompt.format(**final_chain_input)
    final_response = llm(final_prompt_filled)

    return final_response


# Streamlit UI elements
st.title("🤖 RAG Fusion Chatbot")

# User query input
query = st.text_input("Ask a question based on your documents:")

if query:
    with st.spinner("Generating response..."):
        response = rag_fusion_chain_response(query, llm, retriever, top_k)  
        st.markdown(f"### 🤖 AI Response:\n{response}")

    # Optionally, show retrieved documents for transparency
    with st.expander("🔍 Retrieved Context"):
        # Example to display top K retrieved documents
        for i, doc in enumerate(response['context']):
            st.markdown(f"**Chunk {i+1}:** {doc[:500]}...")  # Truncate for display


st.sidebar.markdown("---")
st.sidebar.markdown("🚀 Built with Streamlit & LangChain")
# # # # # # # # #