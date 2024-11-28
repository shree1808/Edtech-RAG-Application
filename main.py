import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'rag-pdf-demo-ollama'

load_dotenv()
# Access the environment variables to verify
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_project = os.getenv('LANGCHAIN_PROJECT')

# Data Load 
DATA_PATH = 'data-pdf\optimizers.pdf'
CHROMA_PATH = 'chroma'

# C:\Users\Shree123\rag-demo\data-pdf\

from load_and_split import load_content, split_content
raw_chunks = load_content(DATA_PATH)

chunks = split_content(raw_chunks)

from vector_store import chroma_vector_store
chroma_db = chroma_vector_store(CHROMA_PATH)


# Generating unique ids for individual chunks
from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(chunks))]
print(f'Number of Chunks : {len(chunks)}')


# Retreival

# retriever = chroma_db.as_retriever(search_type="mmr")
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})

input_query = "Describe in detail the process of SGD"

# docs = retriever.invoke(input_query)
# print(len(docs))

# for res in docs:
#     print(f'Content -> {res.page_content} , source -> {res.metadata}')

# # # # # # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # #

# Generation
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# prompt
template = """
Answer the question based on the provided content only 
Think Step by step before providing a detailed response: 
{context}

Question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)


model_name = "llama3.2"
ollama_url = "http://localhost:11434/"

llm = OllamaLLM(model = model_name, 
                    base_url = ollama_url,
                    temperature = 0.2,
                )

from langchain_core.runnables import RunnablePassthrough
# Chain
retrieval_chain = (
    {"context" : retriever, "question" : RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

response = retrieval_chain.invoke(input_query)

print(response)
