import os
from dotenv import load_dotenv


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



# Loading data from the pdfs'
from load_and_split import load_content, split_content
raw_chunks = load_content(DATA_PATH)
chunks = split_content(raw_chunks)



# Initializing Vector Store
from vector_store import chroma_vector_store
chroma_db = chroma_vector_store(CHROMA_PATH)



# Generating unique ids for individual chunks
from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(chunks))]
print(f'Number of Chunks : {len(chunks)}')



# # Retreival
# # retriever = chroma_db.as_retriever(search_type="mmr")
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})



from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# Passing Prompt for Query Translation -> Multi-Query for perspectives
perspective_template = """
You are an AI language model assistant. Your task is to generate five different versions of the given user question 
to retrieve relevant documents from a vector store/database. By generating multiple perspectives on the user questions, your goal
is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions seperated by newlines. Original Question : {question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(perspective_template)


# Generation of Perspective Questions
model_name = "llama3.2"
ollama_url = "http://localhost:11434/"

llm = OllamaLLM(model = model_name, 
                    base_url = ollama_url,
                    temperature = 0.2,
                )

# Chain for Perspective Questions
generate_questions = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)



# To get the union of the unique documents (context not questions) obtained.
from langchain.load import dumps, loads

def get_unique_docs(documents : list[list]):
    """ Union of all the documents """
    flattened_docs = [dumps(docs) for sublist in documents for docs in sublist]
    unique_docs = list(set(flattened_docs))
    load_unique_docs = [loads(ud) for ud in unique_docs]
    return load_unique_docs



# Retrieve Questions
question = "What are the different optimizers mentioned in the Documents ? Please explain each one of them"
retrieval_chain = generate_questions | retriever.map() | get_unique_docs
# docs = retrieval_chain.invoke({'question' : question})
# print(docs) , print(len(docs))



from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# prompt
template = """
Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


model_name = "llama3.2"
ollama_url = "http://localhost:11434/"

llm = OllamaLLM(model = model_name, 
                    base_url = ollama_url,
                    temperature = 0.2,
                )


# Chain
final_rag_chain = (
    {"context" : retrieval_chain,
    "question" : itemgetter('question')}
    | prompt 
    | llm 
    | StrOutputParser()
)

response = final_rag_chain.invoke({'question': question})

print(response)
