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


# # # # # # # # # # # # # # # # # # # # # 
# Initializing Retriever
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})

# # # # # # # # # # # # # # # # # # # # # 

# RAG Fusion
# K queries -> Retrieve k docs -> Rank docs (RRF) -> 

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

# # # # # # # # # # # # # # # # # # # # # 
question = 'What are the different Optmizers used in Deep Learning ?'

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

response = final_rag_chain.invoke({'question': question})

print(response)
