## RAG Pipeline for confidential Edtech documents
#### Tools Used: LangChain, LangSmith, Chroma and FAISS Vector store, Ollama (llama3.2), Streamlit.



### * Notion of a Prompt :
Placeholder where we pass custom instructions/behavior for the LLM based on the context and question asked by the user.

Example of a normal prompt :
'''
template = 
Answer the question based on the provided content only 
Think Step by step before providing a detailed response: 
{context}
Question : {question}
'''


### * Getting started with Query Translation

It involves approaches like Multi-Query and RAG Fusion:
* Multi-Query involves breaking down the question into different sub-questions and the taking the union of the context obtained for those questions and finally passing it to the original question.
Example Prompt :
'''
You are an AI language model assistant. Your task is to generate five different versions of the given user question 
to retrieve relevant documents from a vector store/database. By generating multiple perspectives on the user questions, your goal
is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions seperated by newlines. Original Question : {question}
'''

Later Steps:
* Create a chain that includes the following:
 prompt_perspectives  
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n")) [Creates a list of questions]

 All this to get those distinct questions

* Later we create a function that would combine the responses to those questions and would be feed as an entire corpus to the original question with the retriever
 generate_questions | retriever.map() | get_unique_docs

 This corpus would serve as context from which the original question would be answered. 
