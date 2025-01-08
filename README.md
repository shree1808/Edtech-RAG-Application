## RAG Pipeline for confidential Edtech documents
#### Tools Used: LangChain, LangSmith, Chroma Vector store, Ollama (llama3.2), Streamlit.



### Notion of a Prompt :
Placeholder where we pass custom instructions/behavior for the LLM based on the context and question asked by the user.

### Getting started with Query Translation
It involves approaches like Multi-Query and RAG Fusion:

#### Multi-Query:
 It involves breaking down the question into different sub-questions and the taking the union of the context obtained for those questions and finally passing it to the original question.


#### RAG-Fusion
Like Multi-query we break down the input query into 'k' different questions, later with the help of retriever we get the context of those k queries. We then map them to the Reciprocal Ranked Fusion function which provides the fused scores for each document.

The top ranked documents are passed again to the llm alongside the original question. This process has less Abstraction.
