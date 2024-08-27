
from langchain_groq import ChatGroq
from rag.config.config import Config
from rag.retriever.query_translation import Bm25, Retriever
from rag.rag_strategy.adaptive_rag import AdaptiveRag 
from init import vars
config = Config()
llm = ChatGroq(
            api_key=config.groq_api_key,
            model="llama-3.1-70b-versatile",
            temperature=0.1,
        )
retriever= Bm25(llm)

print(retriever.get_relevant_documents("What is the weather today in Hanoi?"))