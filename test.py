
from langchain_groq import ChatGroq
from rag.config.config import Config
from rag.retriever.query_translation import Retriever
from rag.rag_strategy.adaptive_rag import AdaptiveRag 
from init import vars
config = Config()
llm = ChatGroq(
            api_key=config.groq_api_key,
            model="llama-3.1-70b-versatile",
            temperature=0.1,
        )
rag = AdaptiveRag(llm, retriever= Retriever(llm))
rag.build_graph()

print(rag.run({"question": "What is the weather today in Hanoi?"}))