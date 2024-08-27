from qdrant.client import Qdrant_Client
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama

from rag.config.config import Config
config = Config()


class InitVariable:

    def __init__(self):
        self.embedding = FastEmbedEmbeddings()
        self.qdrant_client = Qdrant_Client(embeddings=self.embedding)
        self.tool_use_llm = ChatGroq(
            api_key=config.groq_api_key,
            model="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.2,
        )
        self.retriever_llm = ChatGroq(
            api_key=config.groq_api_key,
            model="llama3-70b-8192",
            temperature=0.2,
        )
        # self.retriever_llm = Ollama(model="llama3.1:8b")


vars = InitVariable()

print("Initialized variables!")
