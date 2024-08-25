from qdrant.client import Qdrant_Client
from langchain_groq import ChatGroq
from rag.config.config import Config
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

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


vars = InitVariable()

print("Initialized variables!")
