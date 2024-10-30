from typing import List
import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from rag.extract_documents.text_reader import TextReader
from langchain_text_splitters import CharacterTextSplitter
from rag.config.config import Config
from logs.logging import logger

config = Config()


class Qdrant_Client:
    """Qdrant client for vector databse"""

    vectorstores: List[QdrantVectorStore] = []

    def __init__(self, embeddings) -> None:

        self.url = config.qdrant_url
        self.api_key = config.qdrant_key
        self.embeddings = embeddings
        self.client = qdrant_client.QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        self.text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.vectorstores = []
        self.get_vectorstores()

    def get_vectorstores(self):
        self.vectorstores = []
        try:
            collections = self.client.get_collections().collections
            print(collections)
            for collection in collections:
                vtst = QdrantVectorStore(
                    client=self.client,
                    collection_name=collection.name,
                    embedding=self.embeddings,
                )
                self.vectorstores.append(vtst)
        except Exception as e:
            print("Exception: ", e)
    def create_collection(self, colection_name):
        """Create Qdrant collection"""
        new_vtstr = QdrantVectorStore
        try:
            self.client.create_collection(
                collection_name=colection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            new_vtstr = QdrantVectorStore(
                client=self.client,
                collection_name=colection_name,
                embedding=self.embeddings,
            )
        except Exception as e:
            logger.info(f"Failed to create collection: {str(e)}")
            logger.info("Init from existing collection")
            new_vtstr = QdrantVectorStore.from_existing_collection(
                url=self.url,
                api_key=self.api_key,
                embedding=self.embeddings,
                collection_name=colection_name,
            )
        return new_vtstr

    def retriever(self, text: str, k=5):
        """Get k relevant documents to given input text

        Args:
            text (_type_): _description_
            k (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        if len(text) == 0:
            return []
        docs_with_scores = []
        for vt in self.vectorstores:
            docs_with_scores.extend(vt.similarity_search_with_score(query=text, k=k))
        sorted_docs = [
            doc
            for doc, score in sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        ]
        return sorted_docs[:k]

    def upload_from_text(self, text: TextReader, topic: str):
        """From input text, chunking and saving to Qdrant collection

        Args:
            text (str): input text
            title (str): title of document
        """
        docs = text.create_documents()
        vtstr = self.create_collection(topic)
        vtstr.add_documents(docs)

    def retriever_map(self, queries: list[str]) -> list[list]:
        """From input queries, get relevant documents

        Args:
            queries (list[str]): input queries
        Returns:
            list[list]: relevant documents for each query
        """
        docs = []
        for query in queries:
            response = self.retriever(query)
            docs.append(response)
        return docs
