import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from Rag.extract_documents.text_reader import TextReader
from langchain_text_splitters import CharacterTextSplitter
from Rag.config.config import Config


config = Config()


class Qdrant_Client:
    """Qdrant client for vector databse"""

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
        try:
            collections = self.client.get_collections().collections
            for collection in collections:
                vtst = QdrantVectorStore(
                    client=self.client,
                    collection_name=collection.name,
                    embedding=self.embeddings,
                )
                self.vectorstores.append(vtst)
            print(f"Initialized with {len(self.vectorstores)} collections")
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
            self.vectorstores.append(new_vtstr)
        except Exception as e:
            print(f"Failed to create collection: {str(e)}")
            print("Init from existing collection")
            new_vtstr = QdrantVectorStore.from_existing_collection(
                url=self.url,
                api_key=self.api_key,
                embedding=self.embeddings,
                collection_name=colection_name,
            )
            self.vectorstores.append()

        return new_vtstr

    def retriever(self, text: str, k=3):
        """Get k relevant documents to given input text

        Args:
            text (_type_): _description_
            k (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        if len(text) == 0:
            return []
        docs = []
        for vt in self.vectorstores:
            docs_ = vt.similarity_search(query=text, k=k)
            for doc in docs_:
                docs.append(doc)
        return docs

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
