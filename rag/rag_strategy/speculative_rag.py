from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from torch import embedding
from qdrant.client import Qdrant_Client
from init import vars
from sklearn.cluster._kmeans import KMeans
from langchain_core.language_models import BaseLanguageModel
from rag.rag_strategy.prompt import drafter_prompt
import random
from langchain_core.output_parsers import StrOutputParser


class SpeculativeRag:
    def __init__(
        self,
        C: Embeddings,
        drafter_llm: BaseLanguageModel,
        verifier_llm: BaseLanguageModel,
        m: int = 3,
        k: int = 5,
    ):
        self.drafter_llm = drafter_llm
        self.verifier_llm = verifier_llm
        self.m = m
        self.k = k
        self.C = C
        # self.D = []
        self.client = vars.qdrant_client
        # self.clusters = []

    def retriever(self, Q):
        D = []
        collections = self.client.client.get_collections().collections
        query_dense_embedding = self.C.embed_query(Q)
        for collection in collections:
            results = self.client.client.query_points(
                query=query_dense_embedding,
                collection_name=collection.name,
                with_vectors=True,
            ).points
            D.extend(results)

        return D

    def cluster_docs(self, query):
        D = self.retriever(query)
        vectors = [d.vector for d in D]

        kmeans = KMeans(n_clusters=self.k).fit(vectors)

        clts = kmeans.labels_
        num_cluster = max(clts) + 1
        clusters = [[] for _ in range(num_cluster)]
        for i, doc in enumerate(D):
            #  print(f"Tài liệu: {doc.payload["page_content"]} - Cluster: {clts[i]}")
            clusters[clts[i]].append(doc.payload["page_content"])
        return clusters

    def get_subset(self, query):
        clusters = self.cluster_docs(query=query)
        subsets = []
        while len(subsets) < self.m:
            subset = set()
            for c in clusters:
                random_value = random.choice(c)
                subset.add(random_value)
            subsets.append(subset)

        return subsets

    def drafter_generate(self, query, subset):
        chain = drafter_prompt | self.drafter_llm | StrOutputParser()
        return chain.invoke({"instruction": query, "evidence": subset})
