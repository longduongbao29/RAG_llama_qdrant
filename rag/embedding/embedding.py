from typing import List

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings


class LlamaCppEmbeddings_(LlamaCppEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text)[0] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)[0]
        return list(map(float, embedding))

