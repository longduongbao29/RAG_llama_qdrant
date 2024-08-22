import rag.retriever.templates as templates
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from init import vars
from langchain_core.retrievers import BaseRetriever, Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from typing import List
from logs.loging import logger
from rag.schemas.schemas import ModeEnum


class Retriever(BaseRetriever):
    model: BaseLanguageModel = None
    generate_prompt: ChatPromptTemplate = templates.default_prompt
    query_generate_prompt: ChatPromptTemplate = None
    docs: List[Document] = None
    k: int = 5

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _get_relevant_documents(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of documents related to the question.

        """
        self.docs = vars.qdrant_client.retriever(text=question)
        return self.docs[: self.k]

    def remove_duplicates(self, documents):
        """
        Remove duplicate documents from the list.
        Args:
        documents (List[Document]): The list of documents to remove duplicates from.
        Returns:
        List[Document]: A list of unique documents.
        """
        seen = set()
        unique_documents = []

        for doc in documents:
            id = doc.metadata["_id"]  # Hoặc doc.metadata nếu cần
            if id not in seen:
                seen.add(id)
                unique_documents.append(doc)

        return unique_documents

    def get_page_contents(self, docs):
        """
        Get the page contents of the documents.
        Args:
        docs (List[Document]): The list of documents.
        Returns:
        List[str]: The page contents of the documents.
        """
        page_contents = [doc.page_content for doc in docs]
        return page_contents

    def get_context(self, page_contents):
        """Merge page contents

        Args:
            page_contents (list[str]): page contents from documents.

        Returns:
            str: context
        """
        context = "\n".join(page_contents)
        return context

    def get_input_vars(self, question: str):
        """
        Generate input vars for the prompt.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        dict: input variables for the prompt.
        """
        docs = self._get_relevant_documents(question)
        page_contents = self.get_page_contents(docs)
        context = self.get_context(page_contents)
        return {"question": question, "context": context}

    def flatten_docs(self, docs):
        """Flatten documents' array from retrieved documents

        Args:
            docs (list[list[Document]]): retrieved documents

        Returns:
            list[Document]: flattened documents'array
        """
        flatten = []
        for ds in docs:
            for doc in ds:
                flatten.append(doc)
        return flatten

    def generate_queries(self, question: str, k=5) -> list[str]:
        """
        Generate k queries for the given question
        """
        chain = (
            self.query_generate_prompt
            | self.model
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        queries = chain.invoke({"question": question, "k": k})
        logger.output({"queries": queries[-k:]})
        return queries[-k:]


class MultiQuery(Retriever):

    def __init__(self, model) -> None:
        super().__init__(model)
        self.query_generate_prompt = templates.multiquery_prompt
        self.generate_prompt = templates.default_prompt

    def _get_relevant_documents(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of documents related to the question.
        """
        queries = self.generate_queries(question)
        docs = vars.qdrant_client.retriever_map(queries)
        docs = self.flatten_docs(docs)
        self.docs = self.remove_duplicates(docs)
        return self.docs[: self.k]


class RAGFusion(Retriever):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.query_generate_prompt = templates.rag_fusion_prompt
        self.generate_prompt = templates.default_prompt

    def _get_relevant_documents(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client with rerank documents.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of rerank documents related to the question.
        """
        queries = self.generate_queries(question)
        docs = vars.qdrant_client.retriever_map(queries)
        self.docs = [doc for doc, score in reciprocal_rank_fusion(docs)]
        return self.docs[: self.k]

    def get_input_vars(self, question: str):
        """
        Generate input vars for the prompt.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        dict: input variables for the prompt.
        """
        docs = self._get_relevant_documents(question)
        docs = [doc[0] for doc in docs]
        page_contents = self.get_page_contents(docs)
        context = self.get_context(page_contents)
        return {"question": question, "context": context}


class QueryDecompostion(Retriever):
    decomposition_mode: str = "recursive"

    def __init__(self, model, mode) -> None:
        super().__init__(model)
        self.decomposition_mode = mode
        self.query_generate_prompt = templates.decomposition_prompt
        if mode == "recursive":
            self.generate_prompt = templates.recursive_decomposition_prompt
        else:
            self.generate_prompt = templates.individual_decomposition_prompt

    def format_qa_pairs(self, question, answer):
        """Format Q and A pair"""

        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        logger.output({"Q&A": formatted_string.strip()})
        return formatted_string.strip()

    def retrieve_and_rag(self, question, prompt_rag, sub_question_generator_chain):
        """RAG on each sub-question"""

        # Use our decomposition /
        sub_questions = sub_question_generator_chain(question)

        # Initialize a list to hold RAG chain results
        rag_results = []

        for sub_question in sub_questions:

            # Retrieve documents for each sub-question
            retrieved_docs = self._get_relevant_documents(sub_question)

            # Use retrieved documents and sub-question in RAG chain
            answer = (prompt_rag | self.model | StrOutputParser()).invoke(
                {"context": retrieved_docs, "question": sub_question}
            )
            rag_results.append(answer)

        return rag_results, sub_questions

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain


class StepBack(Retriever):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.query_generate_prompt = templates.step_back_prompt
        self.generate_prompt = templates.generate_step_back_prompt

    def get_input_vars(self, question: str):
        """
        Generate input variables for the prompt based on the given question.

        Parameters:
        question (str): The question to retrieve related documents and generate context.

        Returns:
        dict: A dictionary containing the input variables for the prompt. The dictionary includes:
            - "normal_context": The context derived from the most relevant documents retrieved for the question.
            - "step_back_context": The context derived from the documents retrieved for the generated queries.
            - "question": The original question.
        """
        normal_context = self.get_context(
            self.get_page_contents(self._get_relevant_documents(question))
        )
        queries = self.generate_queries(question)
        step_back_docs = vars.qdrant_client.retriever_map(queries)
        step_back_docs = self.flatten_docs(step_back_docs)
        docs_content = self.get_page_contents(step_back_docs)
        step_back_context = self.get_context(docs_content)
        logger.output(
            {"normal context": normal_context, "step_back_context": step_back_context}
        )
        input_vars = {
            "normal_context": normal_context,
            "step_back_context": step_back_context,
            "question": question,
        }
        return input_vars


class HyDE(Retriever):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.query_generate_prompt = templates.prompt_hyde
        self.generate_prompt = templates.default_prompt

    def generate_docs(self, question: str) -> list[str]:
        chain = self.query_generate_prompt | self.model | StrOutputParser()
        docs = chain.invoke(question)
        return docs

    def _get_relevant_documents(self, question):
        docs_for_retrieval = self.generate_docs(question)
        logger.output({"docs Hyde": docs_for_retrieval})
        self.docs = super()._get_relevant_documents(docs_for_retrieval)
        return self.docs[: self.k]


class Bm25(Retriever):
    def _get_relevant_documents(self, question):
        from langchain_community.retrievers import BM25Retriever

        retriever = BM25Retriever.from_documents(self.get_documents())
        self.docs = retriever.invoke(question)
        return self.docs

    def get_documents(self):
        client = vars.qdrant_client.client
        collections = client.get_collections().collections
        docs = []
        for collection in collections:
            collection_name = collection.name
            page_size = 100
            offset = 0
            while True:
                response = client.scroll(
                    collection_name=collection_name,
                    limit=page_size,
                    offset=offset,
                )
                for r in response[0]:
                    data = r.payload
                    doc = Document(
                        metadata=data["metadata"], page_content=data["page_content"]
                    )
                    docs.append(doc)
                # Nếu số lượng documents trả về ít hơn page_size thì dừng lại
                if len(response[0]) < page_size:
                    break

                # Tăng offset cho lần lặp tiếp theo
                offset += page_size
        return docs


def reciprocal_rank_fusion(results, k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def get_retriever(mode: ModeEnum) -> Retriever:
    """Get retriever from mode"""
    retriever_ = Retriever(vars.retriever_llm)
    if mode == ModeEnum.multi_query:
        retriever_ = MultiQuery(vars.retriever_llm)
    elif mode == ModeEnum.rag_fusion:
        retriever_ = RAGFusion(vars.retriever_llm)
    elif mode == ModeEnum.recursive_decomposition:
        retriever_ = QueryDecompostion(vars.retriever_llm, mode="recursive")
    elif mode == ModeEnum.individual_decomposition:
        retriever_ = QueryDecompostion(vars.retriever_llm, mode="individual")
    elif mode == ModeEnum.step_back:
        retriever_ = StepBack(vars.retriever_llm)
    elif mode == ModeEnum.hyde:
        retriever_ = HyDE(vars.retriever_llm)
    elif mode == ModeEnum.bm25:
        retriever_ = Bm25(vars.retriever_llm)
    return retriever_


def get_multiple_retriever(mode: List[ModeEnum]) -> List[Retriever]:
    """Get multiple retrievers from modes"""
    retrievers = []
    for mode in mode:
        retrievers.append(get_retriever(mode))
    return retrievers


class MultipleRetriever(Retriever):
    retriever_methods: List[Retriever] = []

    def __init__(self, model, retriever_methods) -> None:
        super().__init__(model)
        self.retriever_methods = retriever_methods

    def _get_relevant_documents(self, question):

        docs_ = []
        if len(self.retriever_methods) == 1:
            docs_.extend(self.retriever_methods[0]._get_relevant_documents(question))
        else:
            for retriever in self.retriever_methods:
                docs_.append(retriever._get_relevant_documents(question))
            docs_ = [doc for doc, score in reciprocal_rank_fusion(docs_)][:5]
        self.docs = docs_
        return self.docs
