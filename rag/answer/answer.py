from venv import logger
from rag.retriever.query_translation import (
    Retriever,
    MultiQuery,
    RAGFusion,
    QueryDecompostion,
    StepBack,
    HyDE,
)
from rag.schemas.schemas import ModeEnum
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from init import vars
from langchain_core.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from typing import Dict


class Generate:
    retriever: Retriever = None
    llm: BaseLanguageModel = None
    prompt: ChatPromptTemplate = None
    chain: RunnableSerializable[Dict, str] = None

    def __init__(self, llm, retriever):
        self.retriever = retriever
        self.llm = llm
        self.prompt = retriever.generate_prompt
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, question: str):
        if isinstance(self.retriever, QueryDecompostion):
            response, docs = self.decomposition_generate(question)
        else:
            response, docs = self.default_generate(question)
        return response, docs

    def default_generate(self, question):
        """LLM generate for multi-query, rag-fusion, Stepback, HyDE"""
        docs = self.get_context(self.retriever.invoke(input=question))
        answer = self.chain.invoke({"question": question, "context": docs})
        return answer, docs

    def get_context(self, docs):
        contexts = []
        for doc in docs:
            contexts.append("\n".join([doc.metadata["title"], doc.page_content]))
        return contexts

    def decomposition_generate(self, question):
        """Generate for Query Decomposition"""
        if self.retriever.decomposition_mode == "recursive":
            questions = self.retriever.generate_queries(question)
            answer = ""
            q_a_pairs = ""
            docs = []
            for q in questions:
                context = self.get_context(self.retriever.invoke(input=q))

                answer = self.chain.invoke(
                    {"question": q, "q_a_pairs": q_a_pairs, "context": context}
                )
                q_a_pair = self.retriever.format_qa_pairs(q, answer)
                q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
                docs = context.append(q_a_pairs)
            return answer, docs
        else:
            prompt_rag = hub.pull("rlm/rag-prompt")

            answers, questions, docs = self.retriever.retrieve_and_rag(
                question, prompt_rag, self.retriever.generate_queries
            )
            context = self.retriever.format_qa_pairs(questions, answers)
            final_answer = self.chain.invoke({"context": context, "question": question})
            return final_answer, self.get_context(docs)
