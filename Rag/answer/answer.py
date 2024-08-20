from venv import logger
from Rag.retriever.query_translation import (
    Retriever,
    MultiQuery,
    RAGFusion,
    QueryDecompostion,
    StepBack,
    HyDE,
)
from Rag.schemas.schemas import ModeEnum
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

    def _run(self, question: str):
        response = None
        if isinstance(self.retriever, QueryDecompostion):
            response = self.decomposition_generate(question)
        else:
            response = self.default_generate(question)
        return response

    def default_generate(self, question):
        """LLM generate for multi-query, rag-fusion, Stepback, HyDE"""
        input_vars = self.retriever.get_input_vars(question)
        answer = self.chain.invoke(input_vars)
        return answer

    def decomposition_generate(self, question):
        """Generate for Query Decomposition"""
        if self.retriever.decomposition_mode == "recursive":
            questions = self.retriever.generate_queries(question)
            answer = ""
            q_a_pairs = ""
            for q in questions:
                docs = self.retriever.invoke(question)
                page_contents = self.retriever.get_page_contents(docs)
                context = self.retriever.get_context(page_contents)
                answer = self.chain.invoke(
                    {"question": q, "q_a_pairs": q_a_pairs, "context": context}
                )
                q_a_pair = self.retriever.format_qa_pairs(q, answer)
                q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
            return answer
        else:
            prompt_rag = hub.pull("rlm/rag-prompt")
            answers, questions = self.retriever.retrieve_and_rag(
                question, prompt_rag, self.retriever.generate_queries
            )
            context = self.retriever.format_qa_pairs(questions, answers)
            final_answer = self.chain.invoke({"context": context, "question": question})
            return final_answer
