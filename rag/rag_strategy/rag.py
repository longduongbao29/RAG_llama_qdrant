from abc import abstractmethod
from typing_extensions import TypedDict
from langchain_core.language_models.base import BaseLanguageModel
from rag.rag_strategy.prompt import first_gen_prompt,rag_prompt, grade_prompt, hallucination_prompt, answer_prompt, re_write_prompt, route_prompt
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.graph.state import CompiledStateGraph
from langchain_core.output_parsers import StrOutputParser
from logs.loging import logger
from langchain_community.tools.tavily_search import TavilySearchResults
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    topics: str
    web_search_: str
    chat_history: List

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class Rag():
    app: CompiledStateGraph= None
    web_search_tool = TavilySearchResults()
    def __init__(self, llm: BaseLanguageModel, retriever_: BaseRetriever):
        self.llm = llm
        self.retriever_ = retriever_
        self.first_generate_chain  = first_gen_prompt | llm | StrOutputParser()
        self.rag_chain = rag_prompt | llm | StrOutputParser()
        self.retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)
        self.answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()
        self.hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
        self.question_router = route_prompt | llm.with_structured_output(RouteQuery)
    def retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logger.output("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = []
        documents.extend(self.retriever_._get_relevant_documents(question))
        return {"documents": documents, "question": question}
    def route_question(self,state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        logger.output("---ROUTE QUESTION---")
        question = state["question"]
        topics = state["topics"]
        source = self.question_router.invoke({"question": question, "topics": topics})
        if source.datasource == "web_search":
            logger.output("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            logger.output("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # logger.output("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state["chat_history"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question, "chat_history":chat_history})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        logger.output("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                logger.output("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.output("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        logger.output("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        logger.output("---WEB SEARCH---")
        question = state["question"]
        if not state["documents"]:
            state["documents"] = []
        documents = state["documents"]
        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join(d["content"] for d in docs)
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        
        return {"documents": documents, "question": question}
    
    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        logger.output("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.output(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            logger.output("---DECISION: GENERATE---")
            return "generate"

    def first_generate(self, inputs):
        answer = self.first_generate_chain.invoke(inputs)
        score = self.answer_grader.invoke({"question": inputs["question"], "generation": answer})
        grade = score.binary_score
        logger.output({"grade": grade, "answer":answer})
        return grade, answer
        
    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        logger.output("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            logger.output("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            logger.output("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                logger.output("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                logger.output("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            logger.output("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    def get_retriever_topics(self):
        from init import vars
        collections = vars.qdrant_client.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        return ", ".join(collection_names)
    @abstractmethod
    def build_graph(self) -> None:
        """Build graph 
        """
    def run(self, inputs:dict):
        inputs["topics"] = self.get_retriever_topics()
        grade, answer = self.first_generate(inputs)
        if grade == "yes":
            return answer
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                logger.output(f"Node '{key}':")
                # Optional: logger.output full state at each node
                # plogger.output.plogger.output(value["keys"], indent=2, width=80, depth=None)
            logger.output("\n---\n")

            # Final generation
        return value["generation"]