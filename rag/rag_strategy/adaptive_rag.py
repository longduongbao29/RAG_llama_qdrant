from rag.rag_strategy.rag import Rag, GraphState
from langgraph.graph import END, StateGraph, START


class AdaptiveRag(Rag):
    def build_graph(self) -> None:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("web_search", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        self.app = workflow.compile()
        
    
    def run(self, inputs:dict):
        from logs.logging import logger
        inputs["topics"] = self.get_retriever_topics()
        logger.output(inputs)
        return super().run(inputs)