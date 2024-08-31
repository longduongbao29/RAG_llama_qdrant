from rag.rag_strategy.adaptive_rag import AdaptiveRag
from rag.rag_strategy.c_rag import CRag
from rag.rag_strategy.self_rag import SelfRag
from rag.rag_strategy.rag import Rag
from rag.schemas.schemas import StrategyEnum
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
def get_strategy(strategy: StrategyEnum, llm: BaseLanguageModel, retriever: BaseRetriever) -> Rag:
    if strategy==StrategyEnum.default: 
        return Rag(llm,retriever)
    elif strategy == StrategyEnum.self_rag:
        return SelfRag(llm, retriever)
    elif strategy == StrategyEnum.c_rag:
        return CRag(llm, retriever)
    elif strategy == StrategyEnum.adaptive_rag:
        return AdaptiveRag(llm, retriever)