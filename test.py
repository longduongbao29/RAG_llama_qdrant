from matplotlib.pyplot import draw_if_interactive
from rag.rag_strategy.speculative_rag import SpeculativeRag
from init import vars

rag = SpeculativeRag(
    C=vars.embedding, drafter_llm=vars.retriever_llm, verifier_llm=vars.retriever_llm
)
q = "What is MVO?"
subsets = rag.get_subset(q)
draft = rag.drafter_generate(q, subsets[0])

print(draft)