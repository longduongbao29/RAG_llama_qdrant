# from matplotlib.pyplot import draw_if_interactive
from rag.rag_strategy.speculative_rag import SpeculativeRag
from init import vars
import asyncio
async def main(): 
    rag = SpeculativeRag(
        C=vars.embedding, drafter_llm=vars.retriever_llm, verifier_llm=vars.retriever_llm
    )
    q = "What is theorized regarding transmission?"

    print(await rag.run(q))
asyncio.run(main())
