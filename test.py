# from Rag.answer.answer import Generate
# from Rag.retriever.query_translation import (
#     MultiQuery,
#     RAGFusion,
#     QueryDecompostion,
#     StepBack,
#     HyDE,
# )
# from init import vars
# from Rag.agent.agent import Agent
# from logs.loging import logger

# multi_query = MultiQuery(vars.llm)
# generate = Generate(vars.llm, multi_query)
# print(generate._run("What is pokemon?"))

# rag_fusion = RAGFusion(vars.llm)
# generate = Generate(vars.llm, retriever= rag_fusion)
# print(generate._run("What is pokemon?"))


# decomposition = QueryDecompostion(vars.llm, mode="recursive")
# generate = Generate(vars.llm, decomposition)
# print(generate._run("What is pokemon?"))

# stepback = StepBack(model = vars.llm)
# generate = Generate(vars.llm, stepback)
# print(generate._run("What is pokemon?"))


# hyde = HyDE(vars.llm)
# generate = Generate(vars.llm, hyde)
# print(generate._run("What is pokemon?"))

# multi_query = MultiQuery(vars.llm)
# agent = Agent(llm=vars.llm, retriever=multi_query)
# print(agent.run({"input": "What is the weather tommorow in hanoi?"}))


# import fitz

# my_path = "data/test_rag.pdf"
# doc = fitz.open(my_path)
# blocks = []
# for page in doc:
#     output = page.get_text("blocks")
#     for block in output:
#         if block[6] == 0:
#             # We only take the text
#             text = block[4]
#             if "http://" not in text and "www" not in text and ".com" not in text:
#                 blocks.append(block)
#                 print(block)
#                 print("-" * 80)

# chunks = []
# prev_block = blocks[0]
# is_join = False
# for current_block in blocks[1:]:
#     if is_join:
#         prev_block = current_block
#         is_join = False
#         continue
#     if prev_block[0] > current_block[0]:
#         text = " ".join([prev_block[4], current_block[4]]).replace("\n", " ")
#         chunks.append(text)
#         is_join = True
#     else:
#         chunks.append(prev_block[4].replace("\n", " "))
#     prev_block = current_block

# for chunk in chunks:
#     print(chunk)
#     print("\n\n")

from rag.retriever.query_translation import Retriever
from self_rag.self_rag import SelfRag
from init import vars
self_rag = SelfRag(vars.retriever_llm, retriever= Retriever(vars.retriever_llm))
self_rag.build_graph()

print(self_rag.run({"question": "What is Pokemon?"}))