import pandas as pd
import sys
import os
# Thêm đường dẫn của module 'logs'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.loging import logger
from init import vars
from langchain_core.documents import Document
# Download dataset
splits = {'train': 'covidqa/train-00000-of-00001.parquet', 'test': 'covidqa/test-00000-of-00001.parquet', 'validation': 'covidqa/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["train"])[["id","question","documents","response"]]

documents = ["\n".join(doc) for doc in df["documents"]]
question_answer = [(question, answer) for question, answer in zip(df["question"].to_list(), df["response"].to_list())]
# def create_documents(listdocs):
#     docs = []
#     for doc in listdocs:
#         spl = doc.split("\n")
#         title, content = spl[0], spl[1]
#         docs.append(Document(page_content= content, metadata = {"title":title}))
#     return docs
# #Upload to database
# client = vars.qdrant_client
# vector_store=client.create_collection(colection_name= "covidqa")
# vector_store.add_documents(create_documents(documents))

from rag.retriever.query_translation import Retriever
from rag.answer.answer import Generate
retriever = Retriever(vars.retriever_llm)
generate = Generate(vars.retriever_llm, retriever)

inputs = []
actual_outputs=[]
expected_outputs = []
retrieval_contexts = []
for question, answer in question_answer[:1]:  
    actual_response, docs =  generate.run(question)
    inputs.append(question)
    actual_outputs.append(actual_response)
    retrieval_contexts.append(docs)


