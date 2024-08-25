import pandas as pd
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.loging import logger
from init import vars

# Download dataset
splits = {'train': 'covidqa/train-00000-of-00001.parquet', 'test': 'covidqa/test-00000-of-00001.parquet', 'validation': 'covidqa/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["train"])[["id","question","documents","response"]][:100]
question_answer = [(question, answer) for question, answer in zip(df["question"].to_list(), df["response"].to_list())]

# documents = ["\n".join(doc) for doc in df["documents"]]

# print("Number of documents:", len(documents))

# def create_documents(listdocs):
#     docs = []
#     for doc in listdocs:
#         spl = doc.split("\n")
#         title, content = spl[0], spl[1]
#         docs.append(Document(page_content= content, metadata = {"title":title}))
#     return docs
#Upload to database
# client = vars.qdrant_client
# vector_store=client.create_collection(colection_name= "covidqa")
# vector_store.add_documents(create_documents(documents))


# inputs = []
# actual_outputs=[]
# expected_outputs = []
# retrieval_contexts = []

num_samples = len(question_answer)


import csv

def write_to_csv(file_path, generate):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Ghi header
        writer.writerow(["input", "actual_output", "expected_output", "retrieval_context"])
        i = 0
        for question, answer in question_answer[:100]:  
            try: 
                actual_response, docs =  generate.run(question)
            except Exception as e:
                print("Error generating", str(e))
                actual_response = ""
                docs = ["Error generating"]
            # logger.output(f"\n{i+1}/{num_samples}\nQuestion: {question}\nResponse: {actual_response}\nExpected: {answer}\n" + "-"*80)
            # inputs.append(question)
            # expected_outputs.append(answer)
            # actual_outputs.append(actual_response)
            # retrieval_contexts.append(docs)
            print(f"Generation: {i+1}/{num_samples}", end="\r")
            i+=1
            writer.writerow([question, actual_response, answer, docs])

    print(f"Data saved to {file_path}")


