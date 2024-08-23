from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

default_template = """From following context, answer the question:

{context}

Question: {question}
Return (1-5 sentences):
"""
default_prompt = ChatPromptTemplate.from_template(default_template)

# MULTI-QUERY
multiquery_template = """You are an AI language model assistant. Your task is to generate {k} 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}
Output ({k} questions):
1. question#1
2. question#2
3. question#3
...
Important: Just return questions in the format, do not explain!
"""

multiquery_prompt = ChatPromptTemplate.from_template(multiquery_template)

# RAG-FUSION
rag_fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output ({k} queries):
1. query#1
2. query#2
3. query#3
...
Important: Just return queries in the format, do not explain!
"""
rag_fusion_prompt = ChatPromptTemplate.from_template(rag_fusion_template)

# DECOMPOSITION
decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output ({k} sub-questions): 
1. sub-question#1
2. sub-question#2
3. sub-question#3
Important: Just return questions in the format, do not explain!
"""
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)

recursive_decomposition_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
Return (1-5 sentences):
"""

recursive_decomposition_prompt = ChatPromptTemplate.from_template(
    recursive_decomposition_template
)

individual_decomposition_template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer (1-5 sentences) to the question: {question}
"""
individual_decomposition_prompt = ChatPromptTemplate.from_template(
    individual_decomposition_template
)

# STEP BACK
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
step_back_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

generate_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Return (1-5 sentences):"""
generate_step_back_prompt = ChatPromptTemplate.from_template(generate_prompt_template)

# HyDE
hyde_template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(hyde_template)
