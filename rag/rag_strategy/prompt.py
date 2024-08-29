from langchain_core.prompts import ChatPromptTemplate

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

system = """You are a grader assessing whether an answer addresses / resolves an input sentence or question.\n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question / input sentence.
    
    Example:
    Human: Hello
    System: Hello, what's on your mind today?
    Return: 'yes'"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User input: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
     Return (1 question 3-5 words): """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to topics: {topics}
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Chat history: {chat_history}

Answer:"""

rag_prompt = ChatPromptTemplate.from_template(template=template)

first_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant with extensive knowledge.
    Answer user queries based on your understanding, chat history, and knowledge base.
    Please avoid answering questions related to the following topics: {topics}.
    Always be concise, respectful, and clear in your responses.
    If the question falls into the restricted topics, politely inform the user that you cannot respond.
    Here is the chat history for reference:
    {chat_history}
    Return the answer as follows:
    """,
        ),
        ("human", "{question}"),
    ]
)


drafter_prompt_template = """
Response to the instruction. Also provide rationale for your response.

Instruction: {instruction}

Evidence: 
{evidence}

Return a dictionary including both "Response" and "Rationale".

# EXAMPLE:
## Instruction: Which actress/singer starred as Doralee Rhodes in the 1980 film, "Nine to Five"?
## Evidence:
[1] Diana DeGarmo played the role of Doralee Rhodes in the national tour of "9 to 5", which was launched in Nashville on September 21, 2010. She ended her run as Doralee after the July 2011 Minneapolis tour stop.
[2] Pippa Winslow as Violet Newstead, Louise Olley as Doralee Rhodes and Leo Sene as Franklin Hart Jr, with Samantha Giffard as Roz, Matthew Chase as Joe and Mark Houston, Rachel Ivy, and Blair Anderson. "9 to 5" will play in the West End at the Savoy Theatre from January 29 to August 31, 2019. The production stars Amber Davies (Judy), ...

Return:
{{
    "Response": "Diana DeGarmo",
    "Rationale": "Diana DeGarmo played the role of Doralee Rhodes in the national tour of '9 to 5', which began in September 2010."
}}

# END OF EXAMPLE

Please return the result as a valid JSON dictionary, without adding any extra text or explanations. Only return the result in JSON format with the requested keys and values.
"""

drafter_prompt = ChatPromptTemplate.from_template(template=drafter_prompt_template)

verifier_prompt_template = """From responses and rationales below, choose which pair of response-rationale is the most suitable for the query:

Response-rationale pairs:
{pairs}

Question: {query}
Return:
"""

verifier_prompt = ChatPromptTemplate.from_template(verifier_prompt_template)
