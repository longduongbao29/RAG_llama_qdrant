from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.output_parsers import StrOutputParser
from rag.rag_strategy.rag import Rag
from logs.loging import logger
system = """You are a helpful assistant. From human input, you make a decision to reply or retrieve information.
Give a binary score 'yes' or 'no'. Yes' means that you decide to reply."""
route_prompt = ChatPromptTemplate.from_messages(
    [   
        ("system", system),
        ("human", "{question}"),
        ("placeholder", "{chat_history}")
    ]
)
gen_prompt = ChatPromptTemplate.from_messages(
    [   
        ("system", "Yuu are a helpful assistant"),
        ("human", "{question}"),
        ("placeholder", "{chat_history}")
    ]
)

class RouteQuery(BaseModel):
    """Decision for relying or retrival"""

    reply:str = Field(
        description="Given a user input, make decision to reply or retrieve information, 'yes' or 'no",
    )

class ChatBotGen():
    def __init__(self, llm: BaseLanguageModel, strategy: Rag):  
        self.llm = llm
        self.gen_chain = gen_prompt | llm | StrOutputParser()
        self.router_chain = route_prompt | llm.with_structured_output(RouteQuery)
        self.strategy = strategy
    
    def run(self, inputs):
        decision = self.router_chain.invoke(inputs)
        logger.output({"decision": decision})
        if decision.reply == "yes":
            return self.gen_chain.invoke(inputs)
        else:
            return self.strategy.run(inputs)