from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from Rag.answer.answer import Generate

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. First, try to answer the simple question base on chat history.
            Chat history: {chat_history}
            If you don't have any information, use search tool or retrieval tool and chat history to answer the given question. 
            If you don't know the answer, just say that you don't know.
            """,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class Agent:
    def __init__(self, llm, retriever) -> None:
        self.search_tool = TavilySearchResults()
        self.llm = llm
        self.retriever = retriever
        self.retriever_tool = Generate(llm, retriever)
        self.tools = [self.search_tool, self.retriever_tool]
        # self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.agent = create_tool_calling_agent(self.llm, self.tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=False
        )

    def run(self, input):
        """
        Executes the agent with the given input and returns the result.

        Parameters:
        input (str): The user's question or instruction. The agent will use this input to determine the appropriate tool or action.

        Returns:
        str: The result of executing the agent. This could be the output of a tool, a final answer, or an error message.
        """
        return self.agent_executor.invoke(input)
