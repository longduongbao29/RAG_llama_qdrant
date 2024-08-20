from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from logs.loging import logger

memory = MemorySaver()
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class Agent:
    def __init__(self, llm, retriever) -> None:
        self.search_tool = TavilySearchResults()
        self.llm = llm
        self.retriever = retriever
        self.retriever_tool = create_retriever_tool(
            retriever,
            "search_from_database",
            "",
        )
        self.tools = [self.retriever_tool, self.search_tool]
        self.agent = create_tool_calling_agent(self.llm, self.tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def update_description_retriever_tool(self, client):
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        self.retriever_tool.description = f"""
        A search tool optimized for comprehensive, accurate, and trusted results. 
        Useful for when you need to answer questions about following topics: {", ".join(collection_names)}
        For any questions about topics above, you must use this tool!
        """
        logger.output({"description": collection_names})

    def run(self, input):
        """
        Executes the agent with the given input and returns the result.

        Parameters:
        input (str): The user's question or instruction. The agent will use this input to determine the appropriate tool or action.

        Returns:
        str: The result of executing the agent. This could be the output of a tool, a final answer, or an error message.
        """
        return self.agent_executor.invoke(input)
