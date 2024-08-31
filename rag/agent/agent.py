from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from logs.logging import logger
from init import vars

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For casual questions such as greetings or small talk, respond normally without using any tools. 
            For queries that require information retrieval from a database, use the `retrieve_from_database` tool to fetch relevant documents.
            If the query requires searching the internet, use the `tavily_search_results_json` tool to obtain results.
            Always determine the nature of the query before deciding whether to use a tool or respond directly.""",
        ),
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

    def update_description_retriever_tool(self):
        client =  vars.qdrant_client.client
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        self.retriever_tool.description = f"""Useful for when you need to answer questions about following topics: {", ".join(collection_names)}
For any questions about topics above, you must use this tool!"""
        # logger.output({"description": self.retriever_tool.description})

    def run(self, input):
        """
        Executes the agent with the given input and returns the result.

        Parameters:
        input (str): The user's question or instruction. The agent will use this input to determine the appropriate tool or action.

        Returns:
        str: The result of executing the agent. This could be the output of a tool, a final answer, or an error message.
        """
        self.update_description_retriever_tool()
        return self.agent_executor.invoke(input)
