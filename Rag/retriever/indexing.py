from langchain.prompts import ChatPromptTemplate
from langchain.storage import InMemoryByteStore
from langchain_core.output_parsers import StrOutputParser


class MultiRepresentationIndexing():
    def __init__(self, llm) -> None:
        self.llm = llm
        self.store = InMemoryByteStore()
    def generate_summary(self, docs):
        chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | self.llm
        | StrOutputParser()
    )
        summary = chain.batch(docs, {"max_concurrency": 5})
        return summary
