from fastapi.responses import HTMLResponse
from Rag.agent import agent
from Rag.schemas.schemas import RetrieverSchema, Question
from Rag.retriever.query_translation import (
    MultipleRetriever,
    get_retriever,
    get_multiple_retriever,
)
from Rag.agent.agent import Agent
from Rag.retriever.query_translation import Retriever
from init import vars
from fastapi import APIRouter
from fastapi import UploadFile, File
from Rag.extract_documents.text_reader import TextReader
from logs.loging import logger

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <h2>Hello! Welcome to the model serving api.</h2>
        Check the <a href="/docs">api specs</a>.
    """


@router.post("/retriever")
def retriever(question: Question, mode: RetrieverSchema):
    """
    This function retrieves relevant documents based on the given question and retrieval mode.

    Parameters:
    - question (Question): An object containing the question to be answered.
    - mode (RetrieverSchema): An object specifying the retrieval mode (e.g., "default", "multi-query", "rag-fusion", "recursive-decomposition","individual-decomposition", "step-back", "hyde").

    Returns:
    - List[str]: A list of document that are relevant to the given question.
    """
    try:
        question = question.question
        retriever = MultipleRetriever(
            model=vars.retriever_llm, retriever_methods=get_multiple_retriever(mode.mode)
        )

        docs = retriever.invoke(question)
    except Exception as e:
        return {"message": f"Failed to retrieve documents: {str(e)}"}
    return docs


@router.post("/ask")
async def model_predict(question: Question, retrieval_schema: RetrieverSchema, history):
    """
    This function generates an answer to a given question using an Agent including search_tool and retriever_tool.

    Parameters:
    - question (Question): An object containing the question to be answered. The object should have a 'question' attribute.
    - retrieval_schema (RetrieverSchema): An object specifying the retrieval mode (e.g., "default", "multi-query", "rag-fusion", "recursive-decomposition","individual-decomposition", "step-back", "hyde"). The object should have a 'mode' attribute.

    Returns:
    - str: The generated answer to the given question.
    """
    try:
        question = question.question
        retriever = MultipleRetriever(
            model=vars.retriever_llm,
            retriever_methods=get_multiple_retriever(retrieval_schema.mode),
        )
        agent = Agent(vars.tool_use_llm, retriever)
        agent.update_description_retriever_tool(vars.qdrant_client.client)
        answer = agent.run({"input": question, "chat_history": history})
        logger.output({"question": question, "answer": answer})
    except Exception as e:
        return {"message": f"Failed to generate answer: {str(e)}"}
    return answer


@router.post("/upload")
async def upload_to_database(file: UploadFile = File(...)):
    """
    This function handles file uploads and processes them to vector and save to vector database.

    Parameters:
    - file (UploadFile): The file to be uploaded. This parameter is optional and uses FastAPI's File(...) decorator for handling file uploads.

    Returns:
    - dict: A dictionary containing a success message.
    """
    try:
        contents = await file.read()
        file_path = f"data/{file.filename}"
        text_reader = TextReader(file_path=file_path, doc_name=file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        if ".txt" in file.filename:
            text_reader.text = contents.decode("utf-8")
        else:
            text_reader.readpdf()
        topic = text_reader.get_topics(vars.retriever_llm)
        vars.qdrant_client.upload_from_text(text_reader, topic)
    except Exception as e:
        return {"message": f"Failed to process file: {str(e)}"}
    return {"message": "File uploaded and processed successfully", "topic": topic}
