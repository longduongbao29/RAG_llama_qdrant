from pydantic import BaseModel, Field
from enum import Enum


class ModeEnum(str, Enum):
    default = "default"
    multi_query = "multi-query"
    rag_fusion = "rag-fusion"
    recursive_decomposition = "recursive-decompostion"
    individual_decomposition = "individual-decomposition"
    step_back = "step-back"
    hyde = "hyde"


class Question(BaseModel):
    question: str = Field(examples=["What is your name?"])


class RetrieverSchema(BaseModel):
    mode: ModeEnum


class AskRequest(BaseModel):
    question: Question
    retrieval_schema: RetrieverSchema
