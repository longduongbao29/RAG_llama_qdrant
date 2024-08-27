from pydantic import GenerateSchema
from eval.eval_model import CustomLlama3, Gemma2
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from eval.preprocess import write_to_csv
import ast
from deepeval.test_case import LLMTestCase
from logs.loging import logger
from rag.retriever.query_translation import (
    HyDE,
    MultiQuery,
    Retriever,
    RAGFusion,
    QueryDecompostion,
    StepBack,
)
from rag.answer.answer import Generate
from init import vars

model = Gemma2()

contextual_precision = ContextualPrecisionMetric(model=model, include_reason=False)
contextual_recall = ContextualRecallMetric(model=model, include_reason=False)
contextual_relevancy = ContextualRelevancyMetric(model=model, include_reason=False)
answer_relevancy = AnswerRelevancyMetric(model=model, include_reason=False)
faithfulness = FaithfulnessMetric(model=model, include_reason=False)

import csv

# Đường dẫn đến file CSV
file_path = "data/covidqa_hyde.csv"
retriever = HyDE(vars.retriever_llm)
generate = Generate(vars.retriever_llm, retriever)

write_to_csv(file_path, generate)
# Khởi tạo các mảng rỗng
inputs = []
actual_outputs = []
expected_outputs = []
retrieval_contexts = []

# Đọc dữ liệu từ file CSV
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    # Bỏ qua header
    next(reader)

    # Đọc từng dòng và đưa vào các mảng
    line_count = 0
    for row in reader:
        if line_count >= 100:
            break
        inputs.append(row[0])
        actual_outputs.append(row[1])
        expected_outputs.append(row[2])
        retrieval_context = ast.literal_eval(row[3])
        retrieval_contexts.append(retrieval_context)

        line_count += 1
for i in range(10):
    logger.output(
        f"\nQuestion : {inputs[i]}\nResponse : {actual_outputs[i]}\nExpected : {expected_outputs[i]}\nContext : {retrieval_contexts[i]}\n\n\n"
    )


def create_test_cases():
    test_cases = []
    for input, actual_output, expected_output, retrieval_context in zip(
        inputs, actual_outputs, expected_outputs, retrieval_contexts
    ):
        test_cases.append(
            LLMTestCase(
                input=input,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
            )
        )
    return test_cases


test_cases = create_test_cases()
print("\nTest cases:", len(test_cases))
evaluate(
    test_cases=test_cases,
    ignore_errors=True,
    print_results=False,
    metrics=[
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
        answer_relevancy,
        faithfulness,
    ],
)
