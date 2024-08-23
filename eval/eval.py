from eval_model import CustomLlama3_70B
from preprocess import inputs, expected_outputs, actual_outputs, retrieval_contexts
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric
)

from deepeval.test_case import LLMTestCase

model = CustomLlama3_70B()

contextual_precision = ContextualPrecisionMetric(model=model)
contextual_recall = ContextualRecallMetric(model=model)
contextual_relevancy = ContextualRelevancyMetric(model=model)
answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()

def create_test_cases():
    test_cases = []
    for input, actual_output, expected_output, retrieval_context in zip(inputs, actual_outputs, expected_outputs, retrieval_contexts):
        test_cases.append(LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        ))
    return test_cases
test_cases = create_test_cases()
print("\nTest cases:" ,len(test_cases))
evaluate(
    test_cases=test_cases,
    ignore_errors=True,
    metrics=[contextual_precision, contextual_recall, contextual_relevancy, answer_relevancy, faithfulness]
)

