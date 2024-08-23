from eval_model import CustomLlama3_8B, CustomMistral7B
# from preprocess import inputs, expected_outputs, actual_outputs


from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

model = CustomLlama3_8B()
print(model.generate("Write me a joke"))
# contextual_precision = ContextualPrecisionMetric(model=model)
# contextual_recall = ContextualRecallMetric(model=model)
# contextual_relevancy = ContextualRelevancyMetric(model=model)

# test_case = LLMTestCase(
#     input="I'm on an F-1 visa, gow long can I stay in the US after graduation?",
#     actual_output="You can stay up to 30 days after completing your degree.",
#     expected_output="You can stay up to 60 days after completing your degree.",
#     retrieval_context=[
#         """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
#         your degree, unless you have applied for and been approved to participate in OPT."""
#     ]
# )


# contextual_precision.measure(test_case)
# print("Score: ", contextual_precision.score)
# print("Reason: ", contextual_precision.reason)

# contextual_recall.measure(test_case)
# print("Score: ", contextual_recall.score)
# print("Reason: ", contextual_recall.reason)

# contextual_relevancy.measure(test_case)
# print("Score: ", contextual_relevancy.score)
# print("Reason: ", contextual_relevancy.reason)