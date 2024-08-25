
from deepeval.models import DeepEvalBaseLLM
from groq import Groq
class CustomLlama3(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq()
        self.completion = self.client.chat.completions

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.completion.create(
            model="gemma2-9b-it",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=0.5,
            stream=True,
            stop=None,
        )
        answer = ''
        for chunk in response:
            answer+= chunk.choices[0].delta.content or ""
        return answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "gemma2-9b-it"
