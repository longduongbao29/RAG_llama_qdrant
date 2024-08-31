import os
from dotenv import load_dotenv

load_dotenv()

class Config():
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")