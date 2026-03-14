import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_SOLVER = os.getenv("API_KEY_SOLVER")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"