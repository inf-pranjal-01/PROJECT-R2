import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_SOLVER = os.getenv("API_KEY_SOLVER")
API_KEY_CRITIC = os.getenv("API_KEY_CRITIC")
API_KEY_JUDGE = os.getenv("API_KEY_JUDGE")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"

API_KEY_USER_STATE_TRACKER = os.getenv("API_KEY_USER_STATE_TRACKER")
MODEL_GROQ_REASONING = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"