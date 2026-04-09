from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME"))
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_MODEL_IMAGE_NAME = str(os.getenv("OPENAI_MODEL_IMAGE_NAME"))

# объект llm
llm = LLM(
    model=OPENAI_MODEL_NAME,
    base_url=OPENAI_API_BASE,
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
)
