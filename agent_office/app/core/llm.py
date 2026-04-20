from app.config import config_base_llm
from crewai import LLM

# Объект llm
llm = LLM(
    model=config_base_llm.OPENAI_MODEL_NAME,
    base_url=config_base_llm.OPENAI_API_BASE,
    api_key=config_base_llm.OPENROUTER_API_KEY,
    temperature=0.7,
)
