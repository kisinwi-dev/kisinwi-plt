import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class ConfigURL:
    DMS_URL = f"http://{os.getenv('DMS_DOMEN', 'localhost:6500')}"
    TASKER_URL = f"http://{os.getenv('TASKER_DOMEN', 'localhost:6110')}"
    TRAINER_URL = f"http://{os.getenv('TRAINER_DOMEN', 'localhost:6200')}"

@dataclass
class ConfigBaseLLM:
    OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME"))
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

config_url = ConfigURL()
config_base_llm = ConfigBaseLLM()