import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    hf_token: str | None
    agent_endpoint: str
    gaia_config: str
    gaia_split: str
    request_timeout: int
    max_tasks: int


def load_settings() -> Settings:
    hf_token = os.getenv("HF_TOKEN", None)
    agent_endpoint = os.getenv("AGENT_ENDPOINT", "http://127.0.0.1:8090/invoke")
    gaia_config = os.getenv("GAIA_CONFIG", "2023_all")
    gaia_split = os.getenv("GAIA_SPLIT", "validation")
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "120"))
    max_tasks = int(os.getenv("MAX_TASKS", "0"))

    return Settings(
        hf_token=hf_token,
        agent_endpoint=agent_endpoint,
        gaia_config=gaia_config,
        gaia_split=gaia_split,
        request_timeout=request_timeout,
        max_tasks=max_tasks,
    )
