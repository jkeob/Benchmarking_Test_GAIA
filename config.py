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
    certification_mode: bool


def load_settings() -> Settings:
    hf_token = os.getenv("HF_TOKEN", None)
    agent_endpoint = os.getenv("AGENT_ENDPOINT", "http://127.0.0.1:8090/invoke")
    gaia_config = os.getenv("GAIA_CONFIG", "2023_all")
    gaia_split = os.getenv("GAIA_SPLIT", "test")
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "120"))
    max_tasks = int(os.getenv("MAX_TASKS", "0"))
    certification_mode = os.getenv("CERTIFICATION_MODE", "true").lower() == "true"

    # hard-enforce publish-time policy
    if certification_mode:
        if gaia_split != "test":
            raise ValueError("CERTIFICATION_MODE requires GAIA_SPLIT='test'")
        if max_tasks != 0:
            raise ValueError("CERTIFICATION_MODE requires MAX_TASKS=0 (run all rows)")

    return Settings(
        hf_token=hf_token,
        agent_endpoint=agent_endpoint,
        gaia_config=gaia_config,
        gaia_split=gaia_split,
        request_timeout=request_timeout,
        max_tasks=max_tasks,
        certification_mode=certification_mode,
    )
