from typing import Any, Dict, Tuple
from datasets import load_dataset, DatasetDict


QUESTION_KEYS = ["Question", "question", "prompt", "input"]
ANSWER_KEYS = ["Final answer", "final_answer", "answer", "gold", "target", "label"]
TASK_ID_KEYS = ["task_id", "Task ID", "id", "Id"]


def load_gaia_dataset(config_name: str, token: str | None = None) -> DatasetDict:
    """
    Loads GAIA dataset.
    User-provided pattern confirmed:
      load_dataset("gaia-benchmark/GAIA", "2023_all")
    """
    if token:
        return load_dataset("gaia-benchmark/GAIA", config_name, token=token)
    return load_dataset("gaia-benchmark/GAIA", config_name)


def choose_split(ds: DatasetDict, preferred_split: str) -> str:
    if preferred_split in ds:
        return preferred_split
    # fallback to first available split
    return list(ds.keys())[0]


def extract_task_fields(row: Dict[str, Any], row_index: int) -> Tuple[str, str, str]:
    """
    Returns (task_id, question, gold_answer).
    """
    task_id = ""
    question = ""
    gold = ""

    for k in TASK_ID_KEYS:
        if k in row and row[k] is not None:
            task_id = str(row[k])
            break
    if not task_id:
        task_id = f"row_{row_index}"

    for k in QUESTION_KEYS:
        if k in row and row[k] is not None:
            question = str(row[k])
            break

    for k in ANSWER_KEYS:
        if k in row and row[k] is not None:
            gold = str(row[k])
            break

    return task_id, question, gold

