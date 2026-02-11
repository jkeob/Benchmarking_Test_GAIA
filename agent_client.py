from typing import Any, Dict, Tuple
import requests


def build_payload(prompt: str) -> Dict[str, Any]:
    """
    Payload shape required by your /invoke endpoint.
    Allowed task values:
      - summarize
      - extract_action_items
      - meeting_minutes
    """
    return {
        "task": "summarize",
        "content": prompt
    }


def parse_agent_response(data: Any) -> str:
    """
    Parser customized for your API shape:
    {
      "status":"ok",
      "result":{
        "summary":"...",
        "formatted_minutes":"..."
      },
      ...
    }
    """
    if not isinstance(data, dict):
        return ""

    # if API explicitly says error, return empty so caller marks failure
    if data.get("status") not in (None, "ok"):
        return ""

    # 1) primary: nested result.summary
    result = data.get("result")
    if isinstance(result, dict):
        # best field for your current task
        summary = result.get("summary")
        if summary:
            return str(summary)

        # fallback for meeting-minutes style output
        formatted = result.get("formatted_minutes")
        if formatted:
            return str(formatted)

        # extra fallbacks
        for k in ("answer", "final_answer", "response", "text", "result", "content"):
            if k in result and result[k] is not None:
                return str(result[k])

    # 2) flat fallbacks
    for k in ("final_answer", "answer", "response", "text", "result", "content"):
        if k in data and data[k] is not None and not isinstance(data[k], (dict, list)):
            return str(data[k])

    return ""


def call_agent(
    endpoint: str,
    prompt: str,
    timeout_sec: int
) -> Tuple[str, str, int, Dict[str, Any] | None]:
    """
    Returns:
      predicted_answer, failure_type, latency_ms, raw_response_json
    """
    payload = build_payload(prompt)

    try:
        r = requests.post(endpoint, json=payload, timeout=timeout_sec)
        latency_ms = int(r.elapsed.total_seconds() * 1000)

        if r.status_code != 200:
            return "", "http_error", latency_ms, None

        data = r.json()

        # API-level failure check
        if isinstance(data, dict) and data.get("status") not in (None, "ok"):
            return "", "runtime_error", latency_ms, data

        pred = parse_agent_response(data)
        if not pred:
            return "", "invalid_format", latency_ms, data

        return pred, "", latency_ms, data

    except requests.Timeout:
        return "", "timeout", timeout_sec * 1000, None
    except Exception:
        return "", "runtime_error", 0, None
