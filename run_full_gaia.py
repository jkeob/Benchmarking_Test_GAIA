import json
import os
from datetime import datetime, timezone
from collections import Counter
from tqdm import tqdm

from config import load_settings
from gaia_loader import load_gaia_dataset, choose_split, extract_task_fields
from agent_client import call_agent
from scorer import exact_match_score


def main() -> None:
    settings = load_settings()

    print("Loading GAIA dataset...")
    ds = load_gaia_dataset(settings.gaia_config, token=settings.hf_token)

    print("Available splits:", list(ds.keys()))
    split = choose_split(ds, settings.gaia_split)
    data = ds[split]

    print(f"Using split: {split}")
    print(f"Rows in split: {len(data)}")
    print("Columns:", data.column_names)

    total_rows = len(data)
    rows_to_run = total_rows if settings.max_tasks <= 0 else min(settings.max_tasks, total_rows)

    failure_counts = Counter()
    results = []
    scored = 0
    correct = 0

    print(f"\nRunning benchmark on {rows_to_run} task(s)...")
    for i in tqdm(range(rows_to_run)):
        row = data[i]
        task_id, question, gold = extract_task_fields(row, i)

        if not question:
            failure_counts["missing_question"] += 1
            results.append({
                "task_id": task_id,
                "correct": False,
                "failure_type": "missing_question",
                "latency_ms": 0,
                "gold": gold,
                "predicted": "",
            })
            continue

        if not gold:
            # if this split does not expose gold answers, exact-match scoring can't run locally
            failure_counts["missing_gold_answer"] += 1
            results.append({
                "task_id": task_id,
                "correct": False,
                "failure_type": "missing_gold_answer",
                "latency_ms": 0,
                "gold": "",
                "predicted": "",
            })
            continue

        pred, failure_type, latency_ms, raw_json = call_agent(
            endpoint=settings.agent_endpoint,
            prompt=question,
            timeout_sec=settings.request_timeout,
        )

        if failure_type:
            failure_counts[failure_type] += 1
            results.append({
                "task_id": task_id,
                "correct": False,
                "failure_type": failure_type,
                "latency_ms": latency_ms,
                "gold": gold,
                "predicted": "",
                "raw_response": raw_json,
            })
            continue

        is_correct = exact_match_score(pred, gold)
        scored += 1
        if is_correct:
            correct += 1

        results.append({
            "task_id": task_id,
            "correct": is_correct,
            "failure_type": "",
            "latency_ms": latency_ms,
            "gold": gold,
            "predicted": pred,
            "raw_response": raw_json,
        })

    accuracy = (correct / scored) if scored > 0 else 0.0
    avg_latency = (
        sum(r["latency_ms"] for r in results if r["latency_ms"] > 0) /
        max(1, sum(1 for r in results if r["latency_ms"] > 0))
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gaia_config": settings.gaia_config,
        "gaia_split": split,
        "agent_endpoint": settings.agent_endpoint,
        "total_rows": total_rows,
        "rows_executed": rows_to_run,
        "scored_rows": scored,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "avg_latency_ms": round(avg_latency, 2),
        "failure_counts": dict(failure_counts),
    }

    print("\n=== GAIA RUN SUMMARY ===")
    print(f"Config:           {summary['gaia_config']}")
    print(f"Split:            {summary['gaia_split']}")
    print(f"Agent endpoint:   {summary['agent_endpoint']}")
    print(f"Total rows:       {summary['total_rows']}")
    print(f"Rows executed:    {summary['rows_executed']}")
    print(f"Scored rows:      {summary['scored_rows']}")
    print(f"Correct:          {summary['correct']}")
    print(f"Accuracy:         {summary['accuracy'] * 100:.2f}%")
    print(f"Avg latency:      {summary['avg_latency_ms']} ms")
    print(f"Failures:         {summary['failure_counts']}")

    os.makedirs("reports", exist_ok=True)
    out_path = f"reports/gaia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full report to: {out_path}")


if __name__ == "__main__":
    main()
