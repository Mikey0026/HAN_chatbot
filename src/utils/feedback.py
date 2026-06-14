import json
from pathlib import Path
from datetime import datetime

FEEDBACK_FILE = Path("data/feedback/feedback.jsonl")


def save_feedback(question: str, answer: str, rating: str):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback():
    if not FEEDBACK_FILE.exists():
        return []

    records = []

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    return records