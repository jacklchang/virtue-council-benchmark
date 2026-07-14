"""
Virtue Council Benchmark Suite
Dilemma dataset preparation (v0.1.0)

Downloads DailyDilemmas (Chiu et al. 2024, arXiv:2410.02683, CC-BY-4.0),
maps each action's human-value tags onto the seven council virtues, and keeps
dilemmas where the two actions invoke disjoint, non-empty virtue sets --
structurally guaranteed virtue conflict for council deliberation.

Usage:
    python prepare_dilemmas.py                 # writes datasets/dilemmas.json
    python prepare_dilemmas.py --max-items 50  # cap output size
"""

import argparse
import ast
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "dilemmas.json"

HF_BASE = "https://huggingface.co/datasets/kellycyy/daily_dilemmas/resolve/main"
DILEMMAS_URL = f"{HF_BASE}/dilemma_to_action_to_values_aggregated.csv"
VALUES_URL = f"{HF_BASE}/values.csv"

# DailyDilemmas values.csv tags values with Aristotle's virtues from the
# Nicomachean Ethics taxonomy. Map those onto the council roster, then extend
# with value-name keywords for the two council virtues the taxonomy lacks
# direct tags for (justice, prudence) and to widen recall on the others.
TAXONOMY_TO_COUNCIL = {
    "Courage": "courage",
    "Temperance": "temperance",
    "Truthfulness": "honesty",
    "Liberality": "generosity",
    "Modesty": "humility",
    "Patience": "temperance",
    "Righteous Indignation": "justice",
}

VALUE_KEYWORDS_TO_COUNCIL = {
    "justice": ["fairness", "justice", "equality", "equity", "impartiality",
                "accountability", "rule of law", "respect for rules"],
    "prudence": ["prudence", "wisdom", "foresight", "caution", "judgment",
                 "practicality", "discernment", "risk management"],
    "honesty": ["honesty", "truth", "integrity", "transparency", "sincerity"],
    "courage": ["courage", "bravery", "boldness", "assertiveness"],
    "temperance": ["self-control", "self-discipline", "moderation", "restraint",
                   "patience"],
    "generosity": ["generosity", "charity", "giving", "selflessness", "altruism"],
    "humility": ["humility", "modesty", "humbleness"],
}


def download_csv(url: str) -> list[dict]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def build_value_map(values_rows: list[dict]) -> dict[str, set[str]]:
    """Map each DailyDilemmas value string to a set of council virtues."""
    value_map: dict[str, set[str]] = defaultdict(set)
    for row in values_rows:
        value = row["value"].strip().lower()
        taxonomy_virtue = row.get("Virtue", "").strip()
        if taxonomy_virtue in TAXONOMY_TO_COUNCIL:
            value_map[value].add(TAXONOMY_TO_COUNCIL[taxonomy_virtue])
        if row.get("MFT", "").strip() == "Fairness":
            value_map[value].add("justice")
    all_values = {row["value"].strip().lower() for row in values_rows}
    for value in all_values:
        for virtue, keywords in VALUE_KEYWORDS_TO_COUNCIL.items():
            if any(keyword in value for keyword in keywords):
                value_map[value].add(virtue)
    return dict(value_map)


def virtues_for_action(values_field: str, value_map: dict[str, set[str]]) -> set[str]:
    try:
        values = ast.literal_eval(values_field)
    except (ValueError, SyntaxError):
        return set()
    virtues: set[str] = set()
    for value in values:
        virtues |= value_map.get(value.strip().lower(), set())
    return virtues


def prepare(max_items: int | None) -> None:
    print("Downloading DailyDilemmas...")
    dilemma_rows = download_csv(DILEMMAS_URL)
    values_rows = download_csv(VALUES_URL)
    print(f"  {len(dilemma_rows)} action rows, {len(values_rows)} value definitions")

    value_map = build_value_map(values_rows)

    by_dilemma: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in dilemma_rows:
        by_dilemma[row["dilemma_idx"]][row["action_type"]] = row

    items = []
    for dilemma_idx, actions in sorted(by_dilemma.items(), key=lambda kv: int(kv[0])):
        if "to_do" not in actions or "not_to_do" not in actions:
            continue
        a, b = actions["to_do"], actions["not_to_do"]
        virtues_a = virtues_for_action(a["values_aggregated"], value_map)
        virtues_b = virtues_for_action(b["values_aggregated"], value_map)

        # Keep only structurally guaranteed virtue conflicts: each side must
        # invoke at least one council virtue the other side does not.
        if not (virtues_a - virtues_b) or not (virtues_b - virtues_a):
            continue

        items.append({
            "id": f"council_dd_{int(dilemma_idx):04d}",
            "source": "daily_dilemmas",
            "situation": a["dilemma_situation"],
            "action_a": a["action"],
            "action_b": b["action"],
            "consequence_a": a["negative_consequence"],
            "consequence_b": b["negative_consequence"],
            "virtues_a": sorted(virtues_a),
            "virtues_b": sorted(virtues_b),
            "topic_group": a["topic_group"],
        })

    if max_items:
        items = items[:max_items]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"Kept {len(items)} virtue-conflict dilemmas -> {OUTPUT_PATH}")
    topic_counts = defaultdict(int)
    for item in items:
        topic_counts[item["topic_group"]] += 1
    for topic, count in sorted(topic_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {topic}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare council dilemma dataset")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Cap the number of output dilemmas")
    args = parser.parse_args()
    prepare(args.max_items)


if __name__ == "__main__":
    main()
