"""Build a model-agnostic item set from ConflictQA-popQA.

ConflictQA rows are tied to the model that produced them (its "memory").
Whichever way that model was right or wrong, each row gives us one claim
supported by the ground truth and one claim contradicting it, each with two
supporting passages. We re-key rows into that (true, false) form so any model
can be tested on them; which side is "counter-memory" is decided per model
later, from that model's own closed-book belief.
"""
import ast
import hashlib
import json
import random
from pathlib import Path

# Preferred order when a question appears in several source files.
SOURCES = ["gpt4", "chatgpt", "palm2", "llama2-70b", "qwen7b", "vicuna33b", "llama2-7b", "vicuna7b"]


def aliases(ground_truth):
    return ground_truth if isinstance(ground_truth, list) else ast.literal_eval(ground_truth)


def mentions(text, alias_list):
    text = text.lower()
    return any(a.lower() in text for a in alias_list)


def to_item(row, source):
    fields = ("memory_answer", "parametric_memory", "parametric_memory_aligned_evidence",
              "counter_answer", "counter_memory", "counter_memory_aligned_evidence")
    if not all(row.get(k) for k in fields):
        return None
    al = aliases(row["ground_truth"])
    memory_ok = mentions(row["memory_answer"], al)
    counter_ok = mentions(row["counter_answer"], al)
    if memory_ok == counter_ok:  # both or neither mention the truth: unusable
        return None
    mem = (row["memory_answer"].strip(),
           [row["parametric_memory"].strip(), row["parametric_memory_aligned_evidence"].strip()])
    cnt = (row["counter_answer"].strip(),
           [row["counter_memory"].strip(), row["counter_memory_aligned_evidence"].strip()])
    (true_claim, true_ev), (false_claim, false_ev) = (mem, cnt) if memory_ok else (cnt, mem)
    if not all(true_ev + false_ev):
        return None
    return {
        "id": hashlib.sha1(row["question"].encode()).hexdigest()[:12],
        "question": row["question"],
        "ground_truth": al,
        "popularity": int(row["popularity"]),
        "true_claim": true_claim,
        "true_evidence": true_ev,
        "false_claim": false_claim,
        "false_evidence": false_ev,
        "source": source,
    }


def load_all(conflictqa_dir, dataset="popQA"):
    items = {}
    for src in SOURCES:
        path = Path(conflictqa_dir) / f"conflictQA-{dataset}-{src}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = to_item(json.loads(line), src)
                if item and item["id"] not in items:
                    items[item["id"]] = item
    return list(items.values())


def sample(items, n, seed=0, bins=4):
    """Equal-size sample from each popularity quantile, so the set isn't all long-tail entities."""
    rng = random.Random(seed)
    items = sorted(items, key=lambda it: it["popularity"])
    size = len(items) // bins
    per_bin = n // bins
    out = []
    for b in range(bins):
        chunk = items[b * size:(b + 1) * size if b < bins - 1 else len(items)]
        out += rng.sample(chunk, min(per_bin, len(chunk)))
    # Irrelevant-evidence condition: a true passage about some other question.
    for it in out:
        other = rng.choice([o for o in out if o["id"] != it["id"]])
        it["irrelevant_evidence"] = other["true_evidence"][0]
    rng.shuffle(out)
    return out


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
