"""Turn raw call records into belief categories and per-condition answer rates."""
import csv
from collections import defaultdict
from pathlib import Path

from kc import data

LABELS = ("true", "false", "uncertain", "invalid")

# Headline metrics: (name, condition, belief group, label whose rate is reported, description)
HEADLINE = [
    ("susceptibility", "false", "known", "false", "known fact + fake evidence -> picks fake"),
    ("uptake", "true", "unknown", "true", "unknown fact + true evidence -> picks truth"),
    ("conflict_known", "true+false", "known", "true", "known fact, true vs fake evidence -> picks truth"),
    ("conflict_unknown", "true+false", "unknown", "true", "unknown fact, true vs fake evidence -> picks truth"),
    ("majority_poison", "1true+2false", "known", "false", "known fact, fake evidence outnumbers true -> picks fake"),
    ("own_vs_fake", "own+false", "known", "true", "own reasoning vs fake evidence -> sticks with own (memorization ratio)"),
    ("distraction", "irrelevant", "known", "true", "known fact + irrelevant evidence -> keeps truth"),
]


def beliefs(records):
    """known: free-form answer correct AND no-evidence MC picks the truth in both option orders.
    unknown: free-form wrong AND no-evidence MC never picks the truth. Anything else: mixed."""
    free = {r["item_id"]: r.get("correct") for r in records if r["stage"] == "closed_book"}
    mc = defaultdict(list)
    for r in records:
        if r["stage"] == "belief_mc":
            mc[r["item_id"]].append(r.get("label"))
    out = {}
    for item_id, ok in free.items():
        labels = mc.get(item_id, [])
        if len(labels) < 2:
            continue
        if ok and all(l == "true" for l in labels):
            out[item_id] = "known"
        elif not ok and not any(l == "true" for l in labels):
            out[item_id] = "unknown"
        else:
            out[item_id] = "mixed"
    return out


def tabulate(records):
    """(framing, condition, belief) -> counts per label, plus mean prompt tokens and seconds."""
    b = beliefs(records)
    cells = defaultdict(lambda: {"n": 0, **{l: 0 for l in LABELS}, "prompt_tokens": 0, "seconds": 0.0})
    for r in records:
        if r["stage"] != "conflict" or r["item_id"] not in b:
            continue
        c = cells[(r["framing"], r["condition"], b[r["item_id"]])]
        c["n"] += 1
        c[r.get("label", "invalid")] += 1
        c["prompt_tokens"] += r.get("prompt_tokens") or 0
        c["seconds"] += r.get("seconds") or 0
    return b, cells


def main(items_path, results_dir):
    results_dir = Path(results_dir)
    rows, summary = [], []
    for path in sorted(results_dir.glob("*.jsonl")):
        model = path.stem
        b, cells = tabulate(data.read_jsonl(path))
        counts = {k: sum(1 for v in b.values() if v == k) for k in ("known", "unknown", "mixed")}
        summary.append(f"\n## {model}\n\nbeliefs: {counts}\n")
        for (framing, cond, belief), c in sorted(cells.items()):
            n = c["n"]
            rows.append({"model": model, "framing": framing, "condition": cond, "belief": belief, "n": n,
                         **{f"p_{l}": round(c[l] / n, 3) for l in LABELS},
                         "mean_prompt_tokens": round(c["prompt_tokens"] / n, 1),
                         "mean_seconds": round(c["seconds"] / n, 2)})
        framings = sorted({f for f, _, _ in cells})
        if framings:
            summary.append("| metric | " + " | ".join(framings) + " |")
            summary.append("|---|" + "---|" * len(framings))
        for name, cond, belief, label, desc in HEADLINE:
            vals = []
            for f in framings:
                c = cells.get((f, cond, belief))
                vals.append(f"{c[label] / c['n']:.2f} (n={c['n']})" if c and c["n"] else "-")
            if any(v != "-" for v in vals):
                summary.append(f"| {name}: {desc} | " + " | ".join(vals) + " |")

    if not rows:
        print("no conflict results yet")
        return
    with open(results_dir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = "# Knowledge-conflict results\n" + "\n".join(summary) + "\n"
    (results_dir / "report.md").write_text(text)
    print(text)
    print(f"full table: {results_dir / 'report.csv'}")
