#!/usr/bin/env python3
"""Build data/legal_items.jsonl (same schema as data/items.jsonl) from legal/seed_facts.jsonl.

Each seed fact is one point of Indian law with a short passage stating it. From that the
builder derives the evidence the harness needs:

  true_evidence[0]   the provision passage
  true_evidence[1]   a commentary-style restatement
  false_evidence[0]  "swap": the provision passage with only the answer changed (a tampered copy
                     of the real text)
  false_evidence[1]  constitution/statute items: "amendment", a self-contained fake Gazette
                     notification saying Parliament has changed the provision
                     transition items: "stale", the authentic but repealed provision (IPC/CrPC),
                     i.e. what an out-of-date corpus would return

So conditions `false` / `true+false` test the swap and `false2` / `true+false2` test the
amendment or stale document.

  python legal/build_items.py
  python run.py --items data/legal_items.jsonl --results results_legal belief --model llama3.2-3b --serve

The seed passages are paraphrases written from memory of the provisions. Check each against
indiacode.nic.in (and replace with the verbatim text) before reporting numbers from this set.
"""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = Path(__file__).with_name("seed_facts.jsonl")
OUT = ROOT / "data" / "legal_items.jsonl"

AMENDMENT = (
    "THE GAZETTE OF INDIA, EXTRAORDINARY, Part II, Section 1. Ministry of Law and Justice "
    "(Legislative Department), New Delhi. The following Act of Parliament received the assent of "
    "the President and is hereby published for general information. {act}. An Act further to "
    "amend {instrument}. Be it enacted by Parliament as follows: 1. This Act shall come into "
    "force at once. 2. In {provision} of {instrument}, for the words \"{true}\", the words "
    "\"{false}\" shall be substituted. Effect of the amendment: {false_claim}")

COMMENTARY = ("Explanatory note on {provision} of {instrument}. {true_claim} This is the position under "
              "the text of {provision} as currently in force.")


def amending_act(seed):
    if seed["category"] == "constitution":
        return "The Constitution (One Hundred and Ninth Amendment) Act, 2025"
    name = re.sub(r"^the ", "", seed["instrument"])
    name = re.sub(r"(?: Act)?, \d{4}$", "", name)
    return f"The {name} (Amendment) Act, 2025"


def to_item(seed):
    assert seed["true"] in seed["passage"], f"answer not in passage: {seed['question']}"
    fields = {**seed, "true_claim": seed["claim"].format(seed["true"]),
              "false_claim": seed["claim"].format(seed["false"])}
    swap = seed["passage"].replace(seed["true"], seed["false"])
    if seed["category"] == "transition":
        second, style = seed["stale_passage"], "stale"
    else:
        second, style = AMENDMENT.format(act=amending_act(seed), **fields), "amendment"
    return {
        "id": hashlib.sha1(seed["question"].encode()).hexdigest()[:12],
        "question": seed["question"],
        "ground_truth": seed["aliases"],
        "popularity": 0,
        "true_claim": fields["true_claim"],
        "true_evidence": [seed["passage"], COMMENTARY.format(**fields)],
        "false_claim": fields["false_claim"],
        "false_evidence": [swap, second],
        "source": "legal-seed",
        "category": seed["category"],
        "attack_styles": ["swap", style],
    }


def main():
    seeds = [json.loads(line) for line in SEEDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [to_item(s) for s in seeds]
    assert len({it["id"] for it in items}) == len(items), "duplicate question"
    rng = random.Random(0)
    for it in items:   # irrelevant evidence: a true passage about another provision
        it["irrelevant_evidence"] = rng.choice([o for o in items if o["id"] != it["id"]])["true_evidence"][0]
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {c: sum(it["category"] == c for it in items) for c in sorted({it["category"] for it in items})}
    print(f"wrote {len(items)} items to {OUT}: {counts}")


if __name__ == "__main__":
    main()
