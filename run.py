#!/usr/bin/env python3
"""Knowledge-conflict evaluation harness.

  python run.py prepare  --n 500
  python run.py belief   --model llama3.2-3b --serve --limit 20
  python run.py conflict --model llama3.2-3b --serve --limit 20
  python run.py report

See README.md for the experimental design.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from kc import data, prompts, report
from kc.llm import Client, launch_llama_server

ROOT = Path(__file__).resolve().parent
ITEMS = ROOT / "data" / "items.jsonl"
RESULTS = ROOT / "results"
DEFAULT_CONFLICTQA = ROOT.parent / "LLM-Knowledge-Conflict" / "conflictQA"


def stable_bit(*parts):
    return hashlib.sha1("|".join(parts).encode()).digest()[0] & 1 == 0


def key(r):
    return (r["stage"], r["item_id"], r.get("framing"), r.get("condition"), r.get("true_first"))


class Runner:
    """Appends one JSON record per model call to results/<model>.jsonl and skips calls already made."""

    def __init__(self, model, client):
        self.model = model
        self.client = client
        self.path = RESULTS / f"{model}.jsonl"
        self.records = data.read_jsonl(self.path)
        self.done = {key(r) for r in self.records}

    def call(self, record, prompt, max_tokens):
        if key(record) in self.done:
            return None
        text, p_tok, c_tok, secs = self.client.chat(prompt, max_tokens)
        record.update(model=self.model, prompt=prompt, response=text,
                      prompt_tokens=p_tok, completion_tokens=c_tok, seconds=round(secs, 2))
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self.done.add(key(record))
        return record


def mc(runner, stage, item, framing, condition, evidence, true_first):
    p, mapping = prompts.mc_prompt(item, framing, evidence, true_first)
    rec = {"stage": stage, "item_id": item["id"], "framing": framing,
           "condition": condition, "true_first": true_first}
    out = runner.call(rec, p, max_tokens=16)
    if out:
        letter = prompts.parse_choice(out["response"])
        out["choice"] = letter
        out["label"] = mapping.get(letter, "invalid")
        # rewrite last line with parsed fields
        _rewrite_last(runner)
    return out


def _rewrite_last(runner):
    lines = runner.path.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps(runner.records[-1], ensure_ascii=False)
    runner.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_prepare(args):
    items = data.load_all(args.conflictqa_dir)
    chosen = data.sample(items, args.n, seed=args.seed)
    data.write_jsonl(chosen, ITEMS)
    print(f"{len(items)} usable ConflictQA-popQA items; wrote {len(chosen)} to {ITEMS}")


def cmd_belief(args, runner, items):
    """Closed-book: free-form answer (paper's elicitation) + no-evidence MC in both option orders."""
    for i, it in enumerate(items, 1):
        rec = {"stage": "closed_book", "item_id": it["id"]}
        out = runner.call(rec, prompts.closed_book_prompt(it["question"]), max_tokens=args.closed_book_tokens)
        if out:
            out["correct"] = data.mentions(prompts.first_paragraph(out["response"]), it["ground_truth"])
            _rewrite_last(runner)
        for tf in (True, False):
            mc(runner, "belief_mc", it, None, "none", [], tf)
        print(f"[belief {runner.model}] {i}/{len(items)}", file=sys.stderr)


def cmd_conflict(args, runner, items):
    beliefs = report.beliefs(runner.records)
    own = {r["item_id"]: r["response"] for r in runner.records if r["stage"] == "closed_book"}
    for i, it in enumerate(items, 1):
        b = beliefs.get(it["id"])
        if b is None:
            sys.exit(f"no belief for item {it['id']}: run `belief` for {runner.model} first")
        if b == "mixed" and not args.include_mixed:
            continue
        for framing in args.framings:
            for cond in args.conditions:
                if cond == "own+false" and b != "known":
                    continue
                seed = f"{it['id']}|{framing}|{cond}"
                ev = prompts.shuffled(prompts.passages(it, cond, own.get(it["id"])), seed)
                mc(runner, "conflict", it, framing, cond, ev, stable_bit(seed))
        print(f"[conflict {runner.model}] {i}/{len(items)} ({b})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="sample items from ConflictQA-popQA")
    p.add_argument("--conflictqa-dir", default=DEFAULT_CONFLICTQA)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    for name in ("belief", "conflict"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="name from models.json (also the results file name)")
        p.add_argument("--base-url", help="use an already-running OpenAI-compatible server")
        p.add_argument("--serve", action="store_true", help="launch llama-server for the model's gguf")
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--limit", type=int, help="only the first N items (for smoke tests)")
        if name == "belief":
            p.add_argument("--closed-book-tokens", type=int, default=96)
        else:
            p.add_argument("--framings", nargs="+", default=["info", "context_only"], choices=list(prompts.FRAMINGS))
            p.add_argument("--conditions", nargs="+", default=["true", "false", "true+false", "irrelevant"],
                           choices=list(prompts.CONDITIONS))
            p.add_argument("--include-mixed", action="store_true",
                           help="also run items where the model's belief is inconsistent")

    sub.add_parser("report", help="summarise results/*.jsonl")
    args = ap.parse_args()

    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "report":
        return report.main(ITEMS, RESULTS)

    items = data.read_jsonl(ITEMS)
    if not items:
        sys.exit("no items: run `python run.py prepare` first")
    items = items[:args.limit] if args.limit else items

    server = None
    if args.serve:
        models = json.loads((ROOT / "models.json").read_text())
        RESULTS.mkdir(exist_ok=True)
        server = launch_llama_server(models[args.model]["gguf"], args.port, args.threads,
                                     log_path=RESULTS / f"{args.model}.server.log")
        base_url = f"http://127.0.0.1:{args.port}"
    elif args.base_url:
        base_url = args.base_url
    else:
        sys.exit("pass --serve or --base-url")

    try:
        runner = Runner(args.model, Client(base_url, args.model))
        (cmd_belief if args.cmd == "belief" else cmd_conflict)(args, runner, items)
    finally:
        if server:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
