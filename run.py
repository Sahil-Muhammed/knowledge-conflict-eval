#!/usr/bin/env python3
"""Knowledge-conflict evaluation harness.

  python run.py prepare  --n 500
  python run.py belief   --model llama3.2-3b --serve --limit 20
  python run.py conflict --model llama3.2-3b --serve --limit 20
  python run.py defend   --model llama3.2-3b --serve --limit 20
  python run.py report

See README.md for the experimental design.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from kc import data, defences, prompts, report
from kc.llm import Client, launch_llama_server

ROOT = Path(__file__).resolve().parent
ITEMS = ROOT / "data" / "items.jsonl"
RESULTS = ROOT / "results"
DEFAULT_CONFLICTQA = ROOT.parent / "LLM-Knowledge-Conflict" / "conflictQA"
DEFAULT_MODELS_DIR = "/mnt/comp/models"


def stable_bit(*parts):
    return hashlib.sha1("|".join(parts).encode()).digest()[0] & 1 == 0


def key(r):
    return (r["stage"], r["item_id"], r.get("framing"), r.get("condition"), r.get("true_first"), r.get("step"))


class Runner:
    """Appends one JSON record per model call to <results>/<model>.jsonl and reuses calls already made."""

    def __init__(self, model, client, results_dir):
        self.model = model
        self.client = client
        self.path = Path(results_dir) / f"{model}.jsonl"
        self.records = data.read_jsonl(self.path)
        self.done = {key(r): r for r in self.records}

    def cached(self, record):
        return self.done.get(key(record))

    def call(self, record, prompt, max_tokens, post=None, logprobs=False):
        """Run the call unless it is already recorded; `post` adds parsed fields before saving."""
        if self.cached(record):
            return self.cached(record)
        text, p_tok, c_tok, secs, top = self.client.chat(prompt, max_tokens, logprobs)
        record.update(model=self.model, prompt=prompt, response=text,
                      prompt_tokens=p_tok, completion_tokens=c_tok, seconds=round(secs, 2))
        if top:
            record["top_logprobs"] = top
        if post:
            post(record)
        return self.put(record)

    def put(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self.done[key(record)] = record
        return record


def mc(runner, stage, item, framing, condition, evidence, true_first):
    p, mapping = prompts.mc_prompt(item, framing, evidence, true_first)
    rec = {"stage": stage, "item_id": item["id"], "framing": framing,
           "condition": condition, "true_first": true_first}
    return runner.call(rec, p, max_tokens=16, post=prompts.label_with(mapping), logprobs=True)


def cmd_prepare(args):
    items = data.load_all(args.conflictqa_dir)
    usable = len(items)
    items = [it for it in items if it["popularity"] >= args.min_popularity]
    chosen = data.sample(items, args.n, seed=args.seed)
    data.write_jsonl(chosen, args.items)
    print(f"{usable} usable ConflictQA-popQA items ({len(items)} with popularity >= {args.min_popularity}); "
          f"wrote {len(chosen)} to {args.items}")


def cmd_belief(args, runner, items):
    """Closed-book: free-form answer (paper's elicitation) + no-evidence MC in both option orders."""
    for i, it in enumerate(items, 1):
        def grade(rec, it=it):
            rec["correct"] = data.mentions(prompts.first_paragraph(rec["response"]), it["ground_truth"])
        runner.call({"stage": "closed_book", "item_id": it["id"]}, prompts.closed_book_prompt(it["question"]),
                    max_tokens=args.closed_book_tokens, post=grade)
        for tf in (True, False):
            mc(runner, "belief_mc", it, None, "none", [], tf)
        print(f"[belief {runner.model}] {i}/{len(items)}", file=sys.stderr)


def believed(args, runner, items):
    """(index, item, belief) for the items the conflict/defend stages run on."""
    beliefs = report.beliefs(runner.records)
    for i, it in enumerate(items, 1):
        b = beliefs.get(it["id"])
        if b is None:
            sys.exit(f"no belief for item {it['id']}: run `belief` for {runner.model} first")
        if b != "mixed" or args.include_mixed:
            yield i, it, b


def cmd_conflict(args, runner, items):
    own = {r["item_id"]: r["response"] for r in runner.records if r["stage"] == "closed_book"}
    for i, it, b in believed(args, runner, items):
        for framing in args.framings:
            for cond in args.conditions:
                if cond == "own+false" and b != "known":
                    continue
                seed = f"{it['id']}|{framing}|{cond}"
                ev = prompts.shuffled(prompts.passages(it, cond, own.get(it["id"])), seed)
                mc(runner, "conflict", it, framing, cond, ev, stable_bit(seed))
        print(f"[conflict {runner.model}] {i}/{len(items)} ({b})", file=sys.stderr)


def cmd_defend(args, runner, items):
    """Run each defence on the same passage order and option order as the `info` baseline."""
    own = {r["item_id"]: r["response"] for r in runner.records if r["stage"] == "closed_book"}
    priors = report.priors(runner.records)
    for i, it, b in believed(args, runner, items):
        for cond in args.conditions:
            if cond == "own+false":
                continue
            seed = f"{it['id']}|info|{cond}"
            tagged = prompts.shuffled(prompts.tagged_passages(it, cond), seed)
            for name in args.defences:
                defences.run(runner, name, it, cond, tagged, stable_bit(seed), priors.get(it["id"]), own[it["id"]])
        print(f"[defend {runner.model}] {i}/{len(items)} ({b})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=ITEMS, help="item set (default: data/items.jsonl)")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results directory; use a separate one per item set")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="sample items from ConflictQA-popQA")
    p.add_argument("--conflictqa-dir", default=DEFAULT_CONFLICTQA)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-popularity", type=int, default=0,
                   help="drop rarer entities; small models know few of them, which starves the `known` group")

    for name in ("belief", "conflict", "defend"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="name from models.json (also the results file name)")
        p.add_argument("--base-url", help="use an already-running OpenAI-compatible server")
        p.add_argument("--serve", action="store_true", help="launch llama-server for the model's gguf")
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="directory holding the GGUF files")
        p.add_argument("--server-bin", default="llama-server")
        p.add_argument("--ngl", type=int, default=0, help="layers to offload to GPU (99 = all)")
        p.add_argument("--limit", type=int, help="only the first N items (for smoke tests)")
        if name == "belief":
            p.add_argument("--closed-book-tokens", type=int, default=96)
        else:
            p.add_argument("--conditions", nargs="+", default=["true", "false", "true+false", "irrelevant"],
                           choices=list(prompts.CONDITIONS))
            p.add_argument("--include-mixed", action="store_true",
                           help="also run items where the model's belief is inconsistent")
        if name == "conflict":
            p.add_argument("--framings", nargs="+", default=["info", "context_only"], choices=list(prompts.FRAMINGS))
        if name == "defend":
            p.add_argument("--defences", nargs="+", default=list(defences.DEFENCES), choices=list(defences.DEFENCES))

    sub.add_parser("report", help="summarise results/*.jsonl")
    args = ap.parse_args()

    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "report":
        return report.main(args.items, args.results)

    items = data.read_jsonl(args.items)
    if not items:
        sys.exit("no items: run `python run.py prepare` first")
    items = items[:args.limit] if args.limit else items

    server = None
    if args.serve:
        models = json.loads((ROOT / "models.json").read_text())
        args.results.mkdir(parents=True, exist_ok=True)
        server = launch_llama_server(str(Path(args.models_dir) / models[args.model]["file"]), args.port,
                                     args.threads, log_path=args.results / f"{args.model}.server.log",
                                     server_bin=args.server_bin, ngl=args.ngl)
        base_url = f"http://127.0.0.1:{args.port}"
    elif args.base_url:
        base_url = args.base_url
    else:
        sys.exit("pass --serve or --base-url")

    try:
        runner = Runner(args.model, Client(base_url, args.model), args.results)
        {"belief": cmd_belief, "conflict": cmd_conflict, "defend": cmd_defend}[args.cmd](args, runner, items)
    finally:
        if server:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
