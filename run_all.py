#!/usr/bin/env python3
"""Run belief + conflict (+ defend) for several models, one worker per GPU, then write the report.

Each worker owns one GPU (CUDA_VISIBLE_DEVICES) and one port, and takes models off a shared
queue, so on Kaggle's 2x T4 two models run at a time. Per-model output goes to
results/<model>.run.log.

  python run_all.py --gpus 0 1 --ngl 99 --models-dir /tmp/models --server-bin /tmp/llama-server
  python run_all.py --gpus cpu --limit 2          # local smoke test
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from kc import defences, prompts, report

ROOT = Path(__file__).resolve().parent


def worker(gpu, port, jobs, args, failures):
    env = dict(os.environ)
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    while True:
        try:
            model = jobs.get_nowait()
        except queue.Empty:
            return
        common = ["--model", model, "--serve", "--port", str(port), "--threads", str(args.threads),
                  "--models-dir", args.models_dir, "--server-bin", args.server_bin, "--ngl", str(args.ngl)]
        if args.limit:
            common += ["--limit", str(args.limit)]
        steps = [["belief", *common],
                 ["conflict", *common, "--framings", *args.framings, "--conditions", *args.conditions]]
        if args.defences:
            steps.append(["defend", *common, "--defences", *args.defences, "--conditions", *args.conditions])
        log_path = args.results / f"{model}.run.log"
        print(f"[gpu {gpu}] {model}: started (log: {log_path})", flush=True)
        with open(log_path, "a") as log:
            for step in steps:
                rc = subprocess.call([sys.executable, str(ROOT / "run.py"), "--items", str(args.items),
                                      "--results", str(args.results), *step],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if rc != 0:
                    failures.append(f"{model} {step[0]} (exit {rc}, see {log_path})")
                    break
        print(f"[gpu {gpu}] {model}: finished", flush=True)


def main():
    models = list(json.loads((ROOT / "models.json").read_text()))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=models, choices=models)
    ap.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids, or 'cpu'")
    ap.add_argument("--base-port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--models-dir", default="/mnt/comp/models")
    ap.add_argument("--server-bin", default="llama-server")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--framings", nargs="+", default=list(prompts.FRAMINGS), choices=list(prompts.FRAMINGS))
    ap.add_argument("--conditions", nargs="+", choices=list(prompts.CONDITIONS),
                    default=[c for c in prompts.CONDITIONS if not c.endswith("false2")])
    ap.add_argument("--defences", nargs="*", default=[], choices=list(defences.DEFENCES),
                    help="also run these defences (default: none)")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items.jsonl")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if args.gpus == ["cpu"]:
        args.ngl = 0

    args.items, args.results = args.items.resolve(), args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    for m in args.models:
        jobs.put(m)
    failures = []
    threads = [threading.Thread(target=worker, args=(g, args.base_port + i, jobs, args, failures))
               for i, g in enumerate(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report.main(args.items, args.results)
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
