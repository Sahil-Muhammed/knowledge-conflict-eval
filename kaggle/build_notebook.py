#!/usr/bin/env python3
"""Build kaggle/kc-eval.ipynb: a self-contained Kaggle notebook with the harness embedded.

Re-run this after changing the harness or data/items.jsonl, then re-import the notebook.
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("kc-eval.ipynb")
INCLUDE = ["run.py", "run_all.py", "fetch_models.py", "models.json", "README.md", "kc", "legal",
           "data/items.jsonl", "data/legal_items.jsonl"]


def harness_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip()}


CELLS = [
    md("""
# Knowledge-conflict eval on Kaggle

Runs `knowledge-conflict-eval` (belief + conflict + report) for all models on Kaggle GPUs.

**Before running:** *Settings → Accelerator → GPU T4 x2* and *Settings → Internet → On*.
For the full run use **Save Version → Save & Run All (Commit)** so it keeps going with the tab closed;
results land in this notebook's *Output* (`kc-eval/results/`, `results.zip`).

**Resuming / reusing:** a commit is capped at 12 h. To continue an unfinished run, or skip the
llama.cpp build next time, use *Add Input* to attach a previous version's output of this notebook:
finished calls in `kc-eval/results/*.jsonl` are reused and only the missing ones run, and
`bin/llama-server` is copied in instead of rebuilt.
"""),
    code("""
# ---- configuration ----
MODELS = ["llama3.2-3b", "gemma3-4b", "phi4-mini", "qwen3-4b"]
FRAMINGS = ["info", "info+knowledge", "context_only", "skeptical", "conflict_aware"]
CONDITIONS = ["true", "false", "true+false", "2true+1false", "1true+2false", "irrelevant", "own+false"]
DEFENCES = ["recall", "screen", "vote", "prior_keep", "prior_abstain"]   # [] to skip the defend stage
ITEMS = "data/items.jsonl"   # "data/legal_items.jsonl" for the Indian-law set (add "false2", "true+false2" to CONDITIONS)
RESULTS_DIR = "results"      # use another name per item set, e.g. "results_legal"
LIMIT = None            # e.g. 20 for a quick test run; None = all items
LLAMA_CPP_TAG = "b6814" # same llama.cpp build as the local laptop runs
RESUME = True           # reuse results/*.jsonl from an attached previous output (False: only reuse the llama-server build)
"""),
    code("""
import base64, glob, io, os, shutil, subprocess, sys, tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
HARNESS = WORK / "kc-eval"
BIN = WORK / "bin" / "llama-server"
MODELS_DIR = Path("/tmp/models")   # outside /kaggle/working so the GGUFs aren't saved as output

def sh(cmd):
    \"\"\"Run a shell command, streaming its output into the notebook; raise on failure.\"\"\"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")

gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
assert gpu_info, "No GPU: set Settings -> Accelerator -> GPU T4 x2"
print("\\n".join(gpu_info))
GPUS = [l.split(",")[0].strip() for l in gpu_info]
CUDA_ARCH = gpu_info[0].split(",")[2].strip().replace(".", "")
"""),
    code("""
# ---- unpack the embedded harness; pull in results from an attached previous run ----
HARNESS_TGZ = "__HARNESS_B64__"
HARNESS.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(HARNESS_TGZ)), mode="r:gz") as tar:
    try:
        tar.extractall(HARNESS, filter="data")
    except TypeError:  # Python without tarfile extraction filters
        tar.extractall(HARNESS)
(HARNESS / RESULTS_DIR).mkdir(exist_ok=True)

for prev in glob.glob(f"/kaggle/input/**/kc-eval/{RESULTS_DIR}/*.jsonl", recursive=True) if RESUME else []:
    dest = HARNESS / RESULTS_DIR / Path(prev).name
    if not dest.exists() or dest.stat().st_size < Path(prev).stat().st_size:
        shutil.copy(prev, dest)
        print("resuming from", prev)
print(sorted(p.name for p in HARNESS.iterdir()))
"""),
    code("""
# ---- llama-server with CUDA (reuse a previous build if attached, else build; ~10-15 min) ----
prev_bin = glob.glob("/kaggle/input/**/bin/llama-server", recursive=True)
if not BIN.exists() and prev_bin:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(prev_bin[0], BIN)
    BIN.chmod(0o755)
    print("reusing", prev_bin[0])
if not BIN.exists():
    sh(f"rm -rf /tmp/llama.cpp && git clone -q --depth 1 --branch {LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
    # Kaggle's CUDA toolkit has no libcuda.so (not even the stub) where CMake looks, so the
    # CUDA::cuda_driver target is missing. GGML_CUDA_NO_VMM drops the only direct use of it
    # (virtual-memory GPU pool); outputs are unaffected.
    sh(f"cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
       "-DGGML_CUDA_NO_VMM=ON "
       "-DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1 "
       "|| (grep -A8 'CMake Error' /tmp/cmake.log; tail -15 /tmp/cmake.log; false)")
    sh(f"cmake --build /tmp/llama.cpp/build --target llama-server -j {os.cpu_count()} > /tmp/build.log 2>&1 "
       "|| (tail -40 /tmp/build.log; false)")
    BIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy("/tmp/llama.cpp/build/bin/llama-server", BIN)
sh(f"{BIN} --version")
"""),
    code("""
# ---- model weights (~9.4 GB) ----
sh(f"cd {HARNESS} && python fetch_models.py --models-dir {MODELS_DIR} --models {' '.join(MODELS)}")
"""),
    code("""
# ---- run: one worker per GPU, models taken from a queue; per-model logs in kc-eval/results/<model>.run.log ----
cmd = [sys.executable, "-u", "run_all.py", "--gpus", *GPUS, "--ngl", "99", "--threads", "2",
       "--models-dir", str(MODELS_DIR), "--server-bin", str(BIN),
       "--models", *MODELS, "--framings", *FRAMINGS, "--conditions", *CONDITIONS,
       "--items", ITEMS, "--results", RESULTS_DIR]
if DEFENCES:
    cmd += ["--defences", *DEFENCES]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
sh(f"cd {HARNESS} && " + " ".join(f"'{c}'" for c in cmd))
"""),
    code("""
# ---- results ----
from IPython.display import Markdown, display
display(Markdown((HARNESS / RESULTS_DIR / "report.md").read_text()))
shutil.make_archive(str(WORK / "results"), "zip", HARNESS / RESULTS_DIR)
print("saved", WORK / "results.zip")
"""),
]


def main():
    cells = CELLS[:]
    cells[3] = code(cells[3]["source"].replace("__HARNESS_B64__", harness_b64()))
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i}"
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
