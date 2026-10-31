"""Minimal OpenAI-compatible chat client (stdlib only) and a llama-server launcher."""
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

_THINK = re.compile(r"<think>.*?</think>", re.S)


class Client:
    def __init__(self, base_url, model="local"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, prompt, max_tokens):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens,
            # Qwen3: disable thinking mode so all models answer directly.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        text = _THINK.sub("", resp["choices"][0]["message"]["content"] or "").strip()
        usage = resp.get("usage", {})
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens"), time.time() - t0


def wait_healthy(base_url, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError(f"llama-server at {base_url} not healthy after {timeout}s")


def launch_llama_server(gguf, port=8080, threads=8, ctx=4096, log_path=None):
    cmd = ["llama-server", "-m", gguf, "--port", str(port), "-t", str(threads), "-c", str(ctx),
           "-np", "1", "--jinja", "--reasoning-budget", "0"]
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_healthy(f"http://127.0.0.1:{port}")
    except Exception:
        proc.terminate()
        raise
    print(f"llama-server up on :{port} ({gguf})", file=sys.stderr)
    return proc
