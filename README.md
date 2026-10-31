# knowledge-conflict-eval

How do small local LLMs behave when retrieved context conflicts with what they already know?
This harness reruns the setup of Xie et al., *Adaptive Chameleon or Stubborn Sloth* (ICLR'24),
on Llama-3.2-3B, Gemma-3-4B, Phi-4-mini and Qwen3-4B. It uses their ConflictQA data with
their prompt formats. Standard library only, no pip installs.

## Design

ConflictQA rows are tied to the model that generated them, so `prepare` re-keys each row into
a model-independent item:

```
question, ground_truth aliases, popularity
true_claim  + 2 passages supporting the ground truth
false_claim + 2 passages supporting a wrong answer (LLM-generated fakes)
irrelevant passage (true evidence for a different question)
```

Each model's own belief is then measured closed-book (`belief` stage):

| belief | rule |
|---|---|
| `known` | free-form answer mentions the ground truth **and** the no-evidence multiple-choice question picks the true option in both option orders |
| `unknown` | free-form answer is wrong **and** the MC question never picks the true option |
| `mixed` | anything else; skipped in the conflict stage unless `--include-mixed` |

The `conflict` stage asks a 3-way multiple-choice question (true claim / false claim /
Uncertain) under each **evidence condition** × **instruction framing**. The option order and
the passage order are randomised with a fixed seed per (item, condition, framing).

Conditions (`kc/prompts.py: CONDITIONS`): `true`, `false`, `true+false`, `2true+1false`,
`1true+2false`, `irrelevant`, `own+false` (the model's own closed-book reasoning vs a fake
passage; this is the paper's headline setup and runs on `known` items only).

Framings (`kc/prompts.py: FRAMINGS`):
- `info`: the paper's wording, "according to the given information"
- `info+knowledge`: the paper's wording plus "and your knowledge"
- `context_only`: the "ideal RAG" instruction, answer only from the context
- `skeptical`: a prompt-level defence that warns the context may be poisoned

Headline metrics (`kc/report.py: HEADLINE`):
- **susceptibility**: known fact + fake evidence → picks the fake (poisoning success)
- **uptake**: unknown fact + true evidence → picks the truth (learning from context)
- **conflict_known / conflict_unknown**: true vs fake evidence side by side
- **majority_poison**: fake passages outnumber true ones
- **own_vs_fake**: the paper's memorization ratio
- **distraction**: an irrelevant passage shouldn't change a known answer

## Running

```bash
python run.py prepare --n 500                     # data/items.jsonl, stratified by popularity

# per model; --serve starts llama-server with the GGUF from models.json and stops it afterwards
python run.py belief   --model llama3.2-3b --serve
python run.py conflict --model llama3.2-3b --serve
#   --framings info info+knowledge context_only skeptical
#   --conditions true false true+false 2true+1false 1true+2false irrelevant own+false
#   --limit 20   for smoke tests

python run.py report                              # results/report.md + results/report.csv
```

Results are appended to `results/<model>.jsonl` with one record per model call (prompt,
response, parsed label, tokens, latency). Re-running a command skips calls that are
already recorded, so interrupted runs resume, and adding framings or conditions later only
runs the new cells.

To use another server (vLLM or llama.cpp on Kaggle/Colab, an API), pass
`--base-url http://host:port` instead of `--serve`; `--model` then just names the results
file and is sent as the model name.

## Cost on the laptop (i7 CometLake-U, CPU only, Q4_K_M)

Measured: belief calls take about 3 s and conflict calls about 8–25 s (200–500 prompt tokens).
With the defaults (2 framings × 4 conditions) and 500 items, that is about 10 h per model.
Use `--limit` locally and do full runs on a GPU.

## Known caveats

- **Noisy Wikipedia-aligned passages.** In some ConflictQA items the second "true" passage
  is about a different person with the same name (e.g. Ron Douglas). This affects
  `2true+1false`; spot-check items or use only `*0` passages.
- **Ambiguous aliases.** Alias matching on free-form answers can be wrong (e.g. "football"
  matches American football). The two-order MC check exists to catch these; items it
  catches end up `mixed`.
- **Small models know little long-tail trivia.** Most low-popularity items come out
  `unknown`. Read `known` metrics by popularity bucket and sample more popular items if
  `n` is too small.
- **Option-order bias.** This is averaged out by randomised order. For the belief stage
  both orders are run explicitly.

## Adding the NetSec poisoning set

Write items in the same schema as `data/items.jsonl` (true/false claims plus passages; the
fake chunks in the MCP repo's `src/rag/poison.py` are a starting point) to a separate file
and point `ITEMS` in `run.py` at it, or add a `--items` flag.
