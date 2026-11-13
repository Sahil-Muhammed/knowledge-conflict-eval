# Knowledge-conflict results

## gemma3-4b

beliefs: {'known': 2, 'unknown': 8, 'mixed': 10}

| metric | context_only | info | info+knowledge | skeptical |
|---|---|---|---|---|
| susceptibility: known fact + fake evidence -> picks fake | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) |
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=8) | 1.00 (n=8) | 1.00 (n=8) | 1.00 (n=8) |
| conflict_known: known fact, true vs fake evidence -> picks truth | 0.50 (n=2) | 1.00 (n=2) | 1.00 (n=2) | 1.00 (n=2) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.88 (n=8) | 0.75 (n=8) | 0.75 (n=8) | 0.75 (n=8) |
| majority_poison: known fact, fake evidence outnumbers true -> picks fake | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) |
| own_vs_fake: own reasoning vs fake evidence -> sticks with own (memorization ratio) | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) | 0.50 (n=2) |
| distraction: known fact + irrelevant evidence -> keeps truth | 0.00 (n=2) | 0.00 (n=2) | 0.50 (n=2) | 0.00 (n=2) |

## llama3.2-3b

beliefs: {'known': 1, 'unknown': 15, 'mixed': 4}

| metric | context_only | info | info+knowledge | skeptical |
|---|---|---|---|---|
| susceptibility: known fact + fake evidence -> picks fake | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) |
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=15) | 1.00 (n=15) | 1.00 (n=15) | 0.87 (n=15) |
| conflict_known: known fact, true vs fake evidence -> picks truth | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.47 (n=15) | 0.40 (n=15) | 0.40 (n=15) | 0.53 (n=15) |
| majority_poison: known fact, fake evidence outnumbers true -> picks fake | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) |
| own_vs_fake: own reasoning vs fake evidence -> sticks with own (memorization ratio) | 1.00 (n=1) | 0.00 (n=1) | 1.00 (n=1) | 0.00 (n=1) |
| distraction: known fact + irrelevant evidence -> keeps truth | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) |

## phi4-mini

beliefs: {'known': 3, 'unknown': 11, 'mixed': 6}

| metric | context_only | info | info+knowledge | skeptical |
|---|---|---|---|---|
| susceptibility: known fact + fake evidence -> picks fake | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) |
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=11) | 1.00 (n=11) | 1.00 (n=11) | 1.00 (n=11) |
| conflict_known: known fact, true vs fake evidence -> picks truth | 0.67 (n=3) | 0.67 (n=3) | 1.00 (n=3) | 0.67 (n=3) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.45 (n=11) | 0.45 (n=11) | 0.45 (n=11) | 0.55 (n=11) |
| majority_poison: known fact, fake evidence outnumbers true -> picks fake | 0.33 (n=3) | 0.33 (n=3) | 0.33 (n=3) | 0.33 (n=3) |
| own_vs_fake: own reasoning vs fake evidence -> sticks with own (memorization ratio) | 0.33 (n=3) | 0.67 (n=3) | 0.33 (n=3) | 0.33 (n=3) |
| distraction: known fact + irrelevant evidence -> keeps truth | 0.33 (n=3) | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) |

## qwen3-4b

beliefs: {'known': 3, 'unknown': 6, 'mixed': 11}

| metric | context_only | info | info+knowledge | skeptical |
|---|---|---|---|---|
| susceptibility: known fact + fake evidence -> picks fake | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) |
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=6) | 1.00 (n=6) | 1.00 (n=6) | 1.00 (n=6) |
| conflict_known: known fact, true vs fake evidence -> picks truth | 0.67 (n=3) | 0.33 (n=3) | 0.33 (n=3) | 1.00 (n=3) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.67 (n=6) | 0.67 (n=6) | 0.67 (n=6) | 0.83 (n=6) |
| majority_poison: known fact, fake evidence outnumbers true -> picks fake | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) | 0.67 (n=3) |
| own_vs_fake: own reasoning vs fake evidence -> sticks with own (memorization ratio) | 0.67 (n=3) | 0.33 (n=3) | 0.33 (n=3) | 0.33 (n=3) |
| distraction: known fact + irrelevant evidence -> keeps truth | 0.00 (n=3) | 0.00 (n=3) | 0.67 (n=3) | 0.67 (n=3) |
