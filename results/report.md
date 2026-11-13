# Knowledge-conflict results

## gemma3-4b

beliefs: {'known': 1, 'unknown': 4, 'mixed': 5}

| metric | context_only | info |
|---|---|---|
| susceptibility: known fact + fake evidence -> picks fake | 0.00 (n=1) | 0.00 (n=1) |
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=4) | 1.00 (n=4) |
| conflict_known: known fact, true vs fake evidence -> picks truth | 1.00 (n=1) | 1.00 (n=1) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.75 (n=4) | 0.75 (n=4) |
| distraction: known fact + irrelevant evidence -> picks fake | 0.00 (n=1) | 0.00 (n=1) |

## llama3.2-3b

beliefs: {'known': 0, 'unknown': 3, 'mixed': 0}

| metric | context_only | info |
|---|---|---|
| uptake: unknown fact + true evidence -> picks truth | 1.00 (n=3) | 1.00 (n=3) |
| conflict_unknown: unknown fact, true vs fake evidence -> picks truth | 0.67 (n=3) | 1.00 (n=3) |
