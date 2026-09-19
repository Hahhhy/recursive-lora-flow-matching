# Development-line execution plan

The items below are engineering gates. Passing them does not establish an image-generation result.

| Item | Acceptance criterion | Status |
|---|---|---|
| Unified LoRA target interface | Exact paths, rank and alpha are represented by one validated config | Done |
| Shared LoRA across K | K block calls expose only one A/B pair per target | Done |
| Freeze validation | Only LoRA A/B have `requires_grad=True`; backbone gradients stay absent | Done |
| Per-loop graph validation | Every loop output retains a finite gradient after one backward pass | Done |
| Short overfit | Deterministic toy target loss decreases for ordinary K=1 and recursive K=4 | Done |
| Safe checkpoint | LoRA-only state, config and metadata round-trip; mismatched config is rejected | Done |
| Elastic-ready API | Runtime `forward(..., num_loops=K)` override exists | Done |
| Fair dual path | Ordinary direct block call and recursive Euler call are explicit modes | Done |
| Training loop granularity | Reversible, differentiable layerwise and rangewise patches with call-count/gradient tests | Done |

## Deliberately not claimed yet

- Runtime K override is only an interface; it is not evidence of Elastic training.
- Toy overfit uses a synthetic regression target, not Scale-RAE's flow-matching objective.
- `attn.proj` remains the first auditable target, not a proven optimal insertion point.
- The next gate requires a loaded official Scale-RAE checkpoint on the GPU cluster.
- The immediate GPU gate is a one-batch native flow-matching backward pass, not a full run.
- The controlled first-round matrix is specified in `EXPERIMENT_V0.md`.
