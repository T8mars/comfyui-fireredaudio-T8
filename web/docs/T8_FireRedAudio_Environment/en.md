# Environment diagnostics

Shows host Python, Torch, Transformers, and isolated worker state to prove dependencies are not mixed.

## How to use

1. Run once after install or upgrade.
2. Host Transformers may remain on 4.x while the FireRedAudio worker uses pinned 5.8.0.
3. Attach the report to issues; it is more actionable than a screenshot alone.

## Important

This node is read-only: it does not download models, install packages, or run generation.
