# Acceleration benchmark

Benchmarks backends with the same reference, text, seed, and settings and produces an auditable recommendation.

## How to use

1. Use at least one warm-up and three measured runs.
2. Compare `off,flash_attention,deepspeed` first.
3. Inspect median latency, RTF, VRAM, actual fallback, and reproducible hashes.

## Important

The benchmark never changes the loader. Experimental modes should improve by at least 20% before adoption.
