# Corrective QA retry

Regenerates only failed line IDs while keeping script, voice, and quality goals fixed, with hash and cue-duration gates.

## How to use

1. Use for text errors, generation failures, clipping, or cue overruns.
2. Incremented seeds are recorded for every attempt.
3. Only a passing replacement is merged non-destructively into the AudioBatch.

## Important

This is a corrective path, not an acting-variation sampler. Use Creative Line Candidate Pool for exploration.
