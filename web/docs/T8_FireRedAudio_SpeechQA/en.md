# Rendered speech QA

Runs per-line ASR and checks CER/WER, clipping, silence, and cue overrun.

## How to use

1. Connect `failed_line_ids` to corrective retry or review the evidence line by line.
2. ASR cache reuses only the transcript; thresholds are recalculated every run.
3. A silence flag may be an intentional performance pause, so listen before rejecting.

## Important

QA is an evidence gate, not the final judge of voice, emotion, or acting quality.
