# Zero-shot voice cloning

Reads target text using a reference recording and its transcript. The transcript can be generated automatically by ASR.

## How to use

1. Connect a Full model and `Load Audio`.
2. Provide an exact transcript when available to avoid an extra ASR pass.
3. Connect AUDIO to a save or post-production node.

## Important

Prefer a clean, single-speaker reference and manually verify automatic transcripts.
