# Apply creative candidate

Merges the reviewed selection back into the source AudioBatch while preserving the old path and full provenance.

## How to use

1. Connect the source AudioBatch, reviewed candidates, and selected ID.
2. Only the candidate's `source_line_id` is replaced.
3. Continue with QA, line review, or batch delivery.

## Important

Old WAV files are never overwritten. The adoption manifest records previous/new paths, seed, and human review.
