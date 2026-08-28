# Speech-aware subtitle duration fit

Fits generated speech into SRT cue windows while preserving natural delivery where possible.

## How to use

1. Default `speech_aware` detects and removes excess boundary silence first.
2. Pitch-preserving `atempo` is used only for residual overrun; internal pauses are preserved.
3. If the residual tempo exceeds the safety limit, the line is sent for regeneration instead of truncation.

## Important

The report separates raw tempo, boundary trims, and residual tempo. Source files are always preserved.
