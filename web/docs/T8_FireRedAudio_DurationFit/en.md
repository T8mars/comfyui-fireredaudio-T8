# Speech-aware subtitle duration fit

Fits generated speech into SRT cue windows while preserving natural delivery where possible.

## How to use

1. Default `speech_aware` detects and removes excess boundary silence first.
2. For residual overrun, only speech spans are accelerated while qualifying performance pauses keep their original duration.
3. If the required speech tempo exceeds the independent natural-speed limit, the line is sent for regeneration.

## Important

The report lists protected pauses, speech duration, and actual tempo. Source files are preserved and final naturalness still requires listening.
