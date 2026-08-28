# Resumable batch dubbing

Generates a script in batches and atomically updates the manifest after every line, with interruption recovery and fingerprinted caching.

## How to use

1. Connect Model, ScriptPlan, VoiceBank, and optional settings.
2. Start around batch size 4–8 on a 24 GB GPU.
3. Send AudioBatch to duration fitting, QA, review, and delivery.

## Important

A cache hit requires matching model, text, voice, settings, and an existing file. Stale audio is not reused as a new result.
