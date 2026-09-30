# Upstream audit — 2026-09-30

## Decision

Keep the existing code/model pins and packaged runtime. Close node Issue [#1](https://github.com/T8mars/comfyui-fireredaudio-T8/issues/1) after recording this audit. No model download, runtime upgrade, Registry publication, or desktop repackage is required for these changes.

`manifests/upstream_audit.json` records exact reviewed revisions separately from the runtime manifest. `updates_available` and `--fail-on-update` still report differences from pins. The weekly workflow uses `audit_required` so an already reviewed revision does not repeatedly recreate the issue. A new revision, repository, or runtime pin requires another audit; missing audit data never suppresses an alert, and malformed data fails the check.

## Revision evidence

| Source | Pinned | Reviewed latest | Changes / action |
| --- | --- | --- | --- |
| FireRedTeam/FireRedAudio | `88b826378023eb9a49b297214568398a300e5c32` | `4bf612ee3d97fc7ea82be9fddd4fb87771732ea9` | README.md, independent Gradio app.py, assets/fireredteam_avatar.png, assets/snapshot_gradio.png only. Keep pin. |
| t8star/Firered-Audio-Comfy | `ffd450cf336a929f4ceb1eaaff4c7ba4e4af34bb` | `63effcaa9aa63a02bdfde1667c27866430d29b74` | Both recursive file trees have 13 files. Only README.md differs; all model, decoder, configuration, tokenizer and index objects/sizes are unchanged. Keep pin. |

Evidence: [code diff](https://github.com/FireRedTeam/FireRedAudio/compare/88b826378023eb9a49b297214568398a300e5c32...4bf612ee3d97fc7ea82be9fddd4fb87771732ea9), [model history](https://huggingface.co/t8star/Firered-Audio-Comfy/commits/main).

README.md changed from Git object `dc5df26b112d2750f58308b80e64898c81854289` (3759 bytes) to `0406e7c4b7e4a518af070079e09b5e155b82ffaa` (4064 bytes). Large-file weight LFS OIDs, not just filenames, were compared.

## Official PR #12: RNG concurrency

[PR #12](https://github.com/FireRedTeam/FireRedAudio/pull/12), head `f9b3ef1083c8e94a70f7072b5527f8bc6a49fbb7`, moves `_seed` into `INFER_LOCK` for clone/design/edit in the Gradio demo. It is open as of this audit, so it is not part of the reviewed main-branch revision.

The desktop and bundled ComfyUI Worker already serialize `infer()` and `infer_tts_batch()` using the model lock. `_run_task()` sets the seed inside `infer()`'s lock. Each Worker server owns one runtime shared by its HTTP request threads. The Gradio app is not used by these packages, so no runtime patch is warranted.

Validation exercised both actual runtime source trees in separate processes through public `infer()` for TTS, voice design and speech edit. A fake model avoided loading weights. Threading events forced request B to attempt the lock while A was sampling real Python, NumPy and PyTorch CPU RNGs. All six cases matched their per-seed baseline sequences. Moving seeding outside the lock by fault injection was detected in each source/task case.

This validates lock/seed placement, not CUDA kernel determinism or generated-audio quality. No real-model generation was rerun because neither inference code nor weights changed.

## Other open upstream Issues

Official open Issues: #2, #3, #4, #6, #8, #9 and #11. They concern a demo, language support, backend requests, training and community post-training/fine-tuning projects. None supplied a new reproducible bundled-runtime defect in this audit. In particular, community vLLM work is not evidence of supported full audio-generation acceleration; do not silently enable it or replace weights on the basis of these announcements.

The desktop repository has no open PRs/Issues. The node repository has no open PRs; Issue #1 is the audit reminder handled here. Third-party official Issues and PRs are not closed or merged by this audit.

## Verification

- Full node suite: **76 passed, 9 subtests passed**. Upstream-watch regression tests cover reviewed revisions, either source advancing, changed repository/pin, missing/malformed baseline, return to runtime pins, and preserved `--fail-on-update` behavior.
- Live code/model API query reports `status=audited`, `updates_available=true`, `audit_required=false` for the exact revisions above.
- Run the node suite with `python -m pytest tests` and manually dispatch `upstream-watch` after pushing; verify that it does not recreate the completed reminder.
