from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.creator_tools import fit_audio_batch_to_cues
from runtime.production import AudioBatch, file_digest, wav_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v0.16 speech-aware cue fitting on a real WAV.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--slot", type=float, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = file_digest(source)
    batch = AudioBatch(
        "real-listening-source",
        (
            {
                "line_id": "real-listening",
                "index": 1,
                "speaker": "旁白",
                "status": "complete",
                "output_path": str(source),
                "start_seconds": 0.0,
                "end_seconds": float(args.slot),
            },
        ),
    )
    fitted, report = fit_audio_batch_to_cues(
        batch,
        args.output_dir,
        strategy="speech_aware",
        tolerance_seconds=0.05,
        maximum_speed=1.15,
        edge_silence_threshold_db=-40.0,
        edge_silence_min_seconds=0.05,
        edge_padding_seconds=0.12,
    )
    output = Path(str(fitted.items[0]["output_path"]))
    payload = {
        "source": str(source),
        "source_sha256_before": source_hash,
        "source_sha256_after": file_digest(source),
        "source_metrics": wav_metrics(source),
        "output": str(output),
        "output_metrics": wav_metrics(output),
        "report": report,
    }
    report_path = args.output_dir / "validation-report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["source_sha256_before"] != payload["source_sha256_after"]:
        raise RuntimeError("Speech-aware fit modified the source WAV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
