"""Deterministic tensor and waveform diagnostics for generation failures.

The helpers in this module intentionally do not classify or modify audio.  They
record enough evidence to compare the official CLI, Worker memory modes and
decoder placements without hiding a bad generation behind post-processing.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any

import torch


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe statistics without changing ``value``."""
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    flat = tensor.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "numel": int(value.numel()),
        "finite_fraction": float(finite.float().mean().item()) if flat.numel() else 1.0,
        "sha256_float32": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
    }
    if finite_values.numel() == 0:
        result.update({"min": None, "max": None, "mean": None, "std": None, "rms": None})
        return result
    result.update(
        {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "rms": float(torch.sqrt(torch.mean(finite_values.square())).item()),
        }
    )
    return result


def waveform_stats(value: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    """Extend :func:`tensor_stats` with speech/noise inspection measurements."""
    waveform = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    result = tensor_stats(value)
    result["sample_rate"] = int(sample_rate)
    result["duration_seconds"] = float(waveform.numel() / sample_rate)
    if waveform.numel() == 0:
        result.update(
            {
                "dc_offset": None,
                "zero_crossing_rate": None,
                "clipped_fraction": None,
                "spectral_flatness_mean": None,
                "spectral_centroid_hz_mean": None,
            }
        )
        return result

    finite_waveform = torch.nan_to_num(waveform)
    result["dc_offset"] = float(finite_waveform.mean().item())
    result["zero_crossing_rate"] = (
        float(((finite_waveform[1:] * finite_waveform[:-1]) < 0).float().mean().item())
        if finite_waveform.numel() > 1
        else 0.0
    )
    result["clipped_fraction"] = float((finite_waveform.abs() >= 1.0).float().mean().item())

    n_fft = min(1024, _largest_power_of_two(finite_waveform.numel()))
    if n_fft < 16:
        result["spectral_flatness_mean"] = None
        result["spectral_centroid_hz_mean"] = None
        return result
    spectrum = torch.stft(
        finite_waveform,
        n_fft=n_fft,
        hop_length=max(1, n_fft // 4),
        window=torch.hann_window(n_fft),
        return_complex=True,
        center=False,
    ).abs().square().clamp_min(1e-12)
    flatness = spectrum.log().mean(dim=0).exp() / spectrum.mean(dim=0)
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).unsqueeze(1)
    centroid = (spectrum * frequencies).sum(dim=0) / spectrum.sum(dim=0)
    result["spectral_flatness_mean"] = float(flatness.mean().item())
    result["spectral_centroid_hz_mean"] = float(centroid.mean().item())
    return result


def compare_waveforms(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Compare two decoder results over their common sample span."""
    left = reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    right = candidate.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    common = min(left.numel(), right.numel())
    if common == 0:
        return {"common_samples": common, "max_abs_delta": None, "rms_delta": None, "snr_db": None}
    left = left[:common]
    right = right[:common]
    delta = right - left
    signal_rms = torch.sqrt(torch.mean(left.square()))
    delta_rms = torch.sqrt(torch.mean(delta.square()))
    snr_db = float("inf") if delta_rms.item() == 0 else 20.0 * math.log10(
        max(signal_rms.item(), 1e-20) / delta_rms.item()
    )
    return {
        "common_samples": common,
        "length_delta_samples": int(candidate.numel() - reference.numel()),
        "max_abs_delta": float(delta.abs().max().item()),
        "rms_delta": float(delta_rms.item()),
        "snr_db": snr_db,
    }


def character_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return punctuation-insensitive character error evidence for ASR verification."""
    normalized_reference = _normalize_transcript(reference)
    normalized_hypothesis = _normalize_transcript(hypothesis)
    distance = _levenshtein(normalized_reference, normalized_hypothesis)
    denominator = max(1, len(normalized_reference))
    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "edit_distance": distance,
        "reference_characters": len(normalized_reference),
        "cer": float(distance / denominator),
    }


def _largest_power_of_two(value: int) -> int:
    if value < 1:
        return 0
    return 1 << (value.bit_length() - 1)


def _normalize_transcript(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _levenshtein(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_character in enumerate(right, start=1):
        current = [row]
        for column, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
