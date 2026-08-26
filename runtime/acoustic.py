from __future__ import annotations


def acoustic_instruction(
    operation: str, *, pitch_steps: int = 3, speed: float = 1.2, volume: float = 1.0
) -> str:
    if operation == "pitch":
        if pitch_steps == 0 or not -6 <= pitch_steps <= 6:
            raise ValueError("音高步数必须是 -6…-1 或 1…6")
        unit = "step" if abs(pitch_steps) == 1 else "steps"
        return f"shift the pitch by {pitch_steps} {unit}"
    if operation == "speed":
        if not 0.5 <= speed <= 2.0:
            raise ValueError("速度必须在 0.5…2.0")
        return f"adjust the speed to {speed:.1f}"
    if operation == "volume":
        if not 0.3 <= volume <= 2.0:
            raise ValueError("音量必须在 0.3…2.0")
        return f"adjust the volume to {volume:.1f}"
    raise ValueError(f"未知声学操作：{operation}")
