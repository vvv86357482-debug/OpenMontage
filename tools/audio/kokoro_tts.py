"""Kokoro-82M local text-to-speech provider tool (CPU-only).

Generates narration audio using the Kokoro-82M model with local CPU inference.
Outputs WAV at 44100Hz to match the project audio pipeline standard.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Prevent HuggingFace Hub rate-limiting hangs in shared Codespaces.
# This IP gets 429 on cache-verification HEAD requests; model weights are
# already cached locally, so online verification is unnecessary and harmful.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class KokoroTTS(BaseTool):
    name = "kokoro_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "kokoro"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["cmd:ffmpeg", "pip:kokoro", "pip:misaki[en]", "cmd:espeak-ng"]
    install_instructions = (
        "pip install kokoro misaki[en]\n"
        "sudo apt-get install espeak-ng\n"
        "FFmpeg is required for resampling to 44100Hz."
    )
    fallback_tools = []
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "offline",
        "local_cpu",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": False,
        "offline": True,
        "native_audio": True,
    }
    best_for = [
        "free offline narration",
        "batch voiceover generation",
        "consistent synthetic voices across a channel",
    ]
    not_good_for = [
        "voice cloning or custom voice training",
        "languages outside Kokoro's training set",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech"},
            "voice": {
                "type": "string",
                "description": "Kokoro voice ID (e.g. am_fenrir, am_puck, ef_dora)",
                "default": "am_fenrir",
            },
            "output_path": {"type": "string"},
            "speed": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.5,
                "maximum": 2.0,
                "description": "Speech speed multiplier",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1024, vram_mb=0, disk_mb=500, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["generation_failed"])
    idempotency_key_fields = ["text", "voice", "speed"]
    side_effects = ["writes audio file to output_path"]
    user_visible_verification = ["Listen to generated audio for speech clarity and voice match"]

    DEFAULT_VOICE = "am_fenrir"

    CHANNEL_VOICES = {
        "dark-annals": "am_fenrir",
        "crime-ledger": "am_puck",
        "mind-tactics": "ef_dora",
    }

    def get_status(self) -> ToolStatus:
        try:
            from kokoro import KPipeline  # noqa: F401
            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        text = inputs.get("text", "").strip()
        if not text:
            return ToolResult(success=False, error="No text provided")

        voice = inputs.get("voice", self.DEFAULT_VOICE)
        speed = float(inputs.get("speed", 1.0))
        output_path = Path(inputs.get("output_path", "kokoro_output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        try:
            result = self._generate(text, voice, speed, output_path)
        except Exception as exc:
            return ToolResult(success=False, error=f"Kokoro TTS generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = 0.0
        return result

    def _generate(self, text: str, voice: str, speed: float, output_path: Path) -> ToolResult:
        from kokoro import KPipeline
        from misaki import en

        pipeline = KPipeline(lang_code="a")
        voice_tensor = pipeline.load_voice(voice)

        g2p = en.G2P()
        _, tokens = g2p(text)

        audio_chunks: list = []
        for result in pipeline.generate_from_tokens(tokens, voice_tensor):
            audio_chunks.append(result.audio)

        if not audio_chunks:
            return ToolResult(success=False, error="Kokoro produced no audio output")

        audio = torch.cat(audio_chunks, dim=0)

        tmp_wav = output_path.with_suffix(".tmp_24000.wav")
        self._save_wav(tmp_wav, audio, sample_rate=24000)

        if 24000 == 44100:
            tmp_wav.replace(output_path)
        else:
            self._resample_to_44100(tmp_wav, output_path)
            tmp_wav.unlink(missing_ok=True)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": "Kokoro-82M",
                "voice": voice,
                "speed": speed,
                "text_length": len(text),
                "output": str(output_path),
                "format": "wav",
            },
            artifacts=[str(output_path)],
            model="Kokoro-82M",
        )

    @staticmethod
    def _save_wav(path: Path, audio: "torch.Tensor", sample_rate: int) -> None:
        import wave
        import numpy as np

        audio_np = audio.clamp(-1.0, 1.0).cpu().numpy()
        pcm = (audio_np * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def _resample_to_44100(src: Path, dst: Path) -> None:
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ar", "44100",
                "-ac", "1",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
