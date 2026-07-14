"""T4x2 concurrent test — Chatterbox TTS worker.

Pinned to cuda:1 by the parent driver via CUDA_VISIBLE_DEVICES=1.
Generates a small real batch of TTS clips and writes a per-worker manifest.
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings
from pathlib import Path

import torch
import torchaudio as ta

warnings.filterwarnings("ignore")

DEVICE = "cuda"
OUT = Path("/kaggle/working/tts_out")
OUT.mkdir(parents=True, exist_ok=True)

SCENES = [
    ("tts_01", "February first, 2009. Jeju Island. A childcare teacher boards a taxi at three AM."),
    ("tts_02", "Park drove a white NF Sonata. CCTV places a matching vehicle near the scene at the critical time."),
    ("tts_03", "The appellate court reviews. Same evidence. Same arguments. Same conclusion. Not guilty again."),
    ("tts_04", "Jeju police form a cold case team in 2016. They re-examine the body. The taxi. The timeline."),
    ("tts_05", "No murder weapon is found. No DNA connects Park. The fibers are not unique. The CCTV is probabilistic."),
    ("tts_06", "The Supreme Court explicitly noted the second taxi possibility. Reasonable doubt. The victim may have boarded a third vehicle."),
]


def load_cb():
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=DEVICE)


def main() -> int:
    t0 = time.time()
    cb = load_cb()
    cb_sr = int(cb.sr)
    load_s = round(time.time() - t0, 2)
    print(f"[TTS cuda:1] Chatterbox loaded in {load_s}s sr={cb_sr}", flush=True)

    results = []
    total_gen = 0.0
    total_aud = 0.0
    for sid, text in SCENES:
        start = time.time()
        wav = cb.generate(text)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen = time.time() - start
        dur = int(wav.shape[-1]) / cb_sr
        p = OUT / f"{sid}.wav"
        ta.save(str(p), wav.cpu(), cb_sr)
        total_gen += gen
        total_aud += dur
        results.append({
            "id": sid, "path": str(p),
            "duration_s": round(dur, 3),
            "gen_s": round(gen, 3),
            "success": True,
        })
        print(f"  {sid}: {gen:.2f}s", flush=True)

    manifest = {
        "role": "tts",
        "device": DEVICE,
        "model_load_s": load_s,
        "total_gen_s": round(total_gen, 2),
        "total_audio_s": round(total_aud, 2),
        "rtf": round(total_aud / max(total_gen, 1e-6), 3),
        "scenes": results,
    }
    (OUT / "worker_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[TTS cuda:1] DONE load={load_s}s gen={round(total_gen,2)}s rtf={manifest['rtf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
