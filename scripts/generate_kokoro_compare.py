"""Generate the 10 Kokoro .wav clips for the jeju-cold-case comparison lines.

Local CPU Kokoro (am_puck = crime-ledger voice). Saves at 24000 Hz to match
the Chatterbox clips' sample rate for an apples-to-apples listen. These are
the same 10 narration lines used in the Kaggle Chatterbox comparison.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
import wave
import numpy as np
import torch
from pathlib import Path
from kokoro import KPipeline
from misaki import en

OUT = Path("tools/kaggle/tts_compare/results")
OUT.mkdir(parents=True, exist_ok=True)

VOICE = "am_puck"
SCENES = [
    ("scene_01", "February first, 2009. Jeju Island. A childcare teacher boards a taxi at three AM."),
    ("scene_09", "Park drove a white NF Sonata. CCTV places a matching vehicle near the scene at the critical time."),
    ("scene_18", "The appellate court reviews. Same evidence. Same arguments. Same conclusion. Not guilty again."),
    ("scene_27", "Jeju police form a cold case team in 2016. They re-examine the body. The taxi. The timeline."),
    ("scene_36", "No murder weapon is found. No DNA connects Park. The fibers are not unique. The CCTV is probabilistic."),
    ("scene_45", "The Supreme Court explicitly noted the second taxi possibility. Reasonable doubt. The victim may have boarded a third vehicle."),
    ("scene_54", "Park left Jeju after the first investigation. He lived elsewhere for nine years. The case went cold."),
    ("scene_63", "Jeju's Memories of Murder. The nickname references a Korean film about an unsolved killing. Parallels are unavoidable."),
    ("scene_72", "The family receives no justice. No conviction. No apology. Just a Supreme Court document explaining why evidence fell short."),
    ("scene_81", "The court's reasoning is precise. Circumstantial evidence can convict. But it must exclude all reasonable alternatives."),
]

def _save_wav(path: Path, audio: "torch.Tensor", sample_rate: int) -> None:
    audio_np = audio.clamp(-1.0, 1.0).numpy().astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_np.tobytes())


t0 = time.time()
pipeline = KPipeline(lang_code="a")
g2p = en.G2P()
print(f"Kokoro pipeline ready in {round(time.time()-t0,1)}s", flush=True)

total_gen = 0.0
total_audio = 0.0
for sid, text in SCENES:
    vt = pipeline.load_voice(VOICE)
    _, tokens = g2p(text)
    g = time.time()
    chunks = [r.audio for r in pipeline.generate_from_tokens(tokens, vt)]
    gen = time.time() - g
    audio = torch.cat(chunks, dim=0)
    dur = len(audio) / 24000
    _save_wav(OUT / f"{sid}_kokoro.wav", audio.cpu(), 24000)
    total_gen += gen
    total_audio += dur
    print(f"  {sid}: gen={gen:.2f}s dur={dur:.2f}s -> {OUT / f'{sid}_kokoro.wav'}", flush=True)

print(f"\nKokoro local CPU: total_gen={total_gen:.2f}s total_audio={total_audio:.2f}s "
      f"RTF={total_audio/max(total_gen,1e-6):.3f} (voice={VOICE})")
