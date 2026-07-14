"""Generate per-channel Chatterbox voice-clone REFERENCE clips from Kokoro.

Chatterbox has one built-in voice. To give each channel a distinct identity
(matching the personas in styles/<channel>.yaml), we clone Chatterbox from a
Kokoro reference voice — the same distinct identity set the project already
established (am_puck / am_fenrir / ef_dora). Kokoro runs LOCAL CPU (offline);
these short references are then used as Chatterbox `audio_prompt_path` on T4.

Mapping (fixes the prior am_fenrir collision where mythology-slavic AND
speculative-biology shared one voice):
  crime-ledger       -> am_puck   (Cold Case Detective: clinical, restrained male)
  mythology-slavic   -> am_fenrir (Folklore Scholar: resonant, mystical male)
  speculative-biology-> ef_dora   (Nature Documentarian: warm, Attenborough-adjacent female)

Output: 24kHz mono WAV (matches Chatterbox sr=24000) under tools/kaggle/voice_refs/.
"""
from __future__ import annotations

import os
import time
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from kokoro import KPipeline
from misaki import en

OUT = Path(__file__).resolve().parent.parent / "tools/kaggle/voice_refs"
OUT.mkdir(parents=True, exist_ok=True)

# channel -> (kokoro clone-source voice, reference utterance)
REFS = {
    "crime-ledger": (
        "am_puck",
        "The case file shows the timeline does not hold. We examine the evidence, "
        "then we ask what the record still leaves unresolved.",
    ),
    "mythology-slavic": (
        "am_fenrir",
        "The old chronicle names the spirit of the forest. We read the source, "
        "then we wonder what the old ways still teach us today.",
    ),
    "speculative-biology": (
        "ef_dora",
        "The organism shows a adaptation to its niche. We state the observed fact, "
        "then we consider the evolutionary pressure that shaped it.",
    ),
}


def save_wav(path: Path, audio: "torch.Tensor", sr: int) -> None:
    pcm = (audio.clamp(-1.0, 1.0).cpu().numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    t0 = time.time()
    pipeline = KPipeline(lang_code="a")
    g2p = en.G2P()
    print(f"Kokoro pipeline ready in {round(time.time()-t0,1)}s", flush=True)

    for channel, (voice, text) in REFS.items():
        vt = pipeline.load_voice(voice)
        _, toks = g2p(text)
        chunks = [r.audio for r in pipeline.generate_from_tokens(toks, vt)]
        audio = torch.cat(chunks, dim=0)
        dur = len(audio) / 24000
        out = OUT / f"{channel}_ref.wav"
        save_wav(out, audio.cpu(), 24000)
        print(f"  {channel}: voice={voice} dur={dur:.2f}s -> {out}", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
