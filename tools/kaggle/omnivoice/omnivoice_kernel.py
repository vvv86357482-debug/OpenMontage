"""
OmniVoice zero-shot voice-cloning kernel (Kaggle T4).

Known-good constraints — do NOT "fix" any of these:
  - The exact install line below was missing once and caused a real
    ModuleNotFoundError; do not trim packages from it.
  - Reference voice simon_evers.flac (CC0, OwenTyme/voice-zero) is downloaded
    into the kernel every run; never assume it is cached.
  - machine_shape: NvidiaTeslaT4 in kernel-metadata.json (not "gpu_type" /
    "hardware_tier", which are not real Kaggle API fields).
  - Output must be 24 kHz / 16-bit PCM, verified with ffprobe inside the
    kernel. Mismatch = hard fail.
Outputs go to /kaggle/working/output (kernel output), never /tmp.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUTPUT_DIR = "/kaggle/working/output"
REF_URL = (
    "https://raw.githubusercontent.com/OwenTyme/voice-zero/main/voices/simon_evers.flac"
)
REF_PATH = "/kaggle/working/ref_voices/simon_evers.flac"
EXPECTED_SAMPLE_RATE = 24000

SMOKE_NARRATION = (
    "Every machine remembers its ancestors. "
    "Tonight, we listen to three voices the industry tried to forget."
)


def log(msg):
    print(f"[omnivoice] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_check(wav_path):
    """Real ffprobe verification: expect 24 kHz / 16-bit PCM."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=codec_name,sample_rate,bits_per_raw_sample:format=duration,size",
            "-of", "json", wav_path,
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return {"ffprobe_ok": False, "error": out.stderr.strip()}
    stream = json.loads(out.stdout)["streams"][0]
    fmt = json.loads(out.stdout)["format"]
    ok = (
        stream.get("codec_name") == "pcm_s16le"
        and int(stream.get("sample_rate", 0)) == EXPECTED_SAMPLE_RATE
    )
    return {
        "ffprobe_ok": ok,
        "codec_name": stream.get("codec_name"),
        "sample_rate": stream.get("sample_rate"),
        "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
        "duration_seconds": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "expected": {"codec": "pcm_s16le", "sample_rate": EXPECTED_SAMPLE_RATE},
    }


def main():
    # Exact known-good install line — do not trim (missing it once broke this kernel).
    subprocess.run(
        "pip install -q omnivoice imagehash librosa soundfile",
        shell=True, check=True,
    )

    import imagehash
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    from omnivoice import OmniVoice

    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    if not os.path.isfile(REF_PATH):
        log(f"downloading reference voice: {REF_URL}")
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "-o", REF_PATH, REF_URL], check=True
        )
    ref_sha = sha256(REF_PATH)

    manifest = {
        "model_id": "k2-fsa/OmniVoice",
        "ref_voice_url": REF_URL,
        "ref_voice_sha256": ref_sha,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "num_step": 16,
    }
    log(f"GPU: {manifest['gpu_name']}")

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    audio = model.generate(
        text=SMOKE_NARRATION,
        ref_audio=REF_PATH,
        num_step=manifest["num_step"],
    )
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_path = os.path.join(OUTPUT_DIR, "smoke_narration.wav")
    sf.write(wav_path, np.asarray(audio[0]), EXPECTED_SAMPLE_RATE, subtype="PCM_16")

    probe = ffprobe_check(wav_path)
    duration = librosa.get_duration(path=wav_path)

    mel_png = os.path.join(OUTPUT_DIR, "smoke_narration_mel.png")
    mel = librosa.feature.melspectrogram(y=np.asarray(audio[0]), sr=EXPECTED_SAMPLE_RATE)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm = ((mel_db - mel_db.min()) / max(float(np.ptp(mel_db)), 1e-6) * 255).astype(np.uint8)
    Image.fromarray(norm.T[::-1]).save(mel_png)
    voice_print = str(imagehash.phash(Image.open(mel_png)))

    manifest.update(
        {
            "narration_text": SMOKE_NARRATION,
            "output_wav": wav_path,
            "librosa_duration_seconds": round(duration, 3),
            "elapsed_generation_seconds": round(elapsed, 2),
            "peak_vram_mb": round(peak_mem_mb, 1),
            "voice_print_phash": voice_print,
            **probe,
        }
    )
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"manifest: {json.dumps(manifest)}")

    if not probe.get("ffprobe_ok"):
        log(f"HARD FAIL: output failed 24kHz/16-bit PCM gate: {json.dumps(probe)}")
        sys.exit(2)
    log("GATE PASSED")


if __name__ == "__main__":
    main()
