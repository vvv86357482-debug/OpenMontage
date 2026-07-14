"""Build the Chatterbox 270-scene measurement kernel for Kaggle T4.

Reads scene_plans/*.json and emits tools/kaggle/tts_measure/kernel.ipynb
plus kernel-metadata.json. The notebook loads Chatterbox ONCE (singleton),
then synthesizes every scene's narration text on Tesla T4 and records the
real produced duration — no regression, no extrapolation, 100% generated.

This mirrors scripts/measure_kokoro_durations.py's discipline but runs on
Kaggle GPU (Chatterbox needs CUDA; Kokoro ran LOCAL CPU). Voice = "default"
for all three channels (Chatterbox ships a single built-in voice; this matches
the 10-scene compare kernel that used voice="default" for crime-ledger).
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PLANS_DIR = REPO_ROOT / "scene_plans"
OUT_DIR = REPO_ROOT / "tools/kaggle/tts_measure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# channel -> scene_plan file
PLAN_FILES = {
    "crime-ledger": "jeju-cold-case.json",
    "mythology-slavic": "leshy-forest-lord.json",
    "speculative-biology": "bombardier-beetle-proof.json",
}

payload: dict[str, list[list[str]]] = {}
totals: dict[str, int] = {}
for channel, fname in PLAN_FILES.items():
    plan = json.loads((SCENE_PLANS_DIR / fname).read_text())
    rows: list[list[str]] = []
    for s in plan["scenes"]:
        text = s.get("description") or s.get("narration") or ""
        rows.append([s["id"], text])
    payload[channel] = rows
    totals[channel] = len(rows)

print("Payload scene counts:", totals, "grand total:", sum(totals.values()))


def cell(code: str, md: bool = False) -> nbf.NotebookNode:
    if md:
        return nbf.v4.new_markdown_cell(code)
    return nbf.v4.new_code_cell(code)


cells = []

cells.append(cell(
    "# OpenMontage — Chatterbox TTS full measurement (270 scenes, T4)\n"
    "Synthesizes EVERY scene's narration for all three channels on Tesla T4 with "
    "Chatterbox (singleton load) and records the real produced duration. "
    "100% generated, no extrapolation. Self-reporting: always writes a manifest.",
    md=True,
))

cells.append(cell(
    "import os\n"
    "import sys\n"
    "import time\n"
    "import json\n"
    "import torch\n"
    "from pathlib import Path\n"
    "\n"
    "TEST_FORCE_T4 = True\n"
    "GPU_BRANCH = 'unknown'\n"
    "GPU_NAME = 'unknown'\n"
    "\n"
    "if torch.cuda.is_available():\n"
    "    cc = torch.cuda.get_device_properties(0).major\n"
    "    GPU_NAME = torch.cuda.get_device_name(0)\n"
    "    if TEST_FORCE_T4:\n"
    "        if cc < 7:\n"
    "            print(f'[FATAL] T4 required, got CC={cc} ({GPU_NAME}). Aborting.')\n"
    "            sys.exit(1)\n"
    "        GPU_BRANCH = 't4_or_better'\n"
    "    elif cc >= 7:\n"
    "        GPU_BRANCH = 't4_or_better'\n"
    "    elif cc == 6:\n"
    "        GPU_BRANCH = 'p100'\n"
    "    else:\n"
    "        GPU_BRANCH = 'old_gpu'\n"
    "else:\n"
    "    print('No GPU available.')\n"
    "    sys.exit(1)\n"
    "\n"
    "print(f'GPU: {GPU_NAME} (CC={cc}), branch={GPU_BRANCH}')"
))

cells.append(cell(
    "import subprocess, sys\n"
    "\n"
    "def pip_install(no_deps, *pkgs):\n"
    "    cmd = [sys.executable, '-m', 'pip', 'install', '-q']\n"
    "    if no_deps:\n"
    "        cmd.append('--no-deps')\n"
    "    cmd.extend(pkgs)\n"
    "    print(f'  $ pip install {\"--no-deps \" if no_deps else \"\"}' + ' '.join(pkgs), flush=True)\n"
    "    try:\n"
    "        subprocess.check_call(cmd)\n"
    "        return True\n"
    "    except subprocess.CalledProcessError as e:\n"
    "        print(f'  PIP FAILED: {e}')\n"
    "        return False\n"
    "\n"
    "print('Installing chatterbox-tts with --no-deps (keep Kaggle numpy 1.26 / torch ABI intact)...', flush=True)\n"
    "pip_install(True, 'chatterbox-tts')\n"
    "print('Installing chatterbox runtime deps (--no-deps, no numpy change)...', flush=True)\n"
    "pip_install(True, 'resemble-perth>=1.0.0', 'conformer==0.3.2', 'spacy-pkuseg',\n"
    "           'pykakasi==2.3.0', 'pyloudnorm', 'omegaconf', 's3tokenizer',\n"
    "           'librosa==0.11.0', 'gradio==6.8.0')\n"
    "print('Pinning transformers/diffusers to chatterbox requirements (--no-deps)...', flush=True)\n"
    "pip_install(True, 'transformers==5.2.0', 'diffusers==0.29.0')\n"
    "print('Install step done.', flush=True)"
))

cells.append(cell(
    "import torch, warnings, time, traceback\n"
    "warnings.filterwarnings('ignore')\n"
    "device = 'cuda'\n"
    "\n"
    "print('Loading Chatterbox (singleton, default voice)...', flush=True)\n"
    "cb = None; cb_sr = 24000\n"
    "cb_load_s = None\n"
    "try:\n"
    "    t0 = time.time()\n"
    "    from chatterbox.tts import ChatterboxTTS\n"
    "    cb = ChatterboxTTS.from_pretrained(device=device)\n"
    "    cb_sr = int(cb.sr)\n"
    "    cb_load_s = round(time.time() - t0, 2)\n"
    "    print(f'  Chatterbox loaded in {cb_load_s}s, sr={cb_sr}', flush=True)\n"
    "except Exception as e:\n"
    "    print('  Chatterbox load FAILED:', repr(e), flush=True)\n"
    "    traceback.print_exc()\n"
    "    sys.exit(1)"
))

# Embed the SCENES payload as a JSON literal.
payload_literal = json.dumps(payload, indent=2, ensure_ascii=False)
cells.append(cell(
    "# channel -> list of [scene_id, narration_text]\n"
    "SCENES = " + payload_literal + "\n"
    "\n"
    "OUT = Path('/kaggle/working/measure')\n"
    "OUT.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "def save_wav(path, wav, sr):\n"
    "    import torchaudio as ta\n"
    "    ta.save(str(path), wav.cpu(), sr)\n"
    "\n"
    "all_rows = {}\n"
    "channel_stats = {}\n"
    "errors = []\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.reset_peak_memory_stats()\n"
    "\n"
    "for channel, rows in SCENES.items():\n"
    "    cdir = OUT / channel\n"
    "    cdir.mkdir(parents=True, exist_ok=True)\n"
    "    gen_total = 0.0\n"
    "    aud_total = 0.0\n"
    "    scene_rows = []\n"
    "    for sid, text in rows:\n"
    "        t0 = time.time()\n"
    "        try:\n"
    "            wav = cb.generate(text)\n"
    "            if torch.cuda.is_available():\n"
    "                torch.cuda.synchronize()\n"
    "            gen = time.time() - t0\n"
    "            dur = int(wav.shape[-1]) / cb_sr\n"
    "            p = cdir / f'{sid}.wav'\n"
    "            save_wav(p, wav, cb_sr)\n"
    "            gen_total += gen\n"
    "            aud_total += dur\n"
    "            scene_rows.append({'id': sid, 'dur_s': round(dur, 3),\n"
    "                                'gen_s': round(gen, 3), 'bytes': p.stat().st_size})\n"
    "            print(f'  [{channel}] {sid}: gen={gen:.2f}s dur={dur:.2f}s', flush=True)\n"
    "        except Exception as e:\n"
    "            errors.append(f'{channel} {sid}: ' + repr(e))\n"
    "            scene_rows.append({'id': sid, 'error': repr(e)})\n"
    "    channel_stats[channel] = {\n"
    "        'scenes': len(rows),\n"
    "        'total_gen_s': round(gen_total, 2),\n"
    "        'total_audio_s': round(aud_total, 2),\n"
    "        'rtf': round(aud_total / max(gen_total, 1e-6), 3),\n"
    "    }\n"
    "    all_rows[channel] = scene_rows\n"
    "    print(f'  channel {channel}: gen={gen_total:.1f}s audio={aud_total:.1f}s rtf={channel_stats[channel][\"rtf\"]}', flush=True)\n"
    "\n"
    "manifest = {\n"
    "    'project_id': 'tts-measure-chatterbox-270',\n"
    "    'gpu_name': GPU_NAME,\n"
    "    'gpu_branch': GPU_BRANCH,\n"
    "    'voice': 'default',\n"
    "    'model_load_s': cb_load_s,\n"
    "    'vram_peak_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,\n"
    "    'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),\n"
    "    'errors': errors,\n"
    "    'channel_stats': channel_stats,\n"
    "    'channels': all_rows,\n"
    "}\n"
    "(OUT / 'measure_manifest.json').write_text(json.dumps(manifest, indent=2))\n"
    "print(json.dumps(manifest, indent=2))"
))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

nbpath = OUT_DIR / "kernel.ipynb"
nbf.write(nb, nbpath)
print("Wrote", nbpath)

meta = {
    "id": "forts845/openmontage-tts-measure-chatterbox-270",
    "title": "OpenMontage Chatterbox 270-scene TTS measurement",
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "machine_shape": "NvidiaTeslaT4",
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("Wrote", OUT_DIR / "kernel-metadata.json")
