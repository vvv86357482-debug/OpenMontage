"""Build the Chatterbox v2 measurement kernel — FINAL real voices, Kaggle T4.

Re-measures all 270 scenes (90 x 3 scripts) with the APPROVED final CC0 voices,
cloned via Chatterbox audio_prompt_path (NOT the shared default voice). Mirrors
the v1 discipline: 100% generated, no regression/extrapolation.

Per-channel final voice (approved 2026-07-11):
  crime-ledger       -> simon_evers            (LibriVox reader 1255, CC0)
  mythology-slavic   -> padraig_o'hiceadha-lyrical (LibriVox reader 2588, CC0)
  speculative-biology-> nicholas_james_bridgewater (LibriVox reader 1618, CC0)
References are real CC0 flacs from OwenTyme/voice-zero (downloaded at runtime).
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PLANS_DIR = REPO_ROOT / "scene_plans"
OUT_DIR = REPO_ROOT / "tools/kaggle/tts_measure_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAN_FILES = {
    "crime-ledger": "jeju-cold-case.json",
    "mythology-slavic": "leshy-forest-lord.json",
    "speculative-biology": "bombardier-beetle-proof.json",
}

BASE = "https://raw.githubusercontent.com/OwenTyme/voice-zero/main/voices/"

# Final approved voice per channel: (ref flac name, attribution)
CHANNEL_REFS = {
    "crime-ledger": (
        "simon_evers",
        "LibriVox reader 1255 — Celebration of Dialects and Accents, Vol 2, Track 18 (English, Received Pronunciation)",
    ),
    "mythology-slavic": (
        "padraig_o'hiceadha-lyrical",
        "LibriVox reader 2588 — Celebration of Dialects and Accents, Vol 1, Track 2 (Irish, lyrical)",
    ),
    "speculative-biology": (
        "nicholas_james_bridgewater",
        "LibriVox reader 1618 — Celebration of Dialects and Accents, Vol 2, Track 13 (English, Mid-Atlantic, documentary)",
    ),
}

LICENSE = "CC0 1.0 Universal (public-domain dedication) via OwenTyme/voice-zero; source reading public domain on LibriVox"

payload: dict[str, list[list[str]]] = {}
for channel, fname in PLAN_FILES.items():
    plan = json.loads((SCENE_PLANS_DIR / fname).read_text())
    payload[channel] = [[s["id"], s.get("description") or s.get("narration") or ""] for s in plan["scenes"]]
print("Scene counts:", {k: len(v) for k, v in payload.items()}, "grand", sum(len(v) for v in payload.values()))


def cell(code: str, md: bool = False) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(code) if md else nbf.v4.new_code_cell(code)


cells = []
cells.append(cell(
    "# OpenMontage — Chatterbox v2 measurement (FINAL real CC0 voices, 270 scenes, T4)\n"
    "Clones Chatterbox per channel from the APPROVED real CC0 reference (audio_prompt_path), "
    "not the shared default voice. 100% generated, no extrapolation. Self-reporting.",
    md=True))

cells.append(cell(
    "import os, sys, time, json, torch\nfrom pathlib import Path\n"
    "TEST_FORCE_T4=True; GPU_BRANCH='unknown'; GPU_NAME='unknown'\n"
    "if torch.cuda.is_available():\n"
    "    cc=torch.cuda.get_device_properties(0).major; GPU_NAME=torch.cuda.get_device_name(0)\n"
    "    if TEST_FORCE_T4:\n"
    "        if cc<7: print(f'[FATAL] T4 required, got CC={cc}'); sys.exit(1)\n"
    "        GPU_BRANCH='t4_or_better'\n"
    "    elif cc>=7: GPU_BRANCH='t4_or_better'\n"
    "    elif cc==6: GPU_BRANCH='p100'\n"
    "    else: GPU_BRANCH='old_gpu'\n"
    "else: print('No GPU'); sys.exit(1)\n"
    "print(f'GPU: {GPU_NAME} (CC={cc}) branch={GPU_BRANCH}')"))

cells.append(cell(
    "import subprocess, sys\n"
    "def pip_install(no_deps,*pkgs):\n"
    "    cmd=[sys.executable,'-m','pip','install','-q']\n"
    "    if no_deps: cmd.append('--no-deps')\n"
    "    cmd.extend(pkgs); print('  $ pip install '+' '.join(pkgs),flush=True)\n"
    "    return subprocess.call(cmd)==0\n"
    "pip_install(True,'chatterbox-tts')\n"
    "pip_install(True,'resemble-perth>=1.0.0','conformer==0.3.2','spacy-pkuseg','pykakasi==2.3.0','pyloudnorm','omegaconf','s3tokenizer','librosa==0.11.0','gradio==6.8.0')\n"
    "pip_install(True,'transformers==5.2.0','diffusers==0.29.0')\n"
    "print('install done')"))

cells.append(cell(
    "import torch, warnings, time, traceback\nwarnings.filterwarnings('ignore'); device='cuda'\n"
    "print('Loading Chatterbox (singleton)...',flush=True)\ncb=None; cb_sr=24000\n"
    "try:\n"
    "    t0=time.time(); from chatterbox.tts import ChatterboxTTS\n"
    "    cb=ChatterboxTTS.from_pretrained(device=device); cb_sr=int(cb.sr)\n"
    "    print(f'  loaded in {round(time.time()-t0,2)}s sr={cb_sr}',flush=True)\n"
    "except Exception as e: print('load FAILED',repr(e)); traceback.print_exc(); sys.exit(1)"))

# Embed SCENES + CHANNEL_REFS + LICENSE
refs_payload = {ch: {"name": n, "url": BASE + n + ".flac", "attribution": a} for ch, (n, a) in CHANNEL_REFS.items()}

cells.append(cell(
    "SCENES = " + json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    "CHANNEL_REFS = " + json.dumps(refs_payload, indent=2, ensure_ascii=False) + "\n"
    "LICENSE = " + json.dumps(LICENSE) + "\n"
    "OUT = Path('/kaggle/working/measure_v2'); OUT.mkdir(parents=True, exist_ok=True)\n"
    "import urllib.request\n"
    "import torchaudio as ta\n"
    "def download(url, path):\n"
    "    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})\n"
    "    data=urllib.request.urlopen(req, timeout=60).read(); open(path,'wb').write(data); return len(data)\n"
    "def ref_to_wav(flac, wav, sr=24000):\n"
    "    w,sr0=ta.load(flac)\n"
    "    if w.shape[0]>1: w=w.mean(0,keepdim=True)\n"
    "    if sr0!=sr: w=ta.functional.resample(w,sr0,sr)\n"
    "    ta.save(str(wav), w, sr)\n"
    "all_rows={}; channel_stats={}; voice_meta={}; errors=[]\n"
    "if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()\n"
    "for channel, rows in SCENES.items():\n"
    "    ref=CHANNEL_REFS[channel]; cdir=OUT/channel; cdir.mkdir(parents=True, exist_ok=True)\n"
    "    ref_flac=cdir/(ref['name']+'.flac'); ref_wav=cdir/(ref['name']+'_ref.wav')\n"
    "    try:\n"
    "        download(ref['url'], ref_flac); ref_to_wav(ref_flac, ref_wav)\n"
    "    except Exception as e:\n"
    "        errors.append(f'{channel} ref download: '+repr(e)); print('REF FAIL',channel,repr(e),flush=True); continue\n"
    "    print(f'[{channel}] voice={ref[\"name\"]} ref ready',flush=True)\n"
    "    gen_total=0.0; aud_total=0.0; scene_rows=[]\n"
    "    for sid, text in rows:\n"
    "        t0=time.time()\n"
    "        try:\n"
    "            wav=cb.generate(text, audio_prompt_path=str(ref_wav))\n"
    "            if torch.cuda.is_available(): torch.cuda.synchronize()\n"
    "            gen=time.time()-t0; dur=int(wav.shape[-1])/cb_sr\n"
    "            p=cdir/f'{sid}.wav'; ta.save(str(p), wav.cpu(), cb_sr)\n"
    "            gen_total+=gen; aud_total+=dur\n"
    "            scene_rows.append({'id':sid,'dur_s':round(dur,3),'gen_s':round(gen,3),'bytes':p.stat().st_size})\n"
    "        except Exception as e:\n"
    "            errors.append(f'{channel} {sid}: '+repr(e)); scene_rows.append({'id':sid,'error':repr(e)})\n"
    "    channel_stats[channel]={'scenes':len(rows),'total_gen_s':round(gen_total,2),'total_audio_s':round(aud_total,2),'rtf':round(aud_total/max(gen_total,1e-6),3)}\n"
    "    voice_meta[channel]={'voice':ref['name'],'attribution':ref['attribution'],'license':LICENSE,'ref_url':ref['url']}\n"
    "    all_rows[channel]=scene_rows\n"
    "    print(f'  channel {channel}: gen={gen_total:.1f}s audio={aud_total:.1f}s rtf={channel_stats[channel][\"rtf\"]}',flush=True)\n"
    "manifest={'project_id':'tts-measure-chatterbox-v2-final-voices','gpu_name':GPU_NAME,'gpu_branch':GPU_BRANCH,\n"
    "          'model_load_s': None,'vram_peak_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,\n"
    "          'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'errors':errors,'voice_meta':voice_meta,\n"
    "          'channel_stats':channel_stats,'channels':all_rows}\n"
    "(OUT/'measure_manifest.json').write_text(json.dumps(manifest, indent=2))\n"
    "print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='scenes'} for k,v in channel_stats.items()}, indent=2))\n"
    "print('VOICE META:'); print(json.dumps(voice_meta, indent=2))"))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbpath = OUT_DIR / "kernel.ipynb"
nbf.write(nb, nbpath)
meta = {
    "id": "forts845/openmontage-tts-measure-v2-final-voices",
    "title": "OpenMontage TTS measure v2 final voices",
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
print("Wrote", nbpath)
