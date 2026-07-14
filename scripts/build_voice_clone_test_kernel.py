"""Build the Chatterbox voice-clone candidate TEST kernel (Kaggle T4).

For each of 9 real CC0 candidate voices (3 per channel), clone Chatterbox from
the candidate's reference FLAC (downloaded from OwenTyme/voice-zero, CC0) and
generate ONE line from that channel's actual script. Produces per-candidate test
WAVs + a manifest with source/license/test-line so the human can pick finals.

Candidates are real human CC0 voices (no double-synthesis). License per file is
CC0 1.0 (repo LICENSE.md: voices/ dir always holds CC0; sourced from LibriVox,
whose readings are public domain).
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tools/kaggle/voice_clone_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/OwenTyme/voice-zero/main/voices/"

TEST_LINES = {
    "crime-ledger": "February first, 2009. Jeju Island. A childcare teacher boards a taxi at three AM.",
    "mythology-slavic": "Deep in the Russian forest, something watches. Something ancient. Something conscious.",
    "speculative-biology": "In the Australian outback, a beetle defends itself with chemistry. Explosive chemistry. One milligram",
}

# name -> (flac_file, librivox_source)
CANDIDATES = {
    "crime-ledger": [
        ("david_wales", "LibriVox reader 6454 — Five Tales by John Galsworthy, Track 1"),
        ("cori_samuel", "LibriVox reader 92 — Black Beauty (version 2), Track 1"),
        ("simon_evers", "LibriVox reader 1255 — Celebration of Dialects and Accents Vol 2, Track 18 (English RP)"),
    ],
    "mythology-slavic": [
        ("padraig_o'hiceadha-lyrical", "LibriVox reader 2588 — Celebration of Dialects and Accents Vol 1, Track 2 (Irish, lyrical)"),
        ("caden_vaughn_clegg-gravel", "LibriVox reader 6574 — Frankenstein (version 3), Track 13 (gravelly)"),
        ("andy", "LibriVox reader 2262 — Zadig or the Book of Fate, Track 11 (Scottish)"),
    ],
    "speculative-biology": [
        ("greg_golding", "LibriVox reader 8222 — History of England..., Track 64 (American)"),
        ("nicholas_james_bridgewater", "LibriVox reader 1618 — Celebration of Dialects Vol 2, Track 13 (English Mid-Atlantic)"),
        ("clayton_j_smith", "LibriVox reader 487 — Paradise Lost, Track 9 (American)"),
    ],
}

LICENSE = "CC0 1.0 Universal (public domain dedication) — OwenTyme/voice-zero, sourced from LibriVox (public-domain reading)"


def cell(code: str, md: bool = False) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(code) if md else nbf.v4.new_code_cell(code)


cells = []
cells.append(cell(
    "# OpenMontage — Chatterbox voice-clone candidate TEST (9 real CC0 voices)\n"
    "Clones Chatterbox from each real CC0 reference (OwenTyme/voice-zero) and generates\n"
    "ONE line from that channel's actual script. Human picks finals. Self-reporting.",
    md=True))

cells.append(cell(
    "import os\nimport sys\nimport time\nimport json\nimport torch\nfrom pathlib import Path\n"
    "TEST_FORCE_T4 = True\nGPU_BRANCH='unknown'; GPU_NAME='unknown'\n"
    "if torch.cuda.is_available():\n"
    "    cc=torch.cuda.get_device_properties(0).major; GPU_NAME=torch.cuda.get_device_name(0)\n"
    "    if TEST_FORCE_T4:\n"
    "        if cc<7: print(f'[FATAL] T4 required, got CC={cc}'); sys.exit(1)\n"
    "        GPU_BRANCH='t4_or_better'\n"
    "    elif cc>=7: GPU_BRANCH='t4_or_better'\n"
    "    elif cc==6: GPU_BRANCH='p100'\n"
    "    else: GPU_BRANCH='old_gpu'\n"
    "else:\n"
    "    print('No GPU'); sys.exit(1)\n"
    "print(f'GPU: {GPU_NAME} (CC={cc}) branch={GPU_BRANCH}')"))

cells.append(cell(
    "import subprocess, sys\n"
    "def pip_install(no_deps,*pkgs):\n"
    "    cmd=[sys.executable,'-m','pip','install','-q']\n"
    "    if no_deps: cmd.append('--no-deps')\n"
    "    cmd.extend(pkgs)\n"
    "    print('  $ pip install '+' '.join(pkgs),flush=True)\n"
    "    return subprocess.call(cmd)==0\n"
    "pip_install(True,'chatterbox-tts')\n"
    "pip_install(True,'resemble-perth>=1.0.0','conformer==0.3.2','spacy-pkuseg','pykakasi==2.3.0','pyloudnorm','omegaconf','s3tokenizer','librosa==0.11.0','gradio==6.8.0')\n"
    "pip_install(True,'transformers==5.2.0','diffusers==0.29.0')\n"
    "print('install done')"))

cells.append(cell(
    "import torch, warnings, time, traceback\nwarnings.filterwarnings('ignore')\ndevice='cuda'\n"
    "print('Loading Chatterbox (singleton)...',flush=True)\ncb=None; cb_sr=24000\n"
    "try:\n"
    "    t0=time.time(); from chatterbox.tts import ChatterboxTTS\n"
    "    cb=ChatterboxTTS.from_pretrained(device=device); cb_sr=int(cb.sr)\n"
    "    print(f'  loaded in {round(time.time()-t0,2)}s sr={cb_sr}',flush=True)\n"
    "except Exception as e:\n"
    "    print('load FAILED',repr(e)); traceback.print_exc(); sys.exit(1)"))

# Build the candidate payload literal
payload = []
for ch, cands in CANDIDATES.items():
    for fname, src in cands:
        payload.append({
            "channel": ch,
            "name": fname,
            "url": BASE + fname + ".flac",
            "source": src,
            "test_line": TEST_LINES[ch],
        })

cells.append(cell(
    "BASE=None  # set below\n"
    "CANDIDATES = " + json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    "LICENSE = " + json.dumps(LICENSE) + "\n"
    "TEST_LINES = " + json.dumps(TEST_LINES, indent=2, ensure_ascii=False) + "\n"
    "OUT = Path('/kaggle/working/clone_test'); OUT.mkdir(parents=True, exist_ok=True)\n"
    "import urllib.request\n"
    "def download(url, path):\n"
    "    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})\n"
    "    data=urllib.request.urlopen(req, timeout=60).read()\n"
    "    open(path,'wb').write(data)\n"
    "    return len(data)\n"
    "def to_wav(flac_path, wav_path, sr=24000):\n"
    "    import torchaudio as ta\n"
    "    w,sr0=ta.load(flac_path)\n"
    "    if w.shape[0]>1: w=w.mean(0,keepdim=True)\n"
    "    if sr0!=sr: w=ta.functional.resample(w,sr0,sr)\n"
    "    ta.save(str(wav_path), w, sr)\n"
    "rows=[]\n"
    "for c in CANDIDATES:\n"
    "    cdir=OUT/c['channel']; cdir.mkdir(parents=True, exist_ok=True)\n"
    "    ref_flac=cdir/(c['name']+'.flac'); ref_wav=cdir/(c['name']+'_ref.wav')\n"
    "    test_wav=cdir/(c['name']+'_test.wav')\n"
    "    rec={'channel':c['channel'],'name':c['name'],'source':c['source'],'license':LICENSE,'test_line':c['test_line'],'ref_url':c['url']}\n"
    "    try:\n"
    "        download(c['url'], ref_flac)\n"
    "        to_wav(ref_flac, ref_wav)\n"
    "        t0=time.time(); wav=cb.generate(c['test_line'], audio_prompt_path=str(ref_wav))\n"
    "        if torch.cuda.is_available(): torch.cuda.synchronize()\n"
    "        gen=time.time()-t0; dur=int(wav.shape[-1])/cb_sr\n"
    "        import torchaudio as ta; ta.save(str(test_wav), wav.cpu(), cb_sr)\n"
    "        rec.update({'ok':True,'dur_s':round(dur,3),'gen_s':round(gen,3),'test_wav':str(test_wav),'ref_wav':str(ref_wav),'ref_bytes':ref_flac.stat().st_size})\n"
    "        print(f\"  [{c['channel']}] {c['name']}: gen={gen:.2f}s dur={dur:.2f}s\",flush=True)\n"
    "    except Exception as e:\n"
    "        rec.update({'ok':False,'error':repr(e)})\n"
    "        print(f\"  [{c['channel']}] {c['name']} FAIL: {e}\",flush=True)\n"
    "    rows.append(rec)\n"
    "manifest={'project_id':'voice-clone-candidate-test','gpu_name':GPU_NAME,'gpu_branch':GPU_BRANCH,\n"
    "          'voice':'cloned-per-candidate','generated_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'candidates':rows}\n"
    "(OUT/'clone_test_manifest.json').write_text(json.dumps(manifest, indent=2))\n"
    "print(json.dumps(manifest, indent=2))"))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

nbpath = OUT_DIR / "kernel.ipynb"
nbf.write(nb, nbpath)
meta = {
    "id": "forts845/openmontage-voice-clone-candidate-test",
    "title": "OpenMontage voice clone candidate test (9 CC0 voices)",
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
print("Wrote", nbpath, "with", len(payload), "candidates")
