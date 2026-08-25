"""
Sana-Sprint 1.6B batch generation kernel v5 (Kaggle T4) - ELIZA ep01 Phase 4
complex-scene remediation: 11 flagged scenes regenerated with subject-first
prompts, equipment anchors trimmed where they caused collage, scene-specific
NOT clauses through augment_prompt_for_sana semantics, seed base 2000.

Known-good constraints unchanged from smoke test (do NOT modify):
  machine_shape NvidiaTeslaT4; num_inference_steps=2 exactly; transformer
  bfloat16 + pipe.vae.to(torch.float32); torch.cuda.synchronize() before
  max_memory_allocated(); hard gate >=50KB and pixel std >=5.0 per image;
  prompts pre-augmented with augment_prompt_for_sana() semantics
  (NOT X, use Y instead clauses); outputs to /kaggle/working/output, never /tmp.
"""

import json, os, sys, time
import numpy as np
import torch
from PIL import Image

OUTPUT_DIR = "/kaggle/working/output/batch"
MODEL_ID = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
STEPS = 2
MIN_BYTES, MIN_STD = 50_000, 5.0
SEED_BASE = 2000

STD_NEG = ("NOT modern smartphone, use era-correct equipment instead, "
           "NOT laptop, use era-correct equipment instead, "
           "NOT LED lighting, use era-correct equipment instead, "
           "NOT flat-screen monitor, use era-correct equipment instead, "
           "NOT modern UI, use era-correct equipment instead, "
           "NOT glossy product-render look, use era-correct equipment instead, "
           "NOT contemporary office, use era-correct equipment instead, "
           "NOT neon cyberpunk glow, use era-correct equipment instead, "
           "NOT futuristic hologram, use era-correct equipment instead, "
           "NOT modern photorealism, use era-correct equipment instead, "
           "NOT color photography where period demands monochrome, use era-correct equipment instead.")

STYLE = ("silver-gelatin black-and-white photograph, Bell Labs / Stanford / "
         "MIT-era technical documentation style, available-light realism, film grain.")

BATCH = [
  {"id": "s01_01",
   "prompt": f"dim 1960s university corridor, long perspective, fluorescent glow, linoleum floor, painted walls, {STYLE} NOT control room, use empty corridor instead, NOT computer equipment, use empty corridor instead, {STD_NEG}"},
  {"id": "s01_02",
   "prompt": f"close-up of a single vintage teletype machine, round keys soft in shallow focus, curling paper tape, tungsten light, {STYLE} NOT sharp readable key legends, use soft out-of-focus keys instead, NOT multiple keyboards, use one teletype instead, NOT dense control panels, use a single machine instead, {STD_NEG}"},
  {"id": "s02_03",
   "prompt": f"wide view of a mainframe hall, rows of tall plain cabinets with tape reels and small status lights, raised floor, long perspective, {STYLE} NOT patch panels, use plain cabinet fronts instead, NOT dense buttons, use plain cabinet fronts instead, NOT readable labels, use plain cabinet fronts instead, {STD_NEG}"},
  {"id": "s03_04",
   "prompt": f"period phonetic chart page, ink linework of mouth-position diagrams on graph paper, sparse annotation, slight angle, {STYLE} NOT dense text, use sparse linework instead, NOT computer equipment, use paper chart instead, {STD_NEG}"},
  {"id": "s03_05",
   "prompt": f"vintage archival index card with large hand-lettered title reading PYGMALION 1913, Shaw-era lettering, card stock, slight angle, soft focus, {STYLE} NOT computer equipment, use paper card instead, NOT dense panels, use paper card instead, {STD_NEG}"},
  {"id": "s04_06",
   "prompt": f"ink technical flow diagram, rectangular blocks connected by arrows, sparse hand-lettered labels, graph paper, Bell Systems manual style, {STYLE} NOT dense equipment, use paper diagram instead, NOT readable paragraphs, use short labels instead, {STD_NEG}"},
  {"id": "s04_07",
   "prompt": f"printed terminal transcript page on tractor-feed paper, alternating lowercase lines and capital-letter lines, print slightly soft and illegible, raking light, {STYLE} NOT sharp readable words, use soft illegible print instead, NOT computer equipment, use paper page instead, {STD_NEG}"},
  {"id": "s04_08",
   "prompt": f"tall stack of fanfold line-printer paper, tractor-feed margins, dense print lines as soft texture, shallow depth of field, {STYLE} NOT readable words, use soft print texture instead, NOT computer equipment, use paper stack instead, {STD_NEG}"},
  {"id": "s04_09",
   "prompt": f"time-sharing console terminal with one typewriter-style keyboard suggested in shallow focus, round CRT with soft glow, three-quarter view, laboratory setting, 1960s, {STYLE} NOT multiple keyboards, use one keyboard instead, NOT dense buttons, use plain console surfaces instead, NOT readable labels, use blank surfaces instead, {STD_NEG}"},
  {"id": "s05_10",
   "prompt": f"archival contact sheet, grid of small document photographs, shallow depth of field, lettering soft and illegible, {STYLE} NOT readable signage, use soft letterforms instead, NOT control panels, use paper documents instead, {STD_NEG}"},
  {"id": "s07_13",
   "prompt": f"empty corridor, a single curled paper tape ribbon lying on the polished floor, one shaft of light, long perspective, {STYLE} NOT scattered debris, use one paper ribbon instead, NOT keyboards, use one paper ribbon instead, {STD_NEG}"},
  {"id": "s06_12",
   "prompt": f"lecture hall podium microphone close-up, vintage dynamic microphone on a slim stand, blurred rows of seats in the background, volumetric projector light beam, 1960s, {STYLE} NOT vacuum tubes, use solid-state electronics instead, NOT valve-based electronics, use solid-state electronics instead, NOT dense buttons, use plain microphone body instead, NOT control panels, use lecture hall background instead, NOT tangled cables, use slim stand instead, {STD_NEG}"}
]

def log(m): print(f"[sana_batch] {m}", flush=True)

def main():
    from diffusers import SanaSprintPipeline
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = SanaSprintPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.to(torch.float32)
    log(f"GPU {torch.cuda.get_device_name(0)} | vae={next(pipe.vae.parameters()).dtype} "
        f"transformer={next(pipe.transformer.parameters()).dtype}")
    manifest={"model_id":MODEL_ID,"num_inference_steps":STEPS,"gpu":torch.cuda.get_device_name(0),
              "torch_version":torch.__version__,"items":[]}
    failed=0
    for i,item in enumerate(BATCH):
        seed=SEED_BASE+i
        g=torch.Generator("cuda").manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0=time.time()
        img=pipe(prompt=item["prompt"], num_inference_steps=STEPS,
                 height=1024, width=1024, generator=g).images[0]
        torch.cuda.synchronize()
        el=time.time()-t0
        path=os.path.join(OUTPUT_DIR, item["id"]+".png")
        img.save(path)
        size=os.path.getsize(path)
        std=float(np.asarray(Image.open(path).convert("RGB"),dtype=np.float32).std())
        ok=size>=MIN_BYTES and std>=MIN_STD
        failed+= 0 if ok else 1
        rec={"id":item["id"],"file":path,"bytes":size,"pixel_std":round(std,2),
             "seed":seed,"elapsed_seconds":round(el,2),
             "peak_vram_mb":round(torch.cuda.max_memory_allocated()/1048576,1),"gate_passed":ok}
        manifest["items"].append(rec)
        log(json.dumps(rec))
    with open(os.path.join(OUTPUT_DIR,"manifest.json"),"w") as fh: json.dump(manifest,fh,indent=2)
    log(f"DONE total={len(BATCH)} failed={failed}")
    sys.exit(3 if failed else 0)

if __name__ == "__main__":
    main()
