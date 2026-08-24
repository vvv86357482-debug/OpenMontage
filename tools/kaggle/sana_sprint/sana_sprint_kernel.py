"""
Sana-Sprint 1.6B batch generation kernel v3 (Kaggle T4) - ELIZA ep01 Phase 4.

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

BATCH = [
  {
    "id": "s01_01",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, dim 1960s MIT corridor rendered as silver-gelatin photograph, fluorescent glow, long perspective. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s01_02",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, close-up of teletype console keys and curling paper tape, tungsten light. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s02_03",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, mainframe hall with raised floor, reel-to-reel cabinets and patch panels. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s03_04",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, period phonetic pronunciation chart, ink linework on graph paper. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s03_05",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, typographic index card reading PYGMALION 1913 in Shaw-era lettering. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s04_06",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, ink technical diagram of keyword decomposition and reassembly flow, Bell Systems manual style. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s04_07",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, printed terminal transcript page with lowercase questions and capital-letter replies. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s04_08",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, stack of MAD-SLIP line printer output with margin punch marks. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s04_09",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, AI reconstruction, general period environment (not a specific documented terminal): time-sharing console terminal with typewriter keyboard and CRT, laboratory setting, 1960s. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s05_10",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, contact sheet of psychiatry conference programs and hospital corridor signage, archival. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s05_11",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, student silhouette at glowing terminal in darkened laboratory, low key. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s06_12",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, lecture hall podium microphone close-up, volumetric projector light. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s07_13",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, empty corridor with finished teletype tape lying on floor, single shaft of light. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  },
  {
    "id": "s07_14",
    "prompt": "1950s-1980s laboratory black-and-white photograph, Bell Labs / Stanford / MIT-era technical documentation style, silver-gelatin print, available-light realism, punch cards, oscilloscope traces, reel-to-reel hardware, end card typography FORGOTTEN HISTORY OF AI on archive-card stock. NOT modern smartphone, use era-correct equipment instead, NOT laptop, use era-correct equipment instead, NOT LED lighting, use era-correct equipment instead, NOT flat-screen monitor, use era-correct equipment instead, NOT modern UI, use era-correct equipment instead, NOT glossy product-render look, use era-correct equipment instead, NOT contemporary office, use era-correct equipment instead, NOT neon cyberpunk glow, use era-correct equipment instead, NOT futuristic hologram, use era-correct equipment instead, NOT modern photorealism, use era-correct equipment instead, NOT color photography where period demands monochrome, use era-correct equipment instead."
  }
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
        seed=1000+i
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
