"""Validate a scene_plan against the updated scene_plan.schema.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("jsonschema not installed — attempting install")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonschema", "-q"])
    import jsonschema

SCHEMA_PATH = Path(__file__).parent / "schemas/artifacts/scene_plan.schema.json"
EXAMPLE_PATH = Path(__file__).parent / "tests/fixtures/scene_plan_dark_annals_example.json"

def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def load_example() -> dict:
    with open(EXAMPLE_PATH) as f:
        return json.load(f)

def make_example() -> dict:
    return {
        "version": "1.0",
        "style_playbook": "dark-annals",
        "scenes": [
            {
                "id": "scene-1",
                "type": "generated",
                "shot_type": "body_kenburns",
                "description": "Medieval Strasbourg market square at dusk, torches lighting the cobblestones",
                "image_prompt": "medieval Strasbourg market square at dusk, oil painting texture, warm torchlight from upper-left, amber sepia palette, heavy vignette, 2.35:1 letterbox",
                "stock_search_query": None,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "duration_seconds": 5.0,
                "script_section_id": "hook",
                "narrative_role": "establish_context",
                "information_role": "Set the scene — 1518 Strasbourg, a normal summer evening",
                "hero_moment": True,
                "shot_intent": "Establish period setting with immersive atmosphere",
                "framing": "Wide establishing shot, rule of thirds",
                "movement": "Slow zoom-in 1.0 → 1.04 over 5s",
                "transition_in": "fade",
                "transition_out": "dissolve",
                "overlay_notes": "Heavy vignette, 35mm grain",
                "texture_keywords": ["oil-painting", "film-grain", "vignette"],
                "required_assets": [
                    {"type": "image", "description": "market square establishing shot", "source": "generate"}
                ]
            },
            {
                "id": "scene-2",
                "type": "text_card",
                "shot_type": "text_card",
                "description": "Title card: 'In July 1518, a woman stepped outside and began to dance. She did not stop.'",
                "image_prompt": None,
                "stock_search_query": None,
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "duration_seconds": 5.0,
                "script_section_id": "hook",
                "narrative_role": "deliver_payload",
                "information_role": "Deliver the hook promise verbatim",
                "hero_moment": False,
                "shot_intent": "Deliver the core hook line with maximum impact",
                "framing": "Centered text card",
                "movement": "Static",
                "transition_in": "dissolve",
                "transition_out": "cut",
                "overlay_notes": "Gothic serif font, parchment background, gold border",
                "texture_keywords": ["parchment", "ink"],
                "required_assets": []
            },
            {
                "id": "scene-3",
                "type": "generated",
                "shot_type": "body_kenburns",
                "description": "Close-up of medieval dancer's feet, trembling, with onlookers gathering in background",
                "image_prompt": "medieval dancer feet close-up trembling, oil painting style, amber candlelight, parchment texture, ink details, 2.35:1 letterbox, gothic aesthetic",
                "stock_search_query": None,
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "duration_seconds": 5.0,
                "script_section_id": "body-1",
                "narrative_role": "introduce_subject",
                "information_role": "Show the dancer's physical state",
                "hero_moment": False,
                "shot_intent": "Emotional through-line: involuntary movement",
                "framing": "Close-up on feet, shallow implied depth",
                "movement": "Slow zoom-out 1.04 → 1.0 over 5s",
                "transition_in": "cut",
                "transition_out": "fade",
                "overlay_notes": "Sepia tone, heavy vignette",
                "texture_keywords": ["oil-painting", "sepia"],
                "required_assets": [
                    {"type": "image", "description": "dancer feet close-up", "source": "generate"}
                ]
            },
            {
                "id": "scene-4",
                "type": "generated",
                "shot_type": "hook_stock",
                "description": "Candle flame close-up, flickering in darkness — atmospheric texture B-roll",
                "image_prompt": None,
                "stock_search_query": "candle flame close up dark",
                "start_seconds": 15.0,
                "end_seconds": 20.0,
                "duration_seconds": 5.0,
                "script_section_id": "body-2",
                "narrative_role": "emotional_beat",
                "information_role": "Breathing room — atmospheric pause",
                "hero_moment": False,
                "shot_intent": "Maintain mood between information beats",
                "framing": "Macro shot of flame, shallow depth",
                "movement": "Static with subtle flicker",
                "transition_in": "fade",
                "transition_out": "dissolve",
                "overlay_notes": "",
                "texture_keywords": ["candlelight", "warm-glow"],
                "required_assets": [
                    {"type": "video", "description": "candle flame stock clip", "source": "source"}
                ]
            }
        ],
        "metadata": {
            "total_duration_seconds": 20.0,
            "scene_count": 4
        }
    }

def validate(instance: dict, schema: dict) -> bool:
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"VALIDATION FAILED: {e.message}")
        print(f"  Path: {list(e.absolute_path)}")
        return False

def main() -> int:
    schema = load_schema()
    example = make_example()

    print("=== SCENE PLAN SCHEMA VALIDATION ===\n")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Example: {EXAMPLE_PATH} (generated in-memory)\n")

    ok = validate(example, schema)
    if ok:
        print("PASS — scene_plan validates cleanly against updated schema")
    else:
        print("FAIL — see errors above")

    # Show key fields present
    print("\n=== EXAMPLE SCENE FIELDS ===")
    for scene in example["scenes"]:
        print(f"\n  Scene {scene['id']}:")
        print(f"    type={scene['type']}  shot_type={scene['shot_type']}  duration_seconds={scene['duration_seconds']}")
        print(f"    start={scene['start_seconds']}  end={scene['end_seconds']}")
        if scene.get("stock_search_query"):
            print(f"    stock_search_query={scene['stock_search_query']!r}")
        if scene.get("image_prompt"):
            print(f"    image_prompt={scene['image_prompt'][:60]!r}...")

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
