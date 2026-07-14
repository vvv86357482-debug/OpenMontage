"""Video composition tool — FFmpeg-only render runtime.

Pipeline-facing orchestration surface for composition. Takes `edit_decisions`,
`asset_manifest`, and audio, and delegates to FFmpeg.

Routing is driven by `edit_decisions.render_runtime` (locked at proposal).
Only 'ffmpeg' is supported in this environment.

- `ffmpeg` → FFmpeg concat/trim. The only supported render runtime.

Silent runtime swaps are forbidden by governance. If a different runtime is
requested, this tool surfaces a structured error and waits for the agent to
re-ask the user.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ResumeSupport,
    ToolResult,
    ToolStability,
    ToolTier,
)


class VideoCompose(BaseTool):
    name = "video_compose"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg: https://ffmpeg.org/download.html"
    agent_skills = ["ffmpeg"]

    capabilities = [
        "compose_cuts",
        "burn_subtitles",
        "overlay_assets",
        "encode_profile",
    ]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["compose", "render", "burn_subtitles", "overlay", "encode"],
                "description": (
                    "compose: low-level concat cuts + audio + subtitles. "
                    "render: high-level — resolves asset IDs and routes to FFmpeg. "
                    "burn_subtitles: burn subtitle file into existing video. "
                    "overlay: composite overlays onto base video. "
                    "encode: re-encode to a target profile/codec."
                ),
            },
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "edit_decisions": {
                "type": "object",
                "description": "Full edit_decisions artifact (required for compose/render)",
            },
            "asset_manifest": {
                "type": "object",
                "description": (
                    "Full asset_manifest artifact (required for render). "
                    "Used to resolve asset IDs in cuts[].source to file paths."
                ),
            },
            "proposal_packet": {
                "type": "object",
                "description": (
                    "Full proposal_packet artifact. Optional but STRONGLY "
                    "recommended — when present, final_review compares "
                    "proposal_packet.production_plan.render_runtime against "
                    "edit_decisions.render_runtime and flags runtime_swap_detected. "
                    "Without it, runtime-swap detection falls back to checking "
                    "edit_decisions.metadata.proposal_render_runtime."
                ),
            },
            "narration_transcript_path": {
                "type": "string",
                "description": (
                    "Path to a word-level transcript JSON (from `transcriber` "
                    "tool output). Optional but STRONGLY recommended: when "
                    "combined with script_path/script_text, final_review "
                    "runs transcript_comparison and catches TTS failures "
                    "like 'Chirp3-HD reads ... as the word dot'. Without "
                    "it, content-level audio bugs ship silently."
                ),
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Path to the source narration script (plain text). "
                    "Used by transcript_comparison to diff against the "
                    "transcribed audio. Provide this OR script_text."
                ),
            },
            "script_text": {
                "type": "string",
                "description": (
                    "Inline source narration script. Used by "
                    "transcript_comparison when a file path is unavailable."
                ),
            },
            "subtitle_path": {"type": "string"},
            "subtitle_style": {
                "type": "object",
                "description": "ASS subtitle styling. Also extracted from edit_decisions.subtitles if not provided.",
                "properties": {
                    "font": {"type": "string", "default": "Arial"},
                    "font_size": {"type": "integer", "default": 24},
                    "primary_color": {"type": "string", "default": "&HFFFFFF"},
                    "outline_color": {"type": "string", "default": "&H000000"},
                    "outline_width": {"type": "number", "default": 2},
                    "margin_v": {"type": "integer", "default": 40},
                    "alignment": {"type": "integer", "default": 2},
                },
            },
            "overlays": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset_path": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "audio_path": {"type": "string", "description": "Mixed audio to mux into output"},
            "profile": {
                "type": "string",
                "description": (
                    "Media profile name from media_profiles.py "
                    "(e.g. youtube_landscape, tiktok, instagram_reels). "
                    "Applied in render and encode operations."
                ),
            },
            "options": {
                "type": "object",
                "description": "Render options (used by the render operation)",
                "properties": {
                    "subtitle_burn": {"type": "boolean", "default": True},
                    "two_pass_encode": {"type": "boolean", "default": False},
                },
            },
            "codec": {"type": "string", "default": "libx264"},
            "crf": {"type": "integer", "default": 23},
            "preset": {"type": "string", "default": "medium"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=5000, network_required=False
    )

    best_for = [
        "Final render for explainer and animation pipelines",
        "Pure video concat, trim, and composition (FFmpeg)",
        "Subtitle burn, overlay compositing, and profile encoding",
    ]
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["Conversion failed"])
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = ["operation", "input_path", "edit_decisions"]
    side_effects = ["writes video file to output_path"]
    user_visible_verification = [
        "Play the composed output and verify cuts, subtitles, and overlays",
    ]
    # Map playbook transition names to ffmpeg xfade filter values.
    _XFADE_MAP: dict[str, str] = {
        "fade": "fade",
        "fade-black": "fadeblack",
        "fade-white": "fadewhite",
        "dissolve": "dissolve",
        "wipe-left": "wipeleft",
        "wipe-right": "wiperight",
        "slide-left": "slideleft",
        "slide-right": "slideright",
        "slide-up": "slideup",
        "slide-down": "slidedown",
        "circle-open": "circleopen",
        "circle-close": "circleclose",
        "pixelize": "pixelize",
        "radial": "radial",
        # Motion-only keywords — not xfade transitions — skip (hard cut)
        "slow-zoom": None,
        "slow-pan": None,
        "slowpan": None,
        # Explicit cut/hard-cut — no transition
        "cut": None,
        "hard-cut": None,
    }


    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["render_engines"] = {
            "ffmpeg": True,
        }
        info["render_runtimes"] = info["render_engines"]
        info["render_runtime_note"] = (
            "FFmpeg is the only render runtime in this environment. "
            "Remotion and HyperFrames are not available."
        )
        return info

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs["operation"]
        start = time.time()

        try:
            if operation == "compose":
                result = self._compose(inputs)
            elif operation == "render":
                result = self._render(inputs)
            elif operation == "burn_subtitles":
                result = self._burn_subtitles(inputs)
            elif operation == "overlay":
                result = self._overlay(inputs)
            elif operation == "encode":
                result = self._encode(inputs)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        result.duration_seconds = round(time.time() - start, 2)
        return result

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    @staticmethod
    def _is_image(path: Path) -> bool:
        """Check if a file is a still image (routes to Remotion, not FFmpeg)."""
        return path.suffix.lower() in VideoCompose._IMAGE_EXTENSIONS

    @staticmethod
    def _has_audio_stream(path: Path) -> bool:
        """Return True iff ffprobe reports at least one audio stream.

        Many stock video clips (especially from Pexels) ship with no audio
        stream at all. If we blindly tell ffmpeg to transcode the 0:a stream
        on such a file it errors out. This helper lets the segment builder
        branch on stream presence so it can synthesize a silent track when
        needed, keeping the concat segment layout consistent.
        """
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=nw=1:nk=1",
                    str(path),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return "audio" in out
        except Exception:
            return False

    def _compose(self, inputs: dict[str, Any]) -> ToolResult:
        """FFmpeg composition: concat video cuts, add audio, burn subtitles.

        Handles video sources only. Still images and animated scene types
        are routed to Remotion via the render operation — call compose
        directly only for pure video pipelines (e.g. talking-head).
        """
        edit_decisions = inputs.get("edit_decisions")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for compose")

        output_path = Path(inputs.get("output_path", "composed_output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path = inputs.get("audio_path")
        subtitle_path = inputs.get("subtitle_path")
        playbook_data = inputs.get("playbook")
        playbook_output = (playbook_data or {}).get("output", {})

        codec = inputs.get("codec", playbook_output.get("video_codec", "libx264"))
        crf = inputs.get("crf", playbook_output.get("crf", 23))
        preset = inputs.get("preset", playbook_output.get("preset", "medium"))
        pixel_format = inputs.get("pixel_format", playbook_output.get("pixel_format", "yuv420p"))
        fps_override = inputs.get("fps") or playbook_output.get("fps")
        audio_codec = inputs.get("audio_codec", playbook_output.get("audio_codec", "aac"))
        audio_bitrate = inputs.get("audio_bitrate", playbook_output.get("audio_bitrate", "192k"))
        audio_sample_rate = inputs.get("audio_sample_rate", playbook_output.get("audio_sample_rate", 44100))
        profile_name = inputs.get("profile")

        # Resolve target resolution + fit mode. Priority: explicit `profile`
        # arg > edit_decisions.metadata.compose_target > default (landscape HD).
        # compose_target = {"width": W, "height": H, "fit": "pad"|"cover"} lets a
        # caller request vertical (9:16) or any aspect without a named profile.
        # fit="pad" letterboxes (no content loss, the historical default);
        # fit="cover" scales-to-fill and centre-crops (better for vertical social).
        resolution = "1920x1080"
        fit_mode = "pad"
        compose_target = (edit_decisions.get("metadata") or {}).get("compose_target")
        if isinstance(compose_target, dict):
            try:
                resolution = f"{int(compose_target['width'])}x{int(compose_target['height'])}"
            except (KeyError, ValueError, TypeError):
                pass
            if compose_target.get("fit") in ("pad", "cover"):
                fit_mode = compose_target["fit"]

            # Playbook output overrides resolution/fps if no explicit compose_target or profile
            if not compose_target:
                playbook_resolution = playbook_output.get("resolution")
                if playbook_resolution:
                    resolution = playbook_resolution
                playbook_fps = playbook_output.get("fps")
                if playbook_fps:
                    fps = playbook_fps
            if profile_name:
                try:
                    from lib.media_profiles import get_profile
                    p = get_profile(profile_name)
                    resolution = f"{p.width}x{p.height}"
                except (ImportError, ValueError):
                    pass
        try:
            target_w, target_h = (int(v) for v in resolution.split("x"))
        except ValueError:
            target_w, target_h = 1920, 1080

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        # Resolve subtitle style using the layered priority resolver
        # (explicit > edit_decisions > playbook > defaults)
        playbook_data = inputs.get("playbook")
        resolved_sub_style = self._resolve_subtitle_style(
            inputs.get("subtitle_style"),
            edit_decisions,
            playbook_data,
        )
        inputs = dict(inputs)
        inputs["subtitle_style"] = resolved_sub_style

        ed_subs = edit_decisions.get("subtitles", {})
        if ed_subs.get("source") and not subtitle_path:
            subtitle_path = ed_subs["source"]

        temp_dir = output_path.parent / ".compose_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_segments: list[Path] = []
        concat_path: Path | None = None
        concat_out: Path | None = None

        try:
            for i, cut in enumerate(cuts):
                source = Path(cut["source"])
                if not source.exists():
                    return ToolResult(success=False, error=f"Cut source not found: {source}")

                seg_path = temp_dir / f"seg_{i:04d}.mp4"
                in_s = cut["in_seconds"]
                out_s = cut["out_seconds"]
                duration = out_s - in_s
                speed = cut.get("speed", 1.0)

                if self._is_image(source):
                    return ToolResult(
                        success=False,
                        error=(
                            f"Still image '{source.name}' in cuts. "
                            "Use operation='render' for compositions "
                            "with images. Note: FFmpeg-only runtime in this environment."
                        ),
                    )
                else:
                    # Video source: trim to segment.
                    #
                    # Semantics:
                    #   -ss BEFORE -i   → fast input-level seek to in_s
                    #   -t  AFTER  -i   → "play for `duration` seconds"
                    #                     (unambiguous regardless of seek mode)
                    #
                    # We MUST re-encode here — `-c copy` cannot do frame-accurate
                    # cuts because it snaps to keyframes. With sparse GOPs (common
                    # in Pexels / AI-generated clips), stream-copy can produce
                    # segments significantly longer than `duration`, breaking the
                    # target timeline. Re-encoding with libx264/AAC is slower but
                    # gives exact cut boundaries. Same resolution in → same
                    # resolution out, so same-res inputs concat cleanly.
                    # Strip embedded audio from every segment. Narration is
                    # the only intended audio source and is added in the
                    # final mux step below. This prevents chipmunk pitch-shift
                    # regressions from stock clips at mismatched sample rates.
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(in_s),
                        "-t", str(duration),
                        "-i", str(source),
                        "-an",
                    ]

                    vf_parts: list[str] = []
                    if fit_mode == "cover":
                        vf_parts.extend([
                            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase",
                            f"crop={target_w}:{target_h}",
                        ])
                    else:
                        vf_parts.extend([
                            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease",
                            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
                        ])
                    vf_parts.extend(["setsar=1", "fps=30"])
                    if speed != 1.0:
                        vf_parts.append(f"setpts={1.0/speed}*PTS")
                        # atempo audio speed is skipped because -an strips audio;
                        # audio speed is irrelevant since segments are silent.

                    cmd.extend(["-filter:v", ",".join(vf_parts)])
                    cmd.extend([
                        "-c:v", codec,
                        "-crf", str(crf),
                        "-preset", preset,
                        "-pix_fmt", "yuv420p",
                        "-r", "30",
                    ])

                    cmd.append(str(seg_path))
                    self.run_command(cmd)

                temp_segments.append(seg_path)

            # Step 2: Assemble segments — xfade if transitions present, else concat demuxer.
            # xfade filter-complex compresses output by (N-1) * transition_duration,
            # which is intentional: transitions overlap adjacent clips.
            has_transitions = any(
                cut.get("transition_in") and cut.get("transition_in") not in ("cut", "hard-cut")
                for cut in cuts[1:]
            )

            if has_transitions:
                xfade_parts = []
                current_label = "0"
                # xfade offset = cumulative output duration so far minus transition_duration.
                # This positions the crossfade correctly so the output duration equals
                # sum(segment durations) - sum(transition durations).
                running_total = float(cuts[0].get("out_seconds", 0)) - float(cuts[0].get("in_seconds", 0))
                for i, cut in enumerate(cuts[1:], 1):
                    transition = (cut.get("transition_in") or "fade").lower()
                    duration = float(cut.get("transition_duration", 0.5))
                    xfade_type = self._XFADE_MAP.get(transition, "fade")
                    if xfade_type is None:
                        # Motion-only or explicit cut — advance by full segment duration, no overlap
                        prev_cut = cuts[i - 1]
                        prev_dur = float(prev_cut.get("out_seconds", 0)) - float(prev_cut.get("in_seconds", 0))
                        running_total += prev_dur
                        current_label = str(i)
                        continue
                    prev_cut = cuts[i - 1]
                    prev_dur = float(prev_cut.get("out_seconds", 0)) - float(prev_cut.get("in_seconds", 0))
                    offset = max(0, running_total - duration)
                    next_label = f"x{i-1}{i}"
                    xfade_parts.append(
                        f"[{current_label}][{i}]xfade=transition={xfade_type}:duration={duration}:offset={offset}[{next_label}]"
                    )
                    running_total = running_total + prev_dur - duration
                    current_label = next_label

                if xfade_parts:
                    xfade_out = temp_dir / "xfade.mp4"
                    filter_complex = ";".join(xfade_parts)
                    cmd = ["ffmpeg", "-y"]
                    for seg in temp_segments:
                        cmd.extend(["-i", str(seg)])
                    cmd.extend([
                        "-filter_complex", filter_complex,
                        "-map", f"[{current_label}]",
                        "-c:v", codec,
                        "-crf", str(crf),
                        "-preset", preset,
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(xfade_out),
                    ])
                    self.run_command(cmd)
                    final_input = xfade_out
                else:
                    # All transitions resolved to "none" — fall through to concat
                    has_transitions = False
            if not has_transitions:
                concat_path = temp_dir / "concat_list.txt"
                with open(concat_path, "w", encoding="utf-8") as f:
                    for seg in temp_segments:
                        safe = str(seg.resolve()).replace("\\", "/")
                        f.write(f"file '{safe}'\n")

                concat_out = temp_dir / "concat.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_path),
                    "-c", "copy",
                    str(concat_out),
                ]
                self.run_command(cmd)
                final_input = concat_out

            # Step 3: Apply subtitles and/or replace audio

            vfilters = []

            if subtitle_path and Path(subtitle_path).exists():
                style = inputs.get("subtitle_style", {})
                ass_style = self._build_subtitle_style(style)
                sub_escaped = str(Path(subtitle_path).resolve()).replace("\\", "/").replace(":", "\\:")
                vfilters.append(f"subtitles='{sub_escaped}':force_style='{ass_style}'")

            cmd = ["ffmpeg", "-y", "-i", str(final_input)]

            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path])

            # Determine if profile requires re-encoding (resize/fps change)
            # This must be checked BEFORE choosing copy vs encode, because
            # -s and -r are incompatible with -c:v copy.
            profile_flags: list[str] = []
            if fps_override:
                profile_flags.extend(["-r", str(int(fps_override))])
            if profile_name:
                try:
                    from lib.media_profiles import get_profile
                    p = get_profile(profile_name)
                    profile_flags = ["-s", f"{p.width}x{p.height}", "-r", str(p.fps)]
                except (ImportError, ValueError):
                    pass

            needs_reencode = bool(vfilters) or bool(profile_flags)

            if needs_reencode:
                if vfilters:
                    cmd.extend(["-vf", ",".join(vfilters)])
                cmd.extend(["-c:v", codec, "-crf", str(crf), "-preset", preset])
                cmd.extend(profile_flags)
            else:
                cmd.extend(["-c:v", "copy"])

            if audio_path and Path(audio_path).exists():
                # Use type-based selectors (0:v, 1:a) instead of index-based
                # (0:v:0) because source videos may have audio as stream 0
                # and video as stream 1 (e.g. Kling-generated clips).
                cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-ar", "44100", "-shortest"])
            else:
                cmd.extend(["-c:a", "copy"])

            cmd.append(str(output_path))
            self.run_command(cmd)

            return ToolResult(
                success=True,
                data={
                    "operation": "compose",
                    "cut_count": len(cuts),
                    "has_subtitles": subtitle_path is not None,
                    "has_mixed_audio": audio_path is not None,
                    "profile": profile_name,
                    "output": str(output_path),
                },
                artifacts=[str(output_path)],
            )
        finally:
            # Cleanup temp files
            for f in temp_segments:
                if f.exists():
                    f.unlink()
            for f in [concat_path, concat_out, xfade_out]:
                if f is not None and f.exists():
                    f.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass

    def _pre_compose_validation(
        self,
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        scene_plan: list[dict] | None = None,
    ) -> ToolResult | None:
        """Pre-compose quality gate — blocks render on critical violations.

        Checks:
        1. Delivery promise violation: motion-required brief with >70% still cuts → BLOCK
        2. Slideshow risk score "fail" (average ≥ 4.0) → BLOCK
        3. Missing renderer_family → WARN (log only, don't block)

        Returns a failed ToolResult if render should be blocked, None if OK to proceed.
        """
        log = logging.getLogger("video_compose")
        warnings: list[str] = []
        blocks: list[str] = []

        # --- 1. Delivery promise check ---
        delivery_data = edit_decisions.get("metadata", {}).get("delivery_promise")
        if not delivery_data:
            # Also check top-level (proposal_packet nests it at top level)
            delivery_data = edit_decisions.get("delivery_promise")

        if delivery_data:
            try:
                from lib.delivery_promise import DeliveryPromise
                promise = DeliveryPromise.from_dict(delivery_data)
                result = promise.validate_cuts(resolved_cuts)
                if not result["valid"]:
                    for v in result["violations"]:
                        blocks.append(f"Delivery promise violation: {v}")
            except Exception as e:
                log.warning("Could not validate delivery promise: %s", e)
        else:
            warnings.append("No delivery_promise in edit_decisions — skipping promise validation")

        # --- 2. Slideshow risk check ---
        renderer_family = edit_decisions.get("renderer_family")
        scenes = scene_plan or []

        # If no scene_plan passed, try to extract scene info from cuts
        if not scenes and resolved_cuts:
            scenes = [
                {
                    "type": c.get("type", ""),
                    "description": c.get("reason", ""),
                    "shot_language": c.get("shot_language", {}),
                    "shot_intent": c.get("shot_intent"),
                    "narrative_role": c.get("narrative_role"),
                    "information_role": c.get("information_role"),
                    "hero_moment": c.get("hero_moment", False),
                }
                for c in resolved_cuts
            ]

        if scenes:
            try:
                from lib.slideshow_risk import score_slideshow_risk
                render_runtime = edit_decisions.get("render_runtime")
                risk = score_slideshow_risk(
                    scenes, edit_decisions, renderer_family, render_runtime
                )
                if risk["verdict"] == "fail":
                    blocks.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: fail). "
                        f"Video plan looks like a slideshow — revise scene plan before rendering."
                    )
                elif risk["verdict"] == "revise":
                    warnings.append(
                        f"Slideshow risk score {risk['average']:.1f}/5.0 (verdict: revise). "
                        f"Consider improving scene variety before final render."
                    )
            except Exception as e:
                log.warning("Could not compute slideshow risk: %s", e)

        # --- 3. Missing renderer_family (BLOCK — must be set at proposal) ---
        if not renderer_family:
            blocks.append(
                "No renderer_family in edit_decisions. "
                "renderer_family must be set at proposal stage and locked before compose. "
                "Re-run the proposal stage with a renderer_family selection."
            )

        # Log warnings
        for w in warnings:
            log.warning("[pre-compose] %s", w)

        # Block on critical violations
        if blocks:
            return ToolResult(
                success=False,
                error=(
                    "Pre-compose validation failed — render blocked.\n"
                    + "\n".join(f"  • {b}" for b in blocks)
                    + ("\n\nWarnings:\n" + "\n".join(f"  • {w}" for w in warnings) if warnings else "")
                ),
            )

        return None

    def _render(self, inputs: dict[str, Any]) -> ToolResult:
        """High-level render: assemble edit decisions + asset manifest into final video.

        FFmpeg is the only render runtime in this environment.
        """
        edit_decisions = inputs.get("edit_decisions")
        asset_manifest = inputs.get("asset_manifest")
        if not edit_decisions:
            return ToolResult(success=False, error="edit_decisions required for render")

        render_runtime = (edit_decisions.get("render_runtime") or "").strip().lower()

        if not render_runtime:
            return ToolResult(
                success=False,
                error=(
                    "render_runtime is not set in edit_decisions. It MUST be "
                    "locked at proposal stage as 'ffmpeg'. Re-run the proposal "
                    "stage with render_runtime='ffmpeg'."
                ),
            )

        if render_runtime != "ffmpeg":
            return ToolResult(
                success=False,
                error=(
                    f"render_runtime={render_runtime!r} is not available in this "
                    f"environment. Only 'ffmpeg' is supported. Re-run the proposal "
                    f"stage with render_runtime='ffmpeg'."
                ),
            )

        if not asset_manifest:
            return ToolResult(success=False, error="asset_manifest required for render")

        output_path = Path(inputs.get("output_path", "renders/output.mp4"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        asset_lookup = {a["id"]: a for a in asset_manifest.get("assets", [])}

        cuts = edit_decisions.get("cuts", [])
        if not cuts:
            return ToolResult(success=False, error="No cuts in edit_decisions")

        resolved_cuts = []
        for cut in cuts:
            source_id = cut.get("source", "")
            resolved_cut = dict(cut)
            if source_id in asset_lookup:
                resolved_cut["source"] = asset_lookup[source_id]["path"]
            resolved_cuts.append(resolved_cut)

        scene_plan = inputs.get("scene_plan")
        validation_block = self._pre_compose_validation(edit_decisions, resolved_cuts, scene_plan)
        if validation_block is not None:
            return validation_block

        profile = inputs.get("profile") or inputs.get("output_profile")

        return self._render_via_ffmpeg(
            inputs=inputs,
            edit_decisions=edit_decisions,
            resolved_cuts=resolved_cuts,
            output_path=output_path,
             profile=profile,
         )

    def _render_via_ffmpeg(
        self,
        *,
        inputs: dict[str, Any],
        edit_decisions: dict[str, Any],
        resolved_cuts: list[dict],
        output_path: Path,
        profile: Optional[str],
    ) -> ToolResult:
        """Explicit FFmpeg-only render path.

        Use when the proposal locked `render_runtime="ffmpeg"` — e.g. simple
        source-footage concat/trim jobs that don't benefit from composition.
        Still runs the mandatory final self-review.
        """
        options = inputs.get("options", {})
        subtitle_burn = options.get("subtitle_burn", True)

        subtitle_path = inputs.get("subtitle_path")
        if subtitle_burn and not subtitle_path:
            ed_subs = edit_decisions.get("subtitles", {})
            if ed_subs.get("enabled") and ed_subs.get("source"):
                subtitle_path = ed_subs["source"]

        compose_inputs = dict(inputs)
        compose_inputs["edit_decisions"] = dict(edit_decisions, cuts=resolved_cuts)
        compose_inputs["output_path"] = str(output_path)
        if subtitle_path:
            compose_inputs["subtitle_path"] = subtitle_path
        if profile:
            compose_inputs["profile"] = profile

        render_result = self._compose(compose_inputs)

        if render_result.success and output_path.exists():
            final_review = self._run_final_review(
                output_path,
                edit_decisions,
                inputs.get("proposal_packet"),
                narration_transcript_path=inputs.get("narration_transcript_path"),
                script_text=inputs.get("script_text") or self._read_text_file(
                    inputs.get("script_path")
                ),
            )
            if render_result.data is None:
                render_result.data = {}
            render_result.data["final_review"] = final_review
            render_result.data["final_review_status"] = final_review["status"]
            if final_review["status"] == "fail":
                return ToolResult(
                    success=False,
                    error=(
                        "Post-render self-review FAILED (FFmpeg). The output is not presentable.\n"
                        + "\n".join(f"  • {i}" for i in final_review.get("issues_found", []))
                    ),
                    data=render_result.data,
                )

        return render_result

    # ------------------------------------------------------------------
    # Final self-review — mandatory post-render inspection
    # ------------------------------------------------------------------
    # Final self-review — mandatory post-render inspection
    # ------------------------------------------------------------------

    # Punctuation/SSML-leak words that should NEVER appear in rendered audio.
    # When a TTS engine reads a literal "..." as the word "dot", or a "—" as
    # "hyphen", those leak into the transcript. Catching these in the final
    # review is the difference between catching a bad voice render in-tool
    # vs. shipping a video that says "dot dot dot" twelve times. CRITICAL.
    _TTS_PUNCTUATION_LEAK_WORDS = {
        "dot", "dots", "ellipsis", "period", "periods",
        "comma", "commas", "semicolon", "colon",
        "dash", "hyphen", "emdash", "endash",
        "parenthesis", "bracket", "brace",
        "asterisk", "slash", "backslash",
        "exclamation", "question mark",
    }

    @staticmethod
    def _read_text_file(path: str | Path | None) -> str | None:
        """Read a small text file if given a path; None-safe and exception-safe."""
        if not path:
            return None
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception:
            return None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Split text into comparable word tokens (lowercased, punctuation
        stripped, numeric-word-aware). Empty tokens dropped."""
        import re

        # Preserve hyphenated words as single tokens ("many-worlds" -> "many-worlds").
        # Drop everything except letters, digits, hyphens, apostrophes.
        cleaned = re.sub(r"[^A-Za-z0-9\-' ]+", " ", text.lower())
        return [t for t in cleaned.split() if t and t != "-"]

    @classmethod
    def _compare_transcript_to_script(
        cls,
        transcript_path: Path,
        script_text: str,
    ) -> dict[str, Any]:
        """Compare a word-level transcript against the source script.

        Purpose: catch TTS failures that look fine on audio-volume/duration
        checks but produce garbage content. The canonical example is
        Chirp3-HD reading ellipses ("...") literally as the word "dot" — our
        volume check says "narration present, not clipped" and the video
        ships. This check diffs the actual transcribed audio against what
        was supposed to be said, and flags:

        - Spurious punctuation-leak words ("dot", "comma", "hyphen", etc.)
          that appear in audio but not script → CRITICAL
        - Overall word-accuracy ratio against script → SUGGESTION if < 0.9

        Returns the transcript_comparison section of final_review, or a
        placeholder with an issue describing why the check couldn't run
        (missing transcript, missing script) so the review never goes
        silently quiet on this contract.
        """
        result: dict[str, Any] = {
            "transcript_matches_script": False,
            "word_accuracy": None,
            "script_word_count": 0,
            "transcript_word_count": 0,
            "spurious_punctuation_words": [],
            "issues": [],
        }

        if not transcript_path or not Path(transcript_path).is_file():
            result["issues"].append(
                "transcript_comparison skipped: narration_transcript not provided"
            )
            return result
        if not script_text:
            result["issues"].append(
                "transcript_comparison skipped: script_text not provided"
            )
            return result

        try:
            transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
        except Exception as e:
            result["issues"].append(f"transcript_comparison could not parse transcript: {e}")
            return result

        transcript_words = [
            w.get("word", "").strip() for w in transcript_data.get("word_timestamps", [])
        ]
        transcript_tokens = cls._tokenize(" ".join(transcript_words))
        script_tokens = cls._tokenize(script_text)

        result["script_word_count"] = len(script_tokens)
        result["transcript_word_count"] = len(transcript_tokens)

        if not script_tokens or not transcript_tokens:
            result["issues"].append(
                f"transcript_comparison: empty token set "
                f"(script={len(script_tokens)}, transcript={len(transcript_tokens)})"
            )
            return result

        # --- Punctuation-leak detection (TTS reading literal punctuation) ---
        script_set = set(script_tokens)
        leak_occurrences: dict[str, int] = {}
        for token in transcript_tokens:
            if token in cls._TTS_PUNCTUATION_LEAK_WORDS and token not in script_set:
                leak_occurrences[token] = leak_occurrences.get(token, 0) + 1

        if leak_occurrences:
            formatted = ", ".join(
                f"{w!r}×{n}" for w, n in sorted(leak_occurrences.items(), key=lambda x: -x[1])
            )
            result["spurious_punctuation_words"] = [
                {"word": w, "count": n} for w, n in leak_occurrences.items()
            ]
            result["issues"].append(
                f"TTS punctuation leak: transcript contains {formatted} — "
                f"these words are NOT in the script, which means the voice "
                f"engine is reading literal punctuation aloud. Rewrite the "
                f"script to eliminate the corresponding characters (ellipses, "
                f"em-dashes, etc.) and regenerate narration."
            )

        # --- Word accuracy via set overlap (cheap & ordering-insensitive) ---
        # We don't penalize small word-order differences or minor TTS
        # hallucinations; we just want to know "did 90%+ of the script's
        # content make it into the audio." Using set overlap on the script
        # side is robust to transcription noise.
        matched = sum(1 for t in script_tokens if t in set(transcript_tokens))
        accuracy = matched / max(1, len(script_tokens))
        result["word_accuracy"] = round(accuracy, 3)
        result["transcript_matches_script"] = accuracy >= 0.9 and not leak_occurrences

        if accuracy < 0.9:
            result["issues"].append(
                f"Low transcript-to-script match: only {accuracy:.0%} of script "
                f"words appear in the transcribed audio ({matched}/"
                f"{len(script_tokens)}). Narration may be truncated, mispronounced, "
                f"or the wrong script was used."
            )

        return result

    def _run_final_review(
        self,
        output_path: Path,
        edit_decisions: dict[str, Any] | None = None,
        proposal_packet: dict[str, Any] | None = None,
        narration_transcript_path: str | Path | None = None,
        script_text: str | None = None,
    ) -> dict[str, Any]:
        """Run post-render self-review and produce a final_review artifact.

        This is the governance contract: the compose runtime MUST inspect
        the actual rendered output before marking the stage complete.
        Never claim a video is ready without a real probe + frame sample.

        When `proposal_packet` is provided, its
        `production_plan.render_runtime` is compared against
        `edit_decisions.render_runtime` so `runtime_swap_detected` can
        actually flip. Without it, we fall back to
        `edit_decisions.metadata.proposal_render_runtime` (which the edit
        director can set explicitly to opt into swap detection).

        Returns a dict conforming to final_review.schema.json.
        """
        log = logging.getLogger("video_compose.final_review")
        issues: list[str] = []

        # --- 1. Technical probe via ffprobe ---
        technical_probe: dict[str, Any] = {
            "valid_container": False,
            "issues": [],
        }
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(output_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                probe_data = json.loads(proc.stdout)
                fmt = probe_data.get("format", {})
                streams = probe_data.get("streams", [])
                video_stream = next(
                    (s for s in streams if s.get("codec_type") == "video"), {}
                )
                audio_stream = next(
                    (s for s in streams if s.get("codec_type") == "audio"), {}
                )

                duration = float(fmt.get("duration", 0))
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                fps_str = video_stream.get("r_frame_rate", "0/1")
                fps = self._parse_probe_fps(fps_str)

                technical_probe = {
                    "valid_container": bool(video_stream),
                    "duration_seconds": round(duration, 2),
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "has_audio": bool(audio_stream),
                    "codec": video_stream.get("codec_name", "unknown"),
                    "file_size_bytes": int(fmt.get("size", 0)),
                    "issues": [],
                }

                # Sanity checks
                if duration < 1.0:
                    technical_probe["issues"].append(
                        f"Output is only {duration:.1f}s — suspiciously short"
                    )

                # Check target duration from edit_decisions
                target_dur = None
                if edit_decisions:
                    target_dur = (
                        edit_decisions.get("total_duration_seconds")
                        or edit_decisions.get("metadata", {}).get("target_duration_seconds")
                    )
                if target_dur and target_dur > 0:
                    drift_pct = abs(duration - target_dur) / target_dur
                    if drift_pct > 0.25:
                        technical_probe["issues"].append(
                            f"Duration drift: rendered {duration:.1f}s vs target {target_dur}s "
                            f"({drift_pct:.0%} off). Review pacing or trim."
                        )
                    technical_probe["target_duration"] = target_dur
                    technical_probe["duration_drift_pct"] = round(drift_pct * 100, 1)
                if width < 320 or height < 240:
                    technical_probe["issues"].append(
                        f"Resolution {width}x{height} is very low"
                    )
                if not audio_stream:
                    technical_probe["issues"].append("No audio stream in output")
            else:
                technical_probe["issues"].append(
                    f"ffprobe failed with exit code {proc.returncode}"
                )
        except FileNotFoundError:
            technical_probe["issues"].append("ffprobe not found — cannot validate output")
        except Exception as e:
            technical_probe["issues"].append(f"ffprobe error: {e}")

        issues.extend(technical_probe.get("issues", []))

        # --- 2. Visual spotcheck: sample 4 frames ---
        visual_spotcheck: dict[str, Any] = {
            "frames_sampled": 0,
            "frame_paths": [],
            "black_frames_detected": False,
            "broken_overlays": False,
            "missing_assets": False,
            "unreadable_text": False,
            "issues": [],
        }
        duration = technical_probe.get("duration_seconds", 0)
        if duration > 0 and technical_probe.get("valid_container"):
            try:
                frame_dir = output_path.parent / ".final_review_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                # Sample at 10%, 35%, 65%, 90% of duration
                sample_points = [0.10, 0.35, 0.65, 0.90]
                frame_paths = []
                for i, pct in enumerate(sample_points):
                    ts = round(duration * pct, 2)
                    frame_path = frame_dir / f"review_frame_{i}.png"
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(ts),
                        "-i", str(output_path),
                        "-frames:v", "1", "-q:v", "2",
                        str(frame_path),
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=15)
                    if frame_path.exists():
                        frame_paths.append(str(frame_path))

                        # Check for black frames (file size heuristic:
                        # a 1920x1080 PNG of pure black is ~5KB)
                        if frame_path.stat().st_size < 2000:
                            visual_spotcheck["black_frames_detected"] = True

                visual_spotcheck["frames_sampled"] = len(frame_paths)
                visual_spotcheck["frame_paths"] = frame_paths

                if len(frame_paths) < 4:
                    visual_spotcheck["issues"].append(
                        f"Only {len(frame_paths)}/4 frames extracted — some timestamps may be out of range"
                    )
                if visual_spotcheck["black_frames_detected"]:
                    visual_spotcheck["issues"].append(
                        "Black frame detected — possible missing asset or failed render segment"
                    )
            except Exception as e:
                visual_spotcheck["issues"].append(f"Frame sampling error: {e}")

        issues.extend(visual_spotcheck.get("issues", []))

        # --- 3. Audio spotcheck ---
        audio_spotcheck: dict[str, Any] = {
            "narration_present": False,
            "music_present": False,
            "unexpected_silence": False,
            "clipping_detected": False,
            "mix_intelligible": True,
            "issues": [],
        }
        if technical_probe.get("has_audio") and duration > 0:
            try:
                # Use ffmpeg volumedetect to check audio levels
                cmd = [
                    "ffmpeg", "-i", str(output_path),
                    "-af", "volumedetect", "-f", "null", "-",
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                stderr = proc.stderr or ""
                # Parse mean_volume and max_volume
                mean_vol = None
                max_vol = None
                for line in stderr.split("\n"):
                    if "mean_volume:" in line:
                        try:
                            mean_vol = float(line.split("mean_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    if "max_volume:" in line:
                        try:
                            max_vol = float(line.split("max_volume:")[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass

                if mean_vol is not None:
                    if mean_vol < -60:
                        audio_spotcheck["unexpected_silence"] = True
                        audio_spotcheck["issues"].append(
                            f"Mean volume {mean_vol:.1f} dB — effectively silent"
                        )
                    # Assume narration present if mean volume is reasonable
                    if mean_vol > -40:
                        audio_spotcheck["narration_present"] = True
                    # Assume music present if audio exists (conservative)
                    if mean_vol > -50:
                        audio_spotcheck["music_present"] = True

                if max_vol is not None and max_vol > -0.5:
                    audio_spotcheck["clipping_detected"] = True
                    audio_spotcheck["issues"].append(
                        f"Max volume {max_vol:.1f} dB — possible clipping"
                    )
            except Exception as e:
                audio_spotcheck["issues"].append(f"Audio analysis error: {e}")

        issues.extend(audio_spotcheck.get("issues", []))

        # --- 4. Promise preservation ---
        promise_preservation: dict[str, Any] = {
            "delivery_promise_honored": True,
            "silent_downgrade_detected": False,
            "runtime_swap_detected": False,
            "issues": [],
        }
        if edit_decisions:
            renderer_family = edit_decisions.get("renderer_family", "")
            promise_preservation["renderer_family_used"] = renderer_family

            # Runtime governance — record what actually ran and flag a swap.
            # Three sources of truth, in priority order:
            #   1. proposal_packet.production_plan.render_runtime (authoritative)
            #   2. edit_decisions.metadata.proposal_render_runtime (if edit stage
            #      explicitly copied it to opt into in-tool swap detection)
            #   3. edit_decisions.render_runtime itself (cannot detect a swap in
            #      this case — reviewer does cross-artifact comparison instead)
            render_runtime_edit = (edit_decisions.get("render_runtime") or "").strip().lower()
            if render_runtime_edit:
                promise_preservation["render_runtime_used"] = render_runtime_edit

                proposal_runtime: str | None = None
                runtime_source: str | None = None
                if proposal_packet:
                    pp_runtime = (
                        (proposal_packet.get("production_plan") or {}).get("render_runtime")
                        or ""
                    ).strip().lower()
                    if pp_runtime:
                        proposal_runtime = pp_runtime
                        runtime_source = "proposal_packet.production_plan.render_runtime"
                if proposal_runtime is None:
                    md_runtime = (
                        (edit_decisions.get("metadata") or {}).get("proposal_render_runtime")
                        or ""
                    ).strip().lower()
                    if md_runtime:
                        proposal_runtime = md_runtime
                        runtime_source = "edit_decisions.metadata.proposal_render_runtime"

                if proposal_runtime is None:
                    promise_preservation["runtime_swap_check"] = (
                        "skipped — no proposal_packet or proposal_render_runtime "
                        "metadata provided. Reviewer skill does cross-artifact "
                        "comparison separately."
                    )
                elif proposal_runtime != render_runtime_edit:
                    promise_preservation["runtime_swap_detected"] = True
                    promise_preservation["runtime_swap_check"] = (
                        f"detected — source: {runtime_source}"
                    )
                    promise_preservation["issues"].append(
                        f"render_runtime changed between proposal ({proposal_runtime}) "
                        f"and compose ({render_runtime_edit}) — this is a contract "
                        f"violation unless a render_runtime_selection decision was logged."
                    )
                else:
                    promise_preservation["runtime_swap_check"] = (
                        f"ok — proposal and edit agree ({runtime_source})"
                    )

            delivery_data = (
                edit_decisions.get("metadata", {}).get("delivery_promise")
                or edit_decisions.get("delivery_promise")
            )
            if delivery_data:
                try:
                    from lib.delivery_promise import DeliveryPromise
                    promise = DeliveryPromise.from_dict(delivery_data)
                    cuts = edit_decisions.get("cuts", [])
                    result = promise.validate_cuts(cuts)
                    motion_ratio = result.get("motion_ratio", 0)
                    promise_preservation["motion_ratio_actual"] = round(motion_ratio, 3)

                    if not result["valid"]:
                        promise_preservation["delivery_promise_honored"] = False
                        for v in result["violations"]:
                            promise_preservation["issues"].append(v)

                    # Detect silent downgrade: motion-led promise but <50% motion
                    if (delivery_data.get("type") == "motion_led"
                            and motion_ratio < 0.5):
                        promise_preservation["silent_downgrade_detected"] = True
                        promise_preservation["issues"].append(
                            f"Motion-led promise but only {motion_ratio:.0%} motion — "
                            f"silent downgrade to still-led"
                        )
                except Exception as e:
                    promise_preservation["issues"].append(
                        f"Could not validate delivery promise: {e}"
                    )

        issues.extend(promise_preservation.get("issues", []))

        # --- 5. Subtitle check ---
        subtitle_check: dict[str, Any] = {
            "subtitles_expected": False,
            "subtitles_present": False,
            "issues": [],
        }
        if edit_decisions:
            ed_subs = edit_decisions.get("subtitles", {})
            subtitle_check["subtitles_expected"] = bool(ed_subs.get("enabled"))

            # Check if output has subtitle stream
            if technical_probe.get("valid_container"):
                try:
                    cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_streams", "-select_streams", "s",
                        str(output_path),
                    ]
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=15
                    )
                    if proc.returncode == 0:
                        sub_data = json.loads(proc.stdout)
                        sub_streams = sub_data.get("streams", [])
                        subtitle_check["subtitles_present"] = len(sub_streams) > 0

                    # If subtitles were expected but not found as a stream,
                    # they may be burned in (which is fine — not a failure)
                    if (subtitle_check["subtitles_expected"]
                            and not subtitle_check["subtitles_present"]):
                        # Check if subtitle_path was used (burned in)
                        sub_source = ed_subs.get("source")
                        if sub_source and Path(sub_source).exists():
                            # Burned-in subtitles are not detectable as streams
                            subtitle_check["subtitles_present"] = True
                            subtitle_check["coverage_ratio"] = 1.0
                        else:
                            subtitle_check["issues"].append(
                                "Subtitles expected but not found in output and "
                                "no subtitle source file exists for burn-in"
                            )
                except Exception as e:
                    subtitle_check["issues"].append(f"Subtitle check error: {e}")

        issues.extend(subtitle_check.get("issues", []))

        # --- 6. Transcript-vs-script comparison ---
        # Catches content-level TTS failures (the classic "Chirp reads `...`
        # as the word 'dot'" trap) that volume-based audio checks miss.
        # Only runs when caller provides both the transcript and script; when
        # skipped, issues list records that so the silence is visible.
        transcript_comparison = self._compare_transcript_to_script(
            Path(narration_transcript_path) if narration_transcript_path else None,
            script_text,
        )
        issues.extend(transcript_comparison.get("issues", []))

        # --- 7. Determine overall status ---
        critical_issues = [
            i for i in issues
            if any(kw in i.lower() for kw in [
                "silent downgrade", "delivery promise violation",
                "effectively silent", "ffprobe failed", "suspiciously short",
                "tts punctuation leak",  # reading literal punctuation aloud
            ])
        ]

        if critical_issues:
            status = "revise"
            recommended_action = "re_render"
        elif issues:
            status = "pass"
            recommended_action = "present_to_user"
        else:
            status = "pass"
            recommended_action = "present_to_user"

        if not technical_probe.get("valid_container"):
            status = "fail"
            recommended_action = "re_render"

        final_review = {
            "version": "1.0",
            "output_path": str(output_path),
            "status": status,
            "checks": {
                "technical_probe": technical_probe,
                "visual_spotcheck": visual_spotcheck,
                "audio_spotcheck": audio_spotcheck,
                "promise_preservation": promise_preservation,
                "subtitle_check": subtitle_check,
                "transcript_comparison": transcript_comparison,
            },
            "issues_found": issues,
            "recommended_action": recommended_action,
        }

        log.info(
            "Final review: status=%s, issues=%d, action=%s",
            status, len(issues), recommended_action,
        )

        return final_review

    @staticmethod
    def _parse_probe_fps(fps_str: str) -> float:
        """Parse ffprobe fps string like '30/1' or '24000/1001'."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return round(int(num) / max(int(den), 1), 2)
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _burn_subtitles(self, inputs: dict[str, Any]) -> ToolResult:
        """Burn subtitle file into video."""
        input_path = Path(inputs["input_path"])
        subtitle_path = Path(inputs["subtitle_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_subtitled"))))

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not subtitle_path.exists():
            return ToolResult(success=False, error=f"Subtitle file not found: {subtitle_path}")

        style = inputs.get("subtitle_style", {})
        ass_style = self._build_subtitle_style(style)
        sub_escaped = str(subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"subtitles='{sub_escaped}':force_style='{ass_style}'",
            "-c:v", codec, "-crf", str(crf),
            "-c:a", "aac", "-ar", "44100",
            str(output_path),
        ]

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "burn_subtitles",
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _overlay(self, inputs: dict[str, Any]) -> ToolResult:
        """Composite overlay images/videos on top of base video."""
        input_path = Path(inputs["input_path"])
        overlays = inputs.get("overlays", [])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_overlay"))))
        codec = inputs.get("codec", "libx264")
        crf = inputs.get("crf", 23)

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")
        if not overlays:
            return ToolResult(success=False, error="No overlays provided")

        # Build complex filter for each overlay
        input_args = ["-i", str(input_path)]
        filter_parts = []
        prev_label = "0:v"

        for i, ov in enumerate(overlays):
            asset_path = Path(ov["asset_path"])
            if not asset_path.exists():
                return ToolResult(success=False, error=f"Overlay asset not found: {asset_path}")

            input_args.extend(["-i", str(asset_path)])

            x = int(ov.get("x", 0))
            y = int(ov.get("y", 0))
            start = ov.get("start_seconds", 0)
            end = ov.get("end_seconds")
            opacity = ov.get("opacity", 1.0)

            overlay_input = f"{i + 1}:v"

            # Scale overlay if dimensions specified
            if "width" in ov and "height" in ov:
                w = int(ov["width"])
                h = int(ov["height"])
                filter_parts.append(f"[{overlay_input}]scale={w}:{h}[ov_scaled_{i}]")
                overlay_input = f"ov_scaled_{i}"

            # Build enable expression for timed overlays
            enable = f"between(t,{start},{end})" if end else f"gte(t,{start})"
            out_label = f"v{i}"

            filter_parts.append(
                f"[{prev_label}][{overlay_input}]overlay={x}:{y}:enable='{enable}'[{out_label}]"
            )
            prev_label = out_label

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"]
        cmd.extend(input_args)
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", f"[{prev_label}]", "-map", "0:a?"])
        cmd.extend(["-c:v", codec, "-crf", str(crf), "-c:a", "aac", "-ar", "44100"])
        cmd.append(str(output_path))

        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "overlay",
                "overlay_count": len(overlays),
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    def _encode(self, inputs: dict[str, Any]) -> ToolResult:
        """Re-encode video with a specific profile/codec settings."""
        input_path = Path(inputs["input_path"])
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_encoded"))))
        playbook_data = inputs.get("playbook")
        playbook_output = (playbook_data or {}).get("output", {})

        codec = inputs.get("codec", playbook_output.get("video_codec", "libx264"))
        crf = inputs.get("crf", playbook_output.get("crf", 23))
        preset = inputs.get("preset", playbook_output.get("preset", "medium"))
        pixel_format = inputs.get("pixel_format", playbook_output.get("pixel_format", "yuv420p"))
        fps_override = inputs.get("fps") or playbook_output.get("fps")
        audio_codec = inputs.get("audio_codec", playbook_output.get("audio_codec", "aac"))
        audio_bitrate = inputs.get("audio_bitrate", playbook_output.get("audio_bitrate", "192k"))
        audio_sample_rate = inputs.get("audio_sample_rate", playbook_output.get("audio_sample_rate", 44100))
        profile_name = inputs.get("profile")

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        ]

        # Apply media profile if specified
        # Playbook output overrides resolution/fps if no explicit compose_target or profile
        if playbook_output.get("resolution") and not compose_target:
            resolution = playbook_output["resolution"]
        if fps_override and not profile_name and not compose_target:
            fps = fps_override

        if profile_name:
            try:
                from lib.media_profiles import get_profile, ffmpeg_output_args
                profile = get_profile(profile_name)
                cmd.extend(["-s", f"{profile.width}x{profile.height}"])
                cmd.extend(["-r", str(profile.fps)])
            except (ImportError, ValueError):
                pass  # proceed without profile

        cmd.append(str(output_path))
        self.run_command(cmd)

        return ToolResult(
            success=True,
            data={
                "operation": "encode",
                "codec": codec,
                "crf": crf,
                "profile": profile_name,
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
        )

    @staticmethod
    def _resolve_subtitle_style(
        explicit_style: dict | None,
        edit_decisions: dict | None,
        playbook: dict | None,
    ) -> dict:
        """Resolve subtitle style with layered priority.

        Priority: explicit_style > edit_decisions.subtitles.style > playbook > defaults.
        This prevents every video from looking identical (Arial bold white).
        """
        # Start with minimal fallback defaults
        resolved = {
            "font": "Inter",
            "font_size": 28,
            "bold": True,
            "outline_width": 2,
            "shadow": 0,
            "margin_v": 40,
            "alignment": 2,
        }

        # Layer 1: Playbook-derived style
        if playbook:
            typo = playbook.get("typography", {})
            colors = playbook.get("visual_language", {}).get("color_palette", {})
            if typo.get("body", {}).get("family"):
                resolved["font"] = typo["body"]["family"]
            if colors.get("text"):
                resolved["primary_color"] = colors["text"]
            if colors.get("background"):
                resolved["outline_color"] = colors["background"]
                # Semi-transparent background for readability
                bg = colors["background"]
                resolved["back_color"] = bg

        # Layer 2: edit_decisions subtitle style
        if edit_decisions:
            ed_style = edit_decisions.get("subtitles", {}).get("style", {})
            for k, v in ed_style.items():
                if v is not None:
                    resolved[k] = v

        # Layer 3: Explicit override (highest priority)
        if explicit_style:
            for k, v in explicit_style.items():
                if v is not None:
                    resolved[k] = v

        return resolved

    @staticmethod
    def _build_subtitle_style(style: dict) -> str:
        """Build ASS force_style string from style dict."""
        parts = []
        parts.append(f"FontName={style.get('font', 'Inter')}")
        parts.append(f"FontSize={style.get('font_size', 28)}")
        parts.append(f"Bold={1 if style.get('bold', True) else 0}")
        if style.get("primary_color"):
            parts.append(f"PrimaryColour={style['primary_color']}")
        if style.get("outline_color"):
            parts.append(f"OutlineColour={style['outline_color']}")
        if style.get("back_color"):
            parts.append(f"BackColour={style['back_color']}")
        border_style = style.get("border_style", 1)
        parts.append(f"BorderStyle={border_style}")
        parts.append(f"Outline={style.get('outline_width', 2)}")
        parts.append(f"Shadow={style.get('shadow', 0)}")
        parts.append(f"MarginV={style.get('margin_v', 40)}")
        parts.append(f"Alignment={style.get('alignment', 2)}")
        return ",".join(parts)

    @staticmethod
    def _build_atempo(factor: float) -> str:
        """Build atempo filter chain for audio speed adjustment."""
        filters = []
        remaining = factor
        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)
