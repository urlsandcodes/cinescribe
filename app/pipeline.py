import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from schemas.result import VideoResult
from schemas.audio import AudioAnalysis
from schemas.vision import Scene
from app.config import config
from app.logger import logger

# Import stage workers
from media.downloader import download_if_url
from media import ffmpeg
from workers.executors import run_in_cpu_pool
from audio.detector import analyze_audio
from vision.scenes import detect_scenes
from vision.frames import select_and_extract_frames
from vision.vlm import get_vlm_client
from vision.ocr import extract_ocr_from_vlm_text
from vision.actions import (
    extract_actions_from_vlm_text,
    extract_description_from_vlm_text,
    extract_objects_from_vlm_text
)
from fusion.timeline import build_timeline
from llm.summarize import summarize_video, generate_captions
from llm.prompts import VLM_FRAME_PROMPT

async def run_pipeline(video_id: str, video_source: str, styles: List[str] = None) -> VideoResult:
    """
    Orchestrates the modular analysis pipeline stages for a single video.
    Tracks performance, supports graceful degradation, and cleans up scratch files.
    """
    logger.info(f"Initializing pipeline execution for: {video_source}")
    
    # Track scratch files to delete upon completion
    scratch_files: List[str] = []
    
    # Store performance timings and stage errors
    timings_ms: Dict[str, int] = {}
    stage_errors: Dict[str, str] = {}
    
    # Initialize variables for stages
    local_video_path = None
    metadata = {}
    audio_analysis = None
    audio_segments = []
    scenes = []
    scene_objects: List[Scene] = []
    timeline = []
    summary = None
    detailed_summary = None
    tags = []
    status = "ok"
    
    try:
        # 1. DOWNLOAD STAGE (Required)
        start_time = time.time()
        try:
            local_video_path = await download_if_url(video_source, video_id)
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            
            # If downloaded file resides in our temp directory, register it for deletion
            if local_video_path.startswith(str(Path(config.temp_dir).resolve())):
                scratch_files.append(local_video_path)
        except Exception as e:
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required download stage failed: {e}")
            raise
            
        # 2. METADATA STAGE (Required)
        start_time = time.time()
        try:
            metadata = await run_in_cpu_pool(ffmpeg.extract_metadata, local_video_path)
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required metadata extraction stage failed: {e}")
            raise
            
        duration = metadata.get("duration", 0.0)
        has_audio = metadata.get("has_audio", False)
        
        # 3. AUDIO EXTRACTION STAGE (Optional)
        audio_extracted = False
        audio_output_path = str(Path(config.temp_dir) / f"{video_id}_audio.mp3")
        
        if has_audio:
            start_time = time.time()
            try:
                await run_in_cpu_pool(ffmpeg.extract_audio, local_video_path, audio_output_path)
                scratch_files.append(audio_output_path)
                timings_ms["extract_audio"] = int((time.time() - start_time) * 1000)
                audio_extracted = True
            except Exception as e:
                status = "partial"
                timings_ms["extract_audio"] = int((time.time() - start_time) * 1000)
                stage_errors["extract_audio"] = str(e)
                logger.warning(f"Audio extraction stage failed: {e}")
                
        # 4. TRANSCRIPTION STAGE (Optional)
        if audio_extracted:
            start_time = time.time()
            try:
                audio_analysis, audio_segments = await analyze_audio(audio_output_path)
                timings_ms["transcribe"] = int((time.time() - start_time) * 1000)
            except Exception as e:
                status = "partial"
                timings_ms["transcribe"] = int((time.time() - start_time) * 1000)
                stage_errors["transcribe"] = str(e)
                logger.warning(f"Audio transcription stage failed: {e}")
                
        # 5. SCENE DETECTION STAGE (Optional)
        scenes_detected = False
        start_time = time.time()
        try:
            scenes = await run_in_cpu_pool(detect_scenes, local_video_path, video_id, duration)
            timings_ms["scene_detect"] = int((time.time() - start_time) * 1000)
            scenes_detected = True
        except Exception as e:
            status = "partial"
            timings_ms["scene_detect"] = int((time.time() - start_time) * 1000)
            stage_errors["scene_detect"] = str(e)
            logger.warning(f"Scene detection stage failed: {e}")
            # Fall back to single default scene block for frame extractor
            scenes = [(0.0, duration)]
            
        # 6. FRAME EXTRACTION STAGE (Optional)
        frames_extracted = []
        if scenes:
            start_time = time.time()
            try:
                frames_extracted = await run_in_cpu_pool(select_and_extract_frames, local_video_path, video_id, scenes)
                timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
                # Register frame files for cleanup
                for f in frames_extracted:
                    if f.get("path"):
                        scratch_files.append(f["path"])
            except Exception as e:
                status = "partial"
                timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
                stage_errors["frame_extract"] = str(e)
                logger.warning(f"Frame extraction stage failed: {e}")
                
        # 7. VLM FRAME DESCRIPTION STAGE (Optional)
        if frames_extracted:
            start_time = time.time()
            try:
                vlm_client = get_vlm_client()
                images = [f["bytes"] for f in frames_extracted]
                
                descriptions = await vlm_client.describe_frames_batch(images, VLM_FRAME_PROMPT)
                timings_ms["vlm"] = int((time.time() - start_time) * 1000)
                
                # Parse results for each scene segment
                for frame, raw_desc in zip(frames_extracted, descriptions):
                    desc = extract_description_from_vlm_text(raw_desc)
                    ocr = extract_ocr_from_vlm_text(raw_desc)
                    actions = extract_actions_from_vlm_text(raw_desc)
                    objects = extract_objects_from_vlm_text(raw_desc)
                    
                    scene_objects.append(Scene(
                         scene_id=frame["scene_id"],
                         start=frame["start"],
                         end=frame["end"],
                         description=desc,
                         objects=objects,
                         actions=actions,
                         ocr=ocr
                    ))

                # Log scene description table
                table_lines = [
                    "\n==========================================================================",
                    "DETECTED SCENES & VLM DESCRIPTIONS",
                    "==========================================================================",
                    f"+---------+---------+---------+--------------------------------------------------------+",
                    f"| Scene # | Start   | End     | VLM Frame Description & Details                        |",
                    f"+---------+---------+---------+--------------------------------------------------------+"
                ]
                for sc in scene_objects:
                    # Truncate description if too long for the table cell
                    desc_snippet = sc.description[:52] + "..." if len(sc.description) > 52 else sc.description.ljust(55)
                    table_lines.append(f"| {str(sc.scene_id).ljust(7)} | {f'{sc.start:.2f}s'.ljust(7)} | {f'{sc.end:.2f}s'.ljust(7)} | {desc_snippet.ljust(54)} |")
                    if sc.ocr:
                        table_lines.append(f"|         |         |         |   OCR Detected: {str(sc.ocr)[:50].ljust(50)} |")
                    if sc.actions:
                        table_lines.append(f"|         |         |         |   Actions: {str(sc.actions)[:52].ljust(52)} |")
                table_lines.append(f"+---------+---------+---------+--------------------------------------------------------+")
                table_lines.append("==========================================================================\n")
                logger.info("\n".join(table_lines))

            except Exception as e:
                status = "partial"
                timings_ms["vlm"] = int((time.time() - start_time) * 1000)
                stage_errors["vlm"] = str(e)
                logger.warning(f"VLM processing stage failed: {e}")
                
        # 8. TIMELINE MERGE STAGE (Optional)
        start_time = time.time()
        try:
            timeline = build_timeline(audio_segments, scene_objects)
            timings_ms["timeline"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            status = "partial"
            timings_ms["timeline"] = int((time.time() - start_time) * 1000)
            stage_errors["timeline"] = str(e)
            logger.warning(f"Timeline merge stage failed: {e}")
            
        # 9. LLM SUMMARIZATION / CAPTIONS STAGE (Optional)
        start_time = time.time()
        captions = None
        try:
            # Format timeline events as string for the LLM summarizer context
            timeline_str = ""
            for ev in timeline:
                timeline_str += f"[{ev.time_display}] ({ev.source}) {ev.event}\n"
                
            transcript_text = audio_analysis.transcript if audio_analysis else None

            # Log how we knit the information together for context
            knitting_lines = [
                "\n==========================================================================",
                "KNITTING SCENE & AUDIO TIMELINE CONTEXT FOR GENERATION",
                "==========================================================================",
                f"Transcript Text: {transcript_text or '<No Audio/Speech Found>'}",
                f"Timeline Event Sequence:\n" + "\n".join(f"  - [{ev.time_display}] ({ev.source}) {ev.event}" for ev in timeline),
                "==========================================================================\n"
            ]
            logger.info("\n".join(knitting_lines))
            
            if styles:
                captions = await generate_captions(transcript_text, timeline_str, styles)
                summary = captions.get(styles[0], "Video Caption")
                detailed_summary = "\n".join(f"{s}: {cap}" for s, cap in captions.items())
                tags = styles

                # Log final captions by tone
                caption_lines = [
                    "\n==========================================================================",
                    "FINAL GENERATED CAPTIONS BY TONE",
                    "=========================================================================="
                ]
                for tone, cap in captions.items():
                    caption_lines.append(f"[{tone.upper()}]: {cap}")
                caption_lines.append("==========================================================================\n")
                logger.info("\n".join(caption_lines))
            else:
                summary, detailed_summary, tags = await summarize_video(transcript_text, timeline_str)
                
            timings_ms["summarize"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            if styles:
                raise
            status = "partial"
            timings_ms["summarize"] = int((time.time() - start_time) * 1000)
            stage_errors["summarize"] = str(e)
            logger.warning(f"LLM summarizer stage failed: {e}")
            
    except Exception as e:
        status = "failed"
        stage_errors["pipeline"] = str(e)
        logger.error(f"Video pipeline execution failed critically: {e}")
        
    finally:
        # Clean up scratch files
        logger.info(f"Initiating scratch space cleanup for pipeline: {video_id}")
        for path in scratch_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned scratch file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove scratch file {path}: {e}")
                
    # Build and return the final VideoResult object
    return VideoResult(
        id=video_id,
        source=video_source,
        duration=duration if metadata else 0.0,
        status=status,
        transcript=audio_analysis.transcript if audio_analysis else None,
        timeline=timeline,
        summary=summary,
        detailed_summary=detailed_summary,
        tags=tags,
        captions=captions,
        scenes=scene_objects,
        audio=audio_analysis,
        stage_errors=stage_errors,
        timings_ms=timings_ms
    )
