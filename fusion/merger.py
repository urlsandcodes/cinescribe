from typing import List, Dict, Any
from schemas.timeline import TimelineEvent
from schemas.vision import Scene

def format_timestamp(seconds: float) -> str:
    """Formats float seconds into a user-friendly MM:SS or HH:MM:SS string."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def merge_and_sort_timeline(
    audio_segments: List[Dict[str, Any]],
    scenes: List[Scene]
) -> List[TimelineEvent]:
    """
    Collects events from audio transcripts, scene descriptions, and OCR detections.
    Normalizes time representation, sorts chronologically, and collapses close duplicates.
    """
    raw_events: List[TimelineEvent] = []
    
    # 1. Process audio events
    for segment in audio_segments:
        start_sec = float(segment.get("start", 0.0))
        text = segment.get("text", "").strip()
        if text:
            raw_events.append(TimelineEvent(
                time_seconds=start_sec,
                time_display=format_timestamp(start_sec),
                event=f"Speech: \"{text}\"",
                source="audio"
            ))
            
    # 2. Process scene descriptions & OCR events
    for scene in scenes:
        start_sec = scene.start
        time_str = format_timestamp(start_sec)
        
        # Scene Visual Description
        if scene.description:
            raw_events.append(TimelineEvent(
                time_seconds=start_sec,
                time_display=time_str,
                event=f"Visual: {scene.description}",
                source="vision"
            ))
            
        # OCR Text Detected
        for ocr_text in scene.ocr:
            raw_events.append(TimelineEvent(
                time_seconds=start_sec,
                time_display=time_str,
                event=f"OCR text: \"{ocr_text}\"",
                source="ocr"
            ))
            
    # Sort chronologically by start seconds
    raw_events.sort(key=lambda x: x.time_seconds)
    
    # Collapse near-duplicates (within 2-second window with same source and message content)
    collapsed_events: List[TimelineEvent] = []
    for event in raw_events:
        if not collapsed_events:
            collapsed_events.append(event)
            continue
            
        # Find the last event of the same source in collapsed_events
        last_same_source = None
        for prev in reversed(collapsed_events):
            if prev.source == event.source:
                last_same_source = prev
                break
                
        if last_same_source:
            time_delta = abs(event.time_seconds - last_same_source.time_seconds)
            is_duplicate = (event.event == last_same_source.event)
            if time_delta <= 2.0 and is_duplicate:
                # Skip duplicate within the window
                continue
            
        collapsed_events.append(event)
        
    return collapsed_events
