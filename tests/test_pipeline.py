import pytest
from schemas.timeline import TimelineEvent
from schemas.vision import Scene
from fusion.merger import merge_and_sort_timeline, format_timestamp
from vision.vlm import MockVLMClient
from vision.ocr import extract_ocr_from_vlm_text
from vision.actions import (
    extract_actions_from_vlm_text,
    extract_description_from_vlm_text,
    extract_objects_from_vlm_text
)
from app.config import config

def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(4.5) == "00:04"
    assert format_timestamp(61.5) == "01:01"
    assert format_timestamp(3665) == "01:01:05"

def test_vlm_parsers():
    vlm_text = """
DESCRIPTION: A test scene depicting code on screen.
OCR: ["def test_code", "assert True"]
ACTIONS: ["scrolling", "typing"]
OBJECTS: ["keyboard", "monitor"]
"""
    assert extract_description_from_vlm_text(vlm_text) == "A test scene depicting code on screen."
    assert extract_ocr_from_vlm_text(vlm_text) == ["def test_code", "assert True"]
    assert extract_actions_from_vlm_text(vlm_text) == ["scrolling", "typing"]
    assert extract_objects_from_vlm_text(vlm_text) == ["keyboard", "monitor"]

def test_vlm_parsers_fallback():
    # Irregular formatting fallbacks
    vlm_text = """
OCR: [word1, word2, word3]
ACTIONS: "singing, dancing"
"""
    assert extract_ocr_from_vlm_text(vlm_text) == ["word1", "word2", "word3"]
    assert extract_actions_from_vlm_text(vlm_text) == ["singing", "dancing"]

@pytest.mark.asyncio
async def test_mock_vlm_client():
    client = MockVLMClient()
    res = await client.describe_frame(b"dummy_bytes", "prompt")
    assert "DESCRIPTION:" in res
    assert "OCR:" in res
    assert "ACTIONS:" in res
    assert "OBJECTS:" in res

def test_timeline_merger():
    audio_segments = [
        {"start": 1.5, "end": 4.0, "text": "Hello world!"},
        {"start": 5.0, "end": 7.0, "text": "This is a test speech segment."}
    ]
    
    scenes = [
        Scene(
            scene_id=0,
            start=0.0,
            end=4.0,
            description="Office scene visual representation.",
            ocr=["WHITEBOARD"],
            actions=["writing"]
        ),
        Scene(
            scene_id=1,
            start=4.0,
            end=8.0,
            description="Close up of screen.",
            ocr=["CODE_TEXT"],
            actions=["typing"]
        )
    ]
    
    events = merge_and_sort_timeline(audio_segments, scenes)
    
    # Assert time sorting
    timestamps = [e.time_seconds for e in events]
    assert sorted(timestamps) == timestamps
    
    # Assert sources are unified
    sources = {e.source for e in events}
    assert "audio" in sources
    assert "vision" in sources
    assert "ocr" not in sources
    
    # Assert deduplication
    dup_scenes = [
        Scene(scene_id=0, start=1.0, end=3.0, description="D1", ocr=["DUP_TXT"]),
        Scene(scene_id=1, start=1.1, end=3.1, description="D1", ocr=["DUP_TXT"])
    ]
    collapsed = merge_and_sort_timeline([], dup_scenes)
    
    vis_events = [e for e in collapsed if e.source == "vision"]
    
    # Near-duplicates should collapse
    assert len(vis_events) == 1

@pytest.mark.asyncio
async def test_generate_captions_mock():
    from app.config import config
    from llm.summarize import generate_captions
    
    original_provider = config.vlm_provider
    config.vlm_provider = "mock"
    try:
        styles = ["formal", "sarcastic", "humorous_tech"]
        captions = await generate_captions("mock transcript", "mock timeline", styles)
        assert len(captions) == 3
        assert "formal" in captions
        assert "sarcastic" in captions
        assert "humorous_tech" in captions
        assert isinstance(captions["formal"], str)
        assert len(captions["formal"]) > 5
    finally:
        config.vlm_provider = original_provider
