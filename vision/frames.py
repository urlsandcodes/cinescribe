import os
from pathlib import Path
from app.config import config
from app.logger import logger
from media.ffmpeg import extract_frame

def select_and_extract_frames(video_path: str, video_id: str, scenes: list[tuple[float, float]]) -> list[dict]:
    """
    Selects the midpoint timestamp of each scene as its representative frame.
    Extracts the frame at that timestamp to scratch space and reads its raw bytes.
    Returns a list of dictionaries containing scene metadata, path, and frame bytes.
    """
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_frames = []
    
    for idx, (start, end) in enumerate(scenes):
        # Choose midpoint
        midpoint = start + (end - start) / 2
        
        filename = f"{video_id}_scene_{idx}_at_{midpoint:.2f}.jpg"
        frame_path = temp_dir / filename
        
        success = extract_frame(video_path, midpoint, str(frame_path))
        if success:
            try:
                with open(frame_path, "rb") as f:
                    img_bytes = f.read()
                
                extracted_frames.append({
                    "scene_id": idx,
                    "start": start,
                    "end": end,
                    "timestamp": midpoint,
                    "path": str(frame_path.resolve()),
                    "bytes": img_bytes
                })
            except Exception as e:
                logger.error(f"Could not read extracted frame file {frame_path}: {e}")
        else:
            logger.warning(f"Failed to extract representative frame for scene {idx} at timestamp {midpoint:.2f}s")
            
    logger.info(f"Extracted {len(extracted_frames)} representative frame images out of {len(scenes)} scenes.")
    return extracted_frames
