import json
import httpx
import asyncio
from app.config import config
from app.logger import logger
from llm.prompts import LLM_SUMMARIZE_PROMPT

def clean_special_characters(text: str) -> str:
    """Replaces Unicode em-dashes, en-dashes, and curly quotes with standard ASCII equivalents."""
    if not text:
        return ""
    # Replace em-dashes and en-dashes with a standard hyphen (surrounded by spaces)
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace curly quotes with standard straight quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text

async def summarize_video(transcript: str, timeline_str: str) -> tuple[str, str, list[str]]:
    """
    Calls the Fireworks LLM to summarize the video transcript and event timeline.
    Returns a tuple of (summary, detailed_summary, tags).
    """
    provider = config.vlm_provider
    from llm.prompts import LLM_SUMMARIZE_SYSTEM_PROMPT, LLM_SUMMARIZE_USER_PROMPT
    user_prompt = LLM_SUMMARIZE_USER_PROMPT.format(transcript=transcript or "No transcript available.", timeline=timeline_str)
    
    logger.info(f"Calling LLM summarizer via provider: {provider}")
    if provider == "mock":
        logger.info("Using mock LLM summarizer fallback.")
        summary = "A concise walkthrough showcasing a local video intelligence system in mock execution."
        detailed_summary = (
            "The system executes stages in-memory under concurrency constraints. "
            "It extracts frame representations, runs vision-language prompts to describe scenes, "
            "tracks visual and speech timestamps, and merges everything into a structured report."
        )
        tags = ["prototype", "video-analysis", "architecture", "asyncio"]
        return summary, detailed_summary, tags

    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": LLM_SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "reasoning_effort": "none"
            }
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=60.0
                )
                if resp.status_code == 429:
                    logger.warning(f"Fireworks LLM rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]

            # Clean special characters first
            text = clean_special_characters(text)

            # Clean markdown code blocks from the output if present
            text_clean = text.strip()
            if text_clean.startswith("```"):
                lines = text_clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text_clean = "\n".join(lines).strip()

            data = json.loads(text_clean)
            return (
                data.get("summary", "").strip(),
                data.get("detailed_summary", "").strip(),
                [t.strip() for t in data.get("tags", [])]
            )
        except Exception as e:
            logger.warning(f"LLM summarizer API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0

async def generate_captions(transcript: str, timeline_str: str, styles: list[str]) -> dict[str, str]:
    """
    Calls the Fireworks LLM to generate captions for the video in each of the requested styles.
    """
    from llm.prompts import LLM_CAPTION_SYSTEM_PROMPT, LLM_CAPTION_USER_PROMPT
    provider = config.vlm_provider
    user_prompt = LLM_CAPTION_USER_PROMPT.format(
        transcript=transcript or "No transcript available.",
        timeline=timeline_str or "No visual timeline available.",
        styles=", ".join(styles)
    )

    if provider == "mock":
        logger.info("Using mock LLM captioner fallback.")
        mock_db = {
            "formal": "The video displays a technical prototype execution illustrating async scheduling.",
            "sarcastic": "Oh look, another async Python project that will definitely save the world.",
            "humorous_tech": "When you try to avoid multi-threading in Python so you write 500 lines of asyncio semaphore control.",
            "humorous_non_tech": "A computer screen displaying lots of techy text and coding stuff."
        }
        return {s: mock_db.get(s, "A video caption.") for s in styles}

    logger.info(f"Calling LLM captioner via Fireworks ({config.fireworks_llm_model})")
    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": LLM_CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.6,
                "reasoning_effort": "none"
            }
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=60.0
                )
                if resp.status_code == 429:
                    logger.warning(f"Fireworks LLM rate-limited (429) during captioning. Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]

            # Clean special characters
            text = clean_special_characters(text)

            # Clean markdown code blocks from the output if present
            text_clean = text.strip()
            if text_clean.startswith("```"):
                lines = text_clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text_clean = "\n".join(lines).strip()

            data = json.loads(text_clean)
            logger.info(f"LLM captioner parsed JSON: {data}")
            
            # Find the captions dictionary. Tolerates 'captions', 'captures', or nested dicts.
            captions = {}
            if isinstance(data, dict):
                if "captions" in data and isinstance(data["captions"], dict):
                    captions = data["captions"]
                elif "captures" in data and isinstance(data["captures"], dict):
                    captions = data["captures"]
                else:
                    found_nested = False
                    for k, v in data.items():
                        if isinstance(v, dict):
                            # check if any keys of v match styles
                            for sk in styles:
                                if sk in v or sk.lower().replace("_", "") in [str(x).lower().replace("_", "") for x in v.keys()]:
                                    captions = v
                                    found_nested = True
                                    break
                        if found_nested:
                            break
                    if not found_nested:
                        captions = data
            
            # Normalize keys (strip, lowercase, replace underscores/hyphens)
            def normalize_key(k: str) -> str:
                return str(k).lower().strip().replace("_", "").replace("-", "").replace(" ", "")

            captions_clean = {normalize_key(k): str(v) for k, v in captions.items()}
            logger.info(f"Normalized captions keys: {list(captions_clean.keys())}")
            
            result = {}
            for s in styles:
                norm_s = normalize_key(s)
                # Fallback to key matching or generic string
                val = captions_clean.get(norm_s)
                if not val:
                    # Try partial match (e.g. if style is "humorous_tech" and key is "tech")
                    for k, v in captions_clean.items():
                        if k in norm_s or norm_s in k:
                            val = v
                            break
                result[s] = (val or f"A video description in {s} style.").strip()
            return result
        except Exception as e:
            logger.warning(f"LLM captioner API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0
