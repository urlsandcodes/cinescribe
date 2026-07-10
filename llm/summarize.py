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

async def call_llm_cascade(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = None,
    response_format: dict = None
) -> str:
    """
    Unified LLM call cascade:
    1. Hugging Face Serverless API (Gemma-4-31B)
    2. OpenRouter API (Gemma-4-31B Free)
    3. Fireworks AI API (Default fallback)
    """
    # ==========================================
    # TIER 1: Hugging Face Serverless API
    # ==========================================
    if config.hf_api_key:
        logger.info(f"LLM Cascade: Tier 1 (Hugging Face) calling model {config.hf_model_id}")
        try:
            url = "https://router.huggingface.co/v1/chat/completions"
            payload = {
                "model": config.hf_model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if response_format is not None:
                payload["response_format"] = response_format
                
            headers = {
                "Authorization": f"Bearer {config.hf_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return clean_special_characters(text)
        except Exception as e:
            logger.warning(f"Tier 1 (Hugging Face) failed: {e}. Falling back to Tier 2 (OpenRouter)...")

    # ==========================================
    # TIER 2: OpenRouter API
    # ==========================================
    if config.openrouter_api_key:
        logger.info(f"LLM Cascade: Tier 2 (OpenRouter) calling model {config.openrouter_model_id}")
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": config.openrouter_model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if response_format is not None:
                payload["response_format"] = response_format
                
            headers = {
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return clean_special_characters(text)
        except Exception as e:
            logger.warning(f"Tier 2 (OpenRouter) failed: {e}. Falling back to Tier 3 (Fireworks)...")

    # ==========================================
    # TIER 3: Fireworks AI API
    # ==========================================
    logger.info(f"LLM Cascade: Tier 3 (Fireworks AI) calling model {config.fireworks_llm_model}")
    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "reasoning_effort": "none"
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if response_format is not None:
                payload["response_format"] = response_format
                
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=30.0
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
                return clean_special_characters(text)
        except Exception as e:
            logger.warning(f"LLM cascade Tier 3 (Fireworks) API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0
            
    raise RuntimeError("LLM cascade exhausted all retries and failed.")

async def summarize_video(transcript: str, timeline_str: str) -> tuple[str, str, list[str]]:
    """
    Calls the LLM to summarize the video transcript and event timeline.
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

    try:
        text = await call_llm_cascade(
            system_prompt=LLM_SUMMARIZE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
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
        logger.error(f"LLM summarizer failed critically in cascade: {e}")
        raise

async def generate_single_caption(
    timeline_str: str,
    style: str,
    transcript: str = None,
    facts: dict = None
) -> str:
    """
    Calls the LLM for a single style to prevent style bleed, using the unified LLM cascade.
    """
    from llm.prompts import STYLE_SYSTEM_PROMPTS, STYLE_USER_PROMPT
    
    system_prompt = STYLE_SYSTEM_PROMPTS.get(style)
    if not system_prompt:
        system_prompt = f"Write a single punchy '{style}' style caption."
        
    if facts:
        user_prompt = STYLE_USER_PROMPT.format(
            primary_subject=facts.get("primary_subject", "Not extracted (refer to description)"),
            primary_action=facts.get("primary_action", "Not extracted (refer to description)"),
            notable_detail=facts.get("notable_detail", "Not extracted (refer to description)"),
            setting=facts.get("setting", "Not extracted (refer to description)"),
            timeline=timeline_str,
            transcript=transcript or "No spoken speech detected in audio."
        )
    else:
        user_prompt = STYLE_USER_PROMPT.format(
            primary_subject="Not extracted (refer to description)",
            primary_action="Not extracted (refer to description)",
            notable_detail="Not extracted (refer to description)",
            setting="Not extracted (refer to description)",
            timeline=timeline_str,
            transcript=transcript or "No spoken speech detected in audio."
        )

    try:
        text = await call_llm_cascade(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7 if style != "formal" else 0.2,
            max_tokens=120
        )
        return text.strip().strip('"').strip()
    except Exception as e:
        logger.warning(f"LLM captioner style {style} failed: {e}. Returning fallback caption.")
        return f"A video description in {style} style."

async def extract_scene_facts(timeline_str: str) -> dict | None:
    """
    Calls the LLM once using VLM_EXTRACT_PROMPT to get key facts in JSON format.
    """
    from llm.prompts import VLM_EXTRACT_PROMPT
    prompt = VLM_EXTRACT_PROMPT.format(description=timeline_str)
    
    try:
        text = await call_llm_cascade(
            system_prompt="You are a precise information extraction system. Output JSON only.",
            user_prompt=prompt,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Clean up potential markdown wrapper code blocks if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        
        data = json.loads(text)
        return data
    except Exception as e:
        logger.warning(f"Fact extraction failed in cascade: {e}")
        return None

def validate_extracted_facts(facts: dict) -> bool:
    """
    Validates that the extracted facts is a dictionary and contains all 4 required non-empty string keys.
    """
    if not isinstance(facts, dict):
        return False
    required_keys = ["primary_subject", "primary_action", "notable_detail", "setting"]
    for k in required_keys:
        if k not in facts:
            return False
        if not isinstance(facts[k], str) or not facts[k].strip():
            return False
    return True

async def generate_captions(transcript: str, timeline_str: str, styles: list[str]) -> dict[str, str]:
    """
    Generates captions in each of the requested styles concurrently, incorporating audio transcript if available.
    Uses a single shared extraction step so all style calls work from identical facts.
    """
    provider = config.vlm_provider

    if provider == "mock":
        logger.info("Using mock LLM captioner fallback.")
        mock_db = {
            "formal": "The video displays a technical prototype execution illustrating async scheduling.",
            "sarcastic": "Oh look, another async Python project that will definitely save the world.",
            "humorous_tech": "When you try to avoid multi-threading in Python so you write 500 lines of asyncio semaphore control.",
            "humorous_non_tech": "A computer screen displaying lots of techy text and coding stuff."
        }
        return {s: mock_db.get(s, "A video caption.") for s in styles}

    # Call the VLM extraction step (once per clip)
    logger.info("Calling VLM key facts extraction step (once per clip)...")
    facts = await extract_scene_facts(timeline_str)
    if facts and validate_extracted_facts(facts):
        logger.info(f"Successfully extracted and validated scene facts: {json.dumps(facts)}")
    else:
        logger.warning("Fact extraction failed or returned invalid fields. Falling back to raw scene description.")
        facts = None

    logger.info(f"Generating {len(styles)} style captions concurrently via Multi-Tier Hosted Gemma Fallback Engine")
    
    # Run all style generations concurrently with shared facts to save wall-clock time
    tasks = [generate_single_caption(timeline_str, s, transcript=transcript, facts=facts) for s in styles]
    caption_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    result = {}
    for style, caption in zip(styles, caption_results):
        if isinstance(caption, Exception):
            logger.error(f"Failed to generate caption for style {style}: {caption}")
            result[style] = f"A video description in {style} style."
        else:
            result[style] = caption
            
    return result
