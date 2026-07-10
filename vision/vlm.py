import base64
import httpx
import asyncio
from typing import Protocol, List
from app.config import config
from app.logger import logger

class VLMClient(Protocol):
    async def describe_frame(self, image_bytes: bytes, prompt: str) -> str:
        """Sends a single image and a prompt to the VLM and returns the description."""
        ...

    async def describe_frames_batch(self, images: List[bytes], prompt: str) -> List[str]:
        """Sends a batch of images to the VLM and returns descriptions for each."""
        ...

class FireworksVLMClient:
    """Calls Fireworks AI VLM (e.g. Kimi K2.6) via OpenAI-compatible chat completions API."""
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model or config.fireworks_vlm_model
        self.endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"

    async def describe_frame(self, image_bytes: bytes, prompt: str) -> str:
        # Downscale image if width > 768px to prevent VLM visual hallucinations
        from PIL import Image
        import io
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                if w > 768:
                    ratio = 768 / w
                    new_w = 768
                    new_h = int(h * ratio)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=90)
                    image_bytes = buffer.getvalue()
                    logger.info(f"VLM downscaled input frame from {w}x{h} to {new_w}x{new_h} for resolution compatibility")
        except Exception as e:
            logger.warning(f"Failed to downscale frame image: {e}")

        b64_image = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "reasoning_effort": "none"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        max_retries = 3
        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(self.endpoint, json=payload, headers=headers, timeout=30.0)
                    if resp.status_code == 429:
                        logger.warning(f"Fireworks VLM rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Fireworks VLM API error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0
        raise IOError("Failed to describe frame with Fireworks VLM")

    async def describe_frames_batch(self, images: List[bytes], prompt: str) -> List[str]:
        results = []
        for i, img in enumerate(images):
            if i > 0:
                await asyncio.sleep(1.0)
            desc = await self.describe_frame(img, prompt)
            results.append(desc)
        return results

    async def generate_draft_sequence(self, images: List[bytes], prompt: str, manifest_text: str = None) -> str:
        """
        Sends ALL images in a single API call as a chronological sequence to generate the draft.
        """
        from PIL import Image
        import io

        user_content = []
        if manifest_text:
            user_content.append({
                "type": "text",
                "text": manifest_text
            })
            
        for idx, image_bytes in enumerate(images):
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    w, h = img.size
                    if w > 1024:
                        ratio = 1024 / w
                        new_w = 1024
                        new_h = int(h * ratio)
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        image_bytes = buffer.getvalue()
                        if idx == 0:
                            logger.info(f"VLM downscaled sequence frames from {w}x{h} to {new_w}x{new_h}")
            except Exception as e:
                logger.warning(f"Failed to downscale sequence frame {idx}: {e}")

            b64_image = base64.b64encode(image_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })

        logger.info(f"Sending {len(images)} frames as chronological sequence to VLM in single call")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 2000,
            "reasoning_effort": "none"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(self.endpoint, json=payload, headers=headers, timeout=45.0)
                    if resp.status_code == 429:
                        logger.warning(f"Fireworks VLM sequence rate-limited (429) on draft. Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Fireworks VLM sequence draft error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise IOError(f"Failed to generate draft sequence description: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2.0

    async def describe_frames_sequence(self, images: List[bytes], prompt: str, manifest_text: str = None) -> str:
        """
        Sends ALL images in a single API call as a chronological sequence.
        The VLM sees the full filmstrip and can describe temporal progression, motion, and changes.
        Returns a single unified description covering the whole video.
        """
        draft_content = await self.generate_draft_sequence(images, prompt, manifest_text)
        logger.info(f"VLM Draft Description generated successfully ({len(draft_content)} chars). Initiating self-correction verification.")

        # Rebuild user_content, headers, and max_retries for verification step
        from PIL import Image
        import io
        user_content = []
        for idx, image_bytes in enumerate(images):
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    w, h = img.size
                    if w > 1024:
                        ratio = 1024 / w
                        new_w = 1024
                        new_h = int(h * ratio)
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        image_bytes = buffer.getvalue()
            except Exception as e:
                logger.warning(f"Failed to downscale sequence frame {idx}: {e}")

            b64_image = base64.b64encode(image_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        max_retries = 3

        # Step 2: Self-Correction Verification
        from llm.prompts import VLM_VERIFY_PROMPT
        verify_prompt = VLM_VERIFY_PROMPT.format(draft=draft_content)
        
        verify_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": verify_prompt}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 2000,
            "reasoning_effort": "none"
        }

        backoff = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(self.endpoint, json=verify_payload, headers=headers, timeout=45.0)
                    if resp.status_code == 429:
                        logger.warning(f"Fireworks VLM verification rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    resp.raise_for_status()
                    final_content = resp.json()["choices"][0]["message"]["content"]
                    logger.info("VLM Self-Correction verification loop completed successfully.")
                    return final_content
            except Exception as e:
                logger.warning(f"Fireworks VLM verification error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    logger.warning("Self-correction loop failed critically. Falling back to draft description.")
                    return draft_content
                await asyncio.sleep(backoff)
                backoff *= 2.0
        return draft_content

class MockVLMClient:
    """Simulates a VLM client response for keyless testing and verification."""
    def __init__(self, model: str = "mock"):
        self.model = model
        
    async def describe_frame(self, image_bytes: bytes, prompt: str) -> str:
        import random
        descriptions = [
            "A person explaining a software architecture diagram on a whiteboard.",
            "Close-up of a code editor showing a python async function.",
            "Wide shot of an office desk with a laptop displaying a video pipeline diagram.",
            "A title card presenting 'Video Intelligence Prototype v2'.",
            "A speaker presenting at a technical meetup about large vision models."
        ]
        ocr_texts = [
            '["ARCHITECTURE", "asyncio", "Semaphore(3)"]',
            '["def process_videos", "async with semaphore:", "return results"]',
            '["Video Intelligence", "Gemma 4", "ffmpeg"]',
            '["Local Video Intelligence", "Technical Spec v2"]',
            '["DeepMind", "VLM Swapping", "Hosted vs Local"]'
        ]
        actions = [
            '["writing on whiteboard", "pointing at diagram"]',
            '["scrolling code", "highlighting text"]',
            '["typing on keyboard", "adjusting coffee cup"]',
            '["static title view"]',
            '["gesturing", "advancing slide"]'
        ]
        objects = [
            '["whiteboard", "marker", "diagram"]',
            '["monitor", "IDE", "code text"]',
            '["laptop", "desk", "diagram representation"]',
            '["text", "background"]',
            '["projector screen", "microphone", "speaker"]'
        ]
        
        idx = random.randint(0, len(descriptions) - 1)
        return f"""
DESCRIPTION: {descriptions[idx]}
OCR: {ocr_texts[idx]}
ACTIONS: {actions[idx]}
OBJECTS: {objects[idx]}
"""

    async def describe_frames_batch(self, images: List[bytes], prompt: str) -> List[str]:
        return [await self.describe_frame(img, prompt) for img in images]

import os

class Gemma4VLMClient:
    """
    Cascading Gemma 4 VLM client that queries:
    1. Google AI Studio (Tier 1)
    2. Hugging Face Serverless (Tier 2)
    3. OpenRouter (Tier 3)
    """
    def __init__(self):
        self.gemini_api_key = config.gemini_api_key
        self.hf_api_key = config.hf_api_key
        self.openrouter_api_key = config.openrouter_api_key
        
    async def describe_frame(self, image_bytes: bytes, prompt: str) -> str:
        descs = await self.describe_frames_batch([image_bytes], prompt)
        return descs[0] if descs else ""
        
    async def describe_frames_batch(self, images: List[bytes], prompt: str) -> List[str]:
        results = []
        for img in images:
            desc = await self.describe_frames_sequence([img], prompt)
            results.append(desc)
        return results

    async def describe_frames_sequence(self, images: List[bytes], prompt: str, manifest_text: str = None) -> str:
        full_prompt = prompt
        if manifest_text:
            full_prompt = f"{manifest_text}\n\n{prompt}"

        # 1. TIER 1: Google AI Studio
        if self.gemini_api_key:
            logger.info("Gemma4VLM: Attempting Tier 1 (Google AI Studio)")
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key={self.gemini_api_key}"
                parts = [{"text": full_prompt}]
                for img_bytes in images:
                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64_data
                        }
                    })
                payload = {
                    "contents": [{"parts": parts}]
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45.0)
                    if resp.status_code == 200:
                        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("Gemma4VLM: Google AI Studio Tier 1 Succeeded")
                        return text
                    else:
                        logger.warning(f"Gemma4VLM: Google AI Studio failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Gemma4VLM: Google AI Studio failed with exception: {e}")

        # Limit sequence size to 5 frames to prevent Hugging Face/OpenRouter 413 limits (max 5 images)
        sampled_images = images
        if len(images) > 5:
            indices = [int(i * (len(images) - 1) / 4) for i in range(5)]
            sampled_images = [images[idx] for idx in indices]
            logger.info(f"Gemma4VLM: Sampled down from {len(images)} to {len(sampled_images)} frames to comply with server limits.")

        # 2. TIER 2: Hugging Face Serverless
        if self.hf_api_key:
            logger.info("Gemma4VLM: Attempting Tier 2 (Hugging Face Serverless)")
            try:
                url = "https://router.huggingface.co/v1/chat/completions"
                content = [{"type": "text", "text": full_prompt}]
                for img_bytes in sampled_images:
                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_data}"
                        }
                    })
                payload = {
                    "model": "google/gemma-4-31B-it",
                    "messages": [{"role": "user", "content": content}]
                }
                headers = {
                    "Authorization": f"Bearer {self.hf_api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=45.0)
                    if resp.status_code == 200:
                        text = resp.json()["choices"][0]["message"]["content"]
                        logger.info("Gemma4VLM: Hugging Face Serverless Tier 2 Succeeded")
                        return text
                    else:
                        logger.warning(f"Gemma4VLM: Hugging Face Serverless failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Gemma4VLM: Hugging Face Serverless failed with exception: {e}")

        # 3. TIER 3: OpenRouter
        if self.openrouter_api_key:
            logger.info("Gemma4VLM: Attempting Tier 3 (OpenRouter)")
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                content = [{"type": "text", "text": full_prompt}]
                for img_bytes in sampled_images:
                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_data}"
                        }
                    })
                payload = {
                    "model": config.openrouter_model_id or "google/gemma-4-31b-it:free",
                    "messages": [{"role": "user", "content": content}]
                }
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=45.0)
                    if resp.status_code == 200:
                        text = resp.json()["choices"][0]["message"]["content"]
                        logger.info("Gemma4VLM: OpenRouter Tier 3 Succeeded")
                        return text
                    else:
                        logger.warning(f"Gemma4VLM: OpenRouter failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Gemma4VLM: OpenRouter failed with exception: {e}")

        raise RuntimeError("Gemma4VLM: All cascade tiers failed or keys are not configured.")

def get_vlm_client(provider: str | None = None) -> VLMClient:
    """Returns a configured VLM client instance."""
    provider = provider or config.vlm_provider

    if provider == "fireworks":
        return FireworksVLMClient(
            api_key=config.fireworks_api_key,
            model=config.fireworks_vlm_model
        )
    if provider == "gemma4":
        return Gemma4VLMClient()
    if provider == "mock":
        return MockVLMClient()

    raise ValueError(f"Unknown VLM provider: {provider}")
