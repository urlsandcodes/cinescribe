VLM_FRAME_PROMPT = """Analyze this video frame. Provide your analysis strictly in the following format, with each section on a new line:

DESCRIPTION: A detailed description of what is happening in the frame (1-2 sentences). Focus on:
  - Subject: appearance, clothing, features, characteristics.
  - Setting: background elements, location, weather conditions (e.g., sunny, overcast, rain, interior lights, golden hour).
  - Primary Action: motion, posture, what the subject is actively doing.
OCR: A JSON string list of any readable text visible in the image, e.g. ["text1", "text2"]. If no text is visible, output strictly []. Do not explain why or think out loud.
ACTIONS: A JSON string list of specific actions taking place, e.g. ["talking", "writing"]. If no actions, output strictly [].
OBJECTS: A JSON string list of primary objects visible, e.g. ["whiteboard", "laptop"]. If no objects, output strictly [].

Strict constraint: Do NOT write any introduction, thinking blocks, preamble, self-corrections, or commentary. Start directly with the 'DESCRIPTION:' keyword.
"""

LLM_SUMMARIZE_PROMPT = """Analyze the following transcript and chronological timeline of events extracted from a video.

Transcript:
{transcript}

Timeline of Events:
{timeline}

Generate:
1. A high-level summary (1-2 sentences max).
2. A detailed structured summary (1-2 paragraphs).
3. A list of 3-5 relevant keyword tags.

You must output your response as a valid JSON object in the following format:
{{
  "summary": "your high-level summary here",
  "detailed_summary": "your detailed structured summary here",
  "tags": ["tag1", "tag2", "tag3"]
}}
Ensure your output is valid JSON and nothing else. Do not include markdown formatting.
"""

LLM_CAPTION_PROMPT = """You are an expert video captioning agent. Given a neutral description of a video and its transcript, produce a short caption (1-3 sentences) in each requested style.

=== STYLE GUIDE WITH EXAMPLES ===

**formal**: Professional, objective, factual. Describe only observable events. Do not infer intentions, emotions, or motivations unless directly visible. Avoid dramatic or cinematic wording.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Under bright office interior lighting, a woman wearing a white blazer is seated at a white desk, actively operating a computer mouse near a potted plant."

**sarcastic**: Dry, ironic, lightly mocking. The humor should target ordinary situations, human behavior, or obvious contradictions in the scene rather than inventing unrelated jokes. Keep the sarcasm subtle and conversational.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Another groundbreaking day under fluorescent lights, watching someone in a blazer conquer the wild frontier of moving a computer mouse next to an office plant."

**humorous_tech**: Describe the scene entirely through a consistent technology metaphor such as a game engine, operating system, software debugging, programming, networking, robotics, or APIs. Stay within one metaphor rather than mixing multiple unrelated ones.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "User 'woman_in_blazer' successfully spawned in environment 'office_desk_01'. Fluorescent light shader compiling at 100% brightness. Plant.java object idle while mouse input listener processes coordinates."

**humorous_non_tech**: Use playful observational comedy based on everyday life. Imagine how a witty friend would describe the scene. Humor should come from relatable situations, exaggeration, or amusing comparisons, not sarcasm or technical references.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Looking exceptionally professional in her white blazer under those bright office lights, she tackles the ultimate workday challenge: pretending that scrolling with a mouse next to a desk plant counts as busy work."

=== CRITICAL RULES ===
1. Do not reference the caption generation process or video analysis. Never mention OCR, VLMs, AI models, computer vision, transcripts, timestamps, frames, pipelines, or how the scene was analyzed. Technical and programming terminology is allowed only for the humorous_tech style.
2. If any visible text is specified in the visual description, incorporate it naturally into the caption as it would appear to a human viewer (e.g., "a sign reading 'KOREA LIFES ENGINEERING'", "a display showing...", "a written message saying...").
3. Ensure every caption reads as a natural, seamless, human-written description of the video.
4. Keep humor lighthearted. Do not insult people or make offensive jokes. Avoid political, sexual, or discriminatory humor.

=== VIDEO CONTEXT ===
Transcript: {transcript}
Visual Description: {timeline}

=== TASK ===
Generate a caption for EACH of these styles: {styles}

Output ONLY a valid JSON object in this exact format:
{{"captions": {{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}}}
Do not include markdown formatting, code fences, or any text outside the JSON.
"""
