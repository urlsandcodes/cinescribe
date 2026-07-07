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

**formal**: Professional, objective, factual. No opinions or humor.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Under bright office interior lighting, a woman wearing a white blazer is seated at a white desk, actively operating a computer mouse near a potted plant."

**sarcastic**: Dry, ironic, lightly mocking. Understated wit.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Another groundbreaking day under fluorescent lights, watching someone in a blazer conquer the wild frontier of moving a computer mouse next to an office plant."

**humorous_tech**: Funny with technology/programming references and jargon.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "User 'woman_in_blazer' successfully spawned in environment 'office_desk_01'. Fluorescent light shader compiling at 100% brightness. Plant.java object idle while mouse input listener processes coordinates."

**humorous_non_tech**: Funny, everyday humor. No tech jargon at all.
  Example input: "A woman in a white blazer sits at a white desk in a modern office with plant decorations under bright interior lights, using a computer mouse."
  Example output: "Looking exceptionally professional in her white blazer under those bright office lights, she tackles the ultimate workday challenge: pretending that scrolling with a mouse next to a desk plant counts as busy work."

=== VIDEO CONTEXT ===
Transcript: {transcript}
Visual Description: {timeline}

=== TASK ===
Generate a caption for EACH of these styles: {styles}

Output ONLY a valid JSON object in this exact format:
{{"captions": {{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}}}
Do not include markdown formatting, code fences, or any text outside the JSON.
"""
