"""
commentary_generator.py
-----------------------
Two-LLM pipeline for generating sports commentary from video keyframes.

Pipeline overview
-----------------
LLM-1 (Google Gemini 2.5 Flash  –  multimodal / vision model)
  • Receives each raw keyframe as a Base64 image.
  • Produces a concise factual scene description: what is happening, who is
    involved, what sport, game state visible, notable actions, etc.
  • Acts as a "visual analyst" – it turns pixels into structured text.

LLM-2 (Google Gemini 2.5 Pro  –  language model)
  • Receives ALL scene descriptions produced by LLM-1 plus any extra game
    context the user supplied (sport type, team names, score).
  • Produces fluent, exciting sports commentary in the style of a live
    broadcast commentator – complete with emotion, drama, and timeline stamps.
  • Acts as the "commentator" – it turns analyst notes into broadcast prose.

Justification for model choices
--------------------------------
- Gemini 2.5 Flash (LLM-1): Google's latest multimodal Flash model; supports
  vision+text input, fast and cost-efficient, ideal for per-frame image analysis.
- Gemini 2.5 Pro (LLM-2): Google's most capable language model, superior at
  generating creative, nuanced broadcast-style prose from structured scene
  descriptions. Distinct model family from Flash ensures clear role separation.
Using the `google-genai` SDK (the current, actively-maintained package that
replaces the deprecated `google-generativeai`).
"""

import base64
import time
from typing import List

# Third-party dependency: pip install google-genai
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError(
        "google-genai package is required.\n"
        "Install it with:  pip install google-genai"
    )

from video_processor import FrameInfo


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
VISION_MODEL_ID   = "gemini-2.5-flash"   # LLM-1 – fast multimodal model for per-frame visual analysis
LANGUAGE_MODEL_ID = "gemini-2.5-pro"     # LLM-2 – highest quality language model for commentary generation
# Justification for model choices:
# gemini-2.5-flash (LLM-1): Google's latest multimodal Flash model; supports vision+text input,
#   fast and cost-efficient, ideal for processing many keyframes sequentially.
# gemini-2.5-pro (LLM-2): Google's most capable language model, superior at generating
#   creative, nuanced, and engaging broadcast-style prose from structured scene descriptions.
# Using two distinct model families (Flash vs Pro) satisfies the two-LLM requirement with
# clear role separation: Flash for speed/vision analysis, Pro for language quality.


def get_client(api_key: str) -> genai.Client:
    """
    Create and return an authenticated Google Gemini API client.

    Parameters
    ----------
    api_key : str
        A valid Google AI Studio API key.

    Returns
    -------
    genai.Client
        Authenticated client ready to make API calls.

    Raises
    ------
    ValueError
        If the api_key is empty or None.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Google API key must not be empty.")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# LLM-1: Visual Scene Analyser  (Gemini Flash – multimodal)
# ---------------------------------------------------------------------------

def _build_vision_prompt(timestamp: str, sport: str) -> str:
    """
    Build the prompt sent to the vision model for a single frame.

    Parameters
    ----------
    timestamp : str
        Human-readable timestamp in MM:SS format.
    sport     : str
        Name of the sport (e.g. "football", "basketball") for context.

    Returns
    -------
    str
        Complete prompt string.
    """
    return (
        f"You are a professional sports analyst watching a {sport} match.\n"
        f"This image is a keyframe captured at timestamp {timestamp} in the video.\n\n"
        "Describe in 2-4 concise sentences:\n"
        "  1. What sport is being played and the general scene setting.\n"
        "  2. The key action or event happening at this moment "
        "     (e.g., a goal attempt, a tackle, a free-kick, player positioning).\n"
        "  3. Any visible scoreboard, referee signals, crowd reaction, or other notable detail.\n\n"
        "Be factual and precise. Do NOT generate commentary – just describe what you observe."
    )


def analyse_frame_with_llm1(
    frame: FrameInfo,
    sport: str,
    client: genai.Client,
    retry_delay: float = 3.0,
) -> str:
    """
    Send a single keyframe to LLM-1 (Gemini Flash) for visual scene description.

    The image is passed as inline Base64 bytes alongside the text prompt.
    On rate-limit errors the function sleeps and retries once.

    Parameters
    ----------
    frame        : FrameInfo
        The keyframe object containing base64 image and timestamp.
    sport        : str
        Sport type for contextual prompting.
    client       : genai.Client
        Authenticated Gemini API client.
    retry_delay  : float
        Seconds to wait before retrying after a rate-limit error.

    Returns
    -------
    str
        Plain-text scene description produced by LLM-1.
    """
    prompt_text = _build_vision_prompt(frame.timestamp_str, sport)

    # Decode the base64 string back to raw bytes for the inline image part
    image_bytes = base64.b64decode(frame.base64_image)

    # Build multimodal content: [inline_image, text_prompt]
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    text_part  = genai_types.Part.from_text(text=prompt_text)

    config = genai_types.GenerateContentConfig(
        temperature=0.2,        # Low temperature → factual, deterministic output
        max_output_tokens=250,
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=VISION_MODEL_ID,
                contents=[image_part, text_part],
                config=config,
            )
            return response.text.strip()
        except Exception as exc:
            error_msg = str(exc)
            # Retry once on rate-limit / quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt == 0:
                    print(f"    [LLM-1] Rate limit at {frame.timestamp_str}. Retrying in {retry_delay}s…")
                    time.sleep(retry_delay)
                    continue
            # Non-retryable or second failure – return placeholder
            return f"[Scene at {frame.timestamp_str}: visual analysis unavailable – {error_msg[:80]}]"

    return f"[Scene at {frame.timestamp_str}: max retries exceeded]"


def analyse_all_frames(
    frames: List[FrameInfo],
    sport: str,
    api_key: str,
    request_delay: float = 2.0,
) -> List[dict]:
    """
    Run LLM-1 on every extracted keyframe sequentially.

    A small delay (`request_delay`) is inserted between API calls to respect
    per-minute rate limits on the free Gemini tier.

    Parameters
    ----------
    frames        : List[FrameInfo]
        Keyframes from the video processor.
    sport         : str
        Sport type for contextual prompting.
    api_key       : str
        Google AI Studio API key.
    request_delay : float
        Seconds to sleep between consecutive API calls (default 2.0 s).

    Returns
    -------
    List[dict]
        List of records: {"timestamp": str, "timestamp_sec": float, "description": str}
    """
    client = get_client(api_key)

    print(f"\n[LLM-1 | {VISION_MODEL_ID}] Analysing {len(frames)} frames as visual analyst…")
    scene_records = []

    for i, frame in enumerate(frames, start=1):
        print(f"  Analysing frame {i}/{len(frames)}  t={frame.timestamp_str}")
        description = analyse_frame_with_llm1(frame, sport, client)
        scene_records.append({
            "timestamp":     frame.timestamp_str,
            "timestamp_sec": frame.timestamp_sec,
            "description":   description,
        })
        if i < len(frames):
            time.sleep(request_delay)   # Throttle to avoid hitting rate limits

    print(f"[LLM-1] Scene analysis complete. {len(scene_records)} descriptions generated.\n")
    return scene_records


# ---------------------------------------------------------------------------
# LLM-2: Sports Commentator  (Gemini Flash – language generation role)
# ---------------------------------------------------------------------------

def _build_commentary_prompt(
    scene_records: List[dict],
    sport: str,
    team_a: str,
    team_b: str,
    context: str,
) -> str:
    """
    Assemble the prompt sent to LLM-2 for full commentary generation.

    Parameters
    ----------
    scene_records   : List[dict]
        Output from LLM-1 – list of {"timestamp", "description"} dicts.
    sport, team_a, team_b, context : str
        Game metadata passed through from the CLI.

    Returns
    -------
    str
        Complete prompt string for LLM-2.
    """
    scene_log = "\n".join(
        f"[{rec['timestamp']}] {rec['description']}"
        for rec in scene_records
    )

    prompt = (
        f"You are an expert live sports commentator broadcasting a {sport} match.\n"
        f"Teams: {team_a}  vs  {team_b}.\n"
    )
    if context.strip():
        prompt += f"Additional context: {context}\n"

    prompt += (
        "\nBelow is a chronological timeline of scene descriptions produced by a "
        "visual analyst who watched the same match:\n\n"
        f"{scene_log}\n\n"
        "Your task: Write a full, ENGAGING LIVE commentary script based ONLY on the "
        "scene descriptions above. Requirements:\n"
        "  • Start each commentary line with the exact timestamp [MM:SS].\n"
        "  • Use vivid, exciting broadcast language with emotion and natural pauses.\n"
        "  • Reference both team names and visible player actions.\n"
        "  • Keep each timestamped segment to 2-3 sentences.\n"
        "  • Do NOT invent scores, player names, or events not in the scene log.\n"
        "  • End with a short SUMMARY paragraph.\n\n"
        "Output format:\n"
        "[MM:SS] <commentary text>\n"
        "...\n"
        "SUMMARY: <brief match summary>\n"
    )
    return prompt


def generate_commentary_with_llm2(
    scene_records: List[dict],
    sport: str,
    team_a: str,
    team_b: str,
    context: str,
    api_key: str,
) -> str:
    """
    Send all scene descriptions to LLM-2 for full commentary generation.

    LLM-2 is called with a higher temperature (0.7) to produce creative,
    natural-sounding broadcast commentary from the factual scene log produced
    by LLM-1.

    Parameters
    ----------
    scene_records                   : List[dict]
        Output from `analyse_all_frames`.
    sport, team_a, team_b, context  : str
        Game metadata.
    api_key                         : str
        Google AI Studio API key.

    Returns
    -------
    str
        The full commentary script as a plain-text string.
    """
    client = get_client(api_key)
    prompt = _build_commentary_prompt(scene_records, sport, team_a, team_b, context)

    config = genai_types.GenerateContentConfig(
        temperature=0.7,        # Higher creativity for broadcast-style prose
        max_output_tokens=3000,
    )

    print(f"[LLM-2 | {LANGUAGE_MODEL_ID}] Generating commentary script as broadcaster…")

    retry_delay = 5.0   # Seconds to wait before retrying on rate-limit errors
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=LANGUAGE_MODEL_ID,
                contents=prompt,
                config=config,
            )
            # Guard against None or empty response text
            raw_text = getattr(response, "text", None)
            if not raw_text or not raw_text.strip():
                print("[LLM-2] WARNING: Empty response received from model.")
                return ""   # Caller checks for empty and handles gracefully
            commentary = raw_text.strip()
            print(f"[LLM-2] Commentary generated successfully ({len(commentary)} chars).\n")
            return commentary
        except Exception as exc:
            error_msg = str(exc)
            # Retry once on rate-limit / quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt == 0:
                    print(f"[LLM-2] Rate limit hit. Retrying in {retry_delay}s…")
                    time.sleep(retry_delay)
                    continue
            # Non-retryable error or second failure
            print(f"[LLM-2] ERROR: Commentary generation failed: {exc}")
            return f"[Commentary generation failed: {exc}]"

    return "[Commentary generation failed: max retries exceeded]"
