"""
main.py
-------
Entry point for the Sports Commentator from Video system (Capstone_Project-CS[05]).

System Overview
---------------
This pipeline automatically generates live-style sports commentary from a video
file that has no existing commentary track.  It processes the video in two stages
using two distinct Large Language Models (LLMs):

  Stage 1 – Visual Analysis (LLM-1: Gemini 2.5 Flash)
    Keyframes are extracted from the video at a configurable interval. Each frame
    is sent to a multimodal vision LLM that produces a concise, factual description
    of what is happening at that moment in the match.

  Stage 2 – Commentary Generation (LLM-2: Gemini 2.5 Pro)
    All scene descriptions from Stage 1 are assembled into a single structured
    prompt and sent to a high-quality language model that writes a full, engaging
    sports commentary script with timestamps – mimicking a professional broadcaster.

Output
------
  • Timestamped commentary printed to the terminal.
  • Commentary saved to a text file (default: commentary_output.txt).

Usage
-----
  python main.py --video <path_to_video> --api_key <google_api_key> [options]

  OR (API key from environment variable GOOGLE_API_KEY):
  python main.py --video <path_to_video>

Run `python main.py --help` for full argument documentation.
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Load environment variables from .env file if present (e.g., GOOGLE_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Reads .env file in the current directory automatically
except ImportError:
    pass  # dotenv not installed – API key must be passed via --api_key or env var

# Local modules
from video_processor import extract_keyframes
from commentary_generator import analyse_all_frames, generate_commentary_with_llm2
from tts_engine import speak_commentary   # Optional Step 4: Speech Synthesis


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Define and return the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all required and optional arguments.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Sports Commentator from Video (CS[05])\n"
            "Generates timestamped sports commentary from a video file using two LLMs.\n"
            "\n"
            "LLM-1: Gemini 2.5 Flash  – multimodal visual scene analysis\n"
            "LLM-2: Gemini 2.5 Pro    – natural language commentary generation\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Required --
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input sports video file (e.g., match.mp4).",
    )

    # -- Mode: live vs highlight (Step 5 – System Integration) --
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "highlight"],
        default="highlight",
        help=(
            "Commentary mode:\n"
            "  'live'      – dense coverage, frame every 3s, up to 60 frames "
            "(simulates real-time broadcast).\n"
            "  'highlight' – sparse coverage, frame every 10s, up to 20 frames "
            "(post-game highlight reel). Default: highlight."
        ),
    )

    # -- API key (required unless set as env variable) --
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "Google AI Studio API key for Gemini models. "
            "If not provided, the GOOGLE_API_KEY environment variable is used."
        ),
    )

    # -- Optional game context --
    parser.add_argument(
        "--sport",
        type=str,
        default="football",
        help="Type of sport in the video (default: 'football').",
    )
    parser.add_argument(
        "--team_a",
        type=str,
        default="Team A",
        help="Name of the first / home team (default: 'Team A').",
    )
    parser.add_argument(
        "--team_b",
        type=str,
        default="Team B",
        help="Name of the second / away team (default: 'Team B').",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help=(
            "Optional extra context for the commentator LLM, e.g., "
            "'Quarter-final match, Team A leads 1-0 at half time'."
        ),
    )

    # -- Frame extraction settings --
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Interval in seconds between extracted keyframes (default: 5.0).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="Maximum number of keyframes to extract (default: 40).",
    )

    # -- Output --
    parser.add_argument(
        "--output",
        type=str,
        default="commentary_output.txt",
        help="Path to the output text file for the commentary (default: commentary_output.txt).",
    )
    parser.add_argument(
        "--save_scenes",
        action="store_true",
        help="If set, also save the intermediate LLM-1 scene descriptions to scene_analysis.json.",
    )

    # -- Optional TTS (Step 4) --
    parser.add_argument(
        "--tts",
        action="store_true",
        help=(
            "If set, convert the generated commentary to speech using the "
            "platform built-in TTS engine (no extra install needed on macOS/Windows)."
        ),
    )
    parser.add_argument(
        "--audio_output",
        type=str,
        default="",
        help=(
            "Path to save the TTS audio file (macOS: AIFF format). "
            "Only used when --tts is set. Example: commentary.aiff"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print a decorative banner when the program starts."""
    print("=" * 65)
    print("   SPORTS COMMENTATOR FROM VIDEO  –  Capstone_Project-CS[05]")
    print("   LLM-1: Gemini 2.5 Flash  |  LLM-2: Gemini 2.5 Pro")
    print("=" * 65)


def save_commentary(commentary: str, output_path: str, meta: dict) -> None:
    """
    Write the generated commentary to a plain-text file.

    Parameters
    ----------
    commentary  : str
        The full commentary script returned by LLM-2.
    output_path : str
        Destination file path for the text output.
    meta        : dict
        Metadata dict (sport, teams, video name, timestamp) written as a header.
    """
    header_lines = [
        "=" * 65,
        "  SPORTS COMMENTARY – AUTO-GENERATED",
        f"  Video  : {meta.get('video', 'N/A')}",
        f"  Sport  : {meta.get('sport', 'N/A')}",
        f"  Teams  : {meta.get('team_a', 'N/A')}  vs  {meta.get('team_b', 'N/A')}",
        f"  Generated at: {meta.get('generated_at', '')}",
        "=" * 65,
        "",
    ]
    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write("\n".join(header_lines))
            fout.write(commentary)
            fout.write("\n")
        print(f"\n[Output] Commentary saved to: {output_path}")
    except (OSError, IOError) as exc:
        # Handles permission denied, disk full, invalid path, etc.
        print(f"[WARNING] Could not save commentary to '{output_path}': {exc}")
        print("[WARNING] Commentary is still displayed above in the terminal.")


def save_scene_analysis(scene_records: list, output_path: str) -> None:
    """
    Persist the intermediate LLM-1 scene descriptions to a JSON file.

    Parameters
    ----------
    scene_records : list
        List of dicts from `analyse_all_frames`.
    output_path   : str
        Destination JSON file path.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(scene_records, fout, indent=2, ensure_ascii=False)
        print(f"[Output] Scene analysis saved to: {output_path}")
    except (OSError, IOError) as exc:
        print(f"[WARNING] Could not save scene analysis to '{output_path}': {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace, api_key: str) -> None:
    """
    Execute the full Sports Commentary pipeline end-to-end.

    Steps
    -----
    1. Extract keyframes from the video using OpenCV.
    2. Run LLM-1 (Gemini Flash) on each frame to get scene descriptions.
    3. Run LLM-2 (Gemini Pro) on all descriptions to generate commentary.
    4. Print and save the commentary; optionally convert to speech (TTS).

    Parameters
    ----------
    args    : argparse.Namespace
        Parsed command-line arguments.
    api_key : str
        Resolved Google API key.
    """
    print_banner()

    # ------------------------------------------------------------------ #
    # Mode preset: override interval/max_frames based on --mode           #
    # live      → dense sampling (every 3s, up to 60 frames)             #
    # highlight → sparse sampling (every 10s, up to 20 frames)           #
    # Manual --interval / --max_frames always take priority if provided. #
    # ------------------------------------------------------------------ #
    MODE_PRESETS = {
        "live":      {"interval_sec": 3.0,  "max_frames": 60},
        "highlight": {"interval_sec": 10.0, "max_frames": 20},
    }
    preset = MODE_PRESETS[args.mode]
    # Only apply preset if the user has not manually specified interval/max_frames
    interval  = args.interval   if args.interval  != 5.0  else preset["interval_sec"]
    max_frames = args.max_frames if args.max_frames != 40   else preset["max_frames"]

    print(f"\n[Mode] {args.mode.upper()} mode selected — "
          f"interval={interval}s, max_frames={max_frames}")

    # ------------------------------------------------------------------ #
    # Step 1: Video frame extraction                                       #
    # ------------------------------------------------------------------ #
    print(f"\n[Step 1/3] Extracting keyframes from video: {args.video}")
    print(f"           Interval={interval}s  |  Max frames={max_frames}")

    frames = extract_keyframes(
        video_path=args.video,
        interval_sec=interval,
        max_frames=max_frames,
    )

    if not frames:
        print("[ERROR] No frames could be extracted from the video. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2: Visual analysis with LLM-1                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2/3] Visual scene analysis with LLM-1 (Gemini 2.5 Flash)")
    print(f"           Sport: {args.sport}  |  Frames to analyse: {len(frames)}")

    try:
        scene_records = analyse_all_frames(
            frames=frames,
            sport=args.sport,
            api_key=api_key,
            request_delay=1.5,   # ~1.5s between calls to stay within free-tier limits
        )
    except Exception as exc:
        print(f"[ERROR] Visual analysis (LLM-1) failed unexpectedly: {exc}")
        sys.exit(1)

    if args.save_scenes:
        save_scene_analysis(scene_records, "scene_analysis.json")

    # ------------------------------------------------------------------ #
    # Step 3: Commentary generation with LLM-2                            #
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3/3] Generating commentary with LLM-2 (Gemini 2.5 Pro)")
    print(f"           Teams: {args.team_a}  vs  {args.team_b}")

    try:
        commentary = generate_commentary_with_llm2(
            scene_records=scene_records,
            sport=args.sport,
            team_a=args.team_a,
            team_b=args.team_b,
            context=args.context,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"[ERROR] Commentary generation (LLM-2) failed unexpectedly: {exc}")
        sys.exit(1)

    # Validate commentary is not empty or an error placeholder
    if not commentary or not commentary.strip():
        print("[ERROR] Commentary generation returned empty output. Please try again.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Display and save output                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("  GENERATED COMMENTARY")
    print("=" * 65)
    print(commentary)

    meta = {
        "video": os.path.basename(args.video),
        "sport": args.sport,
        "team_a": args.team_a,
        "team_b": args.team_b,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_commentary(commentary, args.output, meta)

    # ------------------------------------------------------------------ #
    # Step 4 (Optional): Text-to-Speech synthesis                          #
    # ------------------------------------------------------------------ #
    if args.tts:
        print("\n[Step 4/4] Converting commentary to speech (TTS)…")
        try:
            speak_commentary(commentary, audio_output=args.audio_output)
        except Exception as exc:
            # TTS failure is non-fatal – commentary text is already saved
            print(f"[WARNING] TTS failed: {exc}. Commentary text is still available above.")

    print("\n[Done] Pipeline completed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, resolve the API key, and launch the commentary pipeline.

    The Google API key resolution order:
      1. --api_key CLI argument (highest priority)
      2. GOOGLE_API_KEY environment variable (fallback)
    If neither is available the program exits with an informative error message.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    # -- Resolve API key --------------------------------------------------
    api_key = ""
    if not api_key:
        print(
            "[ERROR] Google API key is required.\n"
            "  Provide it via --api_key <key>  OR  set the GOOGLE_API_KEY "
            "environment variable.\n"
            "  Obtain a free key at: https://aistudio.google.com/app/apikey"
        )
        sys.exit(1)

    # -- Validate video path ----------------------------------------------
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    # -- Run the pipeline -------------------------------------------------
    run_pipeline(args, api_key)


if __name__ == "__main__":
    main()
