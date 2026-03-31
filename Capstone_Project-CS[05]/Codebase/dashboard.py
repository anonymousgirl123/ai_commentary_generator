"""
dashboard.py
------------
Interactive Terminal Dashboard for the Sports Commentator from Video system.
(Capstone_Project-CS[05] – Step 5: System Integration and User Interface)

This module provides a colourful, menu-driven terminal UI that lets the user:
  1. Configure all pipeline settings interactively (no need to remember CLI flags).
  2. Choose between LIVE mode and HIGHLIGHT (post-game) mode.
  3. Run the full commentary pipeline and watch progress in real time.
  4. View the generated commentary rendered inside a formatted panel.
  5. Optionally replay the commentary as speech (TTS).
  6. Provide quick feedback (thumbs up / down) on commentary quality.

No external packages are required – the dashboard uses only ANSI escape codes
for colours and formatting, which work in any modern terminal on macOS, Linux,
and Windows 10+.

Run directly:
    python3 dashboard.py
"""

import os
import sys
import time
import subprocess

# ---------------------------------------------------------------------------
# ANSI colour / style helpers  (no external library needed)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Foreground colours
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# Bright foreground colours
BRIGHT_RED    = "\033[91m"
BRIGHT_GREEN  = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE   = "\033[94m"
BRIGHT_MAGENTA= "\033[95m"
BRIGHT_CYAN   = "\033[96m"
BRIGHT_WHITE  = "\033[97m"

# Background colours
BG_BLUE    = "\033[44m"
BG_GREEN   = "\033[42m"
BG_RED     = "\033[41m"
BG_YELLOW  = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_CYAN    = "\033[46m"

TERMINAL_WIDTH = 65   # Fixed width for consistent rendering

DEFAULT_API_KEY = "key here"

def c(text: str, *styles) -> str:
    """
    Apply one or more ANSI style codes to a text string and reset after.

    Parameters
    ----------
    text   : str   The text to style.
    styles : str   One or more ANSI code constants.

    Returns
    -------
    str  Styled text with reset appended.
    """
    return "".join(styles) + text + RESET


def clear_screen() -> None:
    """Clear the terminal screen cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")


def print_divider(char: str = "─", color: str = CYAN) -> None:
    """Print a full-width horizontal divider line."""
    print(c(char * TERMINAL_WIDTH, color))


def print_header() -> None:
    """Print the dashboard banner / title block."""
    clear_screen()
    print_divider("═", BRIGHT_CYAN)
    title  = "  🏆  SPORTS COMMENTATOR FROM VIDEO"
    sub    = "       Capstone Project CS[05]  –  AI Commentary Generator"
    models = "       LLM-1: Gemini 2.5 Flash   LLM-2: Gemini 2.5 Pro"
    print(c(title,  BOLD, BRIGHT_YELLOW))
    print(c(sub,    BRIGHT_WHITE))
    print(c(models, DIM, CYAN))
    print_divider("═", BRIGHT_CYAN)
    print()


def print_section(title: str) -> None:
    """Print a coloured section heading."""
    print()
    print(c(f"  ▶  {title}", BOLD, BRIGHT_CYAN))
    print_divider("─", CYAN)


def print_success(msg: str) -> None:
    """Print a green success message."""
    print(c(f"  ✅  {msg}", BRIGHT_GREEN))


def print_error(msg: str) -> None:
    """Print a red error message."""
    print(c(f"  ❌  {msg}", BRIGHT_RED))


def print_info(msg: str) -> None:
    """Print a cyan info message."""
    print(c(f"  ℹ   {msg}", CYAN))


def print_warning(msg: str) -> None:
    """Print a yellow warning message."""
    print(c(f"  ⚠   {msg}", BRIGHT_YELLOW))


def prompt_input(label: str, default: str = "") -> str:
    """
    Prompt the user for text input with an optional default value.

    Parameters
    ----------
    label   : str   The prompt label shown to the user.
    default : str   Default value used if the user presses Enter.

    Returns
    -------
    str   The user's input or the default value.
    """
    default_hint = f" [{c(default, YELLOW)}]" if default else ""
    raw = input(f"  {c('▸', BRIGHT_CYAN)} {label}{default_hint}: ").strip()
    return raw if raw else default


def prompt_choice(label: str, choices: list, default: str = "") -> str:
    """
    Show a numbered list of choices and return the user's selection.

    Parameters
    ----------
    label   : str    The question to display.
    choices : list   List of (value, description) tuples.
    default : str    Default value if user presses Enter.

    Returns
    -------
    str   The value string of the selected choice.
    """
    print(f"\n  {c(label, BOLD, WHITE)}")
    for i, (val, desc) in enumerate(choices, start=1):
        marker = c("◉", BRIGHT_GREEN) if val == default else c("○", DIM)
        print(f"    {marker} {c(str(i), BRIGHT_YELLOW)}. {c(val, BOLD)} – {c(desc, DIM)}")

    while True:
        raw = input(f"\n  {c('▸', BRIGHT_CYAN)} Enter number [{c(default, YELLOW)}]: ").strip()
        if not raw:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except ValueError:
            pass
        print_error("Invalid choice. Please enter a number from the list.")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    """
    Ask a yes/no question and return a boolean.

    Parameters
    ----------
    label   : str   The question to ask.
    default : bool  Default answer (True = yes, False = no).

    Returns
    -------
    bool
    """
    hint = c("Y/n", YELLOW) if default else c("y/N", YELLOW)
    raw = input(f"  {c('▸', BRIGHT_CYAN)} {label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Dashboard configuration form
# ---------------------------------------------------------------------------

def collect_configuration() -> dict:
    """
    Interactively collect all pipeline configuration from the user.

    Presents a step-by-step form covering: video path, API key, mode,
    sport type, team names, context, and optional TTS settings.

    Returns
    -------
    dict
        Configuration dictionary with keys matching main.py arguments.
    """
    print_header()
    print_section("STEP 1 of 3 — Video & API Setup")

    # Video path
    while True:
        video = prompt_input("Path to sports video file", "")
        if not video:
            print_error("Video path cannot be empty.")
            continue
        if not os.path.isfile(video):
            print_error(f"File not found: {video}")
            continue
        print_success(f"Video found: {os.path.basename(video)}")
        break

    # API key
    # API key (always from code, no prompt)
    api_key = "AIzaSyAIkSxQy-N5ZdpbpltKF9GlKVn3YwgfOqc"

    if not api_key or "PASTE" in api_key:
        print_error("API key not set. Please update DEFAULT_API_KEY in the code.")
    else:
        print_info("Using API key from code (DEFAULT_API_KEY).")
        print()
        print_section("STEP 2 of 3 — Commentary Mode")

    # Mode selection
    mode = prompt_choice(
        "Select commentary mode:",
        choices=[
            ("highlight", "Post-game highlights  – frame every 10s, up to 20 frames (faster, cheaper)"),
            ("live",      "Live broadcast style  – frame every 3s,  up to 60 frames (detailed, slower)"),
        ],
        default="highlight",
    )

    print()
    print_section("STEP 3 of 3 — Match Details")

    # Sport, teams, context
    sport   = prompt_input("Sport type", "football")
    team_a  = prompt_input("Home team name", "Team A")
    team_b  = prompt_input("Away team name", "Team B")
    context = prompt_input("Extra context (score, match stage – optional)", "")

    # TTS
    print()
    tts = prompt_yes_no("Enable text-to-speech (read commentary aloud)?", default=False)
    audio_output = ""
    if tts:
        audio_output = prompt_input(
            "Save audio to file? (leave blank to play only, or enter e.g. commentary.aiff)", ""
        )

    # Output file
    output = prompt_input("Save commentary to file", "commentary_output.txt")

    return {
        "video":        video,
        "api_key":      api_key,
        "mode":         mode,
        "sport":        sport,
        "team_a":       team_a,
        "team_b":       team_b,
        "context":      context,
        "tts":          tts,
        "audio_output": audio_output,
        "output":       output,
    }


# ---------------------------------------------------------------------------
# Configuration summary panel
# ---------------------------------------------------------------------------

def show_config_summary(cfg: dict) -> bool:
    """
    Display a formatted summary of the chosen configuration and ask for
    confirmation before running the pipeline.

    Parameters
    ----------
    cfg : dict   Configuration dict from collect_configuration().

    Returns
    -------
    bool   True if the user confirms, False to reconfigure.
    """
    print_header()
    print_section("CONFIGURATION SUMMARY")

    rows = [
        ("Video",        os.path.basename(cfg["video"])),
        ("Mode",         cfg["mode"].upper()),
        ("Sport",        cfg["sport"]),
        ("Teams",        f"{cfg['team_a']}  vs  {cfg['team_b']}"),
        ("Context",      cfg["context"] or "(none)"),
        ("TTS",          "Enabled" if cfg["tts"] else "Disabled"),
        ("Audio output", cfg["audio_output"] or "(play only)") if cfg["tts"] else ("", ""),
        ("Save to",      cfg["output"]),
        ("LLM-1",        "gemini-2.5-flash  (visual analysis)"),
        ("LLM-2",        "gemini-2.5-pro    (commentary generation)"),
    ]

    for key, val in rows:
        if not key:
            continue
        print(f"    {c(key + ':', BOLD, CYAN):<30} {c(val, BRIGHT_WHITE)}")

    print()
    return prompt_yes_no("Run the commentary pipeline with these settings?", default=True)


# ---------------------------------------------------------------------------
# Run pipeline and display output
# ---------------------------------------------------------------------------

def run_pipeline_dashboard(cfg: dict) -> None:
    """
    Build the CLI command from the config dict and run the commentary
    pipeline, streaming its output live to the dashboard terminal.

    Parameters
    ----------
    cfg : dict   Configuration dict from collect_configuration().
    """
    print_header()
    print_section("RUNNING PIPELINE")
    print_info("Pipeline started — this may take a few minutes depending on video length.")
    print()

    # Build the CLI command
    cmd = [
        sys.executable, "main.py",
        "--video",   cfg["video"],
        "--mode",    cfg["mode"],
        "--sport",   cfg["sport"],
        "--team_a",  cfg["team_a"],
        "--team_b",  cfg["team_b"],
        "--output",  cfg["output"],
    ]
    if cfg["api_key"] and cfg["api_key"] != os.environ.get("GOOGLE_API_KEY", ""):
        cmd += ["--api_key", cfg["api_key"]]
    if cfg["context"]:
        cmd += ["--context", cfg["context"]]
    if cfg["tts"]:
        cmd.append("--tts")
        if cfg["audio_output"]:
            cmd += ["--audio_output", cfg["audio_output"]]

    # Run and stream output live
    env = os.environ.copy()
    if cfg["api_key"]:
        env["GOOGLE_API_KEY"] = cfg["api_key"]

    print_divider("─", DIM)
    try:
        proc = subprocess.run(cmd, env=env)
        returncode = proc.returncode
    except FileNotFoundError as exc:
        print_error(f"Could not launch pipeline – Python interpreter not found: {exc}")
        return
    except Exception as exc:
        print_error(f"Unexpected error while running pipeline: {exc}")
        return
    print_divider("─", DIM)

    if returncode == 0:
        print_success("Pipeline completed successfully!")
    else:
        print_error(f"Pipeline exited with code {returncode}.")


# ---------------------------------------------------------------------------
# Commentary viewer
# ---------------------------------------------------------------------------

def show_commentary_panel(output_file: str) -> None:
    """
    Read the saved commentary file and display it inside a formatted panel.

    Parameters
    ----------
    output_file : str   Path to the commentary text file.
    """
    if not os.path.isfile(output_file):
        print_warning("Commentary output file not found – cannot display.")
        return

    print_section("GENERATED COMMENTARY")
    print_divider("─", YELLOW)

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, IOError) as exc:
        print_warning(f"Could not read commentary file '{output_file}': {exc}")
        return

    for line in lines:
        line = line.rstrip()
        if not line:
            print()
        elif line.startswith("==="):
            print(c(line, DIM, CYAN))
        elif line.startswith("SUMMARY:"):
            print(c(line, BOLD, BRIGHT_YELLOW))
        elif line.startswith("[") and "]" in line:
            # Colour the timestamp separately
            try:
                bracket_end = line.index("]") + 1
                timestamp   = line[:bracket_end]
                rest        = line[bracket_end:]
                print(c(timestamp, BOLD, BRIGHT_GREEN) + c(rest, WHITE))
            except ValueError:
                print(c(line, DIM, WHITE))
        else:
            print(c(line, DIM, WHITE))

    print_divider("─", YELLOW)


# ---------------------------------------------------------------------------
# Feedback collection
# ---------------------------------------------------------------------------

def collect_feedback() -> None:
    """
    Ask the user for quick feedback on commentary quality.

    Feedback is printed to the terminal so the evaluator can see it
    during review. In a production system this would be stored to a log.
    """
    print_section("FEEDBACK")
    print_info("Rate the generated commentary to help improve the system.")
    print()

    rating = prompt_choice(
        "Overall quality of the commentary:",
        choices=[
            ("5", "⭐⭐⭐⭐⭐  Excellent – very accurate and exciting"),
            ("4", "⭐⭐⭐⭐    Good – mostly accurate with minor issues"),
            ("3", "⭐⭐⭐      Average – some inaccuracies or flat language"),
            ("2", "⭐⭐        Poor – many inaccuracies"),
            ("1", "⭐         Very poor – not usable"),
        ],
        default="4",
    )

    note = prompt_input("Any specific comments? (optional)", "")

    print()
    print_success(f"Feedback recorded: {rating}/5 stars.")
    if note:
        print_info(f"Comment: {note}")
    print_info("Thank you! In production, this feedback would be logged for model improvement.")


# ---------------------------------------------------------------------------
# Main dashboard loop
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for the interactive terminal dashboard.

    Runs a full session:
      1. Collect configuration interactively.
      2. Show summary and ask for confirmation.
      3. Run the commentary pipeline.
      4. Display the generated commentary.
      5. Collect user feedback.
      6. Offer to run again or exit.
    """
    while True:
        # -- Configuration form --
        cfg = collect_configuration()

        # -- Confirmation summary --
        confirmed = show_config_summary(cfg)
        if not confirmed:
            print_info("Returning to configuration…")
            time.sleep(1)
            continue

        # -- Run pipeline --
        run_pipeline_dashboard(cfg)

        # -- Show commentary --
        show_commentary_panel(cfg["output"])

        # -- Feedback --
        give_feedback = prompt_yes_no("\nWould you like to rate the commentary quality?", default=True)
        if give_feedback:
            collect_feedback()

        # -- Run again? --
        print()
        again = prompt_yes_no("Run again with a different video or settings?", default=False)
        if not again:
            print()
            print(c("  Thanks for using Sports Commentator from Video! 🏆", BOLD, BRIGHT_YELLOW))
            print()
            break


if __name__ == "__main__":
    main()
