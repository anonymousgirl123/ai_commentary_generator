"""
tts_engine.py
-------------
Optional Speech Synthesis module for the Sports Commentator pipeline.

Converts the generated commentary text into spoken audio using the
platform's built-in text-to-speech engine (no external API or paid service
required).

Platform support
----------------
- macOS   : uses the built-in `say` command (pre-installed on all Macs).
- Windows : uses the built-in `SAPI` engine via PowerShell.
- Linux   : uses `espeak` (install with: sudo apt-get install espeak).

Speech style tuning
-------------------
The commentary is delivered at a slightly faster rate and uses a clear,
energetic voice to mimic a live sports broadcast style.  On macOS the
"Alex" or "Daniel" voice is used (whichever is available).

Output
------
- Plays the commentary aloud in the terminal (live broadcast mode).
- Optionally saves the audio to an AIFF file (macOS) for playback later.
"""

import os
import sys
import re
import subprocess
import platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_commentary_for_speech(commentary: str) -> str:
    """
    Strip timestamp tags and formatting from the commentary text so the
    TTS engine reads only the spoken words (not "[00:05]" etc.).

    Parameters
    ----------
    commentary : str
        Raw commentary string from LLM-2, including [MM:SS] tags.

    Returns
    -------
    str
        Clean plain-text commentary suitable for speech synthesis.
    """
    # Remove [MM:SS] timestamp markers
    text = re.sub(r"\[\d{2}:\d{2}\]", "", commentary)
    # Remove the SUMMARY: label
    text = text.replace("SUMMARY:", "And finally, a match summary.")
    # Collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def _get_best_macos_voice() -> str:
    """
    Return the best available English voice on macOS for sports commentary.

    Tries a list of clear, energetic voices in preference order.
    Falls back to the system default if none are found.

    Returns
    -------
    str
        Voice name string for the macOS `say -v` flag.
    """
    preferred = ["Daniel", "Alex", "Tom", "Fred"]
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True, text=True
        )
        available = result.stdout
        for voice in preferred:
            if voice in available:
                return voice
    except Exception:
        pass
    return ""   # Empty string → system default


# ---------------------------------------------------------------------------
# Platform-specific TTS functions
# ---------------------------------------------------------------------------

def _speak_macos(text: str, rate: int = 210, audio_output: str = "") -> None:
    """
    Use the macOS built-in `say` command for speech synthesis.

    The speech rate is set to 210 words/min (default is ~175) to mimic
    the faster pace of a live sports commentator.

    Parameters
    ----------
    text         : str
        Clean commentary text to speak.
    rate         : int
        Words per minute (default 210 – slightly fast for sports energy).
    audio_output : str
        If non-empty, save the audio to this file path (AIFF format).
    """
    voice = _get_best_macos_voice()
    cmd = ["say", "--rate", str(rate)]
    if voice:
        cmd += ["-v", voice]
    if audio_output:
        cmd += ["-o", audio_output]
    cmd.append(text)
    subprocess.run(cmd, check=True)


def _speak_windows(text: str) -> None:
    """
    Use Windows SAPI via PowerShell for speech synthesis.

    Parameters
    ----------
    text : str
        Clean commentary text to speak.

    Raises
    ------
    FileNotFoundError
        If PowerShell is not available on the system PATH.
    subprocess.CalledProcessError
        If the PowerShell command returns a non-zero exit code.
    """
    # Escape single quotes in the text
    safe_text = text.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 3; "   # Rate range: -10 to 10; 3 = slightly fast
        f"$s.Speak('{safe_text}')"
    )
    subprocess.run(
        ["powershell", "-Command", ps_cmd],
        check=True
    )


def _speak_linux(text: str) -> None:
    """
    Use `espeak` on Linux for speech synthesis.

    Requires: sudo apt-get install espeak

    Parameters
    ----------
    text : str
        Clean commentary text to speak.

    Raises
    ------
    FileNotFoundError
        If `espeak` is not installed or not on the system PATH.
    subprocess.CalledProcessError
        If espeak returns a non-zero exit code.
    """
    subprocess.run(
        ["espeak", "-s", "180", "-v", "en", text],
        check=True
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak_commentary(commentary: str, audio_output: str = "") -> None:
    """
    Convert the generated commentary text to speech and play it aloud.

    Automatically detects the current operating system and uses the
    appropriate built-in TTS engine.  No external packages or API keys
    are required.

    Parameters
    ----------
    commentary   : str
        Full commentary string from LLM-2 (with [MM:SS] timestamps).
    audio_output : str
        Optional file path to save audio output.
        On macOS, saves as AIFF file (e.g., "commentary.aiff").
        On other platforms, file saving is not supported and this
        parameter is ignored.

    Raises
    ------
    RuntimeError
        If the TTS engine is not available on the current platform.
    """
    print("\n[TTS] Preparing speech synthesis…")

    # Strip timestamps and formatting for clean speech
    clean_text = _clean_commentary_for_speech(commentary)

    if not clean_text.strip():
        print("[TTS] No text to speak.")
        return

    os_name = platform.system()
    print(f"[TTS] Platform detected: {os_name}")
    print(f"[TTS] Speaking commentary ({len(clean_text)} characters)…")

    try:
        if os_name == "Darwin":        # macOS
            _speak_macos(clean_text, rate=210, audio_output=audio_output)
            if audio_output:
                print(f"[TTS] Audio saved to: {audio_output}")

        elif os_name == "Windows":
            _speak_windows(clean_text)

        elif os_name == "Linux":
            _speak_linux(clean_text)

        else:
            raise RuntimeError(f"Unsupported platform for TTS: {os_name}")

        print("[TTS] Speech synthesis complete.\n")

    except FileNotFoundError as exc:
        # TTS binary missing – provide platform-specific install guidance
        hints = {
            "Darwin":  "macOS: 'say' is pre-installed. Check System Integrity Protection settings.",
            "Windows": "Windows: ensure PowerShell is on the PATH.",
            "Linux":   "Linux: install espeak with 'sudo apt-get install espeak'.",
        }
        hint = hints.get(os_name, "Check that the TTS engine binary is installed and on your PATH.")
        print(
            f"[TTS] WARNING: TTS engine not found – {exc}\n"
            f"       {hint}\n"
            f"       Commentary text is still saved to the output file."
        )
    except subprocess.CalledProcessError as exc:
        # TTS binary ran but returned a non-zero exit code
        print(
            f"[TTS] WARNING: TTS engine returned an error (exit code {exc.returncode}).\n"
            f"       Commentary text is still saved to the output file."
        )
    except RuntimeError as exc:
        # Unsupported platform
        print(
            f"[TTS] WARNING: {exc}\n"
            f"       Commentary text is still saved to the output file."
        )
