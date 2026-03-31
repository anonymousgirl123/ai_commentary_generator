"""
video_processor.py
------------------
Handles video loading, frame extraction, and preprocessing for the
Sports Commentator pipeline.

Responsibilities:
  - Load a video file using OpenCV.
  - Extract keyframes at a configurable interval (e.g., every N seconds).
  - Encode frames to base64 so they can be sent to vision-capable LLMs.
  - Return a list of FrameInfo objects carrying (timestamp, base64_image).
"""

import cv2
import base64
import os
from dataclasses import dataclass
from typing import List


@dataclass
class FrameInfo:
    """
    Stores metadata and pixel data for a single extracted keyframe.

    Attributes
    ----------
    timestamp_sec : float
        Time in seconds from the start of the video where this frame was taken.
    timestamp_str : str
        Human-readable timestamp in MM:SS format (e.g., "01:23").
    base64_image  : str
        JPEG-encoded frame converted to a Base64 string for API transmission.
    """
    timestamp_sec: float
    timestamp_str: str
    base64_image: str


def seconds_to_mmss(seconds: float) -> str:
    """
    Convert a floating-point second value to a MM:SS string.

    Parameters
    ----------
    seconds : float
        Total elapsed seconds.

    Returns
    -------
    str
        Formatted time string such as "02:45".
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def encode_frame_to_base64(frame) -> str:
    """
    Encode an OpenCV BGR frame to a Base64 JPEG string.

    The frame is first JPEG-compressed (quality 85) to reduce payload size
    before being Base64-encoded for transmission to a vision LLM.

    Parameters
    ----------
    frame : numpy.ndarray
        A single video frame in OpenCV BGR format.

    Returns
    -------
    str
        Base64-encoded JPEG bytes as a UTF-8 string.

    Raises
    ------
    RuntimeError
        If OpenCV fails to encode the frame as JPEG.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success or buffer is None:
        raise RuntimeError("cv2.imencode failed: could not encode frame to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def extract_keyframes(
    video_path: str,
    interval_sec: float = 5.0,
    max_frames: int = 60,
) -> List[FrameInfo]:
    """
    Extract keyframes from a video at a fixed time interval.

    Frames are sampled uniformly at every `interval_sec` seconds.  To avoid
    overwhelming the LLM APIs, the extraction is capped at `max_frames`.

    Parameters
    ----------
    video_path   : str
        Absolute or relative path to the input video file.
    interval_sec : float
        Sampling cadence in seconds between consecutive keyframes (default 5 s).
    max_frames   : int
        Hard upper limit on the number of frames returned (default 60).

    Returns
    -------
    List[FrameInfo]
        Ordered list of FrameInfo objects sorted by ascending timestamp.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist at the given path.
    RuntimeError
        If OpenCV cannot open the video (codec issues, corrupt file, etc.).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    # -- retrieve video metadata ------------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"[VideoProcessor] Video loaded: {os.path.basename(video_path)}")
    print(f"[VideoProcessor] FPS={fps:.2f}  |  Duration={duration_sec:.1f}s  |  Total frames={total_frames}")

    frames: List[FrameInfo] = []
    current_time = 0.0

    while current_time < duration_sec and len(frames) < max_frames:
        # Seek to the target frame number
        target_frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

        ret, frame = cap.read()
        if not ret:
            break  # End of stream or read error

        ts_str = seconds_to_mmss(current_time)
        try:
            b64 = encode_frame_to_base64(frame)
        except RuntimeError as enc_err:
            # Skip frames that cannot be encoded (e.g., corrupt pixel data)
            print(f"  [frame] t={ts_str} – skipped (encoding error: {enc_err})")
            current_time += interval_sec
            continue

        frames.append(FrameInfo(
            timestamp_sec=current_time,
            timestamp_str=ts_str,
            base64_image=b64,
        ))

        print(f"  [frame] t={ts_str}  ({len(frames)}/{max_frames})")
        current_time += interval_sec

    cap.release()
    print(f"[VideoProcessor] Extracted {len(frames)} keyframes.")
    return frames
