import cv2
import os
import logging
from typing import Optional, Generator, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str, output_dir: str, fps: Optional[float] = None, prefix: str = ""
) -> int:
    """
    Extracts frames from a video file.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frames.
        fps: Optional frame rate to sample at. If None, extracts all frames.
        prefix: Optional prefix for the saved frame filenames.

    Returns:
        Number of frames extracted.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1
    if fps:
        frame_interval = max(1, int(round(video_fps / fps)))

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_name = f"{prefix}{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        count += 1

    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_path} to {output_dir}")
    return saved_count


def get_video_stream(
    source: str | int,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields frames from a video source.
    Returns: (frame_idx, original_bgr_frame, rgb_frame)
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    logger.info(
        f"Starting video stream from {source}. Press 'q' in any cv2 window to close."
    )
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, frame, rgb_frame
            frame_idx += 1
    finally:
        cap.release()


def get_image_stream(
    image_path: str,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """
    Generator that acts like a video stream but yields a single image.
    Returns: (0, original_bgr_frame, rgb_frame)
    """
    logger.info(f"Processing image {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yield 0, image, rgb_frame


def get_media_stream(
    source: str | int, is_video: bool = True
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """Helper to return the right stream type based on config."""
    if is_video:
        return get_video_stream(source)
    else:
        return get_image_stream(str(source))
