import os

from app.services.ocr import read_frames


def analyze_frames(frames_dir: str):

    print("Running visual analysis (OCR)...")

    if not os.path.exists(frames_dir):
        print("Frames directory not found.")
        return ""

    visual_text = read_frames(frames_dir)

    print("VISUAL TEXT:", visual_text)

    return visual_text