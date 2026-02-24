import subprocess
import os
import uuid
import whisper

from app.services.frame_extractor import extract_frames


BASE_DIR = "downloads"

print("Loading Whisper model...")
MODEL = whisper.load_model("base")
print("Whisper model loaded.")


def extract(url: str):
    print("Starting extraction...")

    # Remove tracking params
    url = url.split("?")[0]

    os.makedirs(BASE_DIR, exist_ok=True)

    file_id = str(uuid.uuid4())

    video_path = f"{BASE_DIR}/{file_id}.mp4"
    audio_path = f"{BASE_DIR}/{file_id}.wav"
    frames_dir = f"{BASE_DIR}/{file_id}_frames"

    # ---------------------------
    # 1️⃣ Download Reel
    # ---------------------------
    print("Downloading video...")

    subprocess.run(
        [
            "yt-dlp",
            "-f", "mp4",
            "-o", video_path,
            url
        ],
        check=True
    )

    # ---------------------------
    # 2️⃣ Extract Audio
    # ---------------------------
    print("Extracting audio...")

    subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ],
        check=True
    )

    # ---------------------------
    # 3️⃣ Extract Frames
    # ---------------------------
    print("Extracting frames...")

    frame_count = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        interval_sec=2
    )

    print(f"Extracted {frame_count} frames.")

    # ---------------------------
    # 4️⃣ Run Whisper
    # ---------------------------
    print("Running Whisper...")

    result = MODEL.transcribe(audio_path)

    transcript = result["text"]

    print("TRANSCRIPT:", transcript)

    # ---------------------------
    # 5️⃣ Cleanup (Optional)
    # ---------------------------
    # Uncomment later if storage becomes issue
    #
    # os.remove(video_path)
    # os.remove(audio_path)

    return transcript