import subprocess
import os
import uuid
import whisper

BASE_DIR = "downloads"

print("Loading Whisper model...")
MODEL = whisper.load_model("base")
print("Whisper model loaded.")


def extract(url: str):
    print("Starting extraction...")

    url = url.split("?")[0]

    os.makedirs(BASE_DIR, exist_ok=True)

    file_id = str(uuid.uuid4())
    video_path = f"{BASE_DIR}/{file_id}.mp4"
    audio_path = f"{BASE_DIR}/{file_id}.wav"

    print("Downloading video...")

    subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", video_path,
        url
    ], check=True)

    print("Extracting audio...")

    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], check=True)

    print("Running Whisper...")

    result = MODEL.transcribe(audio_path)

    print("TRANSCRIPT:", result["text"])

    return result["text"]