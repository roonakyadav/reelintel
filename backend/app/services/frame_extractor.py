import cv2
import os


def extract_frames(video_path: str, output_dir: str, interval_sec: int = 2):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()

    return saved