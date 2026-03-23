import easyocr
import os

print("Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=True)
print("OCR model loaded.")


def read_frames(frames_dir: str):
    texts = []

    if not os.path.exists(frames_dir):
        return ""

    for file in os.listdir(frames_dir):

        if not file.endswith(".jpg"):
            continue

        path = os.path.join(frames_dir, file)

        results = reader.readtext(path)

        for (_, text, confidence) in results:

            if confidence > 0.4:
                texts.append(text)

    return " ".join(texts)