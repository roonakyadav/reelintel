from fastapi import APIRouter
from app.services import extractor, detector, searcher, verifier
from app.services.vision import analyze_frames

router = APIRouter()


@router.post("/analyze")
def analyze_reel(data: dict):

    url = data.get("url")

    # 1️⃣ Extract (audio + frames)
    extracted_data = extractor.extract(url)

    audio_text = extracted_data["transcript"]
    frames_dir = extracted_data["frames_dir"]

    # 2️⃣ OCR Visual Text
    visual_text = analyze_frames(frames_dir)

    # 3️⃣ Merge Audio + Visual
    final_text = audio_text + " " + visual_text

    # 4️⃣ Detect Tools
    tools = detector.detect(audio_text, visual_text)

    # 5️⃣ Search Links
    links = searcher.search(tools)

    # 6️⃣ Verify Links
    verified = verifier.verify(links)

    return {
        "url": url,
        "tools": tools,
        "results": verified
    }