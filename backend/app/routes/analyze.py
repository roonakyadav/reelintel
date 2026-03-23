from fastapi import APIRouter, HTTPException
from app.services import extractor
from app.services.vision import analyze_frames
from app.services.intelligence import build_fingerprint
from app.services.retriever import build_search_query, search_web

router = APIRouter()


@router.post("/analyze")
def analyze_reel(data: dict):
    try:
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        print("Starting extraction...")

        # 1️⃣ Extract transcript + frames
        extracted_data = extractor.extract(url)

        audio_text = extracted_data.get("transcript", "")
        frames_dir = extracted_data.get("frames_dir", "")

        # 2️⃣ OCR visual text
        visual_text = analyze_frames(frames_dir)

        # 3️⃣ Merge context
        final_text = f"{audio_text} {visual_text}"

        print("Building product fingerprint...")

        # 4️⃣ Build fingerprint
        fingerprint = build_fingerprint(final_text)

        print("Building search query...")

        # 5️⃣ Build search query
        query = build_search_query(fingerprint)

        print("Search Query:", query)

        print("Searching web...")

        # 6️⃣ Search real web
        candidates = search_web(query)

        return {
            "url": url,
            "fingerprint": fingerprint,
            "search_query": query,
            "candidates": candidates
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))