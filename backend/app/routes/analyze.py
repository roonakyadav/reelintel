from fastapi import APIRouter
from app.services import extractor, detector, searcher, verifier

router = APIRouter()

@router.post("/analyze")
def analyze_reel(data: dict):
    reel_url = data.get("url")

    content = extractor.extract(reel_url)
    tools = detector.detect(content)
    links = searcher.search(tools)
    verified = verifier.verify(links)

    return {
        "url": reel_url,
        "tools": tools,
        "results": verified
    }