import os
import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def build_search_query(fingerprint: dict) -> str:
    """
    Convert fingerprint into a strong Google query.
    """

    core = fingerprint.get("core_mechanism") or fingerprint.get("main_value_proposition") or ""
    visuals = " ".join(fingerprint.get("distinctive_visual_clues", []))
    rare_terms = " ".join(fingerprint.get("rare_or_brand_like_terms_from_ocr", []))

    query = f"{core} {visuals} {rare_terms} official website"

    return query.strip()


def search_web(query: str, num_results: int = 10):
    """
    Use Serper.dev to search Google.
    """

    if not SERPER_API_KEY:
        raise Exception("SERPER_API_KEY not set in environment variables")

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": num_results
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Search API Error: {response.text}")

    data = response.json()

    results = []

    for item in data.get("organic", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })

    return results