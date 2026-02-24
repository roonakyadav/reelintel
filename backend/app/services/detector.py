import re


# Known AI / Dev Tools (expand over time)
KNOWN_TOOLS = [
    "lovable",
    "cursor",
    "v0",
    "notion",
    "chatgpt",
    "figma",
    "github",
    "canva",
    "replit",
    "vercel",
    "netlify",
    "openai"
]


def detect(text: str):
    text_lower = text.lower()

    found = set()

    # 1. Match known tools
    for tool in KNOWN_TOOLS:
        if tool in text_lower:
            found.add(tool.capitalize())

    # 2. Extract capitalized words (possible brands)
    words = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", text)

    for w in words:
        if len(w) >= 3:
            found.add(w)

    return list(found)