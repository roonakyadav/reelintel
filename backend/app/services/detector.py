import re
from collections import Counter


def tokenize(text: str):
    return re.findall(r'\b[A-Za-z0-9+\-]{3,}\b', text)


def detect(audio_text: str, visual_text: str):

    audio_tokens = tokenize(audio_text)
    visual_tokens = tokenize(visual_text)

    audio_counter = Counter([w.lower() for w in audio_tokens])
    visual_counter = Counter([w.lower() for w in visual_tokens])

    all_candidates = set(audio_counter.keys()).union(set(visual_counter.keys()))

    results = {}

    for word in all_candidates:

        score = 0

        # Appears in audio
        if word in audio_counter:
            score += 3

        # Appears in OCR
        if word in visual_counter:
            score += 2

        # Frequency bonus
        total_freq = audio_counter[word] + visual_counter[word]
        if total_freq > 2:
            score += 1

        # Context bonus
        if any(ctx in word for ctx in ["ai", "ide", "code", "dev"]):
            score += 2

        if score >= 4:
            results[word.capitalize()] = score

    return results