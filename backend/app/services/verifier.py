def verify(links: dict):
    verified = {}

    for k, v in links.items():

        score = 0

        if "github.com" in v:
            score += 3
        if "official" in v or k.lower() in v.lower():
            score += 2
        if v.startswith("http"):
            score += 1

        if score >= 2:
            verified[k] = {
                "url": v,
                "confidence": score
            }

    return verified