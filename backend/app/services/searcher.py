import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def search(tools):

    results = {}

    for tool in tools:

        try:
            query = f"{tool} official site github"
            url = f"https://duckduckgo.com/html/?q={query}"

            res = requests.get(url, headers=HEADERS, timeout=10)

            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.select(".result__a")

            if links:
                results[tool] = links[0]["href"]

        except Exception as e:
            print(f"Search failed for {tool}: {e}")
            continue

    return results