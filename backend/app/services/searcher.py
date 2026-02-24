import requests
from bs4 import BeautifulSoup
import urllib.parse


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def search(tools: list):
    results = {}

    for tool in tools:

        query = f"{tool} AI tool official site github"
        query = urllib.parse.quote_plus(query)

        url = f"https://duckduckgo.com/html/?q={query}"

        res = requests.get(url, headers=HEADERS, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")

        links = soup.select(".result__a")

        if links:
            results[tool] = links[0]["href"]

    return results