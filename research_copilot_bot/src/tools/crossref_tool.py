import requests
from moya.tools.tool import Tool

def lookup_metadata(title: str):
    """
    Lookup paper metadata using Crossref API.

    :param title: Title of the paper.
    :return: Metadata dictionary or None.
    """
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "rows": 1
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if items:
                item = items[0]
                return {
                    "title": item.get("title", [""])[0],
                    "authors": [f"{a.get('given','')} {a.get('family','')}" for a in item.get("author", [])],
                    "doi": item.get("DOI"),
                    "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                }
    except Exception:
        pass
    return None

crossref_tool = Tool(
    name="lookup_metadata",
    description="Lookup paper metadata using Crossref API.",
    function=lookup_metadata
)
