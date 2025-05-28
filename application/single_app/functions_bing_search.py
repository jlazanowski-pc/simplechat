# functions_bing_search.py  -- Bad name now, we're using Google Search keeping for project coinsistency
import os
import requests
from typing import List, Dict

from config import *              # your project-specific helpers
from functions_settings import *  # get_settings()

###############################################################################
#  Google Search helpers
###############################################################################

GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


def _google_search(
    query: str,
    key: str,
    cx: str,
    top_n: int = 10,
) -> List[Dict]:
    """
    Raw call to Google Custom Search JSON API.
    Returns list[{name,title},{url,link},{snippet}] with keys already
    normalised for the rest of the code-base.
    """
    params = {
        "key": key,
        "cx": cx,
        "q": query,
        "num": min(max(top_n, 1), 10),  # API max = 10 per request
    }

    try:
        resp = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as ex:
        # Log & fail “softly” – return [] so the calling code degrades gracefully
        print(f"[GoogleSearch] query='{query}' failed: {ex}")
        return []

    results: List[Dict] = []
    for it in items:
        results.append(
            {
                "name": it.get("title", "Untitled"),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", ""),
            }
        )
    return results


###############################################################################
#  Public API expected elsewhere in the code-base
###############################################################################

def get_search_suggestions(query: str, top_n: int = 5) -> List[str]:
    """
    Google’s CSE JSON API does not provide “autosuggest”.
    We emulate it the same way the old Bing wrapper did – by returning the
    titles of the first N search results.
    """
    return [r["name"] for r in get_search_results(query, top_n=top_n)]


def get_search_results(query: str, top_n: int = 10) -> List[Dict]:
    """
    Main search entry-point, now backed by Google Programmable Search.
    Keeps the same signature the rest of the app expects.
    """
    settings = get_settings()

    if not settings.get("enable_web_search"):
        print("[GoogleSearch] Web search disabled in settings.")
        return []

    google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx  = os.getenv("GOOGLE_SEARCH_CX")

    if not google_key or not google_cx:
        print("[GoogleSearch] API key or CX missing – check configuration.")
        return []

    return _google_search(query, google_key, google_cx, top_n=top_n)


def process_query_with_bing_and_llm(user_query: str, top_n: int = 10):
    """
    This wrapper name is still referenced elsewhere – we simply redirect
    to Google search so no other files have to change.
    """
    print(f"[GoogleSearch] Searching for: {user_query!r}")
    results = get_search_results(user_query, top_n=top_n)
    print(f"[GoogleSearch] {len(results)} result(s) returned.")
    return results
