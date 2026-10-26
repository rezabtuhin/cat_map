"""Conservative exact keys and broader retrieval text are intentionally separate."""

import re
import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-ampersand-v1"


def exact_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("&", " and ").split())


def search_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", exact_key(text), flags=re.UNICODE))
