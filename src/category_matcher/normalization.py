"""Conservative exact keys and broader retrieval text are intentionally separate."""

import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-ampersand-v1"
RETRIEVAL_TEXT_VERSION = "unicode-marks-ascii-light-plurals-v3"
STOP_WORDS = set("a an the and or for of in on to with all general products goods used".split())


def exact_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("&", " and ").split())


def search_text(text: str) -> str:
    # Python's \w omits combining marks used inside Bangla, Hindi, and other scripts.
    characters = (
        character
        if character == "_" or unicodedata.category(character)[0] in {"L", "N", "M"}
        else " "
        for character in exact_key(text)
    )
    return " ".join("".join(characters).split())


def retrieval_terms(text: str) -> str:
    """Light plural folding for retrieval only; never an exact-match key."""
    words = []
    for word in search_text(text).split():
        if not word.isascii():
            words.append(word)
            continue
        if word in STOP_WORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith(("sses", "ches", "shes", "xes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        words.append(word)
    return " ".join(words)
