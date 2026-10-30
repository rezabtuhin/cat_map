"""Explicit input-language hints; category output remains the English Google taxonomy."""

import re

# Enabled input languages, not a claim of measured accuracy in every language.
MULTILINGUAL_INPUTS = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "pt",
    "ru",
    "tr",
    "zh",
)


def language_supported(language: str, supported: list[str] | tuple[str, ...]) -> bool:
    if (
        not isinstance(language, str)
        or re.fullmatch(r"auto|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language) is None
    ):
        raise ValueError("Language must be auto or a valid language tag")
    return language == "auto" or language.split("-", 1)[0] in supported
