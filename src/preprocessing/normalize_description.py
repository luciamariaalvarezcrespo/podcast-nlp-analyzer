"""Text normalization utilities for podcast episode descriptions.

This module exposes a single public helper, :func:`clean_description`, that
normalizes raw episode text for downstream NLP processing.
"""

import re
import emoji


def clean_description(text: str) -> str:
    """Normalize a podcast description string for NLP.

    The function applies a series of deterministic cleanup steps:
    - lowercases the input
    - removes emoji glyphs
    - removes hashtag tokens
    - removes telephone numbers
    - strips leading/trailing whitespace

    Args:
        text: Raw episode description text, or ``None``.

    Returns:
        A normalized string safe to pass to tokenization or vectorization.
    """
    if text is None:
        return ""

    text = text.lower()
    text = _remove_emojis(text)
    text = _remove_hashtags(text)
    text = _remove_phone_numbers(text)
    return text.strip()


def _remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="")


def _remove_hashtags(text: str) -> str:
    return re.sub(r"#\w+", "", text)


def _remove_phone_numbers(text: str) -> str:
    phone_pattern = re.compile(
        r"(?:\+?\d{1,3}[\s\-.])?(?:\(?\d{2,4}\)?[\s\-.]?){1,4}\d{2,4}"
    )
    return phone_pattern.sub("", text)
