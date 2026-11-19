"""Untrusted free text (notes, comments, order text, any non-dictionary string).

    1. line endings -> '\\n'; control characters dropped except '\\n' and '\\t'
    2. multi-line text -> block, every line prefixed '| '
    3. single-line text: '===' -> '= = ='

No untrusted text can produce a line starting with the reserved section marker '==='.
Structural only: the model may still follow embedded instructions.
"""

import unicodedata

SECTION_MARKER = "==="
DEFUSED_MARKER = "= = ="
BLOCK_PREFIX = "| "
CONTROL_CATEGORY = "Cc"
KEPT_CONTROLS = frozenset({"\n", "\t"})

# Unicode line breaks some tokenizers and models treat as newlines.
_LINE_BREAKS = ("\r\n", "\r", "\u0085", " ", " ", "\v", "\f")


def clean(text: str) -> str:
    """Rule 1, then outer whitespace stripped."""
    for line_break in _LINE_BREAKS:
        text = text.replace(line_break, "\n")
    kept = (c for c in text if c in KEPT_CONTROLS or unicodedata.category(c) != CONTROL_CATEGORY)
    return "".join(kept).strip()


def inline(text: str) -> str:
    """Rule 3 on one line, e.g. 'a === b' -> 'a = = = b'. Repeats until no marker is left ('====')."""
    while SECTION_MARKER in text:
        text = text.replace(SECTION_MARKER, DEFUSED_MARKER)
    return text


def block(text: str) -> list[str]:
    """Rule 2: each line prefixed, e.g. 'a\\nb' -> ['| a', '| b']."""
    return [f"{BLOCK_PREFIX}{line}".rstrip() for line in clean(text).split("\n")]


def split(text: str) -> tuple[str, list[str]]:
    """(inline text, block lines): single-line text stays inline, multi-line text becomes a block."""
    cleaned = clean(text)
    if "\n" not in cleaned:
        return inline(cleaned), []
    return "", block(cleaned)
