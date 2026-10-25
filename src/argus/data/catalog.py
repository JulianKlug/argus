"""Decision catalog: MIMIC keys (itemid, order type) -> decision category.

Drafted by scripts/draft_decision_catalog.py, cross-checked against independent evidence
(mimic-code concepts, model labels), disputed rows reviewed by a clinician.
Prescription drug strings are not catalogued: every non-diluent prescription is a medication
order; only their drug name is normalized here, to link infusions.
"""

import re

from enum import Enum
from pathlib import Path

import pandas as pd

from argus.data.domains import SUBCATEGORY, DecisionCategory

CATALOG_PATH = Path(__file__).parent / "decision_catalog.csv"
CATALOG_COLUMNS = ("source", "group", "key", "label", "category", "subcategory", "drug", "status", "evidence")
NOT_DECISION = "NOT_DECISION"
POE_KEY_SEPARATOR = "|"  # poe key = "order_type|order_subtype", e.g. "Radiology|CT Scan"

_WORD = re.compile(r"[a-z]+")
_STRENGTH = re.compile(r"\d+(?:[.,]\d+)?%")  # '5%', '0.9%', '0,9%'
_KEY_SEPARATOR = " "

# Classification columns returned per key (with SUBCATEGORY).
CATEGORY = "category"
DRUG = "drug"


class Source(Enum):
    """Which MIMIC key a catalog row classifies."""

    D_ITEMS = "d_items"  # itemid of chartevents / inputevents / procedureevents
    INPUT_ORDER_CATEGORY = "inputevents_ordercategory"
    POE = "poe"


class Status(Enum):
    """Trust in a row's category; the CSV stores the name."""

    DRAFT = "draft"  # heuristic only, no independent evidence
    AGREED = "agreed"  # every independent source agrees with the draft
    DISPUTED = "disputed"  # a source disagrees, or the model is unsure: clinician decides
    REVIEWED = "reviewed"  # set by a clinician; kept on redraft


class DecisionCatalog:
    """Read-only lookups over the catalog CSV."""

    def __init__(self, path: Path = CATALOG_PATH) -> None:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)

        # "" -> missing; category name -> DecisionCategory, NOT_DECISION -> missing.
        frame = frame.replace("", None)
        frame[CATEGORY] = frame[CATEGORY].map(lambda name: None if name == NOT_DECISION else DecisionCategory[name])

        self._entries = {
            source: frame[frame["source"] == source.value].set_index("key")[[CATEGORY, SUBCATEGORY, DRUG]]
            for source in Source
        }
        self._drugs = [(key, frozenset(key.split(_KEY_SEPARATOR))) for key in set(self._entries[Source.D_ITEMS][DRUG].dropna())]

    def classify(self, source: Source, keys: pd.Series) -> pd.DataFrame:
        """category / subcategory / drug per key, aligned on `keys`; unknown key -> not a decision."""
        entries = self._entries[source]
        text = keys.astype("string")
        return pd.DataFrame(
            {column: text.map(entries[column]).astype(object) for column in (CATEGORY, SUBCATEGORY, DRUG)},
            index=keys.index,
        )

    def keys(self, source: Source) -> set[str]:
        return set(self._entries[source].index)

    def prescription_drugs(self, drugs: pd.Series) -> pd.Series:
        """Normalized drug per prescription string, comparable to the ICU items' `drug`.

        Most specific ICU drug key whose tokens all occur in the string (ties: earliest), else the first word.
        e.g. 'NORepinephrine' -> 'norepinephrine', 'Piperacillin-Tazobactam' -> 'piperacillin',
             'Dextrose 5%' -> 'dextrose 5%', 'Dextrose 50%' -> 'dextrose 50%' (or 'dextrose': never 'dextrose 5%').
        """
        normalized = {name: self._normalize(name) for name in drugs.dropna().unique()}
        return pd.Series([normalized.get(name) for name in drugs], index=drugs.index, dtype=object)

    def _normalize(self, drug: str) -> str | None:
        text = drug.lower()
        words = _WORD.findall(text)
        if not words:
            return None

        tokens = set(words) | set(_strengths(text))
        known = [(key, parts) for key, parts in self._drugs if parts <= tokens]
        if not known:
            return words[0]

        def specificity(entry: tuple[str, frozenset]) -> tuple[int, int]:
            key, parts = entry
            return len(parts), -words.index(key.split(_KEY_SEPARATOR)[0])

        return max(known, key=specificity)[0]


def drug_key(label: str) -> str | None:
    """ICU item label -> drug key: first word + strengths.

    e.g. 'Midazolam (Versed)' -> 'midazolam', 'Dextrose 5%' -> 'dextrose 5%', 'Saline 0,9%' -> 'saline 0.9%'.
    """
    text = label.lower()
    word = _WORD.search(text)
    if not word:
        return None
    return _KEY_SEPARATOR.join([word.group(), *_strengths(text)])


def _strengths(text: str) -> list[str]:
    return [strength.replace(",", ".") for strength in _STRENGTH.findall(text)]
