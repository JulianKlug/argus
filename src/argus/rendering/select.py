"""Record ownership, selection order and quota filling.

Ownership: a source record is fully rendered at most once; first matching section wins.

    decisions ─► current state ─► trends ─► recent evidence ─► notes ─► past admissions

Quota filling (one section, every counter):

    reserve = heading + worst-case omission line
    for item in selection order:  first variant whose line cost fits ─► take; none fits ─► stop
    exact count of the composed section; over budget ─► drop items from the end, recount
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from argus.data import Domain
from argus.rendering.types import Section, SourceRow

Counters = Mapping[str, Callable[[str], int]]
LINE_END = "\n"


class Ownership:
    """SourceRow -> owning section, assigned in section order."""

    def __init__(self) -> None:
        self._owner: dict[SourceRow, Section] = {}

    def claim(self, source: SourceRow, section: Section) -> bool:
        """True if `section` now owns `source`; False if an earlier section already does."""
        return self._owner.setdefault(source, section) is section

    def owner(self, source: SourceRow) -> Section | None:
        return self._owner.get(source)


@dataclass(frozen=True)
class Item:
    """One selectable unit. `variants`: rendered text in preference order (whole note, then shortened)."""

    variants: tuple[str, ...]
    display: tuple = ()  # display sort key; selection order is the list order
    domain: Domain | None = None  # recent evidence: source domain, for omission counts per domain


@dataclass(frozen=True)
class Choice:
    index: int  # into the candidate list
    variant: int  # 0 = whole

    @property
    def shortened(self) -> bool:
        return self.variant > 0


@dataclass
class Filled:
    """Selected items of one section, composed text and its exact token counts."""

    items: list[Item]
    chosen: list[Choice] = field(default_factory=list)
    costs: list[dict[str, int]] = field(default_factory=list)  # line tokens per chosen item
    text: str = ""
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def omitted(self) -> int:
        return len(self.items) - len(self.chosen)

    @property
    def shortened(self) -> int:
        return sum(c.shortened for c in self.chosen)

    def omitted_by_domain(self) -> dict[Domain, int]:
        """Items not rendered, per source domain (items without a domain are not counted)."""
        chosen = {c.index for c in self.chosen}
        counts: dict[Domain, int] = {}
        for index, item in enumerate(self.items):
            if index not in chosen and item.domain is not None:
                counts[item.domain] = counts.get(item.domain, 0) + 1
        return counts

    def selected(self) -> list[tuple[Item, str]]:
        """(item, chosen text) in display order."""
        pairs = [(self.items[c.index], self.items[c.index].variants[c.variant]) for c in self.chosen]
        return sorted(pairs, key=lambda pair: pair[0].display)


Compose = Callable[[Filled], str]


def count(text: str, counters: Counters) -> dict[str, int]:
    return {name: counter(text) for name, counter in counters.items()}


def fits(tokens: Mapping[str, int], budget: Mapping[str, int]) -> bool:
    return all(tokens[name] <= budget[name] for name in budget)


def fill(items: list[Item], budget: Mapping[str, int], counters: Counters, compose: Compose) -> Filled:
    """Greedy prefix in selection order under `budget` per counter; stops at the first item that does not fit."""
    filled = Filled(items)
    used = count(compose(Filled(items)), counters)  # reserve: heading + worst-case omission line

    for index, item in enumerate(items):
        choice = None
        for variant, text in enumerate(item.variants):
            cost = count(text + LINE_END, counters)
            if all(used[n] + cost[n] <= budget[n] for n in budget):
                choice = Choice(index, variant)
                break
        if choice is None:
            break

        filled.chosen.append(choice)
        filled.costs.append(cost)
        for name in used:
            used[name] += cost[name]

    return trim(filled, budget, counters, compose)


def trim(filled: Filled, budget: Mapping[str, int], counters: Counters, compose: Compose) -> Filled:
    """Exact recount; drop chosen items from the end until the composed section fits (or none is left)."""
    while True:
        filled.text = compose(filled)
        filled.tokens = count(filled.text, counters)
        if fits(filled.tokens, budget) or not filled.chosen:
            return filled

        # Drop at least the worst counter's overshoot in one step, then recount.
        worst = max(budget, key=lambda n: filled.tokens[n] - budget[n])
        overshoot = filled.tokens[worst] - budget[worst]
        dropped = 0
        while filled.chosen and dropped < overshoot:
            filled.chosen.pop()
            dropped += filled.costs.pop()[worst]


def drop(filled: Filled, tokens: int, worst: str, counters: Counters, compose: Compose) -> None:
    """Remove chosen items from the end worth >= `tokens` of counter `worst` (at least one), then recompose."""
    dropped = 0
    while filled.chosen and (dropped < tokens or dropped == 0):
        filled.chosen.pop()
        dropped += filled.costs.pop()[worst]
    filled.text = compose(filled)
    filled.tokens = count(filled.text, counters)
