"""Real-window token measurement for D5 (docs/spec_rendering.md, "Real-window measurement").

SnapshotEngine.get_window() and D4 tokenizers only: no inference. Prints aggregates only:
no patient text, no identifiers. The sampled subject list goes to a local file outside the repo
so D6 can exclude these patients from the validation draw.

    ARGUS_STORE_DIR=/path/to/store uv run python scripts/measure_rendering.py \
        --subjects-out /outside/repo/d5_subjects.json --out /tmp/d5_aggregates.json

Sample:
    random  first --random valid windows among seeded candidates (stay, instant in stay)
    heavy   top --heavy windows by visible current-admission record count among --candidates

    candidates ──► workers (engine + tokenizers each) ──► per-window metrics ──► aggregates JSON
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from argus.data import AmbiguousEncounterError, DecisionCategory, Domain, NotInIcuStayError, SnapshotEngine
from argus.data.domains import EVENT_TIME, IS_PENDING, SOURCE_ROW, SOURCE_TABLE
from argus.llm import policy
from argus.llm.backends import connect
from argus.llm.config import DEFAULT_CONFIG as LLM_CONFIG
from argus.llm.config import _backend
from argus.rendering import RefKind, Renderer, RenderingBudgetError, Section, load_config
from argus.rendering import format as fmt
from argus.rendering.config import Budget
from argus.rendering.references import record_id
from argus.rendering.types import OPTIONAL_SECTIONS, SourceRow

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE_ENV = "ARGUS_STORE_DIR"
SEED = 20261116
RANDOM_WINDOWS = 200
HEAVY_WINDOWS = 50
HEAVY_CANDIDATES = 2000
RANDOM_CANDIDATES = 300  # first RANDOM_WINDOWS valid ones are kept
UNBOUNDED = 10**9
PERCENTILES = (50, 95, 99)
SKIPPED = (AmbiguousEncounterError, NotInIcuStayError)

# Per-process state, set by _init.
_engine: SnapshotEngine | None = None
_counters: dict = {}
_budgeted: Renderer | None = None
_unbudgeted: Renderer | None = None
_config = None


# ── counters and sample ───────────────────────────────────────────────────


def tokenizers(names: tuple[str, ...]) -> tuple[dict, dict]:
    """(backend -> count_text_tokens, backend -> pinned digest); policy and digest checked, no inference."""
    import tomllib

    entries = tomllib.loads(LLM_CONFIG.read_text())["backends"]
    counters, digests = {}, {}
    for name in names:
        config = _backend(name, entries[name])
        policy.check_endpoint(config, date.today())
        backend = connect(config, policy.check_response_status).backend
        server = backend.server_info()
        policy.check_server(config, server.digest, server.remote_host)
        counters[name] = backend.count_text_tokens
        digests[name] = config.digest
    return counters, digests


def stays(store: Path) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    rows = duckdb.sql(
        f"SELECT subject_id, intime, outtime FROM '{store}/icu/icustays.parquet' WHERE outtime > intime ORDER BY stay_id"
    ).fetchall()
    return [(s, pd.Timestamp(i), pd.Timestamp(o)) for s, i, o in rows]


def candidates(all_stays: list, rng: random.Random, n: int) -> list[tuple[int, str]]:
    """(subject_id, at ISO) at a uniform minute-floored instant inside a uniform stay."""
    result = []
    for _ in range(n):
        subject, intime, outtime = rng.choice(all_stays)
        at = max((intime + rng.random() * (outtime - intime)).floor("min"), intime)
        result.append((subject, at.isoformat()))
    return result


# ── worker ────────────────────────────────────────────────────────────────


def _init(store: str) -> None:
    global _engine, _counters, _budgeted, _unbudgeted, _config
    _config = load_config()
    _engine = SnapshotEngine(Path(store))
    _counters, _ = tokenizers(_config.tokenizers)
    unbounded = Budget(UNBOUNDED, UNBOUNDED // 8, UNBOUNDED // 8, UNBOUNDED // 8, UNBOUNDED // 8)
    _budgeted = Renderer(_config, _counters)
    _unbudgeted = Renderer(replace(_config, budget=unbounded), _counters)


def volume(candidate: tuple[int, str]) -> int | None:
    """Visible current-admission record count, None if no valid window."""
    subject, at = candidate
    try:
        window = _engine.get_window(subject, pd.Timestamp(at).to_pydatetime())
    except SKIPPED:
        return None
    return sum(len(window.snapshot.current.get(d)) for d in Domain)


def measure(candidate: tuple[int, str]) -> dict | None:
    """Metrics of one window; None if no valid window. Never returns text."""
    subject, at = candidate
    started = time.perf_counter()
    try:
        window = _engine.get_window(subject, pd.Timestamp(at).to_pydatetime())
    except SKIPPED:
        return None
    window_s = time.perf_counter() - started

    unbudgeted = _unbudgeted.render(window)
    result = {
        "window_s": window_s,
        "unbudgeted": {s.value: unbudgeted.section_tokens[s] for s in Section},
        "unbudgeted_total": unbudgeted.total_tokens,
        "items": {s.value: unbudgeted.rendered_items[s] for s in OPTIONAL_SECTIONS},
        "domains": _domain_tokens(window),
        "categories": _category_tokens(unbudgeted.text),
        "core_present": _core_presence(window),
        "ids": {i: _key(r) for i, r in unbudgeted.refs.items() if r.kind is not RefKind.TREND},
    }

    started = time.perf_counter()
    try:
        rendered = _budgeted.render(window)
    except RenderingBudgetError:
        result["mandatory_failure"] = True
        return result
    result |= {
        "mandatory_failure": False,
        "render_s": time.perf_counter() - started,
        "budgeted": {s.value: rendered.section_tokens[s] for s in Section},
        "budgeted_total": rendered.total_tokens,
        "selected": {s.value: rendered.rendered_items[s] for s in OPTIONAL_SECTIONS},
        "omitted": {s.value: rendered.omissions[s] for s in OPTIONAL_SECTIONS},
        "domain_omitted": {d.value: rendered.domain_omissions.get(d, 0) for d in Domain},
    }
    return result


def _key(ref) -> str:
    """Opaque digest of the ref's key: collisions are counted without keeping row identifiers."""
    raw = "|".join(f"{s.table}:{s.row}" for s in ref.sources) + f"|{ref.category}|{ref.action}|{ref.time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _domain_tokens(window) -> dict:
    """Unbudgeted tokens of every record of each domain (current + past), minimal record lines."""
    at = window.snapshot.at
    result = {}
    for domain in Domain:
        frames = [window.snapshot.current.get(domain), window.snapshot.past.get(domain)]
        frame = pd.concat(frames, ignore_index=True).sort_values([EVENT_TIME, SOURCE_TABLE, SOURCE_ROW])
        lines = []
        for row in frame.to_dict("records"):
            ref = record_id(SourceRow(row[SOURCE_TABLE], int(row[SOURCE_ROW])))
            if domain is Domain.NOTES and fmt.present(row.get("text")):
                lines += fmt.note(row, ref, at, str(row["text"]), None, _config.documented_delay).all()
            else:
                lines += fmt.record(row, ref, at, _config.documented_delay).all()
        text = "\n".join(lines)
        result[domain.value] = {name: counter(text) for name, counter in _counters.items()}
    return result


def _category_tokens(text: str) -> dict:
    """Decision lines per category, cut from the unbudgeted DECISIONS section."""
    section = text.split("=== DECISIONS ===\n", 1)[1].split("\n\n=== ", 1)[0]
    blocks = {}
    for category in DecisionCategory:
        marker = f"--- {category.name} ---\n"
        body = section.split(marker, 1)[1].split("\n--- ", 1)[0]
        body = "" if body == fmt.NO_RECORDS else body
        blocks[category.value] = {name: counter(body) for name, counter in _counters.items()}
    return blocks


def _core_presence(window) -> dict:
    result = {}
    for variable in _config.core_variables:
        domain = Domain.CHARTED_OBSERVATIONS if variable.table == "chartevents" else Domain.LABS
        frame = window.snapshot.current.get(domain)
        rows = frame[(frame[SOURCE_TABLE] == variable.table) & (frame["itemid"] == variable.itemid).fillna(False)]
        result[variable.label] = bool((~rows[IS_PENDING].astype(bool)).any())
    return result


# ── aggregation ───────────────────────────────────────────────────────────


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": float(np.median(array)),
        **{f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES if p != 50},
        "max": float(array.max()),
    }


def aggregate(results: list[dict], names: tuple[str, ...]) -> dict:
    ok = [r for r in results if not r["mandatory_failure"]]
    out = {"windows": len(results), "mandatory_failures": len(results) - len(ok)}

    out["sections"] = {}
    for section in Section:
        key = section.value
        entry = {
            "unbudgeted": {n: stats([r["unbudgeted"][key][n] for r in results]) for n in names},
            "budgeted": {n: stats([r["budgeted"][key][n] for r in ok]) for n in names},
        }
        if section in OPTIONAL_SECTIONS:
            entry["items_available"] = stats([r["items"][key] for r in results])
            entry["selected"] = stats([r["selected"][key] for r in ok])
            entry["omitted"] = stats([r["omitted"][key] for r in ok])
            entry["windows_hitting_quota"] = sum(r["omitted"][key] > 0 for r in ok)
        out["sections"][key] = entry

    out["total"] = {
        "unbudgeted": {n: stats([r["unbudgeted_total"][n] for r in results]) for n in names},
        "budgeted": {n: stats([r["budgeted_total"][n] for r in ok]) for n in names},
    }
    out["domains"] = {d.value: {n: stats([r["domains"][d.value][n] for r in results]) for n in names} for d in Domain}
    out["recent_evidence_omitted_by_domain"] = {d.value: stats([r["domain_omitted"][d.value] for r in ok]) for d in Domain}
    out["categories"] = {
        c.value: {n: stats([r["categories"][c.value][n] for r in results]) for n in names} for c in DecisionCategory
    }
    labels = list(results[0]["core_present"]) if results else []
    out["core_presence"] = {label: statistics.fmean(r["core_present"][label] for r in results) for label in labels}
    out["render_s"] = stats([r["render_s"] for r in ok])
    out["get_window_s"] = stats([r["window_s"] for r in results])
    return out


def collisions(results: list[dict]) -> int:
    """IDs mapping to more than one key across all sampled windows."""
    keys = defaultdict(set)
    for result in results:
        for ref_id, key in result["ids"].items():
            keys[ref_id].add(key)
    return sum(len(k) > 1 for k in keys.values())


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects-out", type=Path, required=True, help="sampled subject list, outside the repo")
    parser.add_argument("--out", type=Path, required=True, help="aggregates JSON")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--random", type=int, default=RANDOM_WINDOWS)
    parser.add_argument("--heavy", type=int, default=HEAVY_WINDOWS)
    parser.add_argument("--candidates", type=int, default=HEAVY_CANDIDATES)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    store = os.environ.get(STORE_ENV)
    if not store:
        sys.exit(f"{STORE_ENV} unset")
    if args.subjects_out.resolve().is_relative_to(REPO_ROOT):
        sys.exit("--subjects-out must be outside the repository")

    config = load_config()
    _, digests = tokenizers(config.tokenizers)
    all_stays = stays(Path(store))
    random_pool = candidates(all_stays, random.Random(args.seed), max(RANDOM_CANDIDATES, args.random))
    heavy_pool = candidates(all_stays, random.Random(args.seed + 1), args.candidates)

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(store,)) as pool:
        random_results = [r for r in pool.map(measure, random_pool, chunksize=4) if r is not None][: args.random]
        print(f"random: {len(random_results)} windows", file=sys.stderr)

        volumes = list(pool.map(volume, heavy_pool, chunksize=16))
        ranked = sorted(
            ((v, c) for v, c in zip(volumes, heavy_pool) if v is not None), key=lambda vc: (-vc[0], vc[1]),
        )
        heavy = [c for _, c in ranked[: args.heavy]]
        heavy_results = [r for r in pool.map(measure, heavy, chunksize=1) if r is not None]
        print(f"heavy: {len(heavy_results)} windows", file=sys.stderr)

    names = config.tokenizers
    store_manifest = Path(store) / "manifest.json"
    report = {
        "command": " ".join(["scripts/measure_rendering.py", *sys.argv[1:]]),
        "seed": args.seed,
        "config_digest": config.digest(),
        "model_digests": digests,
        "store_manifest_sha256": hashlib.sha256(store_manifest.read_bytes()).hexdigest(),
        "random": aggregate(random_results, names),
        "heavy": aggregate(heavy_results, names),
        "all": aggregate(random_results + heavy_results, names),
        "reference_collisions": collisions(random_results + heavy_results),
        "heavy_volume_threshold": ranked[args.heavy - 1][0] if len(ranked) >= args.heavy else None,
    }
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))

    # Validation hygiene: every candidate subject that may have been measured.
    subjects = sorted({s for s, _ in random_pool} | {s for s, _ in heavy})
    args.subjects_out.parent.mkdir(parents=True, exist_ok=True)
    args.subjects_out.write_text(json.dumps({
        "seed": args.seed, "config_digest": config.digest(), "subject_ids": subjects,
    }))
    os.chmod(args.subjects_out, 0o600)
    print(f"aggregates -> {args.out}; {len(subjects)} subjects -> {args.subjects_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
