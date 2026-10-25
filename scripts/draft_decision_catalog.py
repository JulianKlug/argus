"""Draft the decision catalog from the Parquet store and grade it against independent evidence.

    uv run python scripts/draft_decision_catalog.py \
        --store /mnt/data1/klug/argus/mimiciv_3.1 \
        --out src/argus/data/decision_catalog.csv

Every key observed in the store gets a row; heuristics below set its category. Evidence in
scripts/catalog_evidence/ then sets the status:

    draft category ──┬── mimic_code.csv       (itemid -> category, from MIT-LCP/mimic-code 303d26c concepts)
                     ├── model_labels.csv    (blind per-row model labels + confidence)
                     └── external_review.csv (external LLM review of all rows, mapped to this taxonomy)
        all agree            -> AGREED
        any disagrees / low  -> DISPUTED   (clinician decides)
        no evidence          -> DRAFT

Rows already marked REVIEWED in `--out` are kept as they are.
"""

import argparse
import csv
from pathlib import Path

import duckdb
import pandas as pd

from argus.data.catalog import (
    CATALOG_COLUMNS, NOT_DECISION, POE_KEY_SEPARATOR, Source, Status, drug_key,
)
from argus.data.domains import DecisionCategory

MEDICATION = DecisionCategory.MEDICATION.name
DEVICE_SETTING = DecisionCategory.DEVICE_SETTING.name
TEST_ORDER = DecisionCategory.TEST_ORDER.name
PROCEDURE = DecisionCategory.PROCEDURE.name
CONSULT = DecisionCategory.CONSULT.name
PATIENT_MOVEMENT = DecisionCategory.PATIENT_MOVEMENT.name

EVIDENCE_DIR = Path(__file__).parent / "catalog_evidence"
MIMIC_CODE_FILE = "mimic_code.csv"
# Per-row label files (source, key, category, subcategory, confidence, reason), by evidence name.
LABEL_FILES = {
    "model": "model_labels.csv",  # blind per-row model labels
    "external": "external_review.csv",  # external LLM review, mapped to this taxonomy
}
LOW_CONFIDENCE = "low"
ORDER_CATEGORY_GROUP = "inputevents/order category"

# chartevents items that are set by clinicians (not measured), per device.
SETTING_ITEMS: dict[str, tuple[int, ...]] = {
    "VENTILATOR": (
        220339, 223834, 223835, 223848, 223849, 224684, 224688, 224691, 224701, 224702, 224704,
        224705, 224706, 224707, 224709, 224715, 224738, 224865, 226732, 226871, 226873,
        227187, 227287, 227577, 227579, 227580, 227581, 227582, 227583, 227808, 229314, 229659,
        229663, 230045,
    ),
    "DIALYSIS": (
        224144, 224145, 224153, 224154, 224191, 225183, 225806, 225810, 225953, 225958, 225976,
        225977, 227290, 228004, 228005, 228006, 230083, 230084, 230085,
    ),
    "ECMO": (229268, 229277, 229278, 229280, 229840, 229841, 229845, 229846),
    "VAD": (229262, 229263, 229303, 229829, 229836, 229895, 228192, 228195, 230086),
    "IABP": (220120, 227980),
    "IMPELLA": (228156, 229675),
    "PACEMAKER": (220194, 223956, 223964, 223969, 224751, 224753, 224755, 224837, 224838),
}

# Other charted decisions (not device settings): itemid -> (category, subcategory).
CHARTED_ITEMS: dict[int, tuple[str, str]] = {
    **dict.fromkeys((224327, 224328, 224329, 224330, 224331, 224410, 228296), (MEDICATION, "PCA")),
    **dict.fromkeys((224366, 224368, 224411), (MEDICATION, "EPIDURAL")),
    **dict.fromkeys((230050, 230051, 230053, 230057, 230058, 230060, 227812, 227813), (MEDICATION, "NERVE_BLOCK")),
    **dict.fromkeys((224169, 224170, 224172, 224173, 224175, 224176,  # MDI drug / puffs
                     224178, 224181, 227133, 227570, 224375, 224748, 227131,  # nebulizers
                     224749), (MEDICATION, "INHALED")),  # nitric oxide
    **dict.fromkeys((227210, 227211, 227212, 227213, 227214, 227215, 229788, 229789), (MEDICATION, "INTUBATION")),
    **dict.fromkeys((225952, 225959, 227638, 227639), (MEDICATION, "DIALYSATE_ADDITIVE")),
    227521: (MEDICATION, "BOLUS"),  # KCL Bolus
    **dict.fromkeys((224856, 227672, 227673, 227946, 227963), (PROCEDURE, "RESTRAINT")),
    **dict.fromkeys((225575, 228026, 225707, 228036, 225647, 225284, 228046), (PROCEDURE, "LINE")),  # "ordered on POE"
    228163: (PROCEDURE, "DEVICE"),  # Impella Catheter Repositioned
}

# procedureevents d_items.category whose every item is a decision.
PROCEDURE_GROUPS: dict[str, tuple[str, str]] = {
    "Access Lines - Invasive": (PROCEDURE, "LINE"),
    "Access Lines - Peripheral": (PROCEDURE, "LINE"),
}

# procedureevents items that are decisions: itemid -> (category, subcategory).
PROCEDURE_ITEMS: dict[int, tuple[str, str]] = {
    224385: (PROCEDURE, "AIRWAY"),  # Intubation
    227194: (PROCEDURE, "AIRWAY"),  # Extubation
    225448: (PROCEDURE, "AIRWAY"),  # Percutaneous Tracheostomy
    226237: (PROCEDURE, "AIRWAY"),  # Open Tracheostomy
    225792: (DEVICE_SETTING, "VENTILATOR"),  # Invasive Ventilation
    225794: (DEVICE_SETTING, "VENTILATOR"),  # Non-invasive Ventilation
    225441: (DEVICE_SETTING, "DIALYSIS"),  # Hemodialysis
    225802: (DEVICE_SETTING, "DIALYSIS"),  # Dialysis - CRRT
    225803: (DEVICE_SETTING, "DIALYSIS"),  # Dialysis - CVVHD
    225805: (DEVICE_SETTING, "DIALYSIS"),  # Peritoneal Dialysis
    225809: (DEVICE_SETTING, "DIALYSIS"),  # Dialysis - CVVHDF
    225955: (DEVICE_SETTING, "DIALYSIS"),  # Dialysis - SCUF
    229366: (MEDICATION, "INHALED"),  # Epoprostenol (Inhaled)
    229983: (MEDICATION, "INHALED"),  # Isoflurane
    229984: (MEDICATION, "INHALED"),  # Sevoflurane
    229996: (MEDICATION, "INHALED"),  # Nitrous Oxide
    225464: (PROCEDURE, "CARDIOVERSION"),  # Cardioversion/Defibrillation
    **dict.fromkeys((
        225399, 225400, 225428, 225429, 225430, 225433, 225434, 225439, 225442, 225445, 225446, 225447,
        225449, 225450, 225479, 226236, 226474, 226475, 226476, 226477, 227550, 227551, 227711, 227712,
        227713, 227714, 229501, 229576, 229580, 229585, 229586,
    ), (PROCEDURE, None)),  # 4-Procedures except tests (EEG, EKG, EMG), screens, cath lab transfers
    229351: (PROCEDURE, "CATHETER"),  # Foley Catheter
}

INPUT_ORDER_CATEGORIES: dict[str, tuple[str, str | None]] = {
    "01-Drips": (MEDICATION, "DRIP"),
    "02-Fluids (Crystalloids)": (MEDICATION, "FLUID"),
    "03-IV Fluid Bolus": (MEDICATION, "FLUID_BOLUS"),
    "04-Fluids (Colloids)": (MEDICATION, "FLUID"),
    "05-Med Bolus": (MEDICATION, "BOLUS"),
    "06-Insulin (Non IV)": (MEDICATION, "SUBCUTANEOUS"),
    "07-Blood Products": (MEDICATION, "BLOOD_PRODUCT"),
    "08-Antibiotics (IV)": (MEDICATION, "ANTIBIOTIC"),
    "09-Antibiotics (Non IV)": (MEDICATION, "ANTIBIOTIC"),
    "10-Prophylaxis (IV)": (MEDICATION, "PROPHYLAXIS"),
    "11-Prophylaxis (Non IV)": (MEDICATION, "PROPHYLAXIS"),
    "12-Parenteral Nutrition": (MEDICATION, "NUTRITION"),
    "13-Enteral Nutrition": (MEDICATION, "NUTRITION"),
    "14-Oral/Gastric Intake": (NOT_DECISION, None),  # intake documentation
    "15-Supplements": (MEDICATION, "NUTRITION"),
    "16-Pre Admission/Non-ICU": (NOT_DECISION, None),  # given before the ICU
}

# poe order_type default, then (order_type, order_subtype) overrides.
POE_TYPES: dict[str, tuple[str, str | None]] = {
    "Medications": (MEDICATION, None),  # New/Change duplicate prescriptions; only stops used
    "Radiology": (TEST_ORDER, "IMAGING"),
    "Cardiology": (TEST_ORDER, "CARDIOLOGY"),
    "Blood Bank": (MEDICATION, "BLOOD_PRODUCT"),
    "Respiratory": (DEVICE_SETTING, "VENTILATOR"),
    "Hemodialysis": (DEVICE_SETTING, "DIALYSIS"),
    "TPN": (MEDICATION, "NUTRITION"),
    "Critical Care": (TEST_ORDER, "MONITORING"),
    "Neurology": (TEST_ORDER, "NEUROLOGY"),
    "Lab": (TEST_ORDER, "LAB"),  # order time; specimens add draw time and test names
    "Consults": (CONSULT, None),
    "ADT orders": (PATIENT_MOVEMENT, None),
}
POE_SUBTYPES: dict[tuple[str, str], tuple[str, str | None]] = {
    ("Blood Bank", "Blood tests"): (TEST_ORDER, "BLOOD_BANK"),
    ("Cardiology", "LVAD Settings"): (DEVICE_SETTING, "VAD"),
    ("Respiratory", "Oxygen Therapy"): (DEVICE_SETTING, "OXYGEN"),
    ("Respiratory", "Pulmonary Function Test"): (TEST_ORDER, "PULMONARY_FUNCTION"),
    ("Respiratory", "Respiratory Therapy Consult"): (CONSULT, None),
    ("Respiratory", "Special Procedures"): (PROCEDURE, None),
    ("Nutrition", "Nutrition consult"): (CONSULT, None),
    ("General Care", "Restraints"): (PROCEDURE, "RESTRAINT"),
    ("IV therapy", "Access"): (PROCEDURE, "LINE"),
    ("IV therapy", "IV access"): (PROCEDURE, "LINE"),
    ("IV therapy", "IV access request"): (PROCEDURE, "LINE"),
    ("Critical Care", "Suspected VAP Pathway"): (NOT_DECISION, None),
    ("General Care", "Telemetry"): (TEST_ORDER, "MONITORING"),
    ("General Care", "Telemetry: Cardiac, cont SpO2"): (TEST_ORDER, "MONITORING"),
    ("General Care", "Vitals/Monitoring"): (TEST_ORDER, "MONITORING"),
    ("IV therapy", "IV fluids"): (MEDICATION, "FLUID"),
    ("IV therapy", "IV Drip"): (MEDICATION, "DRIP"),
}

def _row(source: Source, group: str, key, label, category: str, subcategory: str | None = None,
         drug: str | None = None) -> dict:
    return dict(source=source.value, group=group, key=str(key), label=label or "", category=category,
                subcategory=subcategory or "", drug=drug or "", status=Status.DRAFT.name, evidence="")


def _d_items(con, store: Path) -> list[dict]:
    """chartevents / inputevents / procedureevents items."""
    settings = {itemid: device for device, itemids in SETTING_ITEMS.items() for itemid in itemids}
    rows = []
    items = con.sql(
        f"SELECT itemid, label, linksto, category FROM '{store}/icu/d_items.parquet' "
        f"WHERE linksto IN ('chartevents', 'inputevents', 'procedureevents') ORDER BY itemid"
    ).fetchall()
    for itemid, label, linksto, item_group in items:
        group = f"{linksto}/{item_group}"  # e.g. 'chartevents/Routine Vital Signs'
        if linksto == "inputevents":
            rows.append(_row(Source.D_ITEMS, group, itemid, label, MEDICATION, drug=drug_key(label)))
            continue
        if linksto == "procedureevents":
            default = PROCEDURE_GROUPS.get(item_group, (NOT_DECISION, None))
            category, subcategory = PROCEDURE_ITEMS.get(itemid, default)
            rows.append(_row(Source.D_ITEMS, group, itemid, label, category, subcategory))
            continue
        device = settings.get(itemid)
        default = (DEVICE_SETTING, device) if device else (NOT_DECISION, None)
        category, subcategory = CHARTED_ITEMS.get(itemid, default)
        rows.append(_row(Source.D_ITEMS, group, itemid, label, category, subcategory))
    return rows


def _input_order_categories(con, store: Path) -> list[dict]:
    names = con.sql(
        f"SELECT DISTINCT ordercategoryname FROM '{store}/icu/inputevents.parquet' "
        f"WHERE ordercategoryname IS NOT NULL ORDER BY 1"
    ).fetchall()
    rows = []
    for (name,) in names:
        category, subcategory = INPUT_ORDER_CATEGORIES.get(name, (NOT_DECISION, None))
        rows.append(_row(Source.INPUT_ORDER_CATEGORY, ORDER_CATEGORY_GROUP, name, name, category, subcategory))
    return rows


def _poe(con, store: Path) -> list[dict]:
    pairs = con.sql(
        f"SELECT DISTINCT order_type, order_subtype FROM '{store}/hosp/poe.parquet' ORDER BY 1, 2"
    ).fetchall()
    rows = []
    for order_type, order_subtype in pairs:
        default = POE_TYPES.get(order_type, (NOT_DECISION, None))
        category, subcategory = POE_SUBTYPES.get((order_type, order_subtype), default)
        key = f"{order_type}{POE_KEY_SEPARATOR}{order_subtype or ''}"
        rows.append(_row(Source.POE, order_type, key, order_subtype or order_type, category, subcategory))
    return rows


def _grade(rows: list[dict], evidence: Path) -> None:
    """Set status and evidence of each row from the independent sources (in place)."""
    mimic = pd.read_csv(evidence / MIMIC_CODE_FILE, dtype=str).set_index("itemid")
    labels = {
        name: pd.read_csv(evidence / file, dtype=str, keep_default_na=False).set_index(["source", "key"])
        for name, file in LABEL_FILES.items()
    }

    for row in rows:
        notes, disputed = [], False

        if row["source"] == Source.D_ITEMS.value and row["key"] in mimic.index:
            concept = mimic.loc[row["key"]]
            disputed |= concept["category"] != row["category"]
            notes.append(f"mimic-code {concept['concept']} -> {concept['category']}")

        for name, frame in labels.items():
            if (row["source"], row["key"]) not in frame.index:
                continue
            label = frame.loc[(row["source"], row["key"])]
            if not label["category"]:  # source has no opinion on this row
                continue
            disputed |= label["category"] != row["category"] or label["confidence"] == LOW_CONFIDENCE
            subcategory = f"/{label['subcategory']}" if label["subcategory"] else ""
            notes.append(f"{name} ({label['confidence']}) {label['category']}{subcategory}: {label['reason']}")

        row["evidence"] = "; ".join(notes)
        if not notes:
            continue
        row["status"] = (Status.DISPUTED if disputed else Status.AGREED).name


def _reviewed(path: Path) -> dict[tuple[str, str], dict]:
    """Rows a clinician already reviewed, keyed by (source, key)."""
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {(r["source"], r["key"]): r for r in csv.DictReader(handle) if r["status"] == Status.REVIEWED.name}


def draft(store: Path, out: Path, evidence: Path) -> int:
    """Write the catalog; return the number of rows."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")

    rows = _d_items(con, store) + _input_order_categories(con, store) + _poe(con, store)
    _grade(rows, evidence)

    # Reviewed rows win; order by group so a whole group can be confirmed at once.
    reviewed = _reviewed(out)
    rows = [reviewed.get((r["source"], r["key"]), r) for r in rows]
    rows.sort(key=lambda r: (r["source"], r["group"], r["key"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()
    print(f"{draft(args.store, args.out, args.evidence)} rows -> {args.out}")


if __name__ == "__main__":
    main()
