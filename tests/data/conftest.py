"""Synthetic MIMIC-IV fixture store, built through scripts/build_store.py.

Subject 1 timeline (AT = 2180-01-10 23:59):

    2179-06-01 ── hadm 100 (prior, discharged 06-10) ──┐
    2180-01-08 08:00 ── hadm 200 (current) ───────────────────────────── 01-20 died
    2180-01-09 ── ICU stay 1000 ─────────────── 01-15
                                         AT ┤ AT+1m: sentinel rows (must never appear)

Every table gets SENTINEL rows (available AT + 1 min) and, where the rule allows,
BOUNDARY rows (available exactly AT). Subject 2 has no ICU stay; subject 3 has two
overlapping admissions.
"""

import csv
import gzip
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_store  # noqa: E402

AT = pd.Timestamp("2180-01-10 23:59:00")
MINUTE = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
SENTINEL_TIME = AT + MINUTE

SUBJECT = 1
SUBJECT_NO_ICU = 2
SUBJECT_AMBIGUOUS = 3
UNKNOWN_SUBJECT = 999
PRIOR_HADM = 100
CURRENT_HADM = 200
SENTINEL_HADM = 300
CURRENT_STAY = 1000
HEART_RATE_ITEM = 220045
LACTATE_ITEM = 50813
NOREPI_ITEM = 221906
URINE_ITEM = 226559
VENT_ITEM = 225792
PEEP_ITEM = 220339

# Decision window (spec_decision_window.md): PEEP kept then changed, norepinephrine started then titrated.
WINDOW = 6 * HOUR
WINDOW_START = AT - WINDOW

# Infusion for criterion 7: 10:00-16:00, 600 mL, observed at 12:00.
INFUSION_START = pd.Timestamp("2180-01-10 10:00")
INFUSION_END = pd.Timestamp("2180-01-10 16:00")
INFUSION_AT = pd.Timestamp("2180-01-10 12:00")
INFUSION_AMOUNT = 600.0

# Lab for criterion 6: drawn AT-30m, resulted AT+30m.
PENDING_LAB_DRAWN = AT - 30 * MINUTE
PENDING_LAB_STORED = AT + 30 * MINUTE
PENDING_LAB_VALUE = 4.2

HEADERS = {
    "hosp/admissions": "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admit_provider_id,admission_location,discharge_location,insurance,language,marital_status,race,edregtime,edouttime,hospital_expire_flag",
    "hosp/d_hcpcs": "code,category,long_description,short_description",
    "hosp/diagnoses_icd": "subject_id,hadm_id,seq_num,icd_code,icd_version",
    "hosp/d_icd_diagnoses": "icd_code,icd_version,long_title",
    "hosp/d_icd_procedures": "icd_code,icd_version,long_title",
    "hosp/d_labitems": "itemid,label,fluid,category",
    "hosp/drgcodes": "subject_id,hadm_id,drg_type,drg_code,description,drg_severity,drg_mortality",
    "hosp/emar": "subject_id,hadm_id,emar_id,emar_seq,poe_id,pharmacy_id,enter_provider_id,charttime,medication,event_txt,scheduletime,storetime",
    "hosp/emar_detail": "subject_id,emar_id,emar_seq,parent_field_ordinal,administration_type,pharmacy_id,barcode_type,reason_for_no_barcode,complete_dose_not_given,dose_due,dose_due_unit,dose_given,dose_given_unit,will_remainder_of_dose_be_given,product_amount_given,product_unit,product_code,product_description,product_description_other,prior_infusion_rate,infusion_rate,infusion_rate_adjustment,infusion_rate_adjustment_amount,infusion_rate_unit,route,infusion_complete,completion_interval,new_iv_bag_hung,continued_infusion_in_other_location,restart_interval,side,site,non_formulary_visual_verification",
    "hosp/hcpcsevents": "subject_id,hadm_id,chartdate,hcpcs_cd,seq_num,short_description",
    "hosp/labevents": "labevent_id,subject_id,hadm_id,specimen_id,itemid,order_provider_id,charttime,storetime,value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag,priority,comments",
    "hosp/microbiologyevents": "microevent_id,subject_id,hadm_id,micro_specimen_id,order_provider_id,chartdate,charttime,spec_itemid,spec_type_desc,test_seq,storedate,storetime,test_itemid,test_name,org_itemid,org_name,isolate_num,quantity,ab_itemid,ab_name,dilution_text,dilution_comparison,dilution_value,interpretation,comments",
    "hosp/omr": "subject_id,chartdate,seq_num,result_name,result_value",
    "hosp/patients": "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
    "hosp/pharmacy": "subject_id,hadm_id,pharmacy_id,poe_id,starttime,stoptime,medication,proc_type,status,entertime,verifiedtime,route,frequency,disp_sched,infusion_type,sliding_scale,lockout_interval,basal_rate,one_hr_max,doses_per_24_hrs,duration,duration_interval,expiration_value,expiration_unit,expirationdate,dispensation,fill_quantity",
    "hosp/poe": "poe_id,poe_seq,subject_id,hadm_id,ordertime,order_type,order_subtype,transaction_type,discontinue_of_poe_id,discontinued_by_poe_id,order_provider_id,order_status",
    "hosp/poe_detail": "poe_id,poe_seq,subject_id,field_name,field_value",
    "hosp/prescriptions": "subject_id,hadm_id,pharmacy_id,poe_id,poe_seq,order_provider_id,starttime,stoptime,drug_type,drug,formulary_drug_cd,gsn,ndc,prod_strength,form_rx,dose_val_rx,dose_unit_rx,form_val_disp,form_unit_disp,doses_per_24_hrs,route",
    "hosp/procedures_icd": "subject_id,hadm_id,seq_num,chartdate,icd_code,icd_version",
    "hosp/provider": "provider_id",
    "hosp/services": "subject_id,hadm_id,transfertime,prev_service,curr_service",
    "hosp/transfers": "subject_id,hadm_id,transfer_id,eventtype,careunit,intime,outtime",
    "icu/caregiver": "caregiver_id",
    "icu/chartevents": "subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valuenum,valueuom,warning",
    "icu/datetimeevents": "subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valueuom,warning",
    "icu/d_items": "itemid,label,abbreviation,linksto,category,unitname,param_type,lownormalvalue,highnormalvalue",
    "icu/icustays": "subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los",
    "icu/ingredientevents": "subject_id,hadm_id,stay_id,caregiver_id,starttime,endtime,storetime,itemid,amount,amountuom,rate,rateuom,orderid,linkorderid,statusdescription,originalamount,originalrate",
    "icu/inputevents": "subject_id,hadm_id,stay_id,caregiver_id,starttime,endtime,storetime,itemid,amount,amountuom,rate,rateuom,orderid,linkorderid,ordercategoryname,secondaryordercategoryname,ordercomponenttypedescription,ordercategorydescription,patientweight,totalamount,totalamountuom,isopenbag,continueinnextdept,statusdescription,originalamount,originalrate",
    "icu/outputevents": "subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valueuom",
    "icu/procedureevents": "subject_id,hadm_id,stay_id,caregiver_id,starttime,endtime,storetime,itemid,value,valueuom,location,locationcategory,orderid,linkorderid,ordercategoryname,ordercategorydescription,patientweight,isopenbag,continueinnextdept,statusdescription,originalamount,originalrate",
    "note/discharge": "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text",
    "note/discharge_detail": "note_id,subject_id,field_name,field_value,field_ordinal",
    "note/radiology": "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,text",
    "note/radiology_detail": "note_id,subject_id,field_name,field_value,field_ordinal",
}
COLUMNS = {table: header.split(",") for table, header in HEADERS.items()}


class Kind(Enum):
    """Role of a fixture row in the leak tests."""

    PLAIN = "plain"
    SENTINEL = "sentinel"  # available after AT: must never be returned at AT
    BOUNDARY = "boundary"  # available exactly at AT: must be returned at AT


@dataclass
class FixtureData:
    """Rows per table plus the source rows each Kind marks, keyed by MIMIC table name."""

    rows: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    marked: dict[Kind, set[tuple[str, int]]] = field(default_factory=lambda: defaultdict(set))

    def add(self, table: str, kind: Kind = Kind.PLAIN, **values) -> int:
        unknown = set(values) - set(COLUMNS[table])
        assert not unknown, f"{table}: unknown columns {unknown}"

        index = len(self.rows[table])
        self.rows[table].append(values)
        self.marked[kind].add((table.split("/")[1], index))
        return index


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def write_source(source: Path, data: FixtureData) -> None:
    """Write every table as csv.gz plus a matching SHA256SUMS.txt."""
    sums = []
    for table, columns in COLUMNS.items():
        path = source / f"{table}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in data.rows.get(table, []):
                writer.writerow([_cell(row.get(c)) for c in columns])
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {table}.csv.gz")
    (source / build_store.CHECKSUM_FILE).write_text("\n".join(sums) + "\n")


# ── fixture content ──────────────────────────────────────────────────────────


def _dictionaries(d: FixtureData) -> None:
    for itemid, label in [(HEART_RATE_ITEM, "Heart Rate"), (NOREPI_ITEM, "Norepinephrine"),
                          (URINE_ITEM, "Foley"), (VENT_ITEM, "Invasive Ventilation"), (224277, "Last dose"),
                          (PEEP_ITEM, "PEEP set")]:
        d.add("icu/d_items", itemid=itemid, label=label)
    d.add("hosp/d_labitems", itemid=LACTATE_ITEM, label="Lactate")
    d.add("hosp/d_icd_diagnoses", icd_code="0389", icd_version=9, long_title="Septicemia")
    d.add("hosp/d_icd_procedures", icd_code="0040", icd_version=9, long_title="Procedure on single vessel")
    d.add("hosp/d_hcpcs", code="99291", short_description="Critical care")
    d.add("hosp/provider", provider_id="P1")
    d.add("icu/caregiver", caregiver_id=1)


def _encounters(d: FixtureData) -> None:
    d.add("hosp/patients", subject_id=SUBJECT, gender="F", anchor_age=60, anchor_year=2178,
          anchor_year_group="2014 - 2016", dod=pd.Timestamp("2180-01-20"))
    d.add("hosp/admissions", subject_id=SUBJECT, hadm_id=PRIOR_HADM,
          admittime=pd.Timestamp("2179-06-01 08:00"), dischtime=pd.Timestamp("2179-06-10 10:00"),
          admission_type="URGENT", admit_provider_id="P1", discharge_location="HOME", insurance="Medicare",
          language="ENGLISH", marital_status="MARRIED", race="WHITE", hospital_expire_flag=0)
    d.add("hosp/admissions", subject_id=SUBJECT, hadm_id=CURRENT_HADM,
          admittime=pd.Timestamp("2180-01-08 08:00"), dischtime=pd.Timestamp("2180-01-20 10:00"),
          deathtime=pd.Timestamp("2180-01-20 10:00"), admission_type="EW EMER.", admit_provider_id="P1",
          admission_location="EMERGENCY ROOM", discharge_location="DIED", insurance="Medicare",
          language="ENGLISH", marital_status="WIDOWED", race="WHITE",
          edregtime=pd.Timestamp("2180-01-08 06:00"), edouttime=pd.Timestamp("2180-01-08 09:00"),
          hospital_expire_flag=1)
    # Sentinel admission/stay last one minute so later `at` values stay unambiguous.
    d.add("hosp/admissions", Kind.SENTINEL, subject_id=SUBJECT, hadm_id=SENTINEL_HADM,
          admittime=SENTINEL_TIME, dischtime=SENTINEL_TIME + MINUTE, admission_type="ELECTIVE")

    d.add("icu/icustays", subject_id=SUBJECT, hadm_id=PRIOR_HADM, stay_id=900, first_careunit="MICU",
          last_careunit="MICU", intime=pd.Timestamp("2179-06-02"), outtime=pd.Timestamp("2179-06-05"), los=3.0)
    d.add("icu/icustays", subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=CURRENT_STAY, first_careunit="MICU",
          last_careunit="SICU", intime=pd.Timestamp("2180-01-09"), outtime=pd.Timestamp("2180-01-15"), los=6.0)
    d.add("icu/icustays", Kind.SENTINEL, subject_id=SUBJECT, hadm_id=SENTINEL_HADM, stay_id=1001,
          first_careunit="CVICU", intime=SENTINEL_TIME, outtime=SENTINEL_TIME + MINUTE, los=0.0)

    d.add("hosp/transfers", Kind.BOUNDARY, subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfer_id=1,
          eventtype="transfer", careunit="SICU", intime=AT, outtime=AT + 2 * HOUR)
    d.add("hosp/transfers", Kind.SENTINEL, subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfer_id=2,
          eventtype="transfer", careunit="CCU", intime=SENTINEL_TIME, outtime=SENTINEL_TIME + DAY)
    d.add("hosp/services", Kind.BOUNDARY, subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfertime=AT,
          curr_service="MED")
    d.add("hosp/services", Kind.SENTINEL, subject_id=SUBJECT, hadm_id=CURRENT_HADM,
          transfertime=SENTINEL_TIME, curr_service="SURG")

    # Subject 2: admitted, never in ICU. Subject 3: two admissions contain AT.
    d.add("hosp/patients", subject_id=SUBJECT_NO_ICU, gender="M", anchor_age=40, anchor_year=2180)
    d.add("hosp/admissions", subject_id=SUBJECT_NO_ICU, hadm_id=400, admittime=AT - DAY,
          dischtime=AT + DAY, admission_type="URGENT")
    d.add("hosp/patients", subject_id=SUBJECT_AMBIGUOUS, gender="M", anchor_age=50, anchor_year=2180)
    for hadm_id in (500, 501):
        d.add("hosp/admissions", subject_id=SUBJECT_AMBIGUOUS, hadm_id=hadm_id, admittime=AT - DAY,
              dischtime=AT + DAY, admission_type="URGENT")
    d.add("icu/icustays", subject_id=SUBJECT_AMBIGUOUS, hadm_id=500, stay_id=5000, first_careunit="MICU",
          intime=AT - HOUR, outtime=AT + HOUR, los=0.1)


def _charted(d: FixtureData) -> None:
    """chartevents, datetimeevents, outputevents: max(charttime, storetime)."""
    stay = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=CURRENT_STAY, caregiver_id=1)
    tables = {
        "icu/chartevents": dict(itemid=HEART_RATE_ITEM, value="90", valuenum=90.0, valueuom="bpm"),
        "icu/datetimeevents": dict(itemid=224277, value="2180-01-09 08:00:00", valueuom="Date"),
        "icu/outputevents": dict(itemid=URINE_ITEM, value=50.0, valueuom="mL"),
    }
    for table, payload in tables.items():
        d.add(table, charttime=AT - HOUR, storetime=AT - HOUR, **stay, **payload)
        d.add(table, Kind.BOUNDARY, charttime=AT - 10 * MINUTE, storetime=AT, **stay, **payload)
        d.add(table, Kind.SENTINEL, charttime=AT - 10 * MINUTE, storetime=SENTINEL_TIME, **stay, **payload)
        # storetime before charttime (10% of chartevents): charttime governs.
        d.add(table, Kind.SENTINEL, charttime=SENTINEL_TIME, storetime=AT - HOUR, **stay, **payload)


def _labs(d: FixtureData) -> None:
    lab = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, itemid=LACTATE_ITEM, value="2.0", valuenum=2.0,
               valueuom="mmol/L", ref_range_lower="0.5", ref_range_upper="2", flag="abnormal", comments="x")
    d.add("hosp/labevents", Kind.BOUNDARY, labevent_id=1, specimen_id=1, charttime=AT - HOUR, storetime=AT, **lab)
    d.add("hosp/labevents", Kind.SENTINEL, labevent_id=2, specimen_id=2, charttime=SENTINEL_TIME,
          storetime=SENTINEL_TIME + HOUR, **lab)
    d.add("hosp/labevents", labevent_id=3, specimen_id=3, charttime=PENDING_LAB_DRAWN,
          storetime=PENDING_LAB_STORED, **(lab | dict(value=str(PENDING_LAB_VALUE), valuenum=PENDING_LAB_VALUE)))
    # Prior admission lab -> past.
    d.add("hosp/labevents", labevent_id=4, specimen_id=4, charttime=pd.Timestamp("2179-06-03 06:00"),
          storetime=pd.Timestamp("2179-06-03 07:00"), **(lab | dict(hadm_id=PRIOR_HADM)))

    micro = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, spec_itemid=70012, spec_type_desc="BLOOD CULTURE",
                 test_itemid=90201, test_name="Blood Culture, Routine", org_itemid=80002, org_name="E. COLI",
                 interpretation="S", ab_name="CEFTRIAXONE", dilution_text="<=1", comments="final")
    d.add("hosp/microbiologyevents", Kind.BOUNDARY, microevent_id=1, micro_specimen_id=10,
          chartdate=pd.Timestamp("2180-01-08"), charttime=pd.Timestamp("2180-01-08 10:00"),
          storedate=AT.normalize(), storetime=AT, **micro)
    # Two isolates of one pending test -> one pending row.
    for microevent_id in (2, 3):
        d.add("hosp/microbiologyevents", microevent_id=microevent_id, micro_specimen_id=11,
              chartdate=AT.normalize(), charttime=AT - HOUR, storedate=(AT + DAY).normalize(),
              storetime=AT + DAY, **micro)
    d.add("hosp/microbiologyevents", Kind.SENTINEL, microevent_id=4, micro_specimen_id=12,
          chartdate=SENTINEL_TIME.normalize(), charttime=SENTINEL_TIME, **micro)
    # Date-only collection: EOD(2180-01-10) = AT + 1 min.
    d.add("hosp/microbiologyevents", Kind.SENTINEL, microevent_id=5, micro_specimen_id=13,
          chartdate=AT.normalize(), **micro)


def _orders(d: FixtureData) -> None:
    order = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, order_type="Medications", order_provider_id="P1",
                 order_status="Inactive")
    d.add("hosp/poe", poe_id="1-1", poe_seq=1, ordertime=AT - DAY, **order)
    d.add("hosp/poe", Kind.BOUNDARY, poe_id="1-2", poe_seq=2, ordertime=AT, discontinued_by_poe_id="1-3", **order)
    d.add("hosp/poe", Kind.SENTINEL, poe_id="1-3", poe_seq=3, ordertime=SENTINEL_TIME,
          discontinue_of_poe_id="1-2", **order)
    for poe_id, poe_seq, kind in (("1-1", 1, Kind.PLAIN), ("1-2", 2, Kind.BOUNDARY), ("1-3", 3, Kind.SENTINEL)):
        d.add("hosp/poe_detail", kind, poe_id=poe_id, poe_seq=poe_seq, subject_id=SUBJECT,
              field_name="Admit to", field_value="ICU")

    rx = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, order_provider_id="P1", drug="Norepinephrine",
              drug_type="MAIN")
    d.add("hosp/prescriptions", Kind.BOUNDARY, pharmacy_id=1, poe_id="1-2", poe_seq=2, starttime=AT - HOUR,
          stoptime=AT + 2 * DAY, **rx)
    # Started before AT but ordered (poe) after AT.
    d.add("hosp/prescriptions", Kind.SENTINEL, pharmacy_id=2, poe_id="1-3", poe_seq=3, starttime=AT - HOUR,
          stoptime=AT + DAY, **rx)
    d.add("hosp/prescriptions", Kind.SENTINEL, pharmacy_id=3, starttime=SENTINEL_TIME, **rx)

    pharm = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, medication="Norepinephrine", status="Discontinued")
    d.add("hosp/pharmacy", Kind.BOUNDARY, pharmacy_id=1, poe_id="1-2", entertime=AT, starttime=AT,
          stoptime=AT + DAY, verifiedtime=AT + MINUTE, expirationdate=AT + 2 * DAY, **pharm)
    d.add("hosp/pharmacy", Kind.SENTINEL, pharmacy_id=2, entertime=SENTINEL_TIME, starttime=AT - HOUR, **pharm)

    emar = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, enter_provider_id="P1", medication="Heparin",
                event_txt="Administered")
    d.add("hosp/emar", Kind.BOUNDARY, emar_id="1-1", emar_seq=1, charttime=AT - 5 * MINUTE, storetime=AT, **emar)
    d.add("hosp/emar", Kind.SENTINEL, emar_id="1-2", emar_seq=2, charttime=AT - 5 * MINUTE,
          storetime=SENTINEL_TIME, **emar)
    for emar_id, kind in (("1-1", Kind.BOUNDARY), ("1-2", Kind.SENTINEL)):
        d.add("hosp/emar_detail", kind, subject_id=SUBJECT, emar_id=emar_id, emar_seq=1, dose_given="5000",
              product_description="Heparin 5000 unit")


def _intervals(d: FixtureData) -> None:
    stay = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=CURRENT_STAY, caregiver_id=1)
    for table in ("icu/inputevents", "icu/ingredientevents"):
        d.add(table, starttime=INFUSION_START, endtime=INFUSION_END, storetime=INFUSION_START, itemid=NOREPI_ITEM,
              amount=INFUSION_AMOUNT, amountuom="mL", statusdescription="FinishedRunning", **stay)
        d.add(table, Kind.BOUNDARY, starttime=AT, endtime=AT + HOUR, storetime=AT, itemid=NOREPI_ITEM,
              amount=60.0, amountuom="mL", statusdescription="FinishedRunning", **stay)
        d.add(table, Kind.SENTINEL, starttime=SENTINEL_TIME, endtime=SENTINEL_TIME + HOUR, storetime=SENTINEL_TIME,
              itemid=NOREPI_ITEM, amount=60.0, amountuom="mL", statusdescription="FinishedRunning", **stay)

    vent = dict(itemid=VENT_ITEM, valueuom="min", statusdescription="FinishedRunning", **stay)
    d.add("icu/procedureevents", Kind.BOUNDARY, starttime=AT, endtime=AT + 2 * HOUR, storetime=AT, value=120.0, **vent)
    d.add("icu/procedureevents", Kind.SENTINEL, starttime=SENTINEL_TIME, endtime=SENTINEL_TIME + HOUR,
          storetime=SENTINEL_TIME, value=60.0, **vent)


def _notes(d: FixtureData) -> None:
    note = dict(subject_id=SUBJECT, note_seq=1, text="report")
    d.add("note/radiology", Kind.BOUNDARY, note_id="R1", hadm_id=CURRENT_HADM, note_type="RR",
          charttime=AT - HOUR, storetime=AT, **note)
    d.add("note/radiology", Kind.SENTINEL, note_id="R2", hadm_id=CURRENT_HADM, note_type="RR",
          charttime=AT - HOUR, storetime=SENTINEL_TIME, **note)
    for note_id, kind in (("R1", Kind.BOUNDARY), ("R2", Kind.SENTINEL)):
        d.add("note/radiology_detail", kind, note_id=note_id, subject_id=SUBJECT, field_name="exam_name",
              field_value="CHEST (PORTABLE AP)", field_ordinal=1)

    d.add("note/discharge", note_id="D1", hadm_id=PRIOR_HADM, note_type="DS",
          charttime=pd.Timestamp("2179-06-10 10:00"), storetime=pd.Timestamp("2179-06-11 10:00"), **note)
    d.add("note/discharge", Kind.SENTINEL, note_id="D2", hadm_id=PRIOR_HADM, note_type="DS",
          charttime=pd.Timestamp("2179-06-10 10:00"), storetime=SENTINEL_TIME, **note)
    # Criterion 9: current admission note planted as stored before AT; still banned.
    d.add("note/discharge", Kind.SENTINEL, note_id="D3", hadm_id=CURRENT_HADM, note_type="DS",
          charttime=AT - DAY, storetime=AT - HOUR, **note)
    for note_id, kind in (("D1", Kind.PLAIN), ("D2", Kind.SENTINEL), ("D3", Kind.SENTINEL)):
        d.add("note/discharge_detail", kind, note_id=note_id, subject_id=SUBJECT, field_name="author",
              field_value="X", field_ordinal=1)


def _outpatient_and_coded(d: FixtureData) -> None:
    d.add("hosp/omr", subject_id=SUBJECT, chartdate=AT.normalize() - DAY, seq_num=1, result_name="Weight (Lbs)",
          result_value="150")
    d.add("hosp/omr", Kind.SENTINEL, subject_id=SUBJECT, chartdate=AT.normalize(), seq_num=1,
          result_name="Weight (Lbs)", result_value="151")

    # Prior admission codes visible; current and sentinel admissions' codes not yet coded.
    for hadm_id, kind in ((PRIOR_HADM, Kind.PLAIN), (CURRENT_HADM, Kind.SENTINEL), (SENTINEL_HADM, Kind.SENTINEL)):
        coded = dict(subject_id=SUBJECT, hadm_id=hadm_id)
        d.add("hosp/diagnoses_icd", kind, seq_num=1, icd_code="0389", icd_version=9, **coded)
        d.add("hosp/drgcodes", kind, drg_type="HCFA", drg_code="871", description="SEPTICEMIA", **coded)
        d.add("hosp/procedures_icd", kind, seq_num=1, chartdate=pd.Timestamp("2179-06-03"), icd_code="0040",
              icd_version=9, **coded)
        d.add("hosp/hcpcsevents", kind, chartdate=pd.Timestamp("2179-06-03"), hcpcs_cd="99291", seq_num=1,
              short_description="Critical care", **coded)


def _decisions(d: FixtureData) -> None:
    """PEEP 5 before window, 5 in window (CONTINUE), 8 later (CHANGE); drip START, CHANGE, STOP."""
    stay = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=CURRENT_STAY, caregiver_id=1)
    peep = dict(itemid=PEEP_ITEM, valueuom="cmH2O", **stay)
    for offset, value in ((-HOUR, 5.0), (HOUR, 5.0), (2 * HOUR, 8.0)):
        time = WINDOW_START + offset
        d.add("icu/chartevents", charttime=time, storetime=time, value=str(value), valuenum=value, **peep)

    drip = dict(itemid=NOREPI_ITEM, amountuom="mg", rateuom="mcg/kg/min", ordercategoryname="01-Drips",
                ordercomponenttypedescription="Main order parameter", linkorderid=7, **stay)
    d.add("icu/inputevents", orderid=7, starttime=WINDOW_START + HOUR, endtime=WINDOW_START + 2 * HOUR,
          storetime=WINDOW_START + HOUR, amount=1.0, rate=0.05, statusdescription="ChangeDose/Rate", **drip)
    d.add("icu/inputevents", orderid=8, starttime=WINDOW_START + 2 * HOUR, endtime=WINDOW_START + 3 * HOUR,
          storetime=WINDOW_START + 2 * HOUR, amount=2.0, rate=0.10, statusdescription="Stopped", **drip)


def fixture_data() -> FixtureData:
    data = FixtureData()
    for builder in (_dictionaries, _encounters, _charted, _labs, _orders, _intervals, _notes, _outpatient_and_coded,
                    _decisions):
        builder(data)
    return data


@dataclass(frozen=True)
class FixtureStore:
    path: Path
    data: FixtureData


@pytest.fixture(scope="session")
def fixture_store(tmp_path_factory) -> FixtureStore:
    root = tmp_path_factory.mktemp("mimic")
    data = fixture_data()
    write_source(root / "source", data)
    build_store.build(root / "source", root / "store", [])
    return FixtureStore(root / "store", data)
