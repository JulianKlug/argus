"""Vocabulary shared by all layers: source tables, clinical domains, common columns."""

from enum import Enum


class Domain(Enum):
    """Dataset-agnostic grouping of clinical records (spec D10)."""

    ENCOUNTERS = "encounters"
    CHARTED_OBSERVATIONS = "charted_observations"
    LABS = "labs"
    MICROBIOLOGY = "microbiology"
    ORDERS = "orders"
    MEDICATION_ORDERS = "medication_orders"
    MEDICATION_ADMINISTRATIONS = "medication_administrations"
    INPUTS = "inputs"
    INPUT_INGREDIENTS = "input_ingredients"
    OUTPUTS = "outputs"
    PROCEDURES = "procedures"
    DIAGNOSES = "diagnoses"
    NOTES = "notes"
    OUTPATIENT_MEASUREMENTS = "outpatient_measurements"


class Table(Enum):
    """MIMIC-IV source table; value = store path without suffix."""

    ADMISSIONS = "hosp/admissions"
    DIAGNOSES_ICD = "hosp/diagnoses_icd"
    DRGCODES = "hosp/drgcodes"
    EMAR = "hosp/emar"
    EMAR_DETAIL = "hosp/emar_detail"
    HCPCSEVENTS = "hosp/hcpcsevents"
    LABEVENTS = "hosp/labevents"
    MICROBIOLOGYEVENTS = "hosp/microbiologyevents"
    OMR = "hosp/omr"
    PATIENTS = "hosp/patients"
    PHARMACY = "hosp/pharmacy"
    POE = "hosp/poe"
    POE_DETAIL = "hosp/poe_detail"
    PRESCRIPTIONS = "hosp/prescriptions"
    PROCEDURES_ICD = "hosp/procedures_icd"
    SERVICES = "hosp/services"
    TRANSFERS = "hosp/transfers"
    CHARTEVENTS = "icu/chartevents"
    DATETIMEEVENTS = "icu/datetimeevents"
    ICUSTAYS = "icu/icustays"
    INGREDIENTEVENTS = "icu/ingredientevents"
    INPUTEVENTS = "icu/inputevents"
    OUTPUTEVENTS = "icu/outputevents"
    PROCEDUREEVENTS = "icu/procedureevents"
    DISCHARGE = "note/discharge"
    DISCHARGE_DETAIL = "note/discharge_detail"
    RADIOLOGY = "note/radiology"
    RADIOLOGY_DETAIL = "note/radiology_detail"

    # Dictionaries: label joins only, never exposed.
    D_ITEMS = "icu/d_items"
    D_LABITEMS = "hosp/d_labitems"
    D_ICD_DIAGNOSES = "hosp/d_icd_diagnoses"
    D_ICD_PROCEDURES = "hosp/d_icd_procedures"

    @property
    def source_name(self) -> str:
        """MIMIC table name, e.g. 'labevents'."""
        return self.value.split("/")[1]


# Every clinical table and the domain it feeds. `patients` feeds Demographics only.
TABLE_DOMAIN: dict[Table, Domain] = {
    Table.ADMISSIONS: Domain.ENCOUNTERS,
    Table.ICUSTAYS: Domain.ENCOUNTERS,
    Table.TRANSFERS: Domain.ENCOUNTERS,
    Table.SERVICES: Domain.ENCOUNTERS,
    Table.CHARTEVENTS: Domain.CHARTED_OBSERVATIONS,
    Table.DATETIMEEVENTS: Domain.CHARTED_OBSERVATIONS,
    Table.LABEVENTS: Domain.LABS,
    Table.MICROBIOLOGYEVENTS: Domain.MICROBIOLOGY,
    Table.POE: Domain.ORDERS,
    Table.POE_DETAIL: Domain.ORDERS,
    Table.PRESCRIPTIONS: Domain.MEDICATION_ORDERS,
    Table.PHARMACY: Domain.MEDICATION_ORDERS,
    Table.EMAR: Domain.MEDICATION_ADMINISTRATIONS,
    Table.EMAR_DETAIL: Domain.MEDICATION_ADMINISTRATIONS,
    Table.INPUTEVENTS: Domain.INPUTS,
    Table.INGREDIENTEVENTS: Domain.INPUT_INGREDIENTS,
    Table.OUTPUTEVENTS: Domain.OUTPUTS,
    Table.PROCEDUREEVENTS: Domain.PROCEDURES,
    Table.PROCEDURES_ICD: Domain.PROCEDURES,
    Table.HCPCSEVENTS: Domain.PROCEDURES,
    Table.DIAGNOSES_ICD: Domain.DIAGNOSES,
    Table.DRGCODES: Domain.DIAGNOSES,
    Table.RADIOLOGY: Domain.NOTES,
    Table.RADIOLOGY_DETAIL: Domain.NOTES,
    Table.DISCHARGE: Domain.NOTES,
    Table.DISCHARGE_DETAIL: Domain.NOTES,
    Table.OMR: Domain.OUTPATIENT_MEASUREMENTS,
}

# Leading columns of every domain DataFrame, in order.
SOURCE_TABLE = "source_table"
SOURCE_ROW = "source_row"
HADM_ID = "hadm_id"
STAY_ID = "stay_id"
EVENT_TIME = "event_time"
AVAILABLE_TIME = "available_time"
IS_PENDING = "is_pending"
IS_TRUNCATED = "is_truncated"
LABEL = "label"

COMMON_COLUMNS = (
    SOURCE_TABLE, SOURCE_ROW, HADM_ID, STAY_ID,
    EVENT_TIME, AVAILABLE_TIME, IS_PENDING, IS_TRUNCATED, LABEL,
)


class DecisionCategory(Enum):
    """Kind of clinician decision in a window (docs/spec_decision_window.md)."""

    MEDICATION = "medication"
    DEVICE_SETTING = "device_setting"
    TEST_ORDER = "test_order"
    PROCEDURE = "procedure"  # lines, intubation, restraints, ...
    CONSULT = "consult"
    PATIENT_MOVEMENT = "patient_movement"  # admission, transfer, discharge orders


class DecisionAction(Enum):
    ORDER = "order"
    START = "start"
    CHANGE = "change"
    CONTINUE = "continue"  # first in-window value equal to the last value before the window
    STOP = "stop"


# Columns added to decision frames after COMMON_COLUMNS.
ACTION = "action"
SUBCATEGORY = "subcategory"
ORDER_TABLE = "order_table"  # linked order, e.g. the prescription behind an infusion segment
ORDER_ROW = "order_row"

DECISION_COLUMNS = (ACTION, SUBCATEGORY, ORDER_TABLE, ORDER_ROW)
