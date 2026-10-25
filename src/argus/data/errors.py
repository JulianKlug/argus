"""Snapshot errors."""


class SnapshotError(Exception):
    """Base for every snapshot failure."""


class SubjectNotFoundError(SnapshotError):
    """`subject_id` absent from the store."""


class NotInIcuStayError(SnapshotError):
    """`at` falls outside every ICU stay of the subject."""


class AmbiguousEncounterError(SnapshotError):
    """More than one admission contains `at`."""
