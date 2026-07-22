"""The vertical's legal-default checklist: data, loader and corpus validator.

The 20-item LAU default (a tenant's protections by law) ships as data in the
vertical package. This package loads it vertical-agnostically and validates that
every anchor resolves and every citation still verifies against the ingested
corpus, so the checklist can never drift silently from the law it cites.
"""

from lexme.checklist.models import (
    CHECKLIST_FILENAME,
    Checklist,
    ChecklistCitation,
    ChecklistError,
    ChecklistItem,
    RuleCharacter,
    SilenceTone,
    checklist_path,
    load_checklist,
)
from lexme.checklist.validator import (
    ChecklistFinding,
    ChecklistReport,
    validate_checklist,
)

__all__ = [
    "CHECKLIST_FILENAME",
    "Checklist",
    "ChecklistCitation",
    "ChecklistError",
    "ChecklistFinding",
    "ChecklistItem",
    "ChecklistReport",
    "RuleCharacter",
    "SilenceTone",
    "checklist_path",
    "load_checklist",
    "validate_checklist",
]
