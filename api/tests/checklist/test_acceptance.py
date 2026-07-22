"""Acceptance: the shipped LAU checklist validates green, and any drift fails it.

Every one of the 20 items' citations must verify directly against the curated LAU
excerpts, and a single synthetic corruption -- a citation whose text no longer
matches the law -- must turn the validator red. This is the guarantee that the
checklist cannot desync silently from the corpus.
"""

from dataclasses import replace

from lexme.checklist import load_checklist, validate_checklist
from tests.checklist.conftest import CHECKLIST_PATH, TARGET_DATE, build_corpus
from tests.checklist.fixtures.lau_excerpts import LAU_EXCERPTS, LAU_NORM_ID


def test_shipped_checklist_validates_against_the_corpus() -> None:
    checklist = load_checklist(CHECKLIST_PATH)
    corpus = build_corpus(LAU_NORM_ID, LAU_EXCERPTS)

    report = validate_checklist(checklist, corpus, TARGET_DATE)

    assert report.is_valid, [f"{f.item_id}/{f.block_id}: {f.message}" for f in report.findings]


def test_a_corrupt_item_fails_validation() -> None:
    checklist = load_checklist(CHECKLIST_PATH)
    corpus = build_corpus(LAU_NORM_ID, LAU_EXCERPTS)

    first = checklist.items[0]
    corrupt_citation = replace(first.citation, text="una cita inventada que la ley no contiene")
    corrupt_items = (replace(first, citation=corrupt_citation), *checklist.items[1:])
    corrupt_checklist = replace(checklist, items=corrupt_items)

    report = validate_checklist(corrupt_checklist, corpus, TARGET_DATE)

    assert not report.is_valid
    assert any(finding.item_id == first.id for finding in report.findings)
