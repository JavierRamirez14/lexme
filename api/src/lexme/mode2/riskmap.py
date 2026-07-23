"""Assemble the risk map: map, classify, cross-check, count -- decided in code.

This is the orchestration the pipeline calls once the gates have passed. Phase A
maps every clause in one batch; code reconciles that map against the real clause
ids so a stray or missing entry can never silence a clause. Each evaluable clause
is then classified (Phase B) and each informative one placed without a model call.
The code cross-check adds the absence whites, and a final fold counts the findings by
level and by coverage. What it never does is fold those counts into a single
verdict: the unit of judgement stays the finding.
"""

from collections.abc import Sequence
from datetime import date

from lexme.checklist import Checklist, ChecklistItem
from lexme.llm import LlmClient
from lexme.mode2.classify import classify_clause
from lexme.mode2.crosscheck import cross_check
from lexme.mode2.mapping import ClauseMap, map_clauses
from lexme.mode2.models import Clause
from lexme.mode2.retrieval import ClauseRetriever
from lexme.mode2.risk import (
    AbsenceFinding,
    ClauseFinding,
    CoverageStatus,
    RiskLevel,
    RiskMap,
)
from lexme.verification import CorpusReader


def build_risk_map(
    clauses: Sequence[Clause],
    document_text: str,
    *,
    llm: LlmClient,
    corpus: CorpusReader,
    retriever: ClauseRetriever,
    checklist: Checklist,
    vertical: str,
    target_date: date,
) -> RiskMap:
    """Build the full risk map for a set of anchored clauses.

    Runs the batch mapping, classifies each evaluable clause and places each
    informative one, cross-checks the checklist for omitted rights, and counts the
    findings. Returns a :class:`RiskMap` whose clause findings follow document
    order, so the coverage view has an entry for every clause and no silent holes.
    """
    mapping = map_clauses(llm, document_text, clauses, checklist)
    items_by_id = {item.id: item for item in checklist.items}
    maps_by_id = _reconcile(mapping.clauses, clauses, items_by_id)
    clauses_by_id = {clause.id: clause for clause in clauses}

    clause_findings = [
        _finding_for(
            clause,
            maps_by_id[clause.id],
            clauses_by_id=clauses_by_id,
            items_by_id=items_by_id,
            llm=llm,
            corpus=corpus,
            retriever=retriever,
            checklist=checklist,
            vertical=vertical,
            target_date=target_date,
        )
        for clause in clauses
    ]
    absence_findings = cross_check(checklist, clause_findings, corpus, target_date)
    return RiskMap(
        clause_findings=clause_findings,
        absence_findings=absence_findings,
        level_counts=_level_counts(clause_findings, absence_findings),
        coverage_counts=_coverage_counts(clause_findings),
    )


def _finding_for(
    clause: Clause,
    clause_map: ClauseMap,
    *,
    clauses_by_id: dict[str, Clause],
    items_by_id: dict[str, ChecklistItem],
    llm: LlmClient,
    corpus: CorpusReader,
    retriever: ClauseRetriever,
    checklist: Checklist,
    vertical: str,
    target_date: date,
) -> ClauseFinding:
    """Place one clause: an informative one without a call, an evaluable one judged."""
    if not clause_map.evaluable:
        return _informative_finding(clause, clause_map)
    related = [clauses_by_id[rid] for rid in clause_map.related_clause_ids if rid in clauses_by_id]
    return classify_clause(
        clause,
        clause_map,
        related=related,
        llm=llm,
        corpus=corpus,
        retriever=retriever,
        items_by_id=items_by_id,
        norm_id=checklist.norm_id,
        vertical=vertical,
        target_date=target_date,
    )


def _informative_finding(clause: Clause, clause_map: ClauseMap) -> ClauseFinding:
    """Place a clause with no evaluable legal content at the informative status."""
    return ClauseFinding(
        clause_id=clause.id,
        heading=clause.heading,
        snippet=clause.text,
        start=clause.start,
        end=clause.end,
        coverage=CoverageStatus.INFORMATIVA,
        chk_ids=list(clause_map.chk_ids),
        related_clause_ids=list(clause_map.related_clause_ids),
    )


def _reconcile(
    proposed: Sequence[ClauseMap],
    clauses: Sequence[Clause],
    items_by_id: dict[str, ChecklistItem],
) -> dict[str, ClauseMap]:
    """Reconcile the model's mapping against the real clauses, dropping the invalid.

    Keeps the first mapping per real clause id, filters checklist ids down to ones
    that exist and self- or unknown clause references out of the related list, and
    supplies an evaluable default for any clause the model failed to map -- so a
    clause is never silenced by a missing or malformed mapping entry.
    """
    valid_ids = {clause.id for clause in clauses}
    reconciled: dict[str, ClauseMap] = {}
    for entry in proposed:
        if entry.clause_id not in valid_ids or entry.clause_id in reconciled:
            continue
        reconciled[entry.clause_id] = ClauseMap(
            clause_id=entry.clause_id,
            evaluable=entry.evaluable,
            chk_ids=[chk_id for chk_id in entry.chk_ids if chk_id in items_by_id],
            related_clause_ids=[
                rid
                for rid in entry.related_clause_ids
                if rid in valid_ids and rid != entry.clause_id
            ],
        )
    for clause in clauses:
        reconciled.setdefault(clause.id, ClauseMap(clause_id=clause.id, evaluable=True))
    return reconciled


def _level_counts(
    clause_findings: Sequence[ClauseFinding], absence_findings: Sequence[AbsenceFinding]
) -> dict[str, int]:
    """Count findings per risk level, with the absences contributing the white count."""
    counts = {level.value: 0 for level in RiskLevel}
    for finding in clause_findings:
        if finding.level is not None:
            counts[finding.level.value] += 1
    counts[RiskLevel.AUSENTE.value] = len(absence_findings)
    return counts


def _coverage_counts(clause_findings: Sequence[ClauseFinding]) -> dict[str, int]:
    """Count clauses per coverage status, so the header recount has no gaps."""
    counts = {status.value: 0 for status in CoverageStatus}
    for finding in clause_findings:
        counts[finding.coverage.value] += 1
    return counts
