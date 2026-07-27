"""The constructed reference set: generators, the clause bank, and the review filter.

No public dataset covers Spanish tenancy law under this project's taxonomy, so the
evaluation reference set is built by construction and filtered by a human. This
package holds the Mode 1 query generator (gold blocks fixed by construction), the
Mode 2 clause bank and contract assembler (per-clause and segmentation ground truth,
deliberate omissions as expected whites), and the candidate store whose review gate
is the only door into the versioned set the harness consumes.
"""

from lexme.refset.assembler import AssemblyError, assemble_contract
from lexme.refset.clause_bank import (
    BankClause,
    ClauseBank,
    ClauseBankError,
    CoverageReport,
    absent_items,
    clause_bank_path,
    load_clause_bank,
    validate_bank_coverage,
)
from lexme.refset.models import (
    Candidate,
    CaseKind,
    ClarificationAnswer,
    ClauseGroundTruth,
    ExpectedAbsence,
    KeyPoint,
    Mode1ReferenceCase,
    Mode2ReferenceCase,
    Provenance,
    ReviewStatus,
)
from lexme.refset.query_generator import (
    QUERY_GENERATION_TASK,
    GenerationError,
    QuerySeed,
    QueryStyle,
    generate_query_case,
)
from lexme.refset.store import CandidateStore, StoreError

__all__ = [
    "QUERY_GENERATION_TASK",
    "AssemblyError",
    "BankClause",
    "Candidate",
    "CandidateStore",
    "CaseKind",
    "ClarificationAnswer",
    "ClauseBank",
    "ClauseBankError",
    "ClauseGroundTruth",
    "CoverageReport",
    "ExpectedAbsence",
    "GenerationError",
    "KeyPoint",
    "Mode1ReferenceCase",
    "Mode2ReferenceCase",
    "Provenance",
    "QuerySeed",
    "QueryStyle",
    "ReviewStatus",
    "StoreError",
    "absent_items",
    "assemble_contract",
    "clause_bank_path",
    "generate_query_case",
    "load_clause_bank",
    "validate_bank_coverage",
]
