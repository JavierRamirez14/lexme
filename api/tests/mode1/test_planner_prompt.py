"""The planning prompt's standing rules, pinned where a fake LLM cannot reach them.

The pipeline tests queue a plan into the fake LLM, so they exercise everything the
planner does with a plan and nothing it asks the model for. These tests read the
composed system prompt instead, which is the only place the decomposition rules
actually live.
"""

from datetime import date

from lexme.mode1.branches import AnswerKind, CriticalBranch
from lexme.mode1.graph.planner import _system_prompt

TODAY = date(2026, 8, 5)


def test_the_prompt_asks_for_the_cause_alongside_the_remedy() -> None:
    """A question about stopping a consequence must plan for the norm behind it too.

    ``mh-02`` measured the cost of leaving this implicit: asked how to stop an
    eviction, the planner wrote two sub-queries about enervacion and none about the
    non-payment that let the landlord resolve, so art. 27 LAU was only ever a
    dense-only rank-7 hit and never survived fusion. The answer explained the
    remedy correctly and never said why the eviction was owed.
    """
    prompt = _system_prompt((), TODAY).lower()

    assert "causa" in prompt
    assert "remedio" in prompt


def test_the_prompt_carries_the_run_clock() -> None:
    assert TODAY.isoformat() in _system_prompt((), TODAY)


def test_the_prompt_lists_the_declared_branches_and_omits_the_section_without_them() -> None:
    branch = CriticalBranch(
        id="uso_vivienda",
        question="¿Es tu vivienda habitual?",
        answer_kind=AnswerKind.TEXT,
        sets_target_date=False,
        fact_template="El uso es {answer}.",
        assumption="Se asume vivienda habitual.",
    )

    with_branches = _system_prompt((branch,), TODAY)
    without_branches = _system_prompt((), TODAY)

    assert "uso_vivienda" in with_branches
    assert "unresolved_branch_ids" in with_branches
    assert "unresolved_branch_ids" not in without_branches
