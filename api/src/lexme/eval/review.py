"""The calibration sample: the judge's own rulings, laid out for a reviewer to confirm.

Calibration needs the rulings themselves, not the run's averages, so the sample is
drawn from the verdicts a versioned run recorded -- the reviewer grades the same
answers the published numbers were computed from, never a re-judging of different
ones. The draw is a seeded random sample of the whole population of rulings, and
both the seed and the population travel with the sample, so the reviewed set can be
redrawn and audited.

The review itself is by exception: the reviewer reads every ruling with the article
in front of them and reports only the numbers they disagree with, which is what
makes a fifty-ruling pass affordable. Numbers identify a ruling within the run's
whole population, so a sample stays traceable to the run it came from.
"""

import json
import random
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from lexme.eval.artifact import RunArtifact
from lexme.eval.calibration import (
    KIND_CLAIM,
    KIND_KEY_POINT,
    CalibrationItem,
    JudgeCalibration,
    build_calibration,
)
from lexme.eval.cases import EvalCase, KeyPoint
from lexme.eval.judge import JUDGE_TASK, JudgeVerdict
from lexme.eval.metrics import CaseResult

_NO_DISAGREEMENT = "none"


class ReviewError(ValueError):
    """Raised when a calibration sample cannot be drawn or read back."""


class ReviewRuling(BaseModel):
    """One judge ruling put in front of the reviewer, numbered within its run.

    ``ref`` identifies the ruling the way the calibration record does -- the key
    point's block reference, or the claim's own text. ``reference`` is the written
    key point the coverage was judged against, empty for a claim ruling, and
    ``quoted`` is the passage under review: the judge's quote from the answer, or
    the claim itself. ``article_ref`` is the article the ruling hangs on and
    ``article_text`` its in-force text, empty when it could not be resolved.
    """

    number: int
    case_id: str
    kind: str
    ref: str
    judge_label: bool
    question: str
    reference: str = ""
    quoted: str = ""
    article_ref: str | None = None
    article_text: str = ""


class ReviewSample(BaseModel):
    """A drawn sample of rulings awaiting the reviewer's labels, with its provenance.

    ``population`` is how many rulings the run held in total and ``seed`` the draw
    that produced this subset, so the sample is reproducible; ``size`` is what the
    published agreement is computed over.
    """

    run: str
    judge_model: str
    seed: int
    population: int
    size: int
    rulings: list[ReviewRuling]

    def write(self, path: Path) -> None:
        """Serialize the sample to ``path`` as indented JSON, creating parents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def collect_rulings(artifact: RunArtifact, cases: Sequence[EvalCase]) -> list[ReviewRuling]:
    """Every ruling the run's judge made, numbered from one in case order.

    Each judged case contributes its key-point coverages and then its claim
    assessments; a case the judge did not rule on contributes nothing.
    """
    key_points = _key_point_claims(cases)
    rulings: list[ReviewRuling] = []
    for case in artifact.cases:
        if case.judge_verdict is None:
            continue
        rulings.extend(_case_rulings(case, case.judge_verdict, key_points, len(rulings)))
    return rulings


def build_sample(
    artifact: RunArtifact,
    cases: Sequence[EvalCase],
    size: int,
    seed: int,
    run: str = "",
) -> ReviewSample:
    """Draw ``size`` rulings from ``artifact`` at random under ``seed``.

    A ``size`` at or above the population takes every ruling. The sample reads in
    run order whatever the draw, so the reviewer walks the cases as they ran.
    Raises :class:`ReviewError` when the run judged nothing or pins no judge model.
    """
    rulings = collect_rulings(artifact, cases)
    if not rulings:
        raise ReviewError(f"run '{run or artifact.suite}' has no judge rulings to calibrate on")
    drawn = random.Random(seed).sample(rulings, min(size, len(rulings)))
    drawn.sort(key=lambda ruling: ruling.number)
    return ReviewSample(
        run=run,
        judge_model=_judge_model(artifact),
        seed=seed,
        population=len(rulings),
        size=len(drawn),
        rulings=drawn,
    )


def with_article_texts(sample: ReviewSample, texts: Mapping[tuple[str, str], str]) -> ReviewSample:
    """A copy of ``sample`` with each ruling's article text filled in from ``texts``.

    ``texts`` is keyed by ``(case_id, article_ref)`` because the same article can
    resolve to different redactions across cases pinned at different dates.
    """
    rulings = [
        ruling.model_copy(
            update={"article_text": texts.get((ruling.case_id, ruling.article_ref or ""), "")}
        )
        for ruling in sample.rulings
    ]
    return sample.model_copy(update={"rulings": rulings})


def read_sample(path: Path) -> ReviewSample:
    """Read a drawn sample back from its JSON file at ``path``.

    Raises :class:`ReviewError` when the file is missing or not a valid sample.
    """
    if not path.is_file():
        raise ReviewError(f"calibration sample not found: {path}")
    try:
        return ReviewSample.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError) as error:
        raise ReviewError(f"calibration sample {path} is malformed: {error}") from error


def parse_disagreements(raw: str) -> tuple[int, ...]:
    """Read a comma-separated list of ruling numbers the reviewer disagreed with.

    An empty list or ``none`` means the reviewer confirmed every ruling. Raises
    :class:`ReviewError` on anything that is not a number.
    """
    stripped = raw.strip()
    if not stripped or stripped.lower() == _NO_DISAGREEMENT:
        return ()
    numbers = []
    for token in stripped.split(","):
        candidate = token.strip()
        if not candidate.isdigit():
            raise ReviewError(f"disagreement '{candidate}' is not a ruling number")
        numbers.append(int(candidate))
    return tuple(numbers)


def calibration_from_review(
    sample: ReviewSample,
    disagreed: Collection[int],
    reviewed_by: str,
    reviewer_kind: str,
    reviewed_at: datetime,
) -> JudgeCalibration:
    """Turn a reviewed sample into the calibration record its agreement is derived from.

    Every ruling in the sample counts as confirmed unless its number is in
    ``disagreed``, in which case the reviewer's label is the opposite of the judge's.
    ``reviewer_kind`` declares what the resulting agreement is evidence of. Raises
    :class:`ReviewError` when a number is not in the sample, so a typo cannot
    silently inflate the agreement.
    """
    numbers = {ruling.number for ruling in sample.rulings}
    unknown = sorted(set(disagreed) - numbers)
    if unknown:
        raise ReviewError(f"disagreed rulings not in the sample: {unknown}")
    items = [
        CalibrationItem(
            case_id=ruling.case_id,
            kind=ruling.kind,
            ref=ruling.ref,
            judge_label=ruling.judge_label,
            reviewer_label=ruling.judge_label != (ruling.number in disagreed),
        )
        for ruling in sample.rulings
    ]
    return build_calibration(sample.judge_model, reviewed_by, reviewer_kind, reviewed_at, items)


def render_sheet(sample: ReviewSample) -> str:
    """Render the sample as the Markdown sheet the reviewer reads and rules on."""
    lines = [
        f"# Judge calibration sample — {sample.run or 'unnamed run'}",
        "",
        f"Judge model: `{sample.judge_model}` · temperature 0.",
        f"Sample: {sample.size} of {sample.population} rulings, drawn with seed {sample.seed}.",
        "",
        "Read every ruling against the article quoted under it and note only the numbers "
        "you disagree with; everything you do not note counts as confirmed.",
        "",
    ]
    for ruling in sample.rulings:
        lines.extend(_render_ruling(ruling))
    return "\n".join(lines)


def _render_ruling(ruling: ReviewRuling) -> list[str]:
    """The Markdown block for one ruling: what was judged, how, and against what."""
    lines = [
        "---",
        "",
        f"## #{ruling.number} · case `{ruling.case_id}` · {ruling.kind.replace('_', ' ')}",
        "",
        f"**Question:** {ruling.question}",
        "",
    ]
    if ruling.kind == KIND_KEY_POINT:
        lines += [
            f"**Reference key point** (`{ruling.ref}`): {ruling.reference}",
            "",
            f"**Judge:** {'COVERED' if ruling.judge_label else 'NOT COVERED'} by the answer",
            "",
            f"**Quoted from the answer:** {ruling.quoted or '(nothing quoted)'}",
            "",
        ]
    else:
        backing = f" by `{ruling.article_ref}`" if ruling.article_ref else ""
        lines += [
            f"**Claim the answer makes:** {ruling.ref}",
            "",
            f"**Judge:** {'SUPPORTED' + backing if ruling.judge_label else 'UNSUPPORTED'}",
            "",
        ]
    lines += _render_article(ruling)
    return lines


def _render_article(ruling: ReviewRuling) -> list[str]:
    """The article the ruling hangs on, quoted, or a note that it is not available."""
    if not ruling.article_ref:
        return ["**Article:** none cited.", ""]
    if not ruling.article_text:
        return [f"**Article `{ruling.article_ref}`:** text not resolved.", ""]
    quoted = "\n".join(f"> {line}" for line in ruling.article_text.splitlines() if line.strip())
    return [f"**Article `{ruling.article_ref}`:**", "", quoted, ""]


def _case_rulings(
    case: CaseResult,
    verdict: JudgeVerdict,
    key_points: Mapping[str, Sequence[KeyPoint]],
    offset: int,
) -> list[ReviewRuling]:
    """One case's rulings, numbered on from ``offset``: coverages then claims."""
    written = _aligned_key_points(key_points.get(case.id, ()), verdict)
    rulings = [
        ReviewRuling(
            number=offset + index + 1,
            case_id=case.id,
            kind=KIND_KEY_POINT,
            ref=_key_point_ref(point.block_ref, index, written),
            judge_label=point.covered,
            question=case.question,
            reference=written[index].claim if written else "",
            quoted=point.evidence,
            article_ref=point.block_ref,
        )
        for index, point in enumerate(verdict.key_points)
    ]
    rulings += [
        ReviewRuling(
            number=offset + len(verdict.key_points) + index + 1,
            case_id=case.id,
            kind=KIND_CLAIM,
            ref=claim.claim,
            judge_label=claim.supported,
            question=case.question,
            quoted=claim.claim,
            article_ref=claim.supporting_block_ref,
        )
        for index, claim in enumerate(verdict.claims)
    ]
    return rulings


def _key_point_claims(cases: Sequence[EvalCase]) -> dict[str, Sequence[KeyPoint]]:
    """The written key points of each case, in the order the judge was shown them."""
    return {case.id: case.key_points for case in cases}


def _aligned_key_points(written: Sequence[KeyPoint], verdict: JudgeVerdict) -> Sequence[KeyPoint]:
    """The case's key points when the verdict rules on them one for one, else nothing.

    Several key points of a case routinely share one gold block, so a block
    reference does not identify which of them a ruling is about; their order does,
    since the judge is shown them in order and answers in the same one. A verdict
    that rules on a different number of points cannot be aligned that way, and the
    rulings then carry no reference text rather than a wrong one.
    """
    return written if len(written) == len(verdict.key_points) else ()


def _key_point_ref(block_ref: str, index: int, written: Sequence[KeyPoint]) -> str:
    """The identity of a key-point ruling: its block, plus its position when needed.

    Two rulings of the same case on the same block would otherwise be the same item
    in the calibration record, and one reviewed label would silently stand for both.
    """
    if not written:
        return block_ref
    return f"{block_ref}#{index + 1}"


def _judge_model(artifact: RunArtifact) -> str:
    """The judge model the run pinned, from its configuration fingerprint."""
    pinned = artifact.fingerprint.models.get(JUDGE_TASK)
    if pinned is None:
        raise ReviewError(f"run '{artifact.suite}' pins no '{JUDGE_TASK}' model to calibrate")
    return pinned.model
