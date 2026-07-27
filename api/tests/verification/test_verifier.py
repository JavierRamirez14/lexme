"""Behaviour tests for the runtime citation verifier.

These drive the verifier through the in-memory corpus (the high seam): a test
programs blocks and their history, feeds proposed citations -- clean, corrupt,
mis-anchored or naming the wrong norm -- and asserts the single verdict and the
code-hydrated anchor.
"""

from datetime import date

import pytest

from lexme.blocks import BlockRef
from lexme.llm import FakeLlmClient, Message
from lexme.verification import (
    CitationVerdict,
    ProposedCitation,
    summarize,
    verify_citations,
)
from lexme.verification.snap import SNAP_SIMILARITY_THRESHOLD
from tests.verification.conftest import FakeCorpusReader

NORM_ID = "BOE-A-1994-26003"
OTHER_NORM_ID = "BOE-A-2000-323"
BEFORE_REFORM = date(2013, 6, 6)
AFTER_REFORM = date(2019, 3, 6)

A9_THREE_YEARS = "El plazo mínimo de duración del arrendamiento será de tres años."
A9_FIVE_YEARS = "El plazo mínimo de duración del arrendamiento será de cinco años."
A10_RENT = "La renta será la que libremente estipulen las partes durante la vigencia del contrato."
A11_RAISE = "El arrendador podrá elevar la renta conforme a lo pactado en el contrato."
LEC_A10 = "Serán considerados partes legítimas quienes comparezcan como titulares de la relación."

DIRECT_QUOTE = "libremente estipulen las partes"
SNAP_QUOTE = "La renta sera la que libremente estipularan las partes"
REANCHOR_QUOTE = "El arrendador podrá elevar la renta conforme a lo pactado"
INVENTED_QUOTE = "el inquilino podrá subarrendar sin límite alguno"


def ref(block_id: str, norm_id: str = NORM_ID) -> str:
    """The norm-qualified reference a citation names a block by."""
    return str(BlockRef(norm_id=norm_id, block_id=block_id))


@pytest.fixture
def lau_corpus(corpus: FakeCorpusReader) -> FakeCorpusReader:
    """A small LAU-shaped corpus: a9 with two redactions, plus a10 and a11."""
    corpus.add_block(
        NORM_ID,
        "a9",
        title="Artículo 9",
        versions=[(BEFORE_REFORM, A9_THREE_YEARS), (AFTER_REFORM, A9_FIVE_YEARS)],
    )
    corpus.add_block(NORM_ID, "a10", title="Artículo 10", versions=[(BEFORE_REFORM, A10_RENT)])
    corpus.add_block(NORM_ID, "a11", title="Artículo 11", versions=[(BEFORE_REFORM, A11_RAISE)])
    return corpus


EVIDENCE = (
    BlockRef(NORM_ID, "a9"),
    BlockRef(NORM_ID, "a10"),
    BlockRef(NORM_ID, "a11"),
)


def test_exact_citation_verifies_directly(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(block_ref=ref("a10"), text="libremente estipulen las partes")

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.VERIFIED_DIRECT
    assert result.text == "libremente estipulen las partes"
    assert result.snap_similarity is None


def test_one_changed_word_is_repaired_by_snap_to_the_real_text(
    lau_corpus: FakeCorpusReader,
) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"), text="La renta sera la que libremente estipularan las partes"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.REPAIRED_SNAP
    assert result.text == "La renta será la que libremente estipulen las partes"
    assert result.snap_similarity is not None
    assert result.snap_similarity >= SNAP_SIMILARITY_THRESHOLD


def test_invented_citation_is_discarded_and_never_shown(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"), text="el inquilino podrá subarrendar sin límite alguno"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.DISCARDED
    assert result.text == ""
    assert result.anchor is None


def test_citation_to_a_block_outside_the_evidence_is_discarded(
    lau_corpus: FakeCorpusReader,
) -> None:
    citation = ProposedCitation(block_ref=ref("a99"), text="cualquier texto")

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.DISCARDED


def test_an_unqualified_block_id_is_discarded(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(block_ref="a10", text="libremente estipulen las partes")

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.DISCARDED


def test_a_citation_naming_another_norms_block_of_the_same_id_is_discarded(
    lau_corpus: FakeCorpusReader,
) -> None:
    lau_corpus.add_block(
        OTHER_NORM_ID, "a10", title="Artículo 10", label="LEC", versions=[(BEFORE_REFORM, LEC_A10)]
    )
    citation = ProposedCitation(
        block_ref=ref("a10", OTHER_NORM_ID), text="libremente estipulen las partes"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.DISCARDED


def test_two_norms_sharing_a_block_id_each_verify_against_their_own_text(
    lau_corpus: FakeCorpusReader,
) -> None:
    lau_corpus.add_block(
        OTHER_NORM_ID, "a10", title="Artículo 10", label="LEC", versions=[(BEFORE_REFORM, LEC_A10)]
    )
    evidence = (*EVIDENCE, BlockRef(OTHER_NORM_ID, "a10"))
    citations = [
        ProposedCitation(block_ref=ref("a10"), text="libremente estipulen las partes"),
        ProposedCitation(block_ref=ref("a10", OTHER_NORM_ID), text="partes legítimas quienes"),
    ]

    lau, lec = verify_citations(citations, evidence, BEFORE_REFORM, lau_corpus)

    assert lau.verdict is CitationVerdict.VERIFIED_DIRECT
    assert lau.anchor is not None and lau.anchor.norm_label == "LAU"
    assert lec.verdict is CitationVerdict.VERIFIED_DIRECT
    assert lec.anchor is not None and lec.anchor.norm_label == "LEC"


def test_valid_ellipsis_verifies(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"), text="La renta será la que […] estipulen las partes"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.VERIFIED_DIRECT
    assert result.text == "La renta será la que […] estipulen las partes"


def test_reordered_ellipsis_segments_do_not_verify(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"), text="estipulen las partes […] La renta será la que"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is not CitationVerdict.VERIFIED_DIRECT


def test_overlapping_ellipsis_segments_do_not_verify(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"),
        text="La renta será la que libremente […] libremente estipulen las partes",
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is not CitationVerdict.VERIFIED_DIRECT


def test_too_short_ellipsis_segments_are_discarded(lau_corpus: FakeCorpusReader) -> None:
    citation = ProposedCitation(block_ref=ref("a10"), text="La […] renta […] partes")

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.DISCARDED


def test_verification_resolves_against_the_point_in_time_redaction(
    lau_corpus: FakeCorpusReader,
) -> None:
    citation = ProposedCitation(block_ref=ref("a9"), text="será de cinco años")

    before = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)[0]
    after = verify_citations([citation], EVIDENCE, AFTER_REFORM, lau_corpus)[0]

    assert before.verdict is CitationVerdict.DISCARDED
    assert after.verdict is CitationVerdict.VERIFIED_DIRECT


def test_anchor_fields_are_hydrated_by_code_from_the_corpus(
    lau_corpus: FakeCorpusReader,
) -> None:
    citation = ProposedCitation(block_ref=ref("a9"), text="será de cinco años")

    [result] = verify_citations([citation], EVIDENCE, AFTER_REFORM, lau_corpus)

    assert result.anchor is not None
    assert result.anchor.norm_id == NORM_ID
    assert result.anchor.eli == "https://www.boe.es/eli/es/l/1994/11/24/29"
    assert result.anchor.title == "Artículo 9"
    assert result.anchor.effective_date == AFTER_REFORM


def test_misanchored_citation_is_re_anchored_to_the_single_matching_block(
    lau_corpus: FakeCorpusReader,
) -> None:
    citation = ProposedCitation(
        block_ref=ref("a10"), text="El arrendador podrá elevar la renta conforme a lo pactado"
    )

    [result] = verify_citations([citation], EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert result.verdict is CitationVerdict.REPAIRED_ANCHOR
    assert result.block_ref == ref("a11")
    assert result.anchor is not None and result.anchor.block_id == "a11"


def test_ambiguous_re_anchor_across_several_blocks_is_discarded(
    corpus: FakeCorpusReader,
) -> None:
    shared = "El contrato se prorrogará por plazos anuales hasta cinco años de duración."
    corpus.add_block(NORM_ID, "a10", versions=[(BEFORE_REFORM, A10_RENT)])
    corpus.add_block(NORM_ID, "a12", versions=[(BEFORE_REFORM, shared)])
    corpus.add_block(NORM_ID, "a13", versions=[(BEFORE_REFORM, shared)])
    evidence = (
        BlockRef(NORM_ID, "a10"),
        BlockRef(NORM_ID, "a12"),
        BlockRef(NORM_ID, "a13"),
    )
    citation = ProposedCitation(block_ref=ref("a10"), text=shared)

    [result] = verify_citations([citation], evidence, BEFORE_REFORM, corpus)

    assert result.verdict is CitationVerdict.DISCARDED


def test_each_result_carries_exactly_one_verdict_covering_all_four(
    lau_corpus: FakeCorpusReader,
) -> None:
    citations = [
        ProposedCitation(block_ref=ref("a10"), text=DIRECT_QUOTE),
        ProposedCitation(block_ref=ref("a10"), text=SNAP_QUOTE),
        ProposedCitation(block_ref=ref("a10"), text=REANCHOR_QUOTE),
        ProposedCitation(block_ref=ref("a10"), text=INVENTED_QUOTE),
    ]

    results = verify_citations(citations, EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert [r.verdict for r in results] == [
        CitationVerdict.VERIFIED_DIRECT,
        CitationVerdict.REPAIRED_SNAP,
        CitationVerdict.REPAIRED_ANCHOR,
        CitationVerdict.DISCARDED,
    ]
    assert summarize(results) == {
        "verificada_directa": 1,
        "reparada_snap": 1,
        "reparada_anclaje": 1,
        "descartada": 1,
    }


def test_corrupt_citations_from_the_llm_seam_are_discarded(
    lau_corpus: FakeCorpusReader,
) -> None:
    synthesizer = FakeLlmClient(
        {
            "mode1_synthesis": [
                ProposedCitation(block_ref=ref("a10"), text="texto fabricado por el modelo"),
                ProposedCitation(block_ref=ref("a404"), text="bloque que nadie recuperó"),
            ]
        }
    )
    prompt = [Message(role="user", content="¿Cuál es la renta?")]
    emitted = [
        synthesizer.complete_structured("mode1_synthesis", prompt, ProposedCitation)
        for _ in range(2)
    ]

    results = verify_citations(emitted, EVIDENCE, BEFORE_REFORM, lau_corpus)

    assert all(r.verdict is CitationVerdict.DISCARDED for r in results)
    assert all(r.anchor is None for r in results)
