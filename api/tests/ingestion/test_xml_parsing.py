"""Parsing tests against a real (trimmed) BOE XML fixture."""

from datetime import UTC, date, datetime

import pytest

from lexme.ingestion.xml_parsing import BoeXmlError, parse_norm_xml


def test_parses_norm_metadata(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml)

    assert norm.metadata.norm_id == "BOE-A-1994-26003"
    assert norm.metadata.eli == "https://www.boe.es/eli/es/l/1994/11/24/29"
    assert norm.metadata.title.startswith("Ley 29/1994")
    assert norm.metadata.updated_at == datetime(2026, 4, 30, 7, 33, 59, tzinfo=UTC)


def test_keeps_only_precepto_blocks(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml)

    assert [block.block_id for block in norm.blocks] == ["a1", "a2", "a9"]


def test_preserves_every_version_of_a_block(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    article_nine = next(block for block in norm.blocks if block.block_id == "a9")

    effective_dates = [version.effective_date for version in article_nine.versions]
    assert effective_dates == [
        date(1995, 1, 1),
        date(2009, 12, 24),
        date(2013, 6, 6),
        date(2018, 12, 19),
        date(2019, 1, 24),
        date(2019, 3, 6),
    ]


def test_version_records_the_amending_norm(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    article_nine = next(block for block in norm.blocks if block.block_id == "a9")

    third_version = article_nine.versions[2]
    assert third_version.amending_norm_id == "BOE-A-2013-5941"
    assert third_version.publication_date == date(2013, 6, 5)


def test_text_content_excludes_amendment_footnotes(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    article_nine = next(block for block in norm.blocks if block.block_id == "a9")

    amended_version = article_nine.versions[2]
    assert "Plazo mínimo" in amended_version.text_content
    assert "Se modifica" not in amended_version.text_content
    assert "Se modifica" in amended_version.html_content


def test_missing_metadatos_raises() -> None:
    xml = "<response><data><texto></texto></data></response>"
    with pytest.raises(BoeXmlError):
        parse_norm_xml(xml)


def test_norm_without_precepto_blocks_raises() -> None:
    xml = (
        "<response><data>"
        "<metadatos>"
        "<identificador>X</identificador>"
        "<url_eli>e</url_eli><titulo>t</titulo>"
        "<url_html_consolidada>u</url_html_consolidada>"
        "<fecha_actualizacion>20240101T000000Z</fecha_actualizacion>"
        "</metadatos>"
        '<texto><bloque id="pr" tipo="preambulo" titulo="P"><version '
        'id_norma="X" fecha_publicacion="20240101" fecha_vigencia="20240101">'
        "<p>x</p></version></bloque></texto>"
        "</data></response>"
    )
    with pytest.raises(BoeXmlError):
        parse_norm_xml(xml)


def test_only_the_named_blocks_are_parsed(lau_sample_xml: str) -> None:
    norm = parse_norm_xml(lau_sample_xml, ["a1", "a9"])

    assert [block.block_id for block in norm.blocks] == ["a1", "a9"]


def test_a_malformed_block_outside_the_selection_does_not_fail_the_parse(
    lau_sample_xml: str,
) -> None:
    broken = lau_sample_xml.replace('fecha_vigencia="19950101"', 'fecha_vigencia=""', 1)

    norm = parse_norm_xml(broken, ["a9"])

    assert [block.block_id for block in norm.blocks] == ["a9"]


def test_a_selection_matching_no_block_raises(lau_sample_xml: str) -> None:
    with pytest.raises(BoeXmlError):
        parse_norm_xml(lau_sample_xml, ["a404"])


def test_two_redactions_in_force_on_the_same_date_collapse_to_the_last(
    lau_sample_xml: str,
) -> None:
    duplicated = lau_sample_xml.replace('fecha_vigencia="20130606"', 'fecha_vigencia="20091224"', 1)

    (block,) = [b for b in parse_norm_xml(duplicated).blocks if b.block_id == "a9"]

    by_date = {version.effective_date: version for version in block.versions}
    assert len(by_date) == len(block.versions)
    assert "tres años" in by_date[date(2009, 12, 24)].text_content
