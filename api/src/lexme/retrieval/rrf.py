"""Reciprocal Rank Fusion: combine several ranked lists into one, score-free.

RRF fuses rankings by position alone, never by the raw scores of the underlying
retrievers, so a cosine distance and a lexical ``ts_rank`` -- which live on
different, incomparable scales -- can be merged without normalization. Each list
contributes ``1 / (k + position)`` to every key it ranks; the constant ``k``
damps the weight of top positions so a single retriever cannot dominate.
"""

from collections.abc import Hashable, Sequence
from typing import TypeVar

RRF_K = 60

KeyT = TypeVar("KeyT", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[KeyT]], *, k: int = RRF_K
) -> list[tuple[KeyT, float]]:
    """Fuse ``rankings`` into one list of ``(key, score)`` ordered by score desc.

    ``rankings`` is a list of ranked lists, each ordered best-first. A key absent
    from a list simply contributes nothing from that list. Ties are broken by the
    order in which a key is first seen, so the fusion is deterministic. ``k`` is
    the RRF damping constant.
    """
    scores: dict[KeyT, float] = {}
    first_seen: dict[KeyT, int] = {}
    order = 0
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            if key not in first_seen:
                first_seen[key] = order
                order += 1
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
