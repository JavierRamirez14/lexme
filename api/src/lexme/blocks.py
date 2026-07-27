"""The corpus-wide identity of a block: which norm it belongs to and which block it is.

A block id is only unique inside one norm -- the LAU, the Código Civil and the Ley
de Enjuiciamiento Civil each have an ``a1`` -- while the corpus holds several norms
at once. Wherever a block has to travel as a single string (the citable token in a
prompt, a gold block in an eval case, the subject of an in-force notice) it travels
as ``<norm_id>:<block_id>``, so a citation can never be resolved against the wrong
law.
"""

from dataclasses import dataclass

BLOCK_REF_SEPARATOR = ":"


@dataclass(frozen=True)
class BlockRef:
    """One block's identity across the whole corpus, norm included."""

    norm_id: str
    block_id: str

    def __str__(self) -> str:
        """Render the reference as the single ``norm:block`` token."""
        return f"{self.norm_id}{BLOCK_REF_SEPARATOR}{self.block_id}"

    @classmethod
    def parse(cls, raw: str) -> "BlockRef | None":
        """Read a ``norm:block`` token, or ``None`` when it is not one.

        Splits on the first separator, since norm ids never contain one and block
        ids may. An unqualified block id returns ``None`` rather than guessing a
        norm: callers treat that as an unusable reference, which is what it is.
        """
        norm_id, separator, block_id = raw.strip().partition(BLOCK_REF_SEPARATOR)
        if not separator or not norm_id or not block_id:
            return None
        return cls(norm_id=norm_id, block_id=block_id)
