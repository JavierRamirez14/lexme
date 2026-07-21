"""The stage names each node stamps on the state as it runs.

These are a contract, not decoration: the SSE stream emits the current step after
every node and the UI decodes it to drive the live timeline. Naming them once here
keeps the nodes, the state default and the stream reading from a single source.
"""

from enum import StrEnum


class Step(StrEnum):
    """The human-readable stage a node reports via ``Mode1State.current_step``."""

    START = "inicio"
    ROUTING = "enrutando"
    PLANNING = "planificando"
    CLARIFYING = "preguntando"
    SITUATING = "situando"
    RETRIEVING = "recuperando"
    ITERATING = "iterando"
    SYNTHESIZING = "sintetizando"
    DECIDING = "decidiendo"
    CHECKING_CURRENCY = "comprobando_vigencia"
