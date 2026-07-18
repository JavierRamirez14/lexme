"""The seam between the interface and a single provider's HTTP API.

An adapter translates a provider-neutral :class:`ProviderRequest` into one HTTP
call and returns the reply text. It knows nothing about tasks or Pydantic; the
routing client resolves the task and the structured layer handles schemas.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lexme.llm.protocol import Message


@dataclass(frozen=True)
class ProviderRequest:
    """A fully-resolved request to one provider.

    ``json_mode`` asks the provider to constrain the reply to a single JSON value;
    the interface still validates it against the caller's schema.
    """

    model: str
    temperature: float
    messages: Sequence[Message]
    json_mode: bool = False


class ProviderAdapter(Protocol):
    """Anything that can complete a :class:`ProviderRequest` into reply text."""

    def complete(self, request: ProviderRequest) -> str: ...
