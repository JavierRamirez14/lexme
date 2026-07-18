"""Structured output owned by the interface, not the provider adapters.

Deriving a JSON schema from a Pydantic model, instructing the model to honour it,
and validating the reply all live here. Adapters only flip a provider's JSON mode
on; this keeps the Pydantic coupling in one place and portable across providers.
"""

import json
import re

from pydantic import BaseModel, ValidationError

from lexme.llm.protocol import ModelT, StructuredOutputError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def json_schema_instruction(response_model: type[BaseModel]) -> str:
    """Return a prompt instruction pinning the reply to ``response_model``'s schema."""
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    return (
        "Respond with a single JSON object and nothing else. No prose, no code "
        f"fence. It must validate against this JSON Schema:\n{schema}"
    )


def parse_model(raw: str, response_model: type[ModelT]) -> ModelT:
    """Parse ``raw`` model output into ``response_model``.

    A surrounding ```` ```json ```` fence is tolerated so a reply that ignores the
    "no fence" instruction still parses. Raises :class:`StructuredOutputError` if
    the text is not JSON or does not satisfy the schema.
    """
    candidate = _FENCE.sub("", raw).strip()
    try:
        return response_model.model_validate_json(candidate)
    except ValidationError as error:
        raise StructuredOutputError(
            f"reply did not match {response_model.__name__}: {error}"
        ) from error
