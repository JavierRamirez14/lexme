"""Task-oriented LLM interface.

Every component that needs a language model calls this package, never a provider
SDK. A call names a *task* (``"mode1_synthesis"``, ``"judge"``); the concrete
provider, pinned model version and temperature for that task live in
``tasks.json``, so swapping providers is a config change, not a code change.

The interface owns structured output (Pydantic schema derivation and validation);
provider adapters only speak HTTP. In tests, :class:`FakeLlmClient` is the single
substitution point for the whole system.
"""

from lexme.llm.factory import build_llm_client
from lexme.llm.fake import FakeLlmClient, RecordedCall
from lexme.llm.protocol import (
    LlmClient,
    LlmError,
    Message,
    Role,
    StructuredOutputError,
    TaskNotConfiguredError,
)
from lexme.llm.registry import (
    SUPPORTED_PROVIDERS,
    LlmConfigError,
    TaskModel,
    TaskRegistry,
    load_task_registry,
)

__all__ = [
    "SUPPORTED_PROVIDERS",
    "FakeLlmClient",
    "LlmClient",
    "LlmConfigError",
    "LlmError",
    "Message",
    "RecordedCall",
    "Role",
    "StructuredOutputError",
    "TaskModel",
    "TaskNotConfiguredError",
    "TaskRegistry",
    "build_llm_client",
    "load_task_registry",
]
