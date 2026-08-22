"""FewShotHead implementations and their registry.

Importing this package registers every baseline head, so make_head resolves any
registry key without the caller importing each module by hand.
"""

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    make_head,
    register,
    registered_heads,
)
from fm_fewshot.services.heads.fm_standard import FmStandardHead
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.services.heads.prototype import PrototypeHead

__all__ = [
    "FewShotHead",
    "FmStandardHead",
    "HeadContext",
    "LinearProbeHead",
    "NotFittedError",
    "PrototypeHead",
    "make_head",
    "register",
    "registered_heads",
]
