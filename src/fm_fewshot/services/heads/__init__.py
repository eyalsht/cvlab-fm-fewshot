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
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.services.heads.prototype import PrototypeHead
from fm_fewshot.services.heads.zeroshot_clip import ZeroShotClipHead

__all__ = [
    "FewShotHead",
    "HeadContext",
    "LinearProbeHead",
    "NotFittedError",
    "PrototypeHead",
    "ZeroShotClipHead",
    "make_head",
    "register",
    "registered_heads",
]
