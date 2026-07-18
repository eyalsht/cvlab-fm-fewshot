"""Device resolution and heavy-job guard per ADR-010.

Standard cached-feature experiments run anywhere. Heavy jobs fail fast on a
CPU-only machine with a message naming the GPU box, overridable by an
explicit flag so an intentional overnight CPU run stays possible.
"""

import torch

HEAVY_JOBS = frozenset(
    {"feature_extraction", "rolled_out_sweep", "stage3_training", "fine_tuning"}
)
LIGHT_JOBS = frozenset({"evaluation", "standard_training", "report"})

OVERRIDE_FLAG = "--allow-heavy-on-cpu"


class HeavyJobOnCpuError(RuntimeError):
    """Raised when a heavy job would run on a CPU-only machine without override."""


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device 'cuda' requested but no CUDA device is available")
        return torch.device("cuda")
    raise ValueError(f"unknown device {device!r}; expected 'auto', 'cpu', or 'cuda'")


def check(job: str, device: str, allow_heavy_on_cpu: bool = False) -> torch.device:
    if job not in HEAVY_JOBS | LIGHT_JOBS:
        raise ValueError(f"unknown job {job!r}; known jobs: {sorted(HEAVY_JOBS | LIGHT_JOBS)}")
    resolved = resolve_device(device)
    if job in HEAVY_JOBS and resolved.type == "cpu" and not allow_heavy_on_cpu:
        raise HeavyJobOnCpuError(
            f"job {job!r} is classified heavy and this machine resolved to CPU. "
            f"Run it on the GPU box over SSH, or pass {OVERRIDE_FLAG} to run "
            "here anyway."
        )
    return resolved
