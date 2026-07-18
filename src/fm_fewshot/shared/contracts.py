"""Data contracts per PLAN section 5.

All contracts are frozen: an episode or a result is a fact, never mutated
after creation. Tensor fields compare by identity under dataclass eq, which
is acceptable because equality checks in tests and the report path only ever
compare the tensor-free contracts (ExperimentConfig, EpisodeResult).
"""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class Episode:
    episode_id: int
    dataset: str  # "mnist" | "cifar10" | "mini_imagenet"
    n_way: int
    k_shot: int
    m_query: int
    class_ids: tuple[int, ...]  # global ids, len == n_way, sorted
    support_idx: Tensor  # int64 [n_way * k_shot], rows into the feature cache
    query_idx: Tensor  # int64 [n_way * m_query], disjoint from support_idx


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    dataset: str
    encoder: str  # "clip_vit_b32"
    head: str  # registry key
    head_params: dict[str, object] = field(default_factory=dict)
    n_way: int = 5
    k_shot: int = 1
    m_query: int = 15
    n_episodes: int = 600
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    l2_normalize: bool = True  # ADR-004
    split: str = "test"


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    accuracy: float
    n_correct: int
    n_query: int
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    config: ExperimentConfig
    git_commit: str
    accuracy_mean: float
    ci95: float  # half-width, 1.96 * SE over episodes
    episode_results: list[EpisodeResult]
    wall_seconds: float
