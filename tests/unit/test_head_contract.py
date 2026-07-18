"""The contract battery every registered head must pass.

The battery is the structural claim of ADR-002 turned into a test: whatever a
head does internally, from the outside it is fit(support) then predict(query)
-> logits [M, n_way] with column j scoring local label j, it refuses predict
before fit, and it is deterministic on fixed input. Each concrete head's test
file imports run_head_battery and drives it with a scenario the head is meant
to solve, so "battery green" is checkable per head. A DummyHead here proves the
battery and the registry mechanics before any real head exists.
"""

import numpy as np
import pytest
import torch

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    make_head,
    register,
)
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


def make_episode(n_way: int, k_shot: int = 2, m_query: int = 3, episode_id: int = 0) -> Episode:
    class_ids = tuple(range(n_way))
    support_idx = torch.arange(n_way * k_shot, dtype=torch.int64)
    query_idx = torch.arange(n_way * k_shot, n_way * (k_shot + m_query), dtype=torch.int64)
    return Episode(
        episode_id=episode_id,
        dataset="synthetic",
        n_way=n_way,
        k_shot=k_shot,
        m_query=m_query,
        class_ids=class_ids,
        support_idx=support_idx,
        query_idx=query_idx,
    )


def make_config(head: str, **head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="head-battery",
        dataset="synthetic",
        encoder="stub",
        head=head,
        head_params=dict(head_params),
        n_way=3,
        k_shot=2,
        m_query=3,
        n_episodes=1,
        seed=0,
    )


def run_head_battery(
    cfg: ExperimentConfig,
    episode: Episode,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    expected_labels: torch.Tensor,
    *,
    context: HeadContext | None = None,
) -> torch.Tensor:
    """Assert the full head contract; return the logits for extra per-head checks."""
    n_way = episode.n_way

    unfitted = make_head(cfg, episode, context)
    with pytest.raises(NotFittedError):
        unfitted.predict(query_x)

    head = make_head(cfg, episode, context)
    head.fit(support_x, support_y)
    logits = head.predict(query_x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (query_x.shape[0], n_way)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.equal(logits.argmax(dim=1), expected_labels)

    twin = make_head(cfg, episode, context)
    twin.fit(support_x, support_y)
    assert torch.equal(twin.predict(query_x), logits)

    return logits


@register("dummy_battery")
class DummyHead(FewShotHead):
    """Nearest class-mean by dot product, the smallest head that fits the ABC."""

    def __init__(self, n_way: int) -> None:
        self._n_way = n_way
        self._means: torch.Tensor | None = None

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "DummyHead":
        return cls(n_way=episode.n_way)

    def fit(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None:
        dim = support_x.shape[1]
        means = torch.zeros(self._n_way, dim)
        for j in range(self._n_way):
            means[j] = support_x[support_y == j].mean(dim=0)
        self._means = means

    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        if self._means is None:
            raise NotFittedError("predict called before fit")
        return (query_x @ self._means.T).float()


def separable_scenario(n_way: int = 3, dim: int = 4):
    """One orthogonal cluster per class; queries sit on their class centroid."""
    centers = torch.eye(n_way, dim)
    support_x = centers.repeat_interleave(2, dim=0)
    support_y = torch.arange(n_way).repeat_interleave(2)
    query_x = centers.clone()
    expected = torch.arange(n_way)
    return support_x, support_y, query_x, expected


class TestBattery:
    def test_dummy_head_passes_the_battery(self) -> None:
        cfg = make_config("dummy_battery")
        episode = make_episode(n_way=3)
        support_x, support_y, query_x, expected = separable_scenario()
        run_head_battery(cfg, episode, support_x, support_y, query_x, expected)


class TestRegistry:
    def test_make_head_unknown_key_raises_listing_registered(self) -> None:
        cfg = make_config("no_such_head")
        with pytest.raises(ValueError, match="unknown head"):
            make_head(cfg, make_episode(n_way=3))

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @register("dummy_battery")
            class _Clash(FewShotHead):
                def fit(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None: ...

                def predict(self, query_x: torch.Tensor) -> torch.Tensor: ...

    def test_from_context_has_no_default_implementation(self) -> None:
        class Bare(FewShotHead):
            def fit(self, support_x: torch.Tensor, support_y: torch.Tensor) -> None: ...

            def predict(self, query_x: torch.Tensor) -> torch.Tensor: ...

        with pytest.raises(NotImplementedError):
            Bare.from_context(make_config("x"), make_episode(2), HeadContext())


class TestDeterminismOnFixedInput:
    def test_repeated_fit_predict_is_bit_identical(self) -> None:
        cfg = make_config("dummy_battery")
        episode = make_episode(n_way=3)
        rng = np.random.default_rng(0)
        support_x = torch.from_numpy(rng.standard_normal((6, 4)).astype(np.float32))
        support_y = torch.arange(3).repeat_interleave(2)
        query_x = torch.from_numpy(rng.standard_normal((5, 4)).astype(np.float32))

        first = make_head(cfg, episode)
        first.fit(support_x, support_y)
        second = make_head(cfg, episode)
        second.fit(support_x, support_y)
        assert torch.equal(first.predict(query_x), second.predict(query_x))
