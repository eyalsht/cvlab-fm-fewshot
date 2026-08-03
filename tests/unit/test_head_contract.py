"""The contract battery every registered head must pass.

The battery is the structural claim of ADR-002 turned into a test: whatever a
head does internally, from the outside it is fit(train, val) then
predict(query) -> logits [M, C] with column j scoring class j, it refuses
predict before fit, and it is deterministic on fixed input. Each concrete head's
test file imports run_head_battery and drives it with a scenario the head is
meant to solve, so "battery green" is checkable per head. A DummyHead here
proves the battery and the registry mechanics independently of any real head.
"""

import inspect

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
from fm_fewshot.shared.contracts import ExperimentConfig


def make_config(head: str, **head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="head-battery",
        dataset="synthetic",
        encoder="stub",
        head=head,
        head_params=dict(head_params),
        k=2,
        subset_seed=0,
        init_seed=0,
        seed=0,
    )


def run_head_battery(
    cfg: ExperimentConfig,
    n_classes: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    query_x: torch.Tensor,
    expected_labels: torch.Tensor,
    *,
    context: HeadContext | None = None,
) -> torch.Tensor:
    """Assert the full head contract; return the logits for extra per-head checks."""
    unfitted = make_head(cfg, n_classes, context)
    with pytest.raises(NotFittedError):
        unfitted.predict(query_x)

    head = make_head(cfg, n_classes, context)
    head.fit(train_x, train_y, val_x, val_y)
    logits = head.predict(query_x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (query_x.shape[0], n_classes)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert torch.equal(logits.argmax(dim=1), expected_labels)

    twin = make_head(cfg, n_classes, context)
    twin.fit(train_x, train_y, val_x, val_y)
    assert torch.equal(twin.predict(query_x), logits)

    return logits


@register("dummy_battery")
class DummyHead(FewShotHead):
    """Nearest class-mean by dot product, the smallest head that fits the ABC."""

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self._means: torch.Tensor | None = None

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext
    ) -> "DummyHead":
        return cls(n_classes=n_classes)

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
    ) -> None:
        dim = train_x.shape[1]
        means = torch.zeros(self._n_classes, dim)
        for j in range(self._n_classes):
            means[j] = train_x[train_y == j].mean(dim=0)
        self._means = means

    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        if self._means is None:
            raise NotFittedError("predict called before fit")
        return (query_x @ self._means.T).float()


def separable_scenario(n_classes: int = 3, dim: int = 4):
    """One orthogonal cluster per class; queries sit on their class centroid."""
    centers = torch.eye(n_classes, dim)
    train_x = centers.repeat_interleave(2, dim=0)
    train_y = torch.arange(n_classes).repeat_interleave(2)
    val_x = centers.clone()
    val_y = torch.arange(n_classes)
    query_x = centers.clone()
    expected = torch.arange(n_classes)
    return train_x, train_y, val_x, val_y, query_x, expected


class TestBattery:
    def test_dummy_head_passes_the_battery(self) -> None:
        cfg = make_config("dummy_battery")
        run_head_battery(cfg, 3, *separable_scenario())


class TestTestSplitIsolation:
    def test_fit_signature_exposes_no_test_tensors(self) -> None:
        """A head must have no route to the test split (PRD_evaluation_protocol)."""
        params = list(inspect.signature(FewShotHead.fit).parameters)
        assert params == ["self", "train_x", "train_y", "val_x", "val_y"]


class TestRegistry:
    def test_make_head_unknown_key_raises_listing_registered(self) -> None:
        cfg = make_config("no_such_head")
        with pytest.raises(ValueError, match="unknown head"):
            make_head(cfg, 3)

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @register("dummy_battery")
            class _Clash(FewShotHead):
                def fit(self, train_x, train_y, val_x, val_y) -> None: ...

                def predict(self, query_x): ...

    def test_from_context_has_no_default_implementation(self) -> None:
        class Bare(FewShotHead):
            def fit(self, train_x, train_y, val_x, val_y) -> None: ...

            def predict(self, query_x): ...

        with pytest.raises(NotImplementedError):
            Bare.from_context(make_config("x"), 2, HeadContext())


class TestDeterminismOnFixedInput:
    def test_repeated_fit_predict_is_bit_identical(self) -> None:
        cfg = make_config("dummy_battery")
        rng = np.random.default_rng(0)
        train_x = torch.from_numpy(rng.standard_normal((6, 4)).astype(np.float32))
        train_y = torch.arange(3).repeat_interleave(2)
        val_x = torch.from_numpy(rng.standard_normal((3, 4)).astype(np.float32))
        val_y = torch.arange(3)
        query_x = torch.from_numpy(rng.standard_normal((5, 4)).astype(np.float32))

        first = make_head(cfg, 3)
        first.fit(train_x, train_y, val_x, val_y)
        second = make_head(cfg, 3)
        second.fit(train_x, train_y, val_x, val_y)
        assert torch.equal(first.predict(query_x), second.predict(query_x))
