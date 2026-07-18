"""Linear probe head tests per PRD_linear_probe_head section 5."""

import numpy as np
import pytest
import torch
from test_head_contract import run_head_battery, separable_scenario

from fm_fewshot.services.heads.base import NotFittedError, make_head
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


def make_episode(n_way: int, k_shot: int = 1, episode_id: int = 0) -> Episode:
    return Episode(
        episode_id=episode_id,
        dataset="synthetic",
        n_way=n_way,
        k_shot=k_shot,
        m_query=1,
        class_ids=tuple(range(n_way)),
        support_idx=torch.arange(n_way * k_shot, dtype=torch.int64),
        query_idx=torch.arange(n_way * k_shot, n_way * k_shot + n_way, dtype=torch.int64),
    )


def make_cfg(n_way: int, **head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="linear-probe-test",
        dataset="synthetic",
        encoder="stub",
        head="linear_probe",
        head_params=dict(head_params),
        n_way=n_way,
        k_shot=1,
        m_query=1,
        n_episodes=1,
        seed=0,
    )


def fit_head(cfg: ExperimentConfig, episode: Episode, support_x, support_y) -> LinearProbeHead:
    head = make_head(cfg, episode)
    head.fit(support_x, support_y)
    return head


def separable_3way(scale: float = 8.0):
    """Three well-separated 2D clusters, five points each, tiny seeded noise."""
    rng = np.random.default_rng(0)
    centers = torch.tensor([[scale, 0.0], [-scale, scale], [-scale, -scale]])
    rows, labels = [], []
    for j, center in enumerate(centers):
        for _ in range(5):
            rows.append(center + 0.1 * torch.from_numpy(rng.standard_normal(2).astype(np.float32)))
            labels.append(j)
    return torch.stack(rows).float(), torch.tensor(labels)


class TestBatteryContract:
    def test_linear_probe_passes_the_battery(self) -> None:
        support_x, support_y, query_x, expected = separable_scenario(n_way=3, dim=4)
        run_head_battery(make_cfg(3), make_episode(3, k_shot=2), support_x, support_y, query_x,
                         expected)


class TestSeparableToy:
    def test_reaches_full_support_accuracy_within_default_budget(self) -> None:
        support_x, support_y = separable_3way()
        head = fit_head(make_cfg(3), make_episode(3), support_x, support_y)
        predicted = head.predict(support_x).argmax(dim=1)
        assert torch.equal(predicted, support_y)


class TestDeterminism:
    def test_two_fits_give_bit_identical_parameters(self) -> None:
        support_x, support_y = separable_3way()
        episode = make_episode(3)
        first = fit_head(make_cfg(3), episode, support_x, support_y)
        second = fit_head(make_cfg(3), episode, support_x, support_y)
        assert torch.equal(first.weight, second.weight)
        assert torch.equal(first.bias, second.bias)


class TestRegularization:
    def test_weight_norm_decreases_as_weight_decay_grows(self) -> None:
        rng = np.random.default_rng(4)
        support_x = torch.from_numpy(rng.standard_normal((10, 32)).astype(np.float32))
        support_y = torch.arange(5).repeat_interleave(2)
        episode = make_episode(5, k_shot=2)
        norms = [
            fit_head(make_cfg(5, weight_decay=wd), episode, support_x, support_y).weight.norm().item()
            for wd in (0.0, 1e-2, 1.0)
        ]
        assert norms[0] >= norms[1] >= norms[2]
        assert norms[0] > norms[2]

    def test_regularized_query_accuracy_at_least_unregularized(self) -> None:
        reg_acc, unreg_acc = [], []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            centers = rng.standard_normal((5, 64)).astype(np.float32) * 2.0
            support_x, support_y, query_x, query_y = [], [], [], []
            for j in range(5):
                support_x.append(centers[j] + rng.standard_normal(64).astype(np.float32))
                support_y.append(j)
                for _ in range(15):
                    query_x.append(centers[j] + rng.standard_normal(64).astype(np.float32))
                    query_y.append(j)
            sx = torch.from_numpy(np.stack(support_x))
            sy = torch.tensor(support_y)
            qx = torch.from_numpy(np.stack(query_x))
            qy = torch.tensor(query_y)
            episode = make_episode(5)
            reg = fit_head(make_cfg(5, weight_decay=1e-2), episode, sx, sy).predict(qx)
            unreg = fit_head(make_cfg(5, weight_decay=0.0), episode, sx, sy).predict(qx)
            reg_acc.append((reg.argmax(dim=1) == qy).float().mean().item())
            unreg_acc.append((unreg.argmax(dim=1) == qy).float().mean().item())
        assert np.mean(reg_acc) >= np.mean(unreg_acc)


class TestDegenerate:
    def test_one_shot_five_way_runs_and_stays_finite(self) -> None:
        rng = np.random.default_rng(9)
        support_x = torch.from_numpy(rng.standard_normal((5, 16)).astype(np.float32))
        support_y = torch.arange(5)
        head = fit_head(make_cfg(5), make_episode(5), support_x, support_y)
        logits = head.predict(torch.from_numpy(rng.standard_normal((7, 16)).astype(np.float32)))
        assert logits.shape == (7, 5)
        assert torch.isfinite(logits).all()


class TestLossTrajectory:
    def test_loss_is_non_increasing_over_final_twenty_steps_smoothed(self) -> None:
        support_x, support_y = separable_3way()
        head = fit_head(make_cfg(3), make_episode(3), support_x, support_y)
        history = torch.tensor(head.loss_history)
        window = 5
        smoothed = history.unfold(0, window, 1).mean(dim=1)
        tail = smoothed[-20:]
        assert torch.all(tail[1:] - tail[:-1] <= 1e-4)


class TestValidation:
    def test_predict_before_fit_raises(self) -> None:
        head = make_head(make_cfg(3), make_episode(3))
        with pytest.raises(NotFittedError):
            head.predict(torch.zeros(2, 4))

    def test_missing_class_in_support_raises(self) -> None:
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])  # class 2 absent
        head = make_head(make_cfg(3), make_episode(3))
        with pytest.raises(ValueError, match="class"):
            head.fit(support_x, support_y)

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"n_steps": 0}, "n_steps"),
            ({"lr": 0.0}, "lr"),
            ({"weight_decay": -1.0}, "weight_decay"),
            ({"optimizer": "sgd"}, "optimizer"),
        ],
    )
    def test_bad_config_raises(self, params: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            make_head(make_cfg(3, **params), make_episode(3))

    def test_non_finite_loss_aborts_with_step_index(self) -> None:
        # A gigantic learning rate diverges the fit; the abort names the step.
        support_x, support_y = separable_3way(scale=1e6)
        head = make_head(make_cfg(3, lr=1e30), make_episode(3))
        with pytest.raises(ValueError, match="step"):
            head.fit(support_x, support_y)
