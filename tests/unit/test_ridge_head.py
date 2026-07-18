"""Ridge head tests (PRD FR13, closed-form baseline).

No dedicated section-5 spec exists for ridge; the formula comes from the linear
probe PRD, W = (X^T X + lam I)^-1 X^T Y on one-hot targets. Tests assert the
closed-form value against the normal equations, determinism, the contract
battery, and the low-shot non-collapse property that motivates the head: ridge
holds where the unregularized linear probe overfits.
"""

import numpy as np
import pytest
import torch
from test_head_contract import run_head_battery, separable_scenario

from fm_fewshot.services.heads.base import NotFittedError, make_head
from fm_fewshot.services.heads.ridge import RidgeHead
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


def make_cfg(n_way: int, head: str = "ridge", **head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="ridge-test",
        dataset="synthetic",
        encoder="stub",
        head=head,
        head_params=dict(head_params),
        n_way=n_way,
        k_shot=1,
        m_query=1,
        n_episodes=1,
        seed=0,
    )


def fit_head(cfg: ExperimentConfig, episode: Episode, support_x, support_y) -> RidgeHead:
    head = make_head(cfg, episode)
    head.fit(support_x, support_y)
    return head


class TestBatteryContract:
    def test_ridge_passes_the_battery(self) -> None:
        support_x, support_y, query_x, expected = separable_scenario(n_way=3, dim=4)
        run_head_battery(make_cfg(3, lam=0.1), make_episode(3, k_shot=2), support_x, support_y,
                         query_x, expected)


class TestClosedForm:
    def test_weight_matches_the_normal_equations(self) -> None:
        rng = np.random.default_rng(0)
        support_x = torch.from_numpy(rng.standard_normal((10, 6)).astype(np.float32))
        support_y = torch.arange(5).repeat_interleave(2)
        lam = 0.5
        head = fit_head(make_cfg(5, lam=lam), make_episode(5, k_shot=2), support_x, support_y)

        one_hot = torch.zeros(10, 5)
        one_hot[torch.arange(10), support_y] = 1.0
        gram = support_x.T @ support_x + lam * torch.eye(6)
        expected = torch.linalg.solve(gram, support_x.T @ one_hot)
        assert torch.allclose(head.weight, expected, atol=1e-5)

    def test_two_fits_are_bit_identical(self) -> None:
        rng = np.random.default_rng(1)
        support_x = torch.from_numpy(rng.standard_normal((10, 6)).astype(np.float32))
        support_y = torch.arange(5).repeat_interleave(2)
        episode = make_episode(5, k_shot=2)
        first = fit_head(make_cfg(5), episode, support_x, support_y)
        second = fit_head(make_cfg(5), episode, support_x, support_y)
        assert torch.equal(first.weight, second.weight)


class TestLowShotNonCollapse:
    def test_ridge_holds_where_unregularized_probe_overfits(self) -> None:
        ridge_acc, probe_acc = [], []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            centers = rng.standard_normal((5, 64)).astype(np.float32) * 2.0
            sx, sy, qx, qy = [], [], [], []
            for j in range(5):
                sx.append(centers[j] + rng.standard_normal(64).astype(np.float32))
                sy.append(j)
                for _ in range(15):
                    qx.append(centers[j] + rng.standard_normal(64).astype(np.float32))
                    qy.append(j)
            sx_t = torch.from_numpy(np.stack(sx))
            sy_t = torch.tensor(sy)
            qx_t = torch.from_numpy(np.stack(qx))
            qy_t = torch.tensor(qy)
            episode = make_episode(5)
            ridge = fit_head(make_cfg(5, lam=1.0), episode, sx_t, sy_t).predict(qx_t)
            probe = make_head(make_cfg(5, head="linear_probe", weight_decay=0.0), episode)
            probe.fit(sx_t, sy_t)
            ridge_acc.append((ridge.argmax(dim=1) == qy_t).float().mean().item())
            probe_acc.append((probe.predict(qx_t).argmax(dim=1) == qy_t).float().mean().item())
        # Comparison row for the phase note: ridge mean accuracy is not below the
        # unregularized probe at 1-shot, the non-collapse property from HW2.
        assert np.mean(ridge_acc) >= np.mean(probe_acc)


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

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lam"):
            make_head(make_cfg(3, lam=-1.0), make_episode(3))
