"""Prototype head tests per PRD_prototype_head section 5."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from test_head_contract import run_head_battery, separable_scenario

from fm_fewshot.services.heads.base import NotFittedError, make_head
from fm_fewshot.services.heads.prototype import PrototypeHead
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


def make_cfg(n_way: int, k_shot: int = 1, **head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="prototype-test",
        dataset="synthetic",
        encoder="stub",
        head="prototype",
        head_params=dict(head_params),
        n_way=n_way,
        k_shot=k_shot,
        m_query=1,
        n_episodes=1,
        seed=0,
    )


def fit_head(cfg: ExperimentConfig, episode: Episode, support_x, support_y) -> PrototypeHead:
    head = make_head(cfg, episode)
    head.fit(support_x, support_y)
    return head


class TestBatteryContract:
    def test_prototype_passes_the_battery(self) -> None:
        support_x, support_y, query_x, expected = separable_scenario(n_way=3, dim=4)
        cfg = make_cfg(n_way=3, k_shot=2, metric="cosine")
        run_head_battery(cfg, make_episode(3, k_shot=2), support_x, support_y, query_x, expected)


class TestHandExample:
    support_x = torch.tensor([[0.0, 1.0], [0.0, 3.0], [4.0, 0.0], [6.0, 0.0]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_x = torch.tensor([[0.0, 2.0], [5.0, 0.0]])

    def test_euclidean_logits_match_hand_computation(self) -> None:
        cfg = make_cfg(n_way=2, k_shot=2, metric="euclidean")
        head = fit_head(cfg, make_episode(2, k_shot=2), self.support_x, self.support_y)
        logits = head.predict(self.query_x)
        # protos (0,2) and (5,0); logits = -squared euclidean distance.
        expected = torch.tensor([[0.0, -29.0], [-29.0, 0.0]])
        assert torch.allclose(logits, expected)
        assert torch.equal(logits.argmax(dim=1), torch.tensor([0, 1]))

    def test_cosine_logits_match_hand_computation(self) -> None:
        cfg = make_cfg(n_way=2, k_shot=2, metric="cosine")
        head = fit_head(cfg, make_episode(2, k_shot=2), self.support_x, self.support_y)
        logits = head.predict(self.query_x)
        # renormalized protos are (0,1) and (1,0); queries lie on the axes.
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(logits, expected, atol=1e-6)
        assert torch.equal(logits.argmax(dim=1), torch.tensor([0, 1]))

    def test_temperature_scales_logits(self) -> None:
        cfg = make_cfg(n_way=2, k_shot=2, metric="euclidean", temperature=2.0)
        head = fit_head(cfg, make_episode(2, k_shot=2), self.support_x, self.support_y)
        logits = head.predict(self.query_x)
        assert torch.allclose(logits, torch.tensor([[0.0, -14.5], [-14.5, 0.0]]))


class TestPermutationInvariance:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_shuffling_support_rows_leaves_logits_identical(self, metric: str) -> None:
        rng = np.random.default_rng(3)
        support_x = torch.from_numpy(rng.standard_normal((10, 8)).astype(np.float32))
        support_y = torch.arange(5).repeat_interleave(2)
        query_x = torch.from_numpy(rng.standard_normal((7, 8)).astype(np.float32))
        cfg = make_cfg(n_way=5, k_shot=2, metric=metric)
        episode = make_episode(5, k_shot=2)

        base = fit_head(cfg, episode, support_x, support_y).predict(query_x)
        perm = torch.from_numpy(rng.permutation(10))
        shuffled = fit_head(cfg, episode, support_x[perm], support_y[perm]).predict(query_x)
        assert torch.equal(base, shuffled)


class TestNormalizedEquivalence:
    def test_euclidean_and_cosine_argmax_agree_on_unit_norm_episodes(self) -> None:
        cfg_cos = make_cfg(n_way=5, k_shot=1, metric="cosine")
        cfg_euc = make_cfg(n_way=5, k_shot=1, metric="euclidean")
        episode = make_episode(5, k_shot=1)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            support_x = F.normalize(
                torch.from_numpy(rng.standard_normal((5, 16)).astype(np.float32)), dim=1
            )
            support_y = torch.arange(5)
            query_x = F.normalize(
                torch.from_numpy(rng.standard_normal((12, 16)).astype(np.float32)), dim=1
            )
            cos = fit_head(cfg_cos, episode, support_x, support_y).predict(query_x)
            euc = fit_head(cfg_euc, episode, support_x, support_y).predict(query_x)
            assert torch.equal(cos.argmax(dim=1), euc.argmax(dim=1))


class TestKMeansPath:
    def bimodal_episode(self):
        """Two classes whose modes interleave so both class means collapse to the origin."""
        rng = np.random.default_rng(11)
        modes = {
            0: [(0.0, 5.0), (0.0, -5.0)],
            1: [(5.0, 0.0), (-5.0, 0.0)],
        }
        support_rows, support_labels, query_rows, query_labels = [], [], [], []
        for label, centers in modes.items():
            for cx, cy in centers:
                support_rows += [
                    [cx + 0.05 * rng.standard_normal(), cy + 0.05 * rng.standard_normal()]
                    for _ in range(3)
                ]
                support_labels += [label] * 3
                query_rows += [
                    [cx + 0.05 * rng.standard_normal(), cy + 0.05 * rng.standard_normal()]
                    for _ in range(5)
                ]
                query_labels += [label] * 5
        support_x = torch.tensor(support_rows, dtype=torch.float32)
        support_y = torch.tensor(support_labels, dtype=torch.int64)
        query_x = torch.tensor(query_rows, dtype=torch.float32)
        query_y = torch.tensor(query_labels, dtype=torch.int64)
        return support_x, support_y, query_x, query_y

    def test_two_clusters_beat_single_prototype_on_bimodal_classes(self) -> None:
        support_x, support_y, query_x, query_y = self.bimodal_episode()
        episode = make_episode(2, k_shot=6)

        single = fit_head(
            make_cfg(2, k_shot=6, metric="euclidean", clusters_per_class=1),
            episode, support_x, support_y,
        ).predict(query_x)
        multi = fit_head(
            make_cfg(2, k_shot=6, metric="euclidean", clusters_per_class=2),
            episode, support_x, support_y,
        ).predict(query_x)

        single_acc = (single.argmax(dim=1) == query_y).float().mean().item()
        multi_acc = (multi.argmax(dim=1) == query_y).float().mean().item()
        assert multi_acc > single_acc
        assert multi_acc > 0.95


class TestDeterminism:
    def test_kmeans_fit_twice_yields_identical_prototypes(self) -> None:
        support_x, support_y, _, _ = TestKMeansPath().bimodal_episode()
        cfg = make_cfg(2, k_shot=6, metric="euclidean", clusters_per_class=2)
        episode = make_episode(2, k_shot=6)
        first = fit_head(cfg, episode, support_x, support_y)
        second = fit_head(cfg, episode, support_x, support_y)
        assert torch.equal(first.prototypes, second.prototypes)


class TestValidation:
    def test_predict_before_fit_raises(self) -> None:
        head = make_head(make_cfg(3, metric="cosine"), make_episode(3))
        with pytest.raises(NotFittedError):
            head.predict(torch.zeros(2, 4))

    def test_missing_class_in_support_raises(self) -> None:
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])  # class 2 absent
        head = make_head(make_cfg(3), make_episode(3))
        with pytest.raises(ValueError, match="class"):
            head.fit(support_x, support_y)

    def test_more_clusters_than_shots_raises(self) -> None:
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])
        head = make_head(make_cfg(2, clusters_per_class=3), make_episode(2))
        with pytest.raises(ValueError, match="cluster"):
            head.fit(support_x, support_y)

    def test_unknown_metric_raises(self) -> None:
        support_x = torch.randn(4, 4)
        support_y = torch.tensor([0, 0, 1, 1])
        head = make_head(make_cfg(2, metric="manhattan"), make_episode(2))
        with pytest.raises(ValueError, match="metric"):
            head.fit(support_x, support_y)

    def test_non_finite_support_raises(self) -> None:
        support_x = torch.tensor([[0.0, 1.0], [float("nan"), 3.0], [4.0, 0.0], [6.0, 0.0]])
        support_y = torch.tensor([0, 0, 1, 1])
        head = make_head(make_cfg(2, metric="cosine"), make_episode(2, k_shot=2))
        with pytest.raises(ValueError, match="finite"):
            head.fit(support_x, support_y)

    def test_prototypes_property_before_fit_raises(self) -> None:
        head = make_head(make_cfg(2), make_episode(2))
        with pytest.raises(NotFittedError):
            _ = head.prototypes
