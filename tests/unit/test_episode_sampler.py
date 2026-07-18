"""Episode sampler tests per PRD_episode_sampler section 5."""

import numpy as np
import pytest
import torch

from fm_fewshot.services.data.sampler import EpisodeSampler
from fm_fewshot.shared.contracts import ExperimentConfig


def make_labels(n_classes: int = 10, per_class: int = 20, shuffle_seed: int = 7) -> np.ndarray:
    labels = np.repeat(np.arange(n_classes), per_class)
    return np.random.default_rng(shuffle_seed).permutation(labels).astype(np.int64)


def make_config(**overrides: object) -> ExperimentConfig:
    fields: dict = {
        "run_name": "sampler-test",
        "dataset": "mnist",
        "encoder": "clip_vit_b32",
        "head": "prototype",
        "head_params": {},
        "n_way": 5,
        "k_shot": 1,
        "m_query": 15,
        "n_episodes": 600,
        "seed": 0,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class TestDeterminism:
    def test_two_samplers_yield_byte_identical_episodes(self) -> None:
        labels = make_labels()
        cfg = make_config()
        first = EpisodeSampler(labels, cfg)
        second = EpisodeSampler(labels, cfg)
        for episode_id in range(600):
            a = first.sample(episode_id)
            b = second.sample(episode_id)
            assert a.class_ids == b.class_ids
            assert a.support_idx.numpy().tobytes() == b.support_idx.numpy().tobytes()
            assert a.query_idx.numpy().tobytes() == b.query_idx.numpy().tobytes()

    def test_seed_changes_episodes(self) -> None:
        labels = make_labels()
        a = EpisodeSampler(labels, make_config(seed=0)).sample(0)
        b = EpisodeSampler(labels, make_config(seed=1)).sample(0)
        assert a.class_ids != b.class_ids or not torch.equal(a.support_idx, b.support_idx)


class TestPrefixStability:
    def test_first_100_episodes_independent_of_run_length(self) -> None:
        labels = make_labels()
        long_run = list(EpisodeSampler(labels, make_config(n_episodes=600)).episodes())
        short_run = list(EpisodeSampler(labels, make_config(n_episodes=100)).episodes())
        assert len(long_run) == 600
        assert len(short_run) == 100
        for a, b in zip(short_run, long_run[:100], strict=True):
            assert a.episode_id == b.episode_id
            assert a.class_ids == b.class_ids
            assert torch.equal(a.support_idx, b.support_idx)
            assert torch.equal(a.query_idx, b.query_idx)


class TestEpisodeStructure:
    def test_support_and_query_disjoint_over_200_episodes(self) -> None:
        labels = make_labels()
        sampler = EpisodeSampler(labels, make_config(n_episodes=200, k_shot=5, m_query=15))
        for episode in sampler.episodes():
            support = set(episode.support_idx.tolist())
            query = set(episode.query_idx.tolist())
            assert not support & query

    def test_balance_and_class_blocking(self) -> None:
        labels = make_labels()
        cfg = make_config(k_shot=3, m_query=4)
        episode = EpisodeSampler(labels, cfg).sample(11)
        support_labels = labels[episode.support_idx.numpy()]
        query_labels = labels[episode.query_idx.numpy()]
        expected_support = np.repeat(episode.class_ids, cfg.k_shot)
        expected_query = np.repeat(episode.class_ids, cfg.m_query)
        assert np.array_equal(support_labels, expected_support)
        assert np.array_equal(query_labels, expected_query)

    def test_class_ids_sorted_without_repeats(self) -> None:
        labels = make_labels()
        sampler = EpisodeSampler(labels, make_config(n_episodes=50))
        for episode in sampler.episodes():
            assert list(episode.class_ids) == sorted(set(episode.class_ids))
            assert len(episode.class_ids) == 5

    def test_local_label_convention(self) -> None:
        labels = make_labels()
        episode = EpisodeSampler(labels, make_config(k_shot=2, m_query=3)).sample(4)
        for j, class_id in enumerate(episode.class_ids):
            support_block = episode.support_idx[j * 2 : (j + 1) * 2].numpy()
            assert (labels[support_block] == class_id).all()

    def test_index_dtype_and_shape(self) -> None:
        labels = make_labels()
        episode = EpisodeSampler(labels, make_config(k_shot=5, m_query=15)).sample(0)
        assert episode.support_idx.dtype == torch.int64
        assert episode.query_idx.dtype == torch.int64
        assert episode.support_idx.shape == (5 * 5,)
        assert episode.query_idx.shape == (5 * 15,)
        assert episode.dataset == "mnist"


class TestValidation:
    def test_class_with_too_few_items_raises_naming_it(self) -> None:
        labels = np.concatenate(
            [np.repeat(np.arange(5), 20), np.full(15, 5)]  # class 5: k+m-1 = 15 items
        ).astype(np.int64)
        cfg = make_config(n_way=5, k_shot=1, m_query=15)
        with pytest.raises(ValueError, match=r"class.*5"):
            EpisodeSampler(labels, cfg)

    def test_n_way_larger_than_class_count_raises(self) -> None:
        labels = make_labels(n_classes=4)
        with pytest.raises(ValueError, match="n_way"):
            EpisodeSampler(labels, make_config(n_way=5))

    def test_non_positive_protocol_integers_raise(self) -> None:
        labels = make_labels()
        for field in ("n_way", "k_shot", "m_query", "n_episodes"):
            with pytest.raises(ValueError, match=field):
                EpisodeSampler(labels, make_config(**{field: 0}))
