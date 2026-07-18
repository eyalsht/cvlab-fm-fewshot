"""Contract tests: frozen dataclasses and config yaml round-trip per PLAN section 5."""

import dataclasses
from pathlib import Path

import pytest
import torch

from fm_fewshot.shared.config import load_config, save_config
from fm_fewshot.shared.contracts import (
    Episode,
    EpisodeResult,
    ExperimentConfig,
    RunSummary,
)


def make_episode() -> Episode:
    return Episode(
        episode_id=0,
        dataset="mnist",
        n_way=2,
        k_shot=1,
        m_query=2,
        class_ids=(3, 7),
        support_idx=torch.tensor([10, 20], dtype=torch.int64),
        query_idx=torch.tensor([11, 12, 21, 22], dtype=torch.int64),
    )


def make_config(**overrides: object) -> ExperimentConfig:
    fields: dict = {
        "run_name": "smoke",
        "dataset": "mnist",
        "encoder": "clip_vit_b32",
        "head": "prototype",
        "head_params": {"metric": "cosine"},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class TestFrozen:
    def test_episode_is_frozen(self) -> None:
        episode = make_episode()
        with pytest.raises(dataclasses.FrozenInstanceError):
            episode.dataset = "cifar10"  # type: ignore[misc]

    def test_experiment_config_is_frozen(self) -> None:
        cfg = make_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.seed = 1  # type: ignore[misc]

    def test_episode_result_is_frozen(self) -> None:
        result = EpisodeResult(
            episode_id=0,
            accuracy=0.5,
            n_correct=2,
            n_query=4,
            fit_seconds=0.01,
            predict_seconds=0.01,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.accuracy = 1.0  # type: ignore[misc]

    def test_run_summary_is_frozen(self) -> None:
        summary = RunSummary(
            run_id="20260718T120000_smoke",
            config=make_config(),
            git_commit="abc1234",
            accuracy_mean=0.5,
            ci95=0.02,
            episode_results=[],
            wall_seconds=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.ci95 = 0.0  # type: ignore[misc]


class TestDefaults:
    def test_experiment_config_defaults_match_plan(self) -> None:
        cfg = make_config()
        assert cfg.n_way == 5
        assert cfg.k_shot == 1
        assert cfg.m_query == 15
        assert cfg.n_episodes == 600
        assert cfg.seed == 0
        assert cfg.device == "auto"
        assert cfg.l2_normalize is True
        assert cfg.split == "test"

    def test_episode_fields_hold_int64_indices(self) -> None:
        episode = make_episode()
        assert episode.support_idx.dtype == torch.int64
        assert episode.query_idx.dtype == torch.int64
        assert episode.class_ids == (3, 7)


class TestConfigYaml:
    def test_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        cfg = make_config(seed=42, k_shot=5, head_params={"sample_steps": 8})
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        assert load_config(path) == cfg

    def test_defaults_fill_omitted_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "run_name: smoke\ndataset: mnist\nencoder: clip_vit_b32\n"
            "head: prototype\nhead_params: {}\n",
            encoding="utf-8",
        )
        cfg = load_config(path)
        assert cfg.n_way == 5
        assert cfg.n_episodes == 600

    def test_unknown_key_raises_with_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "run_name: smoke\ndataset: mnist\nencoder: clip_vit_b32\n"
            "head: prototype\nhead_params: {}\nn_wag: 5\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="n_wag"):
            load_config(path)
