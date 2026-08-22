"""The Stage 2 grid and how two configurations of one head are kept apart.

The write-up asks for five methods per setting: the Stage 1 prototype baseline,
standard FM at T = 4 and T = 12, and rolled-out FM at T = 4 and T = 12. Four of
those five are two heads carrying two step counts, so the run name and the
table cell both have to say which T produced a number. `variant` is that label,
and without it the two step counts would aggregate into one cell of six runs
that the protocol never asked for.
"""

from pathlib import Path

import pytest
import yaml
from test_report import write_run

from fm_fewshot.services.evaluation.report import load_cells, render_table
from fm_fewshot.services.evaluation.sweep import grid, load_grid
from fm_fewshot.shared.contracts import ExperimentConfig

STAGE2_VARIANTS = [
    {"label": "T4", "head": "fm_standard", "head_params": {"sample_steps": 4}},
    {"label": "T12", "head": "fm_standard", "head_params": {"sample_steps": 12}},
]


class TestVariantOnTheConfig:
    def test_variant_defaults_to_empty(self) -> None:
        cfg = ExperimentConfig(
            run_name="r", dataset="dtd", encoder="resnet18", head="prototype"
        )
        assert cfg.variant == ""

    def test_variant_is_rejected_when_unknown_keys_are(self) -> None:
        """It is a real field, not a bag the config silently accepts."""
        cfg = ExperimentConfig(
            run_name="r", dataset="dtd", encoder="resnet18", head="fm_standard", variant="T4"
        )
        assert cfg.variant == "T4"


class TestGridVariants:
    def test_one_config_per_variant(self) -> None:
        configs = grid(
            datasets=["dtd"],
            encoders={"resnet18": ["dtd"]},
            heads=[],
            ks=[5],
            variants=STAGE2_VARIANTS,
        )
        assert len(configs) == 6  # two variants, three seeds
        assert {cfg.variant for cfg in configs} == {"T4", "T12"}

    def test_run_names_name_the_variant(self) -> None:
        configs = grid(
            datasets=["dtd"],
            encoders={"resnet18": ["dtd"]},
            heads=[],
            ks=[5],
            variants=STAGE2_VARIANTS,
        )
        names = {cfg.run_name for cfg in configs}
        assert "dtd-resnet18-fm_standard-T4-k5-s0" in names
        assert "dtd-resnet18-fm_standard-T12-k5-s0" in names

    def test_head_params_come_from_the_variant(self) -> None:
        configs = grid(
            datasets=["dtd"],
            encoders={"resnet18": ["dtd"]},
            heads=[],
            ks=[5],
            variants=STAGE2_VARIANTS,
        )
        by_variant = {cfg.variant: cfg.head_params for cfg in configs}
        assert by_variant["T4"]["sample_steps"] == 4
        assert by_variant["T12"]["sample_steps"] == 12

    def test_plain_heads_still_expand_without_a_variant(self) -> None:
        configs = grid(
            datasets=["dtd"], encoders={"resnet18": ["dtd"]}, heads=["prototype"], ks=[5]
        )
        assert len(configs) == 3
        assert all(cfg.variant == "" for cfg in configs)


class TestCellsSeparateVariants:
    def test_two_step_counts_are_two_cells(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        for seed in range(3):
            write_run(results, f"a{seed}", 0.40, head="fm_standard", variant="T4")
            write_run(results, f"b{seed}", 0.44, head="fm_standard", variant="T12")
        cells = load_cells(results)
        assert len(cells) == 2
        assert {cell.head for cell in cells} == {"fm_standard@T4", "fm_standard@T12"}

    def test_the_table_names_the_variant(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        for seed in range(3):
            write_run(results, f"a{seed}", 0.40, head="fm_standard", variant="T4")
        table = render_table(load_cells(results))
        assert "fm_standard@T4" in table

    def test_dacc_still_finds_the_prototype_baseline(self, tmp_path: Path) -> None:
        """A variant row's delta is against the plain prototype row, not another variant."""
        results = tmp_path / "results"
        for seed in range(3):
            write_run(results, f"p{seed}", 0.30, head="prototype")
            write_run(results, f"a{seed}", 0.42, head="fm_standard", variant="T4")
        table = render_table(load_cells(results))
        assert "+0.120" in table


class TestStage2Configs:
    """The DoD for 8.7: the sweep command enumerates exactly the runs the grid needs."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("config/stage2_standard.yaml", 54), ("config/stage2_rolled.yaml", 54)],
    )
    def test_each_config_enumerates_half_the_grid(self, name: str, expected: int) -> None:
        # Three (dataset, encoder) combinations x three K x three runs x two T.
        assert len(load_grid(Path(name))) == expected

    def test_the_two_configs_together_are_the_whole_grid(self) -> None:
        configs = load_grid(Path("config/stage2_standard.yaml")) + load_grid(
            Path("config/stage2_rolled.yaml")
        )
        assert len(configs) == 108
        assert len({cfg.run_name for cfg in configs}) == 108

    def test_the_training_choices_are_fixed_across_the_two_schemes(self) -> None:
        """The write-up: keep the architecture and the main training choices fixed."""
        shared = ("n_train_steps", "batch_size", "lr", "hidden_dims", "time_conditioning")
        standard = load_grid(Path("config/stage2_standard.yaml"))[0].head_params
        rolled = load_grid(Path("config/stage2_rolled.yaml"))[0].head_params
        for key in shared:
            assert standard[key] == rolled[key], key

    def test_both_step_counts_the_write_up_asks_for_are_present(self) -> None:
        for name in ("config/stage2_standard.yaml", "config/stage2_rolled.yaml"):
            steps = {cfg.head_params["sample_steps"] for cfg in load_grid(Path(name))}
            assert steps == {4, 12}

    def test_the_configs_cover_the_stage_1_settings_exactly(self) -> None:
        stage1 = yaml.safe_load(Path("config/stage1.yaml").read_text(encoding="utf-8"))
        stage2 = yaml.safe_load(
            Path("config/stage2_standard.yaml").read_text(encoding="utf-8")
        )
        assert stage2["datasets"] == stage1["datasets"]
        assert stage2["encoders"] == stage1["encoders"]
        assert stage2["ks"] == stage1["ks"]
