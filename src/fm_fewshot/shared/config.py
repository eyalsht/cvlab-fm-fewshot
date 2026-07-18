"""Load and save ExperimentConfig as yaml.

Unknown keys are rejected by name so a typo in a config file fails the run
immediately instead of silently falling back to a default.
"""

from dataclasses import asdict, fields
from pathlib import Path

import yaml

from fm_fewshot.shared.contracts import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} does not contain a mapping")
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {', '.join(unknown)}")
    return ExperimentConfig(**raw)


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    text = yaml.safe_dump(asdict(cfg), sort_keys=False)
    Path(path).write_text(text, encoding="utf-8")
