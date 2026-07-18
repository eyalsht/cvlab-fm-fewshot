"""Gatekeeper tests per ADR-010: device resolution and the heavy-job guard."""

import pytest
import torch

from fm_fewshot.shared import gatekeeper
from fm_fewshot.shared.gatekeeper import HeavyJobOnCpuError, check, resolve_device


class TestResolveDevice:
    def test_cpu_resolves_to_cpu(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == torch.device("cuda")

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == torch.device("cpu")

    def test_cuda_without_cuda_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="cuda"):
            resolve_device("cuda")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="tpu"):
            resolve_device("tpu")


class TestHeavyJobGuard:
    def test_light_job_passes_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert check("evaluation", "auto") == torch.device("cpu")

    def test_heavy_job_on_cpu_refused_naming_the_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(HeavyJobOnCpuError) as exc_info:
            check("feature_extraction", "auto")
        message = str(exc_info.value)
        assert "feature_extraction" in message
        assert "GPU" in message
        assert "allow-heavy-on-cpu" in message

    def test_heavy_job_on_cpu_allowed_with_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert check("feature_extraction", "auto", allow_heavy_on_cpu=True) == torch.device("cpu")

    def test_heavy_job_passes_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert check("feature_extraction", "auto") == torch.device("cuda")

    def test_unknown_job_raises(self) -> None:
        with pytest.raises(ValueError, match="no_such_job"):
            check("no_such_job", "cpu")

    def test_heavy_job_registry_matches_claude_md(self) -> None:
        expected = frozenset(
            {"feature_extraction", "rolled_out_sweep", "stage3_training", "fine_tuning"}
        )
        assert expected == gatekeeper.HEAVY_JOBS
