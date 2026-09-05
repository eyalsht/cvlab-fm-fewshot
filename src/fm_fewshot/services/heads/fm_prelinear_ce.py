"""Stage 3 Strategy 1: the block trained end to end on the probe's decision (FR19).

Everything but the fit is in `fm_prelinear`, shared with the classifier-guided
head so the comparison the write-up asks for has exactly one moving part. What
this head supplies is a loss: the frozen probe's cross-entropy at the end of
the rollout, backpropagated through all T velocity predictions.

T means here what it means for `fm_rolled` and not what it means for
`fm_standard`: the rollout is the network being trained, so T = 4 and T = 12
are two models (ADR-032). The write-up asks for a single T throughout Stage 3
and the choice between the two is made on validation accuracy, by a rule fixed
before the grid runs.
"""

from torch import Tensor

from fm_fewshot.services.flow.training.base import ValidationSelector
from fm_fewshot.services.flow.training.rolled_out_ce import (
    RolledOutCeConfig,
    train_rolled_out_ce_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import HeadContext, register
from fm_fewshot.services.heads.fm_prelinear import FmPreLinearHead
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.shared.contracts import ExperimentConfig


@register("fm_prelinear_ce")
class FmPreLinearCeHead(FmPreLinearHead):
    def __init__(
        self, n_classes: int, *, ce_config: RolledOutCeConfig | None = None, **kwargs
    ) -> None:
        super().__init__(n_classes, **kwargs)
        self._ce_config = ce_config or RolledOutCeConfig()

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
    ) -> "FmPreLinearCeHead":
        params = cfg.head_params
        return cls(
            ce_config=RolledOutCeConfig(
                lambda_disp=float(params.get("lambda_disp", 0.0)),
                lambda_vel=float(params.get("lambda_vel", 0.0)),
                project_velocity=bool(params.get("project_velocity", False)),
            ),
            **cls.shared_params(cfg, n_classes),
        )

    @property
    def ce_config(self) -> RolledOutCeConfig:
        """The penalties and the projection, as the run's config.yaml records them."""
        return self._ce_config

    def _fit_field(
        self,
        train_x: Tensor,
        train_y: Tensor,
        probe: LinearProbeHead,
        selector: ValidationSelector,
    ) -> tuple[VelocityMLP, list[float]]:
        return train_rolled_out_ce_field(
            train_x,
            train_y,
            probe.weight,
            probe.bias,
            selector,
            sample_steps=self._sample_steps,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            lambda_disp=self._ce_config.lambda_disp,
            lambda_vel=self._ce_config.lambda_vel,
            project_velocity=self._ce_config.project_velocity,
            init_seed=self._init_seed,
            zero_output_init=self._zero_output_init,
        )
