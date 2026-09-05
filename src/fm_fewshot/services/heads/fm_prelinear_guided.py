"""Stage 3 Strategy 2: the block trained on classifier-guided targets (FR20).

Everything but the fit is in `fm_prelinear`, shared with the cross-entropy head
so the comparison the write-up asks for has exactly one moving part. What this
head supplies is a coupling: the frozen probe's gradient defines an improved
target for each training feature, and the field is fitted to it by the same
simulation-free update Stage 2's standard scheme uses.

T means here what it means for `fm_standard` and not what it means for
`fm_rolled`. The update never solves the ODE, so one trained field serves every
step count; T reaches the fit twice over, through the rollout that builds the
target and through the selection metric, which is why Stage 3 trains a separate
field per T rather than reading one field at two resolutions (ADR-032).
"""

from torch import Tensor

from fm_fewshot.services.flow.training.base import ValidationSelector
from fm_fewshot.services.flow.training.classifier_guided import (
    GuidedTargetConfig,
    train_classifier_guided_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import HeadContext, register
from fm_fewshot.services.heads.fm_prelinear import FmPreLinearHead
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.shared.contracts import ExperimentConfig


@register("fm_prelinear_guided")
class FmPreLinearGuidedHead(FmPreLinearHead):
    def __init__(
        self, n_classes: int, *, target_config: GuidedTargetConfig | None = None, **kwargs
    ) -> None:
        super().__init__(n_classes, **kwargs)
        self._target_config = (target_config or GuidedTargetConfig()).validate()

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
    ) -> "FmPreLinearGuidedHead":
        params = cfg.head_params
        return cls(
            target_config=GuidedTargetConfig(
                target_step=str(params.get("target_step", "raw")),
                eta=float(params.get("eta", 1.0)),
                target_steps=int(params.get("target_steps", 1)),
                target_every=int(params.get("target_every", 1)),
                rho=float(params.get("rho", 0.5)),
            ),
            **cls.shared_params(cfg, n_classes),
        )

    @property
    def target_config(self) -> GuidedTargetConfig:
        """How the target is built, as the run's config.yaml records it."""
        return self._target_config

    def _fit_field(
        self,
        train_x: Tensor,
        train_y: Tensor,
        probe: LinearProbeHead,
        selector: ValidationSelector,
    ) -> tuple[VelocityMLP, list[float]]:
        return train_classifier_guided_field(
            train_x,
            train_y,
            probe.weight,
            probe.bias,
            selector,
            sample_steps=self._sample_steps,
            config=self._target_config,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            init_seed=self._init_seed,
            zero_output_init=self._zero_output_init,
        )
