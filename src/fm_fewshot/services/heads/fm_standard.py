"""FM block as the last layer, trained by standard FM (Stage 2, FR10).

Everything but the fit is in `fm_base`, shared with the rolled-out head so the
comparison the write-up asks for has exactly one moving part. Standard training
is simulation-free, so this head never reads `sample_steps` while fitting: one
field serves every T, and T = 4 and T = 12 are that one field read at two
resolutions (ADR-023).
"""

from torch import Tensor

from fm_fewshot.services.flow.training.base import ValidationSelector
from fm_fewshot.services.flow.training.standard import train_standard_field
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import HeadContext, register
from fm_fewshot.services.heads.fm_base import FmHead
from fm_fewshot.shared.contracts import ExperimentConfig


@register("fm_standard")
class FmStandardHead(FmHead):
    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
    ) -> "FmStandardHead":
        return cls(**cls.shared_params(cfg, n_classes))

    def _fit_field(
        self, train_x: Tensor, targets: Tensor, selector: ValidationSelector
    ) -> tuple[VelocityMLP, list[float]]:
        return train_standard_field(
            train_x,
            targets,
            selector,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            init_seed=self._init_seed,
        )
