"""FM block as the last layer, trained by rolling the solver out (Stage 2, FR11).

Same classifier as `fm_standard`, same prototypes, same decision rule, same
inference. The only difference is what the fit optimizes: instead of
supervising individual velocity predictions on the ideal interpolation path,
this head runs the T-step Euler sequence on the training features and minimizes
the distance from where they land to their prototype, backpropagating through
all T predictions.

T therefore means something different here than it does for standard training.
There it is a discretization knob applied after the fact to one field; here it
is the depth of the network being trained, so T = 4 and T = 12 are two models
and `n_rollout_steps` does not exist (ADR-022). A config carrying a separate
training depth is refused rather than quietly reconciled, because the number
that would be silently ignored is the one that decides what was trained. The
phase note must not report the two T values as one axis.
"""

from torch import Tensor

from fm_fewshot.services.flow.training.rolled_out import train_rolled_out_field
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import HeadContext, register
from fm_fewshot.services.heads.fm_base import FmHead
from fm_fewshot.shared.contracts import ExperimentConfig

# Names from the pre-write-up design and the obvious ways someone would try to
# reintroduce it. Listed rather than pattern-matched so the refusal message can
# name the key that was passed.
SEPARATE_DEPTH_KEYS = (
    "n_rollout_steps",
    "rollout_steps",
    "train_sample_steps",
    "training_sample_steps",
    "train_steps_depth",
)


@register("fm_rolled")
class FmRolledHead(FmHead):
    def __init__(self, n_classes: int, *, gradient_checkpointing: bool = False, **kwargs) -> None:
        super().__init__(n_classes, **kwargs)
        self._gradient_checkpointing = gradient_checkpointing

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
    ) -> "FmRolledHead":
        params = cfg.head_params
        refuse_separate_depth(params)
        return cls(
            gradient_checkpointing=bool(params.get("gradient_checkpointing", False)),
            **cls.shared_params(cfg, n_classes),
        )

    def _fit_field(self, train_x: Tensor, targets: Tensor) -> tuple[VelocityMLP, list[float]]:
        return train_rolled_out_field(
            train_x,
            targets,
            sample_steps=self._sample_steps,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            gradient_checkpointing=self._gradient_checkpointing,
            init_seed=self._init_seed,
        )


def refuse_separate_depth(head_params: dict[str, object]) -> None:
    """Reject any attempt to give rolled-out training its own step count."""
    for key in SEPARATE_DEPTH_KEYS:
        if key in head_params:
            raise ValueError(
                f"fm_rolled has no {key!r}: rolled-out training uses sample_steps for "
                "both training and inference depth, so the two cannot be separated "
                "(ADR-022). Run two configs if you want two values of T."
            )
