"""Strategy 1: end-to-end rolled-out classification training (FR19, ADR-034).

    zhat_0 = z,  zhat_{k+1} = zhat_k + (1/T) v_theta(zhat_k, k/T),  k = 0..T-1
    L_cls = CE(W zhat_T + b, y)

The gradient goes back through all T velocity predictions. W and b are the
Stage 1 probe's and are detached, so only theta moves.

This is not `rolled_out.py` with a different target. That module minimizes
|| zhat_T - p_y ||^2 toward a prototype fixed before training; here there is no
prototype and no point in feature space to reach, only a decision to get right.
An endpoint distance and a cross-entropy differ in what they hold the block
responsible for: the first names where to land, the second names only which
side of the frozen map's boundaries to land on. That is the whole difference
between Stage 2's second scheme and this one, and it is why Stage 3 writes a
new loss instead of reusing that one (stage3_alignment section 1.1).

The optional penalties are the two the write-up names, and both carry zero
weight in `results/` (ADR-034):

    L = L_cls + lambda_disp * mean_i || zhat_T - z_i ||^2
              + lambda_vel  * mean_i sum_k || v_theta(zhat_k, k/T) ||^2

`project_velocity` is ours, not his. The classifier is frozen and linear, so
the logits depend on zhat only through its projection onto row(W) and the field
is free in all D dimensions while the loss can see at most C of them. Projecting
the velocity puts Strategy 1 in Strategy 2's search space while keeping its
objective, which is the only way to tell an objective difference from a search
space difference in the comparison the write-up asks for
(stage3_alignment section 3.3). It is an `ablations/` row and never a graded one.

His optional extension, jointly fine-tuning the classifier (FR23), is an
option here rather than a second trainer. `JointClassifier` puts the probe's
own W and b into a second parameter group of the same Adam, at their own
learning rate, and gates them on the step number so they can join late. The
default is off, so every graded row is the fit that produced `results/`, and
the tests that pin the frozen path assert exactly that.

There is one integrator in this repository and this module uses it. The
velocities the penalty needs are collected by recording what the solver asks
for, not by writing a second rollout that could drift from the first.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.solver import VelocityField
from fm_fewshot.services.flow.training.base import (
    StepObserver,
    ValidationSelector,
    train_field,
    transport,
)
from fm_fewshot.services.flow.training.classifier_guided import (
    GuidedTargetConfig,
    guided_target,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


@dataclass(frozen=True)
class RolledOutCeConfig:
    """The knobs past the shared training block. Every default is the graded row.

    The write-up's Strategy 1 is the plain cross-entropy, so the penalties are
    off, the velocity is unprojected and the classifier is frozen unless an
    ablation config says otherwise.

    The last three fields are his optional extension (FR23). `joint_classifier`
    off is the graded row and the reason every existing configuration is
    bit-identical to the one that produced `results/`. `classifier_lr` is the
    "different learning rates" variant and follows the field's rate when unset;
    `unfreeze_at` is the "delayed unfreezing" one, counted in optimizer steps,
    so the classifier is untouched for steps 1..N and moves from step N+1.
    """

    lambda_disp: float = 0.0
    lambda_vel: float = 0.0
    project_velocity: bool = False
    joint_classifier: bool = False
    classifier_lr: float | None = None
    unfreeze_at: int = 0
    # The hybrid, ours and not his (stage3_alignment 3.3). mu = 0 is the graded
    # row and the reason every Strategy 1 fit is bit-identical to the one that
    # produced results/: at zero the term is not merely weighted out, it is not
    # computed and it draws no randomness.
    mu: float = 0.0
    coupling: GuidedTargetConfig = field(default_factory=GuidedTargetConfig)

    def __post_init__(self) -> None:
        # The projector is built once from W. A W that moves makes it stale
        # after the first step, so the ablation and the extension would answer
        # neither of their questions; refused rather than silently wrong.
        if self.joint_classifier and self.project_velocity:
            raise ValueError(
                "project_velocity projects onto row(W) of a classifier fixed before "
                "training; it cannot be combined with joint_classifier, whose W moves"
            )


class JointClassifier:
    """The probe's W and b, reopened for training beside the field (FR23).

    Holds the probe's own tensors rather than copies, so the optimizer, the
    checkpoint selector, `predict` and `classifier_digest` all see one map.

    Frozen means `requires_grad = False`, not a zero learning rate. Adam skips
    a parameter whose gradient is None, so during the delay the classifier
    accumulates no optimizer state at all and the boundary is exact: it is the
    Stage 1 map through step N and a trained one from step N+1, with nothing in
    between carried over from the frozen phase.
    """

    def __init__(self, weight: Tensor, bias: Tensor, *, unfreeze_at: int = 0) -> None:
        if unfreeze_at < 0:
            raise ValueError(f"unfreeze_at must be >= 0, got {unfreeze_at}")
        self.weight = weight
        self.bias = bias
        self.unfreeze_at = unfreeze_at
        self._set_trainable(unfreeze_at == 0)

    @property
    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def open_at(self, step: int) -> None:
        """Let the classifier move from step `unfreeze_at + 1` on. Steps are 1-based."""
        self._set_trainable(step > self.unfreeze_at)

    def snapshot(self) -> tuple[Tensor, Tensor]:
        return (self.weight.detach().clone(), self.bias.detach().clone())

    def restore(self, state: object) -> None:
        weight, bias = state  # type: ignore[misc]
        # In place, so the tensors the probe and the optimizer hold are the
        # ones that get the selected checkpoint's values.
        with torch.no_grad():
            self.weight.copy_(weight)
            self.bias.copy_(bias)

    def _set_trainable(self, trainable: bool) -> None:
        self.weight.requires_grad_(trainable)
        self.bias.requires_grad_(trainable)


def row_space_projector(weight: Tensor) -> Tensor:
    """The orthogonal projector onto row(W), [D, D].

    Built from an orthonormal basis of the row space rather than from
    W^T (W W^T)^-1 W, which is the same matrix but needs W to have full row
    rank. A probe fitted at K=5 on 47 classes can be rank deficient, and a
    singular inverse there would be a crash in an ablation rather than a
    projector onto the space that actually exists.
    """
    _, singular, right = torch.linalg.svd(weight, full_matrices=False)
    tolerance = singular.max() * max(weight.shape) * torch.finfo(weight.dtype).eps
    basis = right[singular > tolerance]  # [rank, D], orthonormal rows
    return basis.T @ basis


class _Recorder(torch.nn.Module):
    """The field, plus the velocities the solver asked it for.

    Wrapping the field rather than reimplementing the rollout keeps the
    training path and the inference path the same integrator, which is what
    makes "trained through the rollout it is scored on" true rather than
    intended. Gradients pass through untouched; the recorded tensors are the
    same objects the solver used.
    """

    def __init__(self, field: VelocityField) -> None:
        super().__init__()
        self.field = field
        self.velocities: list[Tensor] = []

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        velocity = self.field(x, t)
        self.velocities.append(velocity)
        return velocity


def rolled_out_ce_loss(
    field: VelocityField,
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    sample_steps: int,
    lambda_disp: float = 0.0,
    lambda_vel: float = 0.0,
    freeze_classifier: bool = True,
    mu: float = 0.0,
    coupling: GuidedTargetConfig | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """L_cls for one batch, with the two optional penalties and the hybrid term.

    `weight` and `bias` are detached here as well as frozen by the head, so a
    caller that hands over tensors still attached to a graph cannot train the
    classifier through this loss by accident. `freeze_classifier=False` is the
    optional extension (FR23) and the only way past that: the gradient then
    reaches W and b, and whether it moves them is decided by their own
    `requires_grad`, which is what the delayed-unfreezing schedule flips.
    """
    if freeze_classifier:
        weight, bias = weight.detach(), bias.detach()
    recorder = _Recorder(field)
    endpoint = transport(recorder, x0, sample_steps=sample_steps)
    loss = F.cross_entropy(endpoint @ weight.T + bias, y)
    if lambda_disp:
        loss = loss + lambda_disp * ((endpoint - x0) ** 2).sum(dim=1).mean()
    if lambda_vel:
        velocity = torch.stack([(v**2).sum(dim=1) for v in recorder.velocities])
        loss = loss + lambda_vel * velocity.sum(dim=0).mean()
    if mu:
        # Strategy 2's coupling, supervising the whole path, added to Strategy
        # 1's decision at the endpoint. Computed per batch, so a cached
        # recompute cadence would be quietly ignored rather than honoured.
        config = (coupling or GuidedTargetConfig()).validate()
        if config.target_every != 1:
            raise ValueError(
                f"the hybrid computes its coupling per batch, so target_every must be 1, "
                f"got {config.target_every}; a cached cadence cannot be honoured here"
            )
        if generator is None:
            raise ValueError("the hybrid term samples t and needs a generator")
        target = guided_target(
            field, x0, y, weight, bias, sample_steps=sample_steps, config=config
        )
        t = torch.rand(x0.shape[0], generator=generator)
        loss = loss + mu * cfm_loss(field, x0, target, t)
    return loss


def train_rolled_out_ce_field(
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    selector: ValidationSelector | None = None,
    *,
    sample_steps: int,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_disp: float = 0.0,
    lambda_vel: float = 0.0,
    project_velocity: bool = False,
    mu: float = 0.0,
    coupling: GuidedTargetConfig | None = None,
    joint: "JointClassifier | None" = None,
    classifier_lr: float | None = None,
    init_seed: int = 0,
    zero_output_init: bool = True,
    init_state: dict[str, Tensor] | None = None,
    on_step: StepObserver | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta by rolling the solver out and scoring the probe's decision.

    With `joint` given, the probe's decision is not frozen: its W and b join
    the optimization in a parameter group of their own at `classifier_lr`,
    defaulting to the field's `lr`. `joint` must wrap the same `weight` and
    `bias` tensors passed here, which is what the head guarantees by handing
    over the probe's own tensors to both.
    """
    if sample_steps < 1:
        raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
    # One projector for the whole fit, built into the field itself: W is
    # frozen, so this is one SVD, and a field that projects only inside the
    # loss would stop projecting the moment inference integrated it.
    projector = row_space_projector(weight) if project_velocity else None

    def batch_loss(
        field: VelocityMLP, batch_x0: Tensor, batch_y: Tensor, generator: torch.Generator
    ) -> Tensor:
        # The generator is unused at mu = 0: the rolled-out objective has no
        # random time (ADR-021 carries over, the rollout is the whole path and
        # nothing is sampled along it). The hybrid term does draw t, which is
        # why it is passed on and why a mu = 0 fit stays bit-identical.
        return rolled_out_ce_loss(
            field,
            batch_x0,
            batch_y,
            weight,
            bias,
            sample_steps=sample_steps,
            lambda_disp=lambda_disp,
            lambda_vel=lambda_vel,
            freeze_classifier=joint is None,
        )

    extra_param_groups = (
        None
        if joint is None
        else [{"params": joint.parameters, "lr": lr if classifier_lr is None else classifier_lr}]
    )
    return train_field(
        x0,
        y,
        batch_loss,
        selector,
        hidden_dims=hidden_dims,
        time_conditioning=time_conditioning,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        init_seed=init_seed,
        zero_output_init=zero_output_init,
        output_projector=projector,
        init_state=init_state,
        extra_param_groups=extra_param_groups,
        on_before_step=None if joint is None else joint.open_at,
        on_step=on_step,
    )
