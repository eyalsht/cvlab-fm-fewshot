"""What both training schemes share: the fitting loop and the inference rollout.

The write-up asks us to keep the network architecture and the main training
choices fixed when comparing standard and rolled-out training, so the two
schemes must not own two copies of the loop that could drift apart. They own a
loss and nothing else. Everything around it, the seeded field, the Adam
optimizer, the batch draw and the loss curve, lives here and is called by both.

The per-batch loss is handed the batch generator rather than a pre-drawn time.
Standard training needs t ~ U(0, 1) and draws it from that stream; rolled-out
training has no t anywhere in its objective (ADR-021) and draws nothing, which
is the honest way to say so in code. The row stream is the same call either
way, so the schemes agree on the first batch and the comparison starts fair.

The optimizer is ours, not his: he asks only for stable training, so Adam at
1e-3 carries over from the Phase 7 toy and every setting of it is a config
field recorded in the run.

Checkpoint selection lives here for the same reason the loop does. It is one
class, constructed by the head and driven by the shared loop, so neither scheme
can be selected on a different rule, a different split or a different schedule
than the other (ADR-028).
"""

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP

# (field, batch_x0, batch_x1, generator) -> scalar loss. batch_x1 is whatever
# per-row payload the caller paired with x0: endpoints in Stage 2, labels in
# Stage 3.
BatchLoss = Callable[[VelocityMLP, Tensor, Tensor, torch.Generator], Tensor]
# (1-based step, global gradient norm) -> None. See `train_field`.
StepObserver = Callable[[int, float], None]


def _gradient_norm(field: VelocityMLP) -> float:
    """The norm of the concatenated gradient over every parameter that has one.

    One number, so a 2000-step fit summarizes as a distribution rather than a
    wall of per-tensor norms. Parameters with no gradient are skipped rather
    than counted as zero, which would deflate the norm silently.
    """
    total = torch.zeros(())
    for parameter in field.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().pow(2).sum()
    return float(total.sqrt())


def require_paired_endpoints(x0: Tensor, x1: Tensor) -> None:
    """Refuse a coupling whose two endpoints do not live in the same space.

    The loop itself checks only the row count, because Stage 3 pairs each
    feature with a label rather than with an endpoint. A scheme whose payload
    really is the other end of a transport calls this, so the mismatch is
    still caught at the door and named for what it is.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            "paired endpoints must match in shape, got "
            f"{tuple(x0.shape)} and {tuple(x1.shape)}"
        )


class Classifier(Protocol):
    """What the selector needs of a decision rule: logits from features.

    Typed structurally so this module keeps knowing nothing about heads. In
    Stage 2 the object passed is the head's own fitted PrototypeHead, which is
    what makes the selection metric the metric the run reports (ADR-020).
    """

    def predict(self, query_x: Tensor) -> Tensor: ...


class ValidationSelector:
    """Keep the field that scored best on the validation split (ADR-028).

    Every `eval_every` optimizer steps the current field transports `val_x` at
    the run's own `sample_steps` and the fitted classifier scores where it
    lands. The best-scoring state dict is held in memory and restored into the
    field when training ends, so the head predicts with the checkpoint that was
    selected rather than the one training happened to stop on.

    `eval_every = 0` disables the mechanism entirely: nothing is evaluated,
    nothing is recorded, nothing is restored, and the fit is bit-identical to
    the one this repository ran before selection existed.

    Two properties the fairness of the comparison rests on. Selection never
    touches the training RNG stream, the weights or the optimizer, so the loss
    curve of a selected run is the curve of the unselected run. And a tie keeps
    the earlier step, the rule the linear probe has used since ADR-014.

    The step numbers recorded are 1-based, matching the rows of
    `loss_curve.csv`, which is where the loop writes them.
    """

    def __init__(
        self,
        val_x: Tensor,
        val_y: Tensor,
        classifier: Classifier,
        *,
        sample_steps: int,
        eval_every: int,
    ) -> None:
        if eval_every < 0:
            raise ValueError(f"eval_every must be >= 0, got {eval_every}")
        self.val_x = val_x
        self.val_y = val_y
        self.classifier = classifier
        self.sample_steps = sample_steps
        self.eval_every = eval_every
        self._history: list[tuple[int, float]] = []
        self._best_step: int | None = None
        self._best_accuracy = -1.0
        self._best_state: dict[str, Tensor] | None = None

    @property
    def enabled(self) -> bool:
        return self.eval_every > 0

    @property
    def history(self) -> list[tuple[int, float]]:
        """(step, val top-1) on the evaluation grid, for the loss_curve.csv column."""
        return list(self._history)

    @property
    def best_step(self) -> int | None:
        """The step whose field was kept; None when nothing was selected."""
        return self._best_step

    def observe(self, step: int, field: VelocityMLP) -> None:
        """Score the field at `step` if it falls on the grid, and keep it if it wins."""
        if not self.enabled or step % self.eval_every:
            return
        accuracy = self._accuracy(field)
        self._history.append((step, accuracy))
        # Strictly greater keeps the earlier step on a tie (ADR-014).
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            self._best_step = step
            self._best_state = {
                name: tensor.detach().clone() for name, tensor in field.state_dict().items()
            }

    def restore(self, field: VelocityMLP) -> None:
        """Put the selected weights back into the field. A no-op if none were kept."""
        if self._best_state is not None:
            field.load_state_dict(self._best_state)

    def _accuracy(self, field: VelocityMLP) -> float:
        was_training = field.training
        field.eval()
        try:
            with torch.no_grad():
                landed = transport(field, self.val_x, sample_steps=self.sample_steps)
                predicted = self.classifier.predict(landed).argmax(dim=1)
        finally:
            field.train(was_training)
        return float((predicted == self.val_y).float().mean())


def train_field(
    x0: Tensor,
    x1: Tensor,
    batch_loss: BatchLoss,
    selector: ValidationSelector | None = None,
    *,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    init_seed: int = 0,
    zero_output_init: bool = False,
    on_step: StepObserver | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta on the paired coupling (x0[i] -> x1[i]); return it and its loss curve.

    `x1` is the per-row payload paired with `x0`, and what it holds is the
    objective's business. Both Stage 2 schemes pass endpoints, [N, D] against
    [N, D], and both Stage 3 strategies pass labels, [N], because neither has an
    endpoint that exists before training starts: Strategy 1 optimizes a decision
    rather than a destination, and Strategy 2 rebuilds its target as the field
    moves. Only the row count is checked here; a loss that is handed the wrong
    shape fails in its own terms, where the error names the objective.

    `selector` is passed positionally, next to the loss, because those two are
    the only arguments that carry anything scheme-specific: the loss is the
    objective, the selector is the validation split and the depth to score at.
    Everything after them is the training configuration both schemes share.
    With no selector, or one built with `eval_every = 0`, the field returned is
    the field the last optimizer step left.

    `zero_output_init` is Stage 3's near-identity start (ADR-030) and defaults
    to off, which is the Stage 2 field. It reaches the network and nothing
    else: the loop, the optimizer and the batch stream do not read it.

    `on_step` is a diagnostic and defaults to off. When given, it is handed the
    1-based step number and the global gradient norm, read after `backward` and
    before `optimizer.step()`, which is the only point at which the gradient
    the optimizer is about to apply exists. It reads buffers and consumes no
    randomness, so an observed fit stays bit-identical to an unobserved one.
    """
    if x0.shape[0] != x1.shape[0]:
        raise ValueError(
            "the per-row payload must have one row per source point, got "
            f"{tuple(x0.shape)} and {tuple(x1.shape)}"
        )
    if n_train_steps < 0:
        raise ValueError(f"n_train_steps must be >= 0, got {n_train_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    field = VelocityMLP(
        dim=x0.shape[1],
        hidden_dims=tuple(hidden_dims),
        time_conditioning=time_conditioning,
        seed=init_seed,
        zero_output_init=zero_output_init,
    )
    if n_train_steps == 0:
        return field, []

    # One stream for the weights, one for the batches, both from init_seed, so
    # the seed alone reproduces the fit (ADR-012).
    generator = torch.Generator().manual_seed(init_seed)
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    history: list[float] = []

    for step in range(n_train_steps):
        rows = torch.randint(0, x0.shape[0], (batch_size,), generator=generator)
        loss = batch_loss(field, x0[rows], x1[rows], generator)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        if on_step is not None:
            on_step(step + 1, _gradient_norm(field))
        optimizer.step()
        history.append(float(loss.detach()))
        if selector is not None:
            # 1-based, so the step numbering agrees with loss_curve.csv.
            selector.observe(step + 1, field)

    if selector is not None:
        selector.restore(field)
    return field, history


def transport(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """T Euler steps from x, the write-up's inference rule.

    Both schemes classify this way; rolled-out training also trains through it,
    which is why there is one integrator and not two.
    """
    return solve_ode(field, x, n_steps=sample_steps, method="euler")


def transport_trajectory(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """The same integration, keeping every state: [T + 1, N, D] for the S4 figures."""
    _, states = solve_ode(field, x, n_steps=sample_steps, method="euler", return_trajectory=True)
    return states
