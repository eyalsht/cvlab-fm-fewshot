"""2D toy transport problem: the Stage 2 mechanism in miniature.

Train an unconditional velocity field on (example -> its class prototype)
pairs, then transport a held-out point and classify it by proximity to the
prototypes. This is the smallest complete instance of what Stage 2 does on real
features, and it is where the machinery gets checked before any of it touches
DTD or Aircraft.

The field is unconditional on purpose. At inference the class is unknown, so
the field has to infer from position alone which basin a point belongs to.
Whether that works at all is the Stage 2 research question; here it is only
being confirmed that the pieces fit together.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


@dataclass(frozen=True)
class ToyProblem:
    train_x: Tensor  # [N, 2]
    train_y: Tensor  # [N]
    test_x: Tensor
    test_y: Tensor
    prototypes: Tensor  # [C, 2], class means
    n_classes: int


def make_toy_problem(
    seed: int = 0,
    n_classes: int = 3,
    per_class: int = 128,
    spread: float = 0.55,
    radius: float = 3.0,
    anisotropy: float = 1.0,
) -> ToyProblem:
    """Gaussians on a circle.

    anisotropy = 1.0 gives isotropic blobs, which nearest-prototype already
    solves perfectly; that regime tests the plumbing and nothing else.

    anisotropy > 1 stretches every class along one shared direction, which is
    the geometry the Stage 1 diagnosis found in real DINOv2 features
    (docs/notes/prototype_gap_diagnosis.md): classes that are linearly
    separable but not compact around their means, where a centroid rule loses
    accuracy it cannot recover. That regime is the one that says something.
    """
    generator = torch.Generator().manual_seed(seed)
    angles = torch.arange(n_classes, dtype=torch.float32) * (2 * torch.pi / n_classes)
    centers = torch.stack([angles.cos(), angles.sin()], dim=1) * radius
    # One shared stretch direction, so class covariances are equal and the
    # optimal fix is a single whitening, exactly as on the real features.
    stretch = torch.tensor([anisotropy, 1.0 / anisotropy])

    def draw(count: int) -> tuple[Tensor, Tensor]:
        y = torch.arange(n_classes).repeat_interleave(count)
        noise = torch.randn(n_classes * count, 2, generator=generator) * spread * stretch
        return centers.repeat_interleave(count, dim=0) + noise, y

    train_x, train_y = draw(per_class)
    test_x, test_y = draw(per_class // 2)
    prototypes = torch.stack([train_x[train_y == c].mean(0) for c in range(n_classes)])
    return ToyProblem(train_x, train_y, test_x, test_y, prototypes, n_classes)


def train_toy_field(
    problem: ToyProblem,
    *,
    seed: int = 0,
    steps: int = 800,
    batch_size: int = 128,
    lr: float = 1e-2,
    hidden_dims: tuple[int, ...] = (64, 64),
) -> VelocityMLP:
    """Standard CFM training on (example, its class prototype) pairs."""
    field = VelocityMLP(dim=2, hidden_dims=hidden_dims, time_embed_dim=16, seed=seed)
    if steps == 0:
        return field

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    x1_all = problem.prototypes[problem.train_y]

    for _ in range(steps):
        rows = torch.randint(
            0, problem.train_x.shape[0], (batch_size,), generator=generator
        )
        t = torch.rand(batch_size, generator=generator)
        loss = cfm_loss(field, problem.train_x[rows], x1_all[rows], t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return field


def classify_by_nearest_prototype(x: Tensor, prototypes: Tensor, y: Tensor) -> float:
    """Accuracy of nearest-prototype classification by euclidean distance."""
    predicted = torch.cdist(x, prototypes).argmin(dim=1)
    return float((predicted == y).float().mean())
