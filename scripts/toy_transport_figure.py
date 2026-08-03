"""Regenerate the 2D toy transport figure into assets/.

    uv run python scripts/toy_transport_figure.py

Shows the Stage 2 mechanism on stretched 2D classes: where the points start,
where the flow moves them, and the accuracy before and after. Deterministic
from the seeds below.
"""

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fm_fewshot.services.evaluation.figures import PALETTE  # noqa: E402
from fm_fewshot.services.flow.solver import solve_ode  # noqa: E402
from fm_fewshot.services.flow.toy import (  # noqa: E402
    classify_by_nearest_prototype,
    make_toy_problem,
    train_toy_field,
)

SEED = 0
ANISOTROPY = 5.0
STEPS = 8
OUT = Path("assets/toy_transport.png")


def main() -> None:
    problem = make_toy_problem(seed=SEED, anisotropy=ANISOTROPY)
    field = train_toy_field(problem, seed=SEED, steps=1200)

    with torch.no_grad():
        moved, trajectory = solve_ode(
            field, problem.test_x, n_steps=STEPS, method="euler", return_trajectory=True
        )

    before = classify_by_nearest_prototype(problem.test_x, problem.prototypes, problem.test_y)
    after = classify_by_nearest_prototype(moved, problem.prototypes, problem.test_y)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=200, sharex=True, sharey=True)
    for ax, points, title in (
        (axes[0], problem.test_x, f"before transport, accuracy {before:.3f}"),
        (axes[1], moved, f"after {STEPS} Euler steps, accuracy {after:.3f}"),
    ):
        for c in range(problem.n_classes):
            mask = problem.test_y == c
            ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.55,
                       color=PALETTE[c], linewidths=0, label=f"class {c}")
            ax.scatter(problem.prototypes[c, 0], problem.prototypes[c, 1], s=220,
                       marker="*", color=PALETTE[c], edgecolors="black", linewidths=0.8,
                       zorder=5)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    # A few trajectories, to show the paths rather than only the endpoints.
    step = max(1, problem.test_x.shape[0] // 40)
    for i in range(0, problem.test_x.shape[0], step):
        axes[1].plot(trajectory[:, i, 0], trajectory[:, i, 1],
                     color="0.35", linewidth=0.4, alpha=0.5, zorder=1)

    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(
        f"unconditional flow toward class prototypes, anisotropy {ANISOTROPY:g}. "
        "Stars are prototypes; the stretch is the geometry found in real features.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"{OUT}  before={before:.4f}  after={after:.4f}  gain={after - before:+.4f}")


if __name__ == "__main__":
    main()
