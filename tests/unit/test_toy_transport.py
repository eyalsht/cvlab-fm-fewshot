"""2D toy transport: does an unconditional flow toward prototypes classify?

This is the Stage 2 mechanism in miniature and the smallest honest check that
the machinery works end to end. The field never sees a label, so at inference
it must infer from position alone which basin a point belongs to. If this fails
on well-separated 2D Gaussians, nothing on real features will work.

Seeded and deterministic; asserts a transport-accuracy floor, not a number.
"""

import pytest
import torch

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.toy import (
    ToyProblem,
    classify_by_nearest_prototype,
    make_toy_problem,
    train_toy_field,
)

# Anisotropic on purpose. Isotropic blobs are already classified perfectly by
# nearest prototype, so transport there can only fail to hurt; it cannot show
# that the mechanism does anything. This is the geometry the Stage 1 diagnosis
# found in real features (docs/notes/prototype_gap_diagnosis.md).
ANISOTROPY = 5.0


@pytest.fixture(scope="module")
def trained() -> tuple[ToyProblem, torch.nn.Module]:
    problem = make_toy_problem(seed=0, anisotropy=ANISOTROPY)
    field = train_toy_field(problem, seed=0, steps=1200)
    return problem, field


class TestToyProblem:
    def test_shapes_and_prototypes(self) -> None:
        problem = make_toy_problem(seed=0)
        assert problem.train_x.shape[1] == 2
        assert problem.prototypes.shape == (problem.n_classes, 2)
        assert problem.test_x.shape[0] == problem.test_y.shape[0]

    def test_is_seeded(self) -> None:
        assert torch.equal(make_toy_problem(seed=1).train_x, make_toy_problem(seed=1).train_x)
        assert not torch.equal(
            make_toy_problem(seed=1).train_x, make_toy_problem(seed=2).train_x
        )

    def test_prototypes_are_the_class_means(self) -> None:
        problem = make_toy_problem(seed=0)
        for c in range(problem.n_classes):
            expected = problem.train_x[problem.train_y == c].mean(0)
            assert torch.allclose(problem.prototypes[c], expected, atol=1e-6)


class TestTransport:
    def test_transport_classifies_at_least_95_percent_at_8_steps(self, trained) -> None:
        """The PRD's acceptance criterion for the toy."""
        problem, field = trained
        with torch.no_grad():
            moved = solve_ode(field, problem.test_x, n_steps=8, method="euler")
        accuracy = classify_by_nearest_prototype(moved, problem.prototypes, problem.test_y)
        assert accuracy >= 0.95, accuracy

    def test_transport_recovers_the_anisotropy_loss(self, trained) -> None:
        """The Stage 2 hypothesis in miniature.

        Nearest prototype loses accuracy on stretched classes. If an
        unconditional flow toward prototypes can recover that loss here, the
        mechanism is at least capable of learning what the Stage 1 diagnosis
        says the real gap is.
        """
        problem, field = trained
        before = classify_by_nearest_prototype(
            problem.test_x, problem.prototypes, problem.test_y
        )
        with torch.no_grad():
            moved = solve_ode(field, problem.test_x, n_steps=8, method="euler")
        after = classify_by_nearest_prototype(moved, problem.prototypes, problem.test_y)
        assert before < 0.95, f"toy is too easy to be informative: {before}"
        assert after > before + 0.05, (before, after)

    def test_isotropic_problem_has_nothing_to_recover(self) -> None:
        """Control: with round classes the centroid rule is already optimal."""
        problem = make_toy_problem(seed=0, anisotropy=1.0)
        before = classify_by_nearest_prototype(
            problem.test_x, problem.prototypes, problem.test_y
        )
        assert before > 0.99

    def test_points_end_up_near_their_prototype(self, trained) -> None:
        problem, field = trained
        with torch.no_grad():
            moved = solve_ode(field, problem.test_x, n_steps=8, method="euler")
        travelled = (problem.test_x - problem.prototypes[problem.test_y]).norm(dim=1)
        remaining = (moved - problem.prototypes[problem.test_y]).norm(dim=1)
        assert float(remaining.mean()) < float(travelled.mean())

    @pytest.mark.parametrize("n_steps", [1, 2, 4, 8, 16, 32])
    def test_logits_stay_finite_at_every_step_count(self, trained, n_steps: int) -> None:
        problem, field = trained
        with torch.no_grad():
            moved = solve_ode(field, problem.test_x, n_steps=n_steps, method="euler")
        assert torch.isfinite(moved).all()

    def test_midpoint_agrees_with_euler_at_high_step_counts(self, trained) -> None:
        problem, field = trained
        with torch.no_grad():
            euler = solve_ode(field, problem.test_x, n_steps=64, method="euler")
            mid = solve_ode(field, problem.test_x, n_steps=64, method="midpoint")
        assert float((euler - mid).norm(dim=1).mean()) < 0.05


class TestDeterminism:
    def test_two_trainings_at_one_seed_are_bit_identical(self) -> None:
        problem = make_toy_problem(seed=0, anisotropy=ANISOTROPY)
        a = train_toy_field(problem, seed=0, steps=50)
        b = train_toy_field(problem, seed=0, steps=50)
        with torch.no_grad():
            assert torch.equal(a(problem.test_x, torch.zeros(len(problem.test_x))),
                               b(problem.test_x, torch.zeros(len(problem.test_x))))

    def test_training_reduces_the_objective(self) -> None:
        problem = make_toy_problem(seed=0, anisotropy=ANISOTROPY)
        field = train_toy_field(problem, seed=0, steps=400)
        x0 = problem.train_x
        x1 = problem.prototypes[problem.train_y]
        t = torch.rand(x0.shape[0], generator=torch.Generator().manual_seed(5))
        untrained = train_toy_field(problem, seed=0, steps=0)
        with torch.no_grad():
            assert float(cfm_loss(field, x0, x1, t)) < float(cfm_loss(untrained, x0, x1, t))
