"""The row(W) / null(W) displacement split, as arithmetic (ADR-036).

`W` here is `[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]`: `C = 2`, `D = 5`, and its two
rows are already orthonormal, so `row(W) = span(e0, e1)` and
`null(W) = span(e2, e3, e4)` by inspection. No SVD or extra reasoning is
needed to know which axes are which, which is what makes this a closed-form
check: the ground truth for "is this vector in row(W)" comes from reading
off coordinates, not from trusting the same `row_space_projector` call the
module under test also uses.
"""

import pytest
import torch

from fm_fewshot.services.evaluation.rowspace_diagnostics import (
    margins,
    norm_fractions,
    split_displacement,
)
from fm_fewshot.services.flow.training.rolled_out_ce import row_space_projector

W = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ]
)


class TestSplitDisplacementAgainstAClosedFormW:
    def test_components_sum_to_the_total_displacement(self) -> None:
        projector = row_space_projector(W)
        displacement = torch.randn(6, 5, generator=torch.Generator().manual_seed(0))
        row_component, null_component = split_displacement(displacement, projector)
        assert torch.allclose(row_component + null_component, displacement, atol=1e-6)

    def test_components_are_orthogonal(self) -> None:
        projector = row_space_projector(W)
        displacement = torch.randn(6, 5, generator=torch.Generator().manual_seed(0))
        row_component, null_component = split_displacement(displacement, projector)
        dot = (row_component * null_component).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(6), atol=1e-5)

    def test_norms_satisfy_pythagoras(self) -> None:
        # The split is orthogonal, so ||row||^2 + ||null||^2 == ||total||^2
        # follows from the orthogonality test above; asserted separately
        # because this is the identity the summary statistics rely on.
        projector = row_space_projector(W)
        displacement = torch.randn(6, 5, generator=torch.Generator().manual_seed(0))
        row_component, null_component = split_displacement(displacement, projector)
        total_norm = displacement.norm(dim=-1)
        row_norm = row_component.norm(dim=-1)
        null_norm = null_component.norm(dim=-1)
        assert torch.allclose(row_norm**2 + null_norm**2, total_norm**2, atol=1e-5)

    def test_a_pure_row_space_displacement_has_zero_null_fraction(self) -> None:
        # e0, e1 span row(W) by inspection. This is the case that would catch
        # a projector applied on the wrong side or a transposed W: either bug
        # would put nonzero mass in the null component here.
        projector = row_space_projector(W)
        displacement = torch.tensor([[3.0, -2.0, 0.0, 0.0, 0.0]])
        row_component, null_component = split_displacement(displacement, projector)
        assert torch.allclose(null_component, torch.zeros(1, 5), atol=1e-6)
        row_norm = row_component.norm(dim=-1)
        null_norm = null_component.norm(dim=-1)
        total_norm = displacement.norm(dim=-1)
        _, null_fraction = norm_fractions(row_norm, null_norm, total_norm)
        assert null_fraction.item() == pytest.approx(0.0, abs=1e-6)

    def test_a_pure_null_space_displacement_has_zero_row_fraction(self) -> None:
        # e2, e3, e4 span null(W) by inspection, the same reasoning in reverse.
        projector = row_space_projector(W)
        displacement = torch.tensor([[0.0, 0.0, 1.0, 4.0, -1.0]])
        row_component, null_component = split_displacement(displacement, projector)
        assert torch.allclose(row_component, torch.zeros(1, 5), atol=1e-6)
        row_norm = row_component.norm(dim=-1)
        null_norm = null_component.norm(dim=-1)
        total_norm = displacement.norm(dim=-1)
        row_fraction, _ = norm_fractions(row_norm, null_norm, total_norm)
        assert row_fraction.item() == pytest.approx(0.0, abs=1e-6)


class TestNormFractionsAtZeroDisplacement:
    def test_zero_total_norm_gives_zero_fractions_not_nan(self) -> None:
        # A point the field left exactly where it was (zero-init before
        # training, or n_train_steps=0) has a real 0/0 here: both components
        # are zero along with the total. Scored as zero mass in both row and
        # null rather than NaN, since "capacity spent" is exactly what did
        # not happen at that point.
        row_norm = torch.tensor([0.0])
        null_norm = torch.tensor([0.0])
        total_norm = torch.tensor([0.0])
        row_fraction, null_fraction = norm_fractions(row_norm, null_norm, total_norm)
        assert torch.isfinite(row_fraction).all()
        assert torch.isfinite(null_fraction).all()
        assert row_fraction.item() == 0.0
        assert null_fraction.item() == 0.0

    def test_nonzero_total_norm_is_unaffected_by_the_zero_guard(self) -> None:
        row_norm = torch.tensor([3.0, 0.0])
        null_norm = torch.tensor([4.0, 0.0])
        total_norm = torch.tensor([5.0, 0.0])
        row_fraction, null_fraction = norm_fractions(row_norm, null_norm, total_norm)
        assert row_fraction[0].item() == pytest.approx(0.6, abs=1e-6)
        assert null_fraction[0].item() == pytest.approx(0.8, abs=1e-6)
        assert row_fraction[1].item() == 0.0
        assert null_fraction[1].item() == 0.0


class TestMargins:
    def test_matches_hand_computation(self) -> None:
        logits = torch.tensor([[3.0, 1.0, 0.5], [0.2, 5.0, 4.0]])
        y = torch.tensor([0, 1])
        # row 0: true class score 3.0, best of the rest is 1.0 -> margin 2.0
        # row 1: true class score 5.0, best of the rest is 4.0 -> margin 1.0
        assert torch.allclose(margins(logits, y), torch.tensor([2.0, 1.0]), atol=1e-6)

    def test_a_wrong_prediction_gives_a_negative_margin(self) -> None:
        logits = torch.tensor([[0.0, 9.0]])
        y = torch.tensor([0])
        assert margins(logits, y).item() == pytest.approx(-9.0, abs=1e-6)
