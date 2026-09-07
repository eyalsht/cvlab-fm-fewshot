"""Starting Stage 3 from a trained Stage 2 field (extra).

**Not in the write-up.** `PRD_stage3_prelinear.md` recorded this as a question
for after the main grid rather than a variant of it, because the two fields
transport toward different things: a Stage 2 field was fitted to reach the
image prototypes, and Stage 3 has no prototypes. It is an `ablations/` line
under ADR-016 and never a graded row.

The property it gives up is worth stating in a test rather than a comment.
ADR-030 makes the untrained Stage 3 system exactly the linear probe, which is
what lets the reported change measure what training added. A field carried over
from Stage 2 is not the identity, so a run that starts from one no longer has
that guarantee, and the head must not pretend otherwise.
"""

import pytest
import torch
from test_head_contract import make_config

from fm_fewshot.services.heads.base import make_head

PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 3,
    "hidden_dims": [16, 16],
    "batch_size": 8,
    "lr": 1e-3,
    "eval_every": 0,
    "probe_params": {"max_epochs": 10},
}


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    return train_x, labels, train_x, labels


def head(**overrides):
    return make_head(make_config("fm_prelinear_ce", **dict(PARAMS, **overrides)), 3)


class TestItIsOffByDefault:
    def test_the_default_is_the_identity_start(self) -> None:
        assert head().init_from == ""

    def test_a_default_run_still_starts_at_identity(self) -> None:
        h = head(n_train_steps=0)
        h.fit(*problem())
        query = problem()[0]
        assert torch.equal(h.transport(query), query)


class TestCarryingTheFieldOver:
    def test_the_field_starts_from_the_stage_2_weights(self) -> None:
        """The point of the option: step 0 is a trained Stage 2 field."""
        carried = head(init_from="fm_standard", n_train_steps=0, stage2_train_steps=5)
        carried.fit(*problem())
        reference = make_head(
            make_config(
                "fm_standard", sample_steps=4, n_train_steps=5,
                hidden_dims=[16, 16], batch_size=8, lr=1e-3, eval_every=0,
            ),
            3,
        )
        reference.fit(*problem())
        for a, b in zip(carried.field.parameters(), reference.field.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_it_no_longer_starts_at_identity(self) -> None:
        """The guarantee ADR-030 gives up, asserted rather than noted."""
        carried = head(init_from="fm_standard", n_train_steps=0, stage2_train_steps=5)
        carried.fit(*problem())
        query = problem()[0]
        assert not torch.equal(carried.transport(query), query)

    def test_the_head_reports_that_it_did_not_start_at_identity(self) -> None:
        """A run store has to be readable without rerunning the fit."""
        carried = head(init_from="fm_standard", n_train_steps=0, stage2_train_steps=5)
        assert carried.starts_at_identity is False
        assert head().starts_at_identity is True

    def test_the_probe_is_still_the_stage_1_probe(self) -> None:
        """Only the field is carried over; the classifier is untouched."""
        carried = head(init_from="fm_standard", n_train_steps=0, stage2_train_steps=5)
        plain = head(n_train_steps=0)
        for h in (carried, plain):
            h.fit(*problem())
        assert carried.probe.classifier_digest == plain.probe.classifier_digest

    def test_it_is_deterministic_at_a_fixed_seed(self) -> None:
        query = problem()[0]
        a = head(init_from="fm_standard", stage2_train_steps=5)
        b = head(init_from="fm_standard", stage2_train_steps=5)
        a.fit(*problem())
        b.fit(*problem())
        assert torch.equal(a.predict(query), b.predict(query))

    def test_training_moves_off_the_carried_field(self) -> None:
        carried = head(init_from="fm_standard", n_train_steps=6, lr=5e-2, stage2_train_steps=5)
        start = head(init_from="fm_standard", n_train_steps=0, stage2_train_steps=5)
        carried.fit(*problem())
        start.fit(*problem())
        assert any(
            not torch.equal(a, b)
            for a, b in zip(carried.field.parameters(), start.field.parameters(), strict=True)
        )

    @pytest.mark.parametrize("key", ["fm_standard", "fm_rolled"])
    def test_either_stage_2_scheme_can_be_carried_over(self, key: str) -> None:
        h = head(init_from=key, n_train_steps=0, stage2_train_steps=3)
        h.fit(*problem())
        assert h.field is not None

    def test_an_unknown_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="init_from"):
            head(init_from="linear_probe", stage2_train_steps=3).fit(*problem())
