"""Zero-shot CLIP head tests per PRD_zeroshot_clip_head section 5.

The real text-tower cache does not exist yet, so class text embeddings arrive
through the HeadContext seam (a stub text cache); the head never calls
open_clip. text_embeddings are shaped [n_way, D] for a single template or
[n_way, n_templates, D] for prompt ensembling, ordered to match class_ids.
"""

import numpy as np
import pytest
import torch
from test_head_contract import run_head_battery

from fm_fewshot.services.heads.base import HeadContext, NotFittedError, make_head
from fm_fewshot.services.heads.zeroshot_clip import ZeroShotClipHead
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


def make_episode(class_ids: tuple[int, ...], episode_id: int = 0) -> Episode:
    n_way = len(class_ids)
    return Episode(
        episode_id=episode_id,
        dataset="synthetic",
        n_way=n_way,
        k_shot=1,
        m_query=1,
        class_ids=class_ids,
        support_idx=torch.arange(n_way, dtype=torch.int64),
        query_idx=torch.arange(n_way, 2 * n_way, dtype=torch.int64),
    )


def make_cfg(**head_params: object) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="zeroshot-test",
        dataset="synthetic",
        encoder="stub",
        head="zeroshot_clip",
        head_params=dict(head_params),
        n_way=3,
        k_shot=1,
        m_query=1,
        n_episodes=1,
        seed=0,
    )


def fit_head(cfg: ExperimentConfig, episode: Episode, text: torch.Tensor) -> ZeroShotClipHead:
    head = make_head(cfg, episode, HeadContext(text_embeddings=text))
    head.fit(torch.zeros(episode.n_way, text.shape[-1]), torch.arange(episode.n_way))
    return head


class TestBatteryContract:
    def test_zeroshot_passes_the_battery(self) -> None:
        text = torch.eye(3, 4)
        support_x = text.repeat_interleave(2, dim=0)
        support_y = torch.arange(3).repeat_interleave(2)
        query_x = torch.eye(3, 4)
        expected = torch.arange(3)
        run_head_battery(
            make_cfg(), make_episode((0, 1, 2), episode_id=0), support_x, support_y, query_x,
            expected, context=HeadContext(text_embeddings=text),
        )


class TestStatelessWrtSupport:
    def test_mutating_support_between_fit_and_predict_cannot_change_logits(self) -> None:
        text = torch.eye(3, 5)
        episode = make_episode((0, 1, 2))
        head = make_head(make_cfg(), episode, HeadContext(text_embeddings=text))
        support_x = torch.randn(3, 5)
        head.fit(support_x, torch.arange(3))
        query_x = torch.randn(6, 5)
        before = head.predict(query_x)
        support_x.add_(100.0)  # in-place mutation of the fitted support
        after = head.predict(query_x)
        assert torch.equal(before, after)


class TestCorrectRows:
    def test_query_equal_to_class_embedding_scores_that_class_with_margin(self) -> None:
        text = torch.eye(3, 3)
        episode = make_episode((0, 1, 2))
        head = fit_head(make_cfg(), episode, text)
        logits = head.predict(torch.eye(3, 3))
        assert torch.equal(logits.argmax(dim=1), torch.arange(3))
        # each true class scores logit_scale, every other column zero.
        assert torch.allclose(logits, 100.0 * torch.eye(3), atol=1e-4)


class TestEnsembling:
    def test_two_templates_average_then_normalize_before_scoring(self) -> None:
        # class 0 templates average to (2,0) -> unit (1,0); class 1 to (0,2) -> (0,1).
        text = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0]], [[0.0, 1.0], [0.0, 3.0]]]
        )  # [n_way=2, n_templates=2, D=2]
        episode = make_episode((0, 1))
        head = fit_head(make_cfg(), episode, text)
        logits = head.predict(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        assert torch.allclose(logits, 100.0 * torch.eye(2), atol=1e-4)


class TestOrdering:
    def test_shuffling_class_ids_permutes_logit_columns(self) -> None:
        rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        query_x = torch.randn(7, 3)
        base = fit_head(make_cfg(), make_episode((0, 1, 2)), rows).predict(query_x)

        perm = [2, 0, 1]
        permuted = fit_head(
            make_cfg(), make_episode(tuple(perm)), rows[perm]
        ).predict(query_x)
        assert torch.allclose(permuted, base[:, perm], atol=1e-5)


class TestDeterminism:
    def test_repeated_predicts_are_bit_identical(self) -> None:
        text = torch.randn(3, 8)
        episode = make_episode((0, 1, 2))
        head = fit_head(make_cfg(), episode, text)
        query_x = torch.randn(9, 8)
        assert torch.equal(head.predict(query_x), head.predict(query_x))


class TestConfiguration:
    def test_logit_scale_scales_logits(self) -> None:
        text = torch.eye(3, 3)
        episode = make_episode((0, 1, 2))
        head = fit_head(make_cfg(logit_scale=10.0), episode, text)
        logits = head.predict(torch.eye(3, 3))
        assert torch.allclose(logits, 10.0 * torch.eye(3), atol=1e-4)


class TestValidation:
    def test_predict_before_fit_raises(self) -> None:
        text = torch.eye(3, 3)
        head = make_head(make_cfg(), make_episode((0, 1, 2)), HeadContext(text_embeddings=text))
        with pytest.raises(NotFittedError):
            head.predict(torch.eye(3, 3))

    def test_missing_text_cache_error_names_build_features(self) -> None:
        with pytest.raises(FileNotFoundError, match="build_features"):
            make_head(make_cfg(), make_episode((0, 1, 2)))

    def test_text_dim_mismatch_raises(self) -> None:
        text = torch.eye(3, 4)  # dim 4 text
        head = fit_head(make_cfg(), make_episode((0, 1, 2)), text)
        with pytest.raises(ValueError, match="dim"):
            head.predict(torch.randn(5, 8))  # dim 8 query

    def test_non_finite_query_produces_no_nan_when_normalized(self) -> None:
        text = torch.eye(3, 3)
        head = fit_head(make_cfg(), make_episode((0, 1, 2)), text)
        rng = np.random.default_rng(0)
        logits = head.predict(torch.from_numpy(rng.standard_normal((4, 3)).astype(np.float32)))
        assert torch.isfinite(logits).all()
