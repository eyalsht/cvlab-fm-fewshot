"""Zero-shot CLIP classifier (Radford et al. 2021), Stage 1 baseline.

The head ignores support features: it scores a query by cosine similarity
against the episode's class text embeddings, showing what pretraining alone
solves. Text rows arrive at construction through the HeadContext seam, ordered
to match episode.class_ids, so no head loads open_clip inside an experiment.
Multiple prompt templates are averaged then renormalized (the CLIP paper's
prompt ensembling). fit is a no-op, but predict still refuses to run before it,
keeping the head-contract battery uniform across every method.
"""

import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    register,
)
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


@register("zeroshot_clip")
class ZeroShotClipHead(FewShotHead):
    def __init__(self, text_embeddings: Tensor, *, logit_scale: float = 100.0) -> None:
        emb = text_embeddings
        if emb.ndim == 2:  # [n_way, D] -> a single template
            emb = emb.unsqueeze(1)
        averaged = emb.mean(dim=1)  # ensemble templates before scoring
        self._text = F.normalize(averaged, dim=1)  # [n_way, D], renormalized
        self._logit_scale = logit_scale
        self._fitted = False

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "ZeroShotClipHead":
        if context.text_embeddings is None:
            raise FileNotFoundError(
                f"no text cache for ({cfg.dataset}, {cfg.encoder}); zeroshot_clip needs class "
                "text embeddings, run build_features to create them"
            )
        return cls(
            context.text_embeddings,
            logit_scale=float(cfg.head_params.get("logit_scale", 100.0)),
        )

    def fit(self, support_x: Tensor, support_y: Tensor) -> None:
        # Zero-shot: nothing to learn from support. Marking fitted keeps the
        # predict-before-fit guard uniform with the other heads.
        self._fitted = True

    def predict(self, query_x: Tensor) -> Tensor:
        if not self._fitted:
            raise NotFittedError("predict called before fit")
        if query_x.shape[1] != self._text.shape[1]:
            raise ValueError(
                f"query feature dim {query_x.shape[1]} != text embedding dim {self._text.shape[1]}"
            )
        query = F.normalize(query_x, dim=1)
        return (self._logit_scale * query @ self._text.T).float()
