"""Prototypical classifier (Snell et al. 2017), Stage 1 baseline.

Each episode class becomes one or more prototypes (clusters_per_class); a query
scores against its nearest prototype under the chosen metric. fit is closed-form
arithmetic with no gradient training. With clusters_per_class > 1 a seeded
Lloyd's K-means runs per class; the seed is derived from the episode so two fits
on the same support give identical prototypes.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    register,
)
from fm_fewshot.shared.contracts import Episode, ExperimentConfig

_KMEANS_MAX_ITER = 100


@register("prototype")
class PrototypeHead(FewShotHead):
    def __init__(
        self,
        n_way: int,
        *,
        metric: str = "cosine",
        temperature: float = 1.0,
        clusters_per_class: int = 1,
        seed: int = 0,
    ) -> None:
        self._n_way = n_way
        self._metric = metric
        self._temperature = temperature
        self._clusters = clusters_per_class
        self._seed = seed
        self._prototypes: Tensor | None = None  # [n_way, clusters, D]

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "PrototypeHead":
        params = cfg.head_params
        return cls(
            n_way=episode.n_way,
            metric=str(params.get("metric", "cosine")),
            temperature=float(params.get("temperature", 1.0)),
            clusters_per_class=int(params.get("clusters_per_class", 1)),
            seed=cfg.seed ^ (episode.episode_id + 1),
        )

    @property
    def prototypes(self) -> Tensor:
        if self._prototypes is None:
            raise NotFittedError("prototypes are undefined before fit")
        return self._prototypes

    def fit(self, support_x: Tensor, support_y: Tensor) -> None:
        if self._metric not in ("cosine", "euclidean"):
            raise ValueError(f"unknown metric {self._metric!r}; use 'cosine' or 'euclidean'")
        if not torch.isfinite(support_x).all():
            raise ValueError("non-finite values in support features")

        dim = support_x.shape[1]
        prototypes = torch.empty(self._n_way, self._clusters, dim)
        for j in range(self._n_way):
            class_x = support_x[support_y == j]
            if class_x.shape[0] == 0:
                raise ValueError(f"class {j} has no support example; every class must appear")
            if self._clusters > class_x.shape[0]:
                raise ValueError(
                    f"clusters_per_class={self._clusters} exceeds the {class_x.shape[0]} "
                    f"support examples of class {j}"
                )
            prototypes[j] = self._class_prototypes(class_x, j)
        self._prototypes = prototypes

    def _class_prototypes(self, class_x: Tensor, class_index: int) -> Tensor:
        if self._clusters == 1:
            return class_x.mean(dim=0, keepdim=True)
        return self._kmeans(class_x, class_index)

    def _kmeans(self, class_x: Tensor, class_index: int) -> Tensor:
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, class_index]))
        init = rng.permutation(class_x.shape[0])[: self._clusters]
        centers = class_x[torch.from_numpy(init)].clone()
        for _ in range(_KMEANS_MAX_ITER):
            assign = torch.cdist(class_x, centers).argmin(dim=1)
            updated = centers.clone()
            for c in range(self._clusters):
                members = class_x[assign == c]
                if members.shape[0] > 0:
                    updated[c] = members.mean(dim=0)
            if torch.equal(updated, centers):
                break
            centers = updated
        return centers

    def predict(self, query_x: Tensor) -> Tensor:
        if self._prototypes is None:
            raise NotFittedError("predict called before fit")
        if self._metric == "cosine":
            query = F.normalize(query_x, dim=1)
            protos = F.normalize(self._prototypes, dim=2)
            sims = torch.einsum("md,jkd->mjk", query, protos)
            scores = sims.max(dim=2).values
            logits = scores / self._temperature
        else:
            n_way, clusters, dim = self._prototypes.shape
            flat = self._prototypes.reshape(n_way * clusters, dim)
            sq = torch.cdist(query_x, flat).pow(2).reshape(query_x.shape[0], n_way, clusters)
            logits = -sq.min(dim=2).values / self._temperature
        return logits.float()
