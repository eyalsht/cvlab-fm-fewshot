"""Deterministic episode sampler per PRD_episode_sampler and ADR-003.

Each episode draws from its own generator seeded with
SeedSequence([seed, episode_id]), so episode i is identical no matter how
many episodes a run requests (prefix stability) and episodes can be
generated out of order. Episodes are a property of (labels, seed, protocol)
alone, never of the head under evaluation.
"""

from collections.abc import Iterator

import numpy as np
import torch

from fm_fewshot.shared.contracts import Episode, ExperimentConfig


class EpisodeSampler:
    def __init__(self, labels: np.ndarray, cfg: ExperimentConfig) -> None:
        for field in ("n_way", "k_shot", "m_query", "n_episodes"):
            if getattr(cfg, field) < 1:
                raise ValueError(f"{field} must be >= 1, got {getattr(cfg, field)}")
        self._labels = np.asarray(labels, dtype=np.int64)
        self._cfg = cfg
        self._classes = np.unique(self._labels)
        if cfg.n_way > self._classes.size:
            raise ValueError(
                f"n_way={cfg.n_way} exceeds the {self._classes.size} classes in the split"
            )
        needed = cfg.k_shot + cfg.m_query
        counts = {int(c): int(np.sum(self._labels == c)) for c in self._classes}
        short = {c: n for c, n in counts.items() if n < needed}
        if short:
            detail = "; ".join(f"class {c} has {n} items" for c, n in short.items())
            raise ValueError(f"classes with fewer than k_shot + m_query = {needed} items: {detail}")
        self._class_indices = {int(c): np.flatnonzero(self._labels == c) for c in self._classes}

    def sample(self, episode_id: int) -> Episode:
        cfg = self._cfg
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, episode_id]))
        ways = np.sort(rng.choice(self._classes, size=cfg.n_way, replace=False))
        support: list[np.ndarray] = []
        query: list[np.ndarray] = []
        for c in ways:
            idx = rng.permutation(self._class_indices[int(c)])
            support.append(idx[: cfg.k_shot])
            query.append(idx[cfg.k_shot : cfg.k_shot + cfg.m_query])
        support_idx = np.concatenate(support)
        query_idx = np.concatenate(query)
        assert np.intersect1d(support_idx, query_idx).size == 0
        return Episode(
            episode_id=episode_id,
            dataset=cfg.dataset,
            n_way=cfg.n_way,
            k_shot=cfg.k_shot,
            m_query=cfg.m_query,
            class_ids=tuple(int(c) for c in ways),
            support_idx=torch.from_numpy(support_idx.astype(np.int64)),
            query_idx=torch.from_numpy(query_idx.astype(np.int64)),
        )

    def episodes(self) -> Iterator[Episode]:
        for episode_id in range(self._cfg.n_episodes):
            yield self.sample(episode_id)
