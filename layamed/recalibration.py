"""Recalibration by the amount of text written: probabilities that grow as the case is written.

A Laya trained only on complete cases is overconfident (in the most frequent class) when there is little text.
This layer does not touch the model: in logit space, it mixes the Laya output with the training class frequencies,
with a weight that grows as the case is written:

    w       = sigmoid(a + b·log(1 + typed tokens) + e·[exam written] + g·[labs written])
    logit_k = w·log P_k + (1 − w)·log freq_k
    P'      = softmax(logit)

"Typed tokens" = record tokens minus those of the empty case (the "Not documented." template).
The 4 parameters are fitted by maximum likelihood on prefixes of validation cases, with the true label
(notebook, part 6).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from . import inference as inf

# Cut points (fraction of the case words, in the order history → exam → labs). Denser at the start.
FRACTIONS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
_RE_WORD = re.compile(r"\S+")


def prefixes(text: str, fractions: Sequence[float] = FRACTIONS) -> list[str]:
    """Versions of the case cut at each fraction of its words, as if it were being typed.

    Sections not reached yet are empty ("Not documented.", as in training). Fraction 1 returns the original text
    unchanged.
    """
    sections = inf.split_sections(text)
    # (section, end offset of each word) in the order the case is written
    ends = [(s, m.end()) for s in inf.SECTIONS for m in _RE_WORD.finditer(sections[s])]
    total = len(ends)
    out = []
    for f in fractions:
        if f >= 1.0:
            out.append(text)
            continue
        k = int(round(f * total))
        parts = {s: "" for s in inf.SECTIONS}
        for s, end in ends[:k]:
            parts[s] = sections[s][:end]
        out.append(inf.build_text(parts["hpi"], parts["pe"], parts["lab"], clean=False))
    return out


def _features(n_tokens, texts: Sequence[str], n_empty: int) -> np.ndarray:
    """Matrix [n, 4]: constant, log(1 + typed tokens), exam written, labs written."""
    n = np.atleast_1d(np.asarray(n_tokens, dtype=float))
    sections = [inf.split_sections(t) for t in texts]
    has = {s: np.array([d[s] != inf.EMPTY_SECTION[s] for d in sections], dtype=float) for s in ("pe", "lab")}
    return np.c_[np.ones_like(n), np.log1p(np.maximum(n - n_empty, 0.0)), has["pe"], has["lab"]]


def _softmax(logit: np.ndarray) -> np.ndarray:
    logit = logit - logit.max(axis=1, keepdims=True)
    e = np.exp(logit)
    return e / e.sum(axis=1, keepdims=True)


class Recalibrator:
    """Mixes Laya × class frequencies with a learned weight (see the module docstring)."""

    NAMES = ("a (constant)", "b (log typed tokens)", "e (exam written)", "g (labs written)")

    def __init__(self, theta: Sequence[float], freq: Sequence[float], n_empty: int, meta: dict | None = None):
        self.theta = np.asarray(theta, dtype=float)
        self.freq = np.asarray(freq, dtype=float)
        self.n_empty = int(n_empty)
        self.meta = meta or {}

    def weight(self, n_tokens, texts: Sequence[str]) -> np.ndarray:
        """w: weight of Laya in the mix (0 = class frequencies only, 1 = Laya only)."""
        x = _features(n_tokens, texts, self.n_empty)
        return 1.0 / (1.0 + np.exp(-(x * self.theta).sum(axis=1)))

    def predict(self, P: np.ndarray, n_tokens, texts: Sequence[str]) -> np.ndarray:
        """Laya P [n, 4], record tokens [n] and model inputs [n] → recalibrated probabilities."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        w = self.weight(n_tokens, texts)[:, None]
        return _softmax(w * np.log(np.clip(P, 1e-7, 1.0)) + (1.0 - w) * np.log(self.freq))

    @classmethod
    def fit(cls, P: np.ndarray, n_tokens, texts: Sequence[str], y, freq: Sequence[float], n_empty: int,
            meta: dict | None = None, steps: int = 4000) -> "Recalibrator":
        """Fits the 4 weight parameters by maximum likelihood on prefixes with the true label.

        Analytic gradient + Adam, using only element-wise operations: in a conda environment with pip torch,
        calling BLAS (numpy @, scipy.optimize, np.linalg) in the same process as torch can crash Python, because
        torch (pip) and MKL (conda) ship different copies of OpenMP.
        """
        x = _features(n_tokens, texts, n_empty)
        log_freq = np.log(np.asarray(freq, dtype=float))
        D = np.log(np.clip(np.asarray(P, dtype=float), 1e-7, 1.0)) - log_freq  # logit = w·D + log freq
        y = np.asarray(y)
        rows = np.arange(len(y))
        D_y = D[rows, y]

        def loss_and_grad(theta):
            w = 1.0 / (1.0 + np.exp(-(x * theta).sum(axis=1)))
            p = _softmax(w[:, None] * D + log_freq)
            nll = -np.log(np.clip(p[rows, y], 1e-300, 1.0)).mean()
            # d nll / d w = −(D_y − Σ_k p_k D_k);  d w / d theta = w(1 − w)·x
            dw = -(D_y - (p * D).sum(axis=1)) * w * (1.0 - w)
            return nll, (dw[:, None] * x).mean(axis=0)

        theta = np.array([4.0, 0.0, 0.0, 0.0])  # starts at w ≈ 0.98 (almost Laya only)
        m, v, lr, b1, b2 = np.zeros(4), np.zeros(4), 0.05, 0.9, 0.999
        for t in range(1, steps + 1):
            nll, g = loss_and_grad(theta)
            m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
            theta = theta - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        nll, g = loss_and_grad(theta)
        if np.abs(g).max() > 1e-4:
            raise RuntimeError(f"Fit did not converge: max gradient {np.abs(g).max():.2e}")
        return cls(theta, freq, n_empty, meta={**(meta or {}), "val_nll": float(nll)})

    def save(self, path: Path) -> None:
        data = {"form": "w = sigmoid(theta · [1, log1p(tokens - n_empty), pe, lab]); logit = w·log P + (1 − w)·log freq",
                "theta": self.theta.tolist(), "freq": self.freq.tolist(), "n_empty": self.n_empty,
                "classes": inf.CLASSES, "meta": self.meta}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Recalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("classes") != inf.CLASSES or "theta" not in data:
            raise ValueError(f"{path} is from another recalibration version: fit it again (notebook, part 6)")
        return cls(data["theta"], data["freq"], data["n_empty"], meta=data.get("meta"))
