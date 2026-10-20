"""Recalibration by the amount of text written (layamed/recalibration.py)."""
import numpy as np

from layamed import inference as inf
from layamed import recalibration as rc


def test_prefixes_follow_writing_order():
    text = inf.build_text("a b c d e f g h", "i j k l", "m n o p")
    first, half, full = rc.prefixes(text, (0.0, 0.5, 1.0))
    assert first == inf.build_text("", "", "", clean=False)
    assert inf.split_sections(half) == {"hpi": "a b c d e f g h", "pe": inf.EMPTY_SECTION["pe"],
                                        "lab": inf.EMPTY_SECTION["lab"]}
    assert full == text


def _toy(n=600, seed=0):
    # Laya is right and confident with much text, and confidently wrong with little text
    r = np.random.default_rng(seed)
    y = r.integers(0, 4, n)
    tokens = r.integers(40, 600, n)
    texts = [inf.build_text("pain", "tender" if t > 300 else "", "") for t in tokens]
    right = r.random(n) < np.clip((tokens - 40) / 300, 0.1, 1.0)
    pred = np.where(right, y, (y + 1) % 4)
    P = np.full((n, 4), 0.02)
    P[np.arange(n), pred] = 0.94
    return P, tokens, texts, y


def test_fit_moderates_short_cases(tmp_path):
    P, tokens, texts, y = _toy()
    freq = np.full(4, 0.25)
    recal = rc.Recalibrator.fit(P, tokens, texts, y, freq, n_empty=32)
    w = recal.weight(np.array([40, 600]), [texts[0], texts[0]])
    assert w[0] < w[1]  # less text → closer to the class frequencies
    nll = lambda Q: -np.log(Q[np.arange(len(y)), y]).mean()
    assert nll(recal.predict(P, tokens, texts)) < nll(P)

    path = tmp_path / "recalibration.json"
    recal.save(path)
    loaded = rc.Recalibrator.load(path)
    np.testing.assert_allclose(loaded.predict(P, tokens, texts), recal.predict(P, tokens, texts))
