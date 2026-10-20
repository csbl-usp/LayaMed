"""Clinical vocabulary of the out-of-scope detector (layamed/scope.py needs torch: skipped without it)."""
import pytest

pytest.importorskip("torch")

from layamed import inference as inf  # noqa: E402
from layamed.scope import count_clinical_terms  # noqa: E402


def test_clinical_text_has_terms_and_recipe_has_none():
    clinical = inf.build_text("Abdominal pain with nausea and fever.", "Tender RLQ, HR 110.", "Lipase: 30 IU/L")
    recipe = inf.build_text("Mix the flour with two eggs and bake for forty minutes.", "", "")
    assert count_clinical_terms(clinical) >= 6
    assert count_clinical_terms(recipe) == 0
