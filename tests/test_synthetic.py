"""Synthetic cases (synthetic/generate_cases.py): splits per base case, no leaking terms, reproducible."""
import pandas as pd
import pytest

from layamed import inference as inf
from synthetic import generate_cases as gen


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    gen.OUTPUT_DIR = tmp_path_factory.mktemp("synthetic")
    return pd.concat(gen.generate().values(), ignore_index=True)


def test_shape(cases):
    assert len(cases) == len(inf.CLASSES) * gen.BASE_CASES_PER_DISEASE * len(gen.STAGES)
    assert set(cases["split"]) == {"train", "val", "test"}


def test_each_base_case_stays_in_one_split(cases):
    assert (cases.groupby("case_id")["split"].nunique() == 1).all()
    assert (cases.groupby("case_id")["disease"].nunique() == 1).all()


def test_no_terms_the_model_must_not_see(cases):
    for text in cases["model_text"]:
        found = inf.audit_text(text)
        assert not (found["disease_or_surgery_name"] or found["imaging"] or found["postop_finding"]), text


def test_model_text_is_the_training_format(cases):
    for row in cases.sample(200, random_state=0).itertuples():
        assert row.model_text == inf.build_text(row.hpi, row.pe, row.lab)


def test_reproducible(cases, tmp_path):
    gen.OUTPUT_DIR = tmp_path
    again = pd.concat(gen.generate().values(), ignore_index=True)
    pd.testing.assert_frame_equal(cases, again)
