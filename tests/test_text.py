"""Text preparation (layamed/inference.py): the same functions build the training data and the server input."""
import re
import time

from layamed import inference as inf

# The original single-regex version of the post-operative clause removal (quadratic); kept here as the reference
OLD_POSTOP = re.compile(r"[^,;.\n]*(?:" + inf._RE_POSTOP + r")[^,;.\n]*[,;.]?", re.IGNORECASE)


def test_build_and_split_sections_roundtrip():
    text = inf.build_text("Pain in the RLQ.", "Tender RLQ.", "WBC 14")
    assert inf.split_sections(text) == {"hpi": "Pain in the RLQ.", "pe": "Tender RLQ.", "lab": "WBC 14"}
    empty = inf.split_sections(inf.build_text("", "", ""))
    assert empty == {s: inf.EMPTY_SECTION[s] for s in inf.SECTIONS}


def test_disease_and_surgery_names_are_masked():
    out = inf.clean_section("Mother had appendicitis. Lap appy in 2010. Pain RLQ.", "hpi")
    assert "appendicitis" not in out and "appy" not in out
    assert out.count(inf.MASK_TOKEN) == 2
    assert inf.clean_section("CT shows pancreatitis", "lab") == f"CT shows {inf.MASK_TOKEN}"


def test_imaging_sentences_are_dropped():
    assert inf.clean_section("CT showed inflammation. Pain in RLQ.", "hpi") == "Pain in RLQ."
    assert inf.clean_section("RUQ US with stones. Nausea.", "hpi") == "Nausea."


def test_postop_clauses_are_dropped_from_exam():
    assert inf.clean_section("Incision clean, abdomen tender in RLQ, no rebound.", "pe") == \
        "abdomen tender in RLQ, no rebound."
    assert inf.clean_section("Abdomen tender, s/p cholecystectomy 2010, Murphy negative.", "pe") == \
        "Abdomen tender, Murphy negative."


def test_postop_removal_matches_original_regex():
    samples = ["", "wound", "Incision c/d/i, soft.", "Soft\nJP drain in place; tender RLQ.",
               "port sites clean,, tender; no rebound", "Post-op day 1. Abdomen soft\n\nwound ok",
               "s/p lap chole, staples in place, abdomen soft, nontender."]
    for s in samples:
        assert inf._drop_postop_clauses(s) == OLD_POSTOP.sub(" ", s), s


def test_cleaning_is_linear_on_long_text_without_punctuation():
    text = ("abcdefghi " * 6000).strip()  # 60 thousand characters, the server limit
    t0 = time.perf_counter()
    inf.clean_section(text, "pe")
    assert time.perf_counter() - t0 < 2.0


def test_split_free_text():
    out = inf.split_free_text("Patient with pain.\nPE: tender RLQ\nLabs: WBC 14\nhistory: prior episodes")
    assert out == {"hpi": "Patient with pain.\nprior episodes", "pe": "tender RLQ", "lab": "WBC 14"}
    assert inf.split_free_text("no titles at all") == {"hpi": "no titles at all", "pe": "", "lab": ""}


def test_inline_titles_are_reported():
    assert inf.inline_titles("HPI: 30yo with pain. Exam: soft. Lab: lipase 3000") == ["Exam", "Lab"]
    assert inf.inline_titles("Rectal exam: normal") == []
    assert inf.inline_titles("History:\npain\nExam:\nsoft") == []


def test_audit_text():
    w = inf.audit_text("s/p appendectomy. CT negative. Tender gallbladder, history of diverticulosis.")
    assert w["disease_or_surgery_name"] == ["appendectomy"]
    assert w["imaging"] == ["CT"]
    assert w["postop_finding"] == ["s/p"]
    assert w["organ_name"] == ["diverticulosis", "gallbladder"]
    assert not any(inf.audit_text("Abdominal pain with nausea, T 38.2 C, WBC 14.").values())
