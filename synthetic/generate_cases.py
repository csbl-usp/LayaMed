"""Generates synthetic clinical cases (English) of the 4 diseases, in partial versions, to train and test LayaMed
without real data.

Nothing here comes from real records: cases are assembled by rules from typical textbook clinical ranges.
Each base case becomes 4 rows, in the order a case is written:
    1. hpi_partial  the first sentences of the history (30–60% of the sentences, at least 2)
    2. hpi          the full history
    3. hpi_pe       history + physical exam
    4. complete     history + physical exam + labs
With 250 base cases per disease, each CSV has 1,000 rows.

Measures so the model does not learn wording shortcuts (a first version was 100% separable by keywords):
    - non-specific sentences (associated symptoms, negatives, past history, medications, general exam) come from the
      SAME banks for the 4 diseases; only sentences with a real clinical signal are disease-specific;
    - ~50% of the histories open with just "abdominal pain" and localize the pain later (the partial version may
      lack the location, as in real life);
    - ~35% of the cases "mimic" the disease they are most confused with: they get 1–2 history sentences typical of
      it (appendicitis ↔ diverticulitis, cholecystitis ↔ pancreatitis); the `mimics` column says which;
    - ~25% atypical presentations (mildly raised lipase in cholecystitis, right-sided pain in diverticulitis,
      isolated epigastric pain in cholecystitis, pancreatitis without back radiation, etc.);
    - no disease or surgery names, no imaging and no post-operative findings (checked with inference.audit_text;
      generation fails otherwise);
    - several wordings per sentence and shuffled order (the first history sentence is always the opening).
The `split` column (train/val/test 70/15/15) is drawn per base case: the 4 rows of a case stay in the same set, so
the answer does not leak between training and test.

Usage (from the repository folder):
    python synthetic/generate_cases.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from layamed import inference as inf  # noqa: E402

SEED = 42
BASE_CASES_PER_DISEASE = 250            # × 4 stages = 1,000 rows per disease
P_GENERIC_OPENING = 0.5
P_ATYPICAL = 0.25
OUTPUT_DIR = ROOT / "data" / "synthetic"
STAGES = ["hpi_partial", "hpi", "hpi_pe", "complete"]
LABEL_ID = {c: i for i, c in enumerate(inf.CLASSES)}


# ==============================================================================
# HELPERS
# ==============================================================================
def one(r: random.Random, options):
    return r.choice(options)


def some(r: random.Random, options, kmin: int, kmax: int):
    return r.sample(options, r.randint(kmin, min(kmax, len(options))))


def unif(r: random.Random, a: float, b: float, decimals: int = 1) -> str:
    return f"{r.uniform(a, b):.{decimals}f}"


def en_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def patient(r: random.Random, age: tuple[int, int], p_woman: float) -> dict:
    woman = r.random() < p_woman
    return {"age": r.randint(*age), "woman": woman, "who": "woman" if woman else "man",
            "pronoun": "She" if woman else "He"}


def duration(r: random.Random, hours_min: int, hours_max: int) -> str:
    h = r.randint(hours_min, hours_max)
    if h < 24:
        return one(r, [f"{h} hours", f"about {h} hours", f"{h}h"])
    d = round(h / 24)
    return f"{d} day{'s' if d > 1 else ''}" if d < 7 else "1 week"


# ==============================================================================
# BANKS SHARED BY THE 4 DISEASES (no disease-specific signal)
# ==============================================================================
COMMON_PMH = ["no significant past medical history", "no past medical history", "a history of hypertension",
              "a history of asthma", "a history of hypothyroidism", "a history of anxiety",
              "a history of gastroesophageal reflux", "a history of hyperlipidemia", "a history of type 2 diabetes",
              "obesity", "a history of depression", "a history of coronary artery disease",
              "obesity and type 2 diabetes", "obesity and hypertension", "hypertension and chronic constipation"]
COMMON_SYMPTOMS = ["nausea", "vomiting", "one episode of emesis", "anorexia", "decreased appetite",
                   "subjective fevers", "chills", "malaise", "bloating", "fatigue"]
COMMON_NEGATIVES = ["No diarrhea.", "Denies dysuria or hematuria.", "No rectal bleeding.", "Denies chest pain or shortness of breath.",
                    "No blood in the stool.", "No urinary symptoms.", "Denies black stools.", "No recent weight loss.",
                    "No cough.", "No headache."]
COMMON_SOCIAL = ["No known drug allergies.", "Allergic to penicillin (rash).", "Takes no regular medications.",
                 "Takes lisinopril and atorvastatin.", "Takes levothyroxine.", "Takes an occasional ibuprofen.",
                 "No recent travel or sick contacts.", "Nonsmoker.", "Former smoker.",
                 "Takes metformin.", "Lives with family and works full time."]
COMMON_CHARACTER = ["The pain is constant.", "The pain has been gradually worsening.", "The pain is sharp.",
                    "The pain is dull and aching.", "The pain is severe, 8 out of 10.", "The pain is moderate, 6 out of 10.",
                    "The pain is worse with movement.", "Nothing makes the pain better.",
                    "Acetaminophen at home did not help."]
GENERAL_EXAM = ["Alert and oriented, in mild distress.", "Uncomfortable, lying still on the stretcher.",
                "Appears unwell but nontoxic.", "In moderate distress.", "Lungs clear bilaterally.",
                "Heart with regular rate and rhythm, no murmurs.", "Mucous membranes dry.",
                "No costovertebral angle tenderness.", "No hernias palpated.", "Diaphoretic.",
                "Bowel sounds present."]
COMMON_ABDOMEN = ["voluntary guarding", "no rebound", "mild distension", "no peritoneal signs", "hypoactive bowel sounds",
                  "normal bowel sounds"]


def associated(r: random.Random, own: list[str]) -> str:
    """Associated symptoms: 1–3 from the shared bank and 0–1 specific to the disease, mixed."""
    items = some(r, COMMON_SYMPTOMS, 1, 3) + some(r, own, 0, 1)
    r.shuffle(items)
    return "Associated with " + en_list(items) + "."


def vital_signs(r: random.Random, t: tuple[float, float], hr: tuple[int, int], sbp: tuple[int, int]) -> str:
    vitals = [f"T {unif(r, *t)} C", f"HR {r.randint(*hr)}", f"BP {r.randint(*sbp)}/{r.randint(58, 92)}",
              f"RR {r.randint(14, 22)}", f"SpO2 {r.randint(95, 100)}% on room air"]
    return ", ".join(vitals[: r.randint(3, 5)]) + "."


def abdominal_exam(r: random.Random, main: str, own: list[str]) -> str:
    items = [main] + some(r, own, 0, 2) + some(r, COMMON_ABDOMEN, 1, 2)
    return one(r, ["Abdomen soft, with ", "Abdomen with ", "On abdominal exam, "]) + en_list(items) + "."


def labs(r: random.Random, specific: list[tuple[str, str]], wbc: tuple[float, float]) -> str:
    """Lab line in the training format ('Name: value unit, ...'), with general tests shared by the 4 diseases."""
    general = [("Neutrophils", f"{unif(r, 62, 90)} %"), ("Hemoglobin", f"{unif(r, 11.8, 15.6)} g/dL"),
               ("Platelet Count", f"{r.randint(160, 390)}.0 K/uL"), ("Sodium", f"{r.randint(133, 143)}.0 mEq/L"),
               ("Potassium", f"{unif(r, 3.4, 4.8)} mEq/L"), ("Creatinine", f"{unif(r, 0.6, 1.3)} mg/dL"),
               ("Glucose", f"{r.randint(85, 165)}.0 mg/dL"), ("Urea Nitrogen", f"{r.randint(8, 26)}.0 mg/dL")]
    items = [("White Blood Cells", f"{unif(r, *wbc)} K/uL")] + specific + some(r, general, 2, 5)
    r.shuffle(items)
    return ", ".join(f"{name}: {value}" for name, value in items)


def normal_lipase(r): return ("Lipase", f"{r.randint(12, 58)}.0 IU/L")
def normal_alt(r): return ("Alanine Aminotransferase (ALT)", f"{r.randint(10, 42)}.0 IU/L")
def normal_ast(r): return ("Aspartate Aminotransferase (AST)", f"{r.randint(10, 38)}.0 IU/L")
def normal_bili(r): return ("Bilirubin, Total", f"{unif(r, 0.3, 1.2)} mg/dL")
def normal_alp(r): return ("Alkaline Phosphatase", f"{r.randint(45, 115)}.0 IU/L")
def crp(r, a, b): return ("C-Reactive Protein", f"{r.randint(a, b)}.0 mg/L")


def opening(r: random.Random, p: dict, pmh: str, dur: str, specific_pain: str, generic_pain: str) -> tuple[str, bool]:
    """First history sentence. With P_GENERIC_OPENING the pain is not localized here (a later sentence does it)."""
    generic = r.random() < P_GENERIC_OPENING
    pain = generic_pain if generic else specific_pain
    template = one(r, ["{a}-year-old {w} with {h} presents with {d} of {pain}.",
                       "{a}-year-old {w} with {h} presenting with {pain} for {d}.",
                       "{a} y/o {w} with {h} who comes in with {d} of {pain}."])
    return template.format(a=p["age"], w=p["who"], h=pmh, d=dur, pain=pain), generic


COMMON_GYN = ["Last menstrual period two weeks ago, denies vaginal discharge.",
              "Regular menses, no vaginal bleeding or discharge.", "Postmenopausal, no vaginal bleeding."]


def build_hpi(r: random.Random, first: str, specific: list[str], woman: bool) -> list[str]:
    """Opening + disease-specific sentences + shared sentences, in shuffled order.

    The gynecologic sentence applies to any woman with any disease (otherwise it would become a shortcut).
    """
    rest = specific + some(r, COMMON_CHARACTER, 1, 2) + some(r, COMMON_NEGATIVES, 1, 3) + some(r, COMMON_SOCIAL, 1, 2)
    if woman and r.random() < 0.6:
        rest.append(one(r, COMMON_GYN))
    r.shuffle(rest)
    return [first] + rest


# ==============================================================================
# APPENDICITIS
# ==============================================================================
def appendicitis_case(r: random.Random):
    p = patient(r, (18, 60), 0.5)
    atypical = r.random() < P_ATYPICAL
    site = one(r, ["right flank pain", "suprapubic pain", "diffuse lower abdominal pain"]) if atypical else \
        one(r, ["right lower quadrant pain", "right lower abdominal pain"])
    first, generic = opening(r, p, one(r, COMMON_PMH), duration(r, 6, 60), site,
                             one(r, ["abdominal pain", "lower abdominal pain"]))
    specific = [one(r, ["The pain began around the umbilicus and migrated to the right lower quadrant.",
                        "It started as vague periumbilical discomfort and is now in the right lower abdomen.",
                        "The pain started periumbilically and moved to the right side over several hours."]
                       if not atypical else
                       ["The pain is poorly localized.", "The pain started in the lower abdomen and is now worse on the right.",
                        "The pain is mostly in the right flank."]),
                associated(r, ["anorexia that preceded the vomiting", "pain with bumps in the car"])]
    if generic and not atypical:
        specific.append(one(r, ["The pain is now localized to the right lower quadrant.",
                                "It is now sharpest in the right lower quadrant."]))

    main = one(r, ["right lower quadrant tenderness", "focal tenderness in the right lower quadrant"]) if not atypical \
        else one(r, ["right lower quadrant and suprapubic tenderness", "mild right flank tenderness",
                     "diffuse lower abdominal tenderness, worse on the right"])
    pe = [vital_signs(r, (36.9, 38.7), (80, 118), (104, 140)), *some(r, GENERAL_EXAM, 1, 2),
          abdominal_exam(r, main, ["tenderness at McBurney's point", "rebound tenderness", "positive Rovsing sign",
                                   "positive psoas sign", "positive obturator sign"])]
    lab = labs(r, [normal_lipase(r), normal_alt(r), normal_bili(r)] + some(r, [crp(r, 8, 140), normal_alp(r), normal_ast(r)], 0, 2),
               (8.5, 19.5))
    return first, specific, pe, lab


# ==============================================================================
# CHOLECYSTITIS
# ==============================================================================
def cholecystitis_case(r: random.Random):
    p = patient(r, (25, 85), 0.68)
    atypical = r.random() < P_ATYPICAL
    site = one(r, ["epigastric pain", "upper abdominal pain"]) if atypical else \
        one(r, ["right upper quadrant pain", "right upper quadrant and epigastric pain",
                "constant right upper quadrant pain", "constant right upper quadrant and epigastric pain"])
    pmh = one(r, COMMON_PMH + ["known gallstones", "obesity and known gallstones", "obesity and type 2 diabetes"])
    first, generic = opening(r, p, pmh, duration(r, 10, 110), site, one(r, ["abdominal pain", "upper abdominal pain"]))
    specific = [one(r, ["The pain began a few hours after a fatty meal.", "Symptoms started after a fried dinner.",
                        "The pain started after a large meal and has not resolved.", "Onset was gradual over the evening."]),
                one(r, ["It radiates to the right shoulder.", "It radiates to the right scapula.",
                        "It radiates around to the back.", "It does not radiate."]),
                associated(r, ["fevers after the pain started", "intolerance to fatty food"]),
                one(r, [f"{p['pronoun']} reports similar but shorter episodes after fatty meals over the past months.",
                        "Prior episodes of postprandial pain that resolved on their own.", "No prior similar episodes."])]
    if generic and not atypical:
        specific.append(one(r, ["The pain is now mostly under the right ribs.", "It is now localized to the right upper quadrant."]))
    specific.append(one(r, ["No jaundice, no change in stool or urine color.", "Denies dark urine or pale stools.",
                            "Family noticed slight yellowing of the eyes.", "Denies alcohol use."]))

    main = "epigastric and right upper quadrant tenderness" if atypical else "right upper quadrant tenderness"
    pe = [vital_signs(r, (37.0, 38.9), (78, 112), (110, 156)), *some(r, GENERAL_EXAM, 1, 2),
          abdominal_exam(r, main, ["positive Murphy's sign", "inspiratory arrest on deep palpation",
                                   "a palpable tender gallbladder"]),
          one(r, ["No scleral icterus.", "Mild scleral icterus.", "Skin without jaundice."])]
    lipase = r.randint(60, 190) if atypical and r.random() < 0.5 else r.randint(12, 60)  # mimics pancreatitis
    lab = labs(r, [("Alkaline Phosphatase", f"{r.randint(80, 260)}.0 IU/L"),
                   ("Alanine Aminotransferase (ALT)", f"{r.randint(22, 210)}.0 IU/L"),
                   ("Aspartate Aminotransferase (AST)", f"{r.randint(18, 180)}.0 IU/L"),
                   ("Bilirubin, Total", f"{unif(r, 0.5, 3.2)} mg/dL"), ("Lipase", f"{lipase}.0 IU/L")], (8.5, 17.5))
    return first, specific, pe, lab


# ==============================================================================
# PANCREATITIS
# ==============================================================================
def pancreatitis_case(r: random.Random):
    cause = r.choices(["alcohol", "biliary", "triglyceride", "other"], weights=[50, 33, 10, 7])[0]
    p = patient(r, (24, 82), 0.3 if cause == "alcohol" else 0.6)
    atypical = r.random() < P_ATYPICAL
    pmh = {"alcohol": one(r, ["alcohol use disorder", "daily alcohol use", "a history of hypertension"]),
           "biliary": one(r, ["known gallstones", "obesity", "a history of type 2 diabetes"]),
           "triglyceride": one(r, ["hypertriglyceridemia", "poorly controlled type 2 diabetes"]),
           "other": one(r, COMMON_PMH)}[cause]
    site = one(r, ["right upper quadrant pain", "diffuse abdominal pain"]) if atypical else \
        one(r, ["severe epigastric pain", "sudden epigastric pain", "constant epigastric pain"])
    first, generic = opening(r, p, pmh, duration(r, 4, 72), site, one(r, ["abdominal pain", "upper abdominal pain"]))
    specific = [one(r, ["It does not clearly radiate.", "It radiates to the right side."] if atypical else
                       ["The pain radiates straight through to the back.", "It bands across the upper abdomen to the back.",
                        "It radiates to the back."]),
                one(r, ["It is worse lying flat and improves when leaning forward.", "Leaning forward gives some relief.",
                        "The pain is constant and boring."]),
                associated(r, ["persistent vomiting", "inability to keep fluids down"]),
                one(r, ["No prior similar episodes.", "Had a similar episode two years ago."])]
    if generic and not atypical:
        specific.append(one(r, ["The pain is centered in the epigastrium.", "It is now mostly epigastric."]))
    specific.append({"alcohol": one(r, ["Drinks about 8 beers a day.", "Reports a weekend of heavy drinking before onset.",
                                        "Drinks a pint of vodka daily.", "Binge drinking the night before."]),
                     "biliary": one(r, ["Denies alcohol use.", "Rarely drinks alcohol.", "The pain began after a large fatty meal."]),
                     "triglyceride": one(r, ["Ran out of lipid-lowering medication.", "Does not drink alcohol."]),
                     "other": one(r, ["Recently started a new medication.", "Denies alcohol use."])}[cause])

    main = one(r, ["right upper quadrant and epigastric tenderness", "diffuse upper abdominal tenderness"]) if atypical \
        else one(r, ["marked epigastric tenderness", "epigastric tenderness"])
    pe = [vital_signs(r, (36.8, 38.4), (92, 128), (92, 140)), *some(r, GENERAL_EXAM, 1, 2),
          abdominal_exam(r, main, ["periumbilical tenderness", "decreased bowel sounds"]),
          one(r, ["No scleral icterus.", "Mild scleral icterus.", "Skin without jaundice."])]
    lipase = r.randint(220, 540) if atypical and r.random() < 0.4 else r.randint(600, 6500)  # lipase only mildly raised
    items = [("Lipase", f"{lipase}.0 IU/L"), ("Calcium, Total", f"{unif(r, 7.5, 9.4)} mg/dL"),
             ("Hematocrit", f"{unif(r, 39, 53)} %")]
    if cause == "biliary":
        items += [("Alanine Aminotransferase (ALT)", f"{r.randint(110, 420)}.0 IU/L"),
                  ("Aspartate Aminotransferase (AST)", f"{r.randint(90, 360)}.0 IU/L"),
                  ("Bilirubin, Total", f"{unif(r, 1.2, 3.6)} mg/dL"), ("Alkaline Phosphatase", f"{r.randint(110, 280)}.0 IU/L")]
    else:
        items += [normal_alt(r), normal_bili(r)] + ([normal_ast(r)] if r.random() < 0.5 else [])
    if cause == "triglyceride":
        items.append(("Triglycerides", f"{r.randint(900, 3200)}.0 mg/dL"))
    return first, specific, pe, labs(r, items, (8.5, 20.0))


# ==============================================================================
# DIVERTICULITIS
# ==============================================================================
def diverticulitis_case(r: random.Random):
    p = patient(r, (38, 88), 0.5)
    atypical = r.random() < P_ATYPICAL
    site = one(r, ["right lower quadrant pain", "suprapubic pain", "lower abdominal pain"]) if atypical else \
        one(r, ["left lower quadrant pain", "left-sided lower abdominal pain"])
    pmh = one(r, COMMON_PMH + ["chronic constipation", "hypertension and chronic constipation"])
    first, generic = opening(r, p, pmh, duration(r, 30, 140), site, one(r, ["abdominal pain", "lower abdominal pain"]))
    specific = [associated(r, ["low-grade fevers", "nausea without vomiting"]),
                one(r, ["Constipated for the past 2 days.", "Change in bowel habits with intermittent loose stools.",
                        "No bowel movement for 3 days.", "Bowel movements are normal."]),
                one(r, ["A similar milder episode last year resolved with oral antibiotics.",
                        "Two prior similar episodes treated as an outpatient.", "No prior similar episodes.",
                        "A colonoscopy years ago showed small outpouchings in the sigmoid colon."]),
                one(r, ["Mild urinary frequency.", "Denies dysuria.", "No urinary symptoms."])]
    if generic and not atypical:
        specific.append(one(r, ["The pain is now localized to the left lower quadrant.",
                                "It is now worst in the left lower abdomen."]))

    main = one(r, ["right lower quadrant tenderness", "suprapubic tenderness", "lower abdominal tenderness"]) if atypical \
        else "left lower quadrant tenderness"
    pe = [vital_signs(r, (36.9, 38.7), (76, 110), (118, 168)), *some(r, GENERAL_EXAM, 1, 2),
          abdominal_exam(r, main, ["localized guarding", "a vague tender fullness in the lower abdomen"]),
          one(r, ["Rectal exam without gross blood.", "Bowel sounds present."])]
    lab = labs(r, [crp(r, 30, 230), normal_lipase(r)] + some(r, [normal_alt(r), normal_ast(r), normal_bili(r), normal_alp(r)], 0, 2),
               (9.5, 18.5))
    return first, specific, pe, lab


GENERATORS = {"appendicitis": appendicitis_case, "cholecystitis": cholecystitis_case,
              "pancreatitis": pancreatitis_case, "diverticulitis": diverticulitis_case}
# Disease most often confused with each one (lower pain × lower pain; upper pain × upper pain)
CONFUSED_WITH = {"appendicitis": "diverticulitis", "diverticulitis": "appendicitis",
                 "cholecystitis": "pancreatitis", "pancreatitis": "cholecystitis"}
P_MIMIC = 0.35  # cases that get 1–2 history sentences typical of the disease they are confused with
# Sentences that cannot be borrowed: findings that would contradict the exam (jaundice) or menstruation
_EXCLUDED_BORROW = ("menstrual", "menses", "yellowing", "jaundice")


def borrowed_sentences(r: random.Random, disease: str, woman: bool) -> list[str]:
    """1–2 specific sentences from a case of the confusable disease (adapting pronouns to the patient)."""
    _, specific, _, _ = GENERATORS[CONFUSED_WITH[disease]](random.Random(r.random()))
    pronoun = "She " if woman else "He "
    candidates = [s.replace("She ", pronoun).replace("He ", pronoun)
                  for s in specific if not any(t in s for t in _EXCLUDED_BORROW)]
    return some(r, candidates, 1, 2) if candidates else []


# ==============================================================================
# PARTIAL CASES AND CSVs
# ==============================================================================
def case_rows(r: random.Random, disease: str, n: int, split: str) -> list[dict]:
    first, specific, pe_sentences, lab = GENERATORS[disease](r)
    woman = " woman " in first
    mimic = r.random() < P_MIMIC
    if mimic:
        specific = specific + borrowed_sentences(r, disease, woman)
    hpi_sentences = build_hpi(r, first, specific, woman=woman)
    k = max(2, round(len(hpi_sentences) * r.uniform(0.3, 0.6)))
    hpi_full, pe_full = " ".join(hpi_sentences), " ".join(pe_sentences)
    found = {kind: terms for kind, terms in inf.audit_text("\n".join([hpi_full, pe_full, lab])).items()
             if terms and kind != "organ_name"}  # organ names are legitimate findings ("tender gallbladder")
    if found:
        raise ValueError(f"case {disease} {n}: terms the model must not see: {found}")
    versions = {"hpi_partial": (" ".join(hpi_sentences[:k]), "", ""), "hpi": (hpi_full, "", ""),
                "hpi_pe": (hpi_full, pe_full, ""), "complete": (hpi_full, pe_full, lab)}
    case_id = f"{disease[:4]}_{n:04d}"
    out = []
    for order, stage in enumerate(STAGES, 1):
        hpi, pe, lb = versions[stage]
        out.append({"row_id": f"{case_id}_{order}", "case_id": case_id, "disease": disease,
                    "label_id": LABEL_ID[disease], "stage": stage, "stage_order": order, "split": split,
                    "mimics": CONFUSED_WITH[disease] if mimic else "",
                    "hpi": hpi, "pe": pe, "lab": lb,
                    "model_text": inf.build_text(hpi, pe, lb),
                    "n_words": len(" ".join([hpi, pe, lb]).split())})
    return out


def generate() -> dict[str, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_val = n_test = round(BASE_CASES_PER_DISEASE * 0.15)
    tables = {}
    for disease in inf.CLASSES:
        r = random.Random(f"{SEED}-{disease}")  # one seed per disease: a CSV does not change if another one does
        splits = ["val"] * n_val + ["test"] * n_test + ["train"] * (BASE_CASES_PER_DISEASE - n_val - n_test)
        r.shuffle(splits)
        rows = [row for n, s in enumerate(splits, 1) for row in case_rows(r, disease, n, s)]
        df = pd.DataFrame(rows)
        path = OUTPUT_DIR / f"partial_cases_{disease}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        tables[disease] = df
        print(f"{disease:15s} {len(df)} rows ({df['case_id'].nunique()} base cases) -> {path}")
    return tables


if __name__ == "__main__":
    generate()
