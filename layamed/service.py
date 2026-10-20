"""Analysis of a case while it is being written: the logic that the local server (server.py) exposes to the interface.

Takes the 3 sections as the user wrote them and returns a plain-data dictionary (no HTML): state (empty / waiting
for minimum context / out of scope / result), recalibrated and original probabilities, case understanding,
management, token budget, warnings about terms never seen in training and the exact text Laya reads.

Order: minimum context (tokenizer only) → out-of-scope detector (Laya encoder + clinical vocabulary, scope.py) →
Laya + recalibration (recalibration.py). A text that does not look like a clinical case gets no probability at all.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import inference as inf
from .recalibration import Recalibrator
from .scope import ScopeDetector

LABELS = {c: c.capitalize() for c in inf.CLASSES}  # Appendicitis, Cholecystitis, ...

# Minimum context to switch inference on (record tokens, counting titles and "Not documented."; the empty case has
# ~32). 100 tokens ≈ 50 typed words: with less text the scope detector cannot separate clinical text from any text.
# Below the minimum, the model does not run and no probability is shown.
MIN_CONTEXT_TOKENS = 100
# Case understanding (0–100%), which multiplies the probabilities in the bars so they start near 1% and grow:
#   size          = (tokens − minimum) / (260 − minimum), clipped to [0, 1]
#   sections      = (1 + exam + labs) / 3, each section full at 30 tokens
#   understanding = max(1%, size × sections)
# It is not a probability: it is a display scale, explained on screen. The probability is shown on the card and on hover.
FULL_CASE_TOKENS = 260
FULL_SECTION_TOKENS = 30
MIN_UNDERSTANDING = 0.01


class Service:
    """Loads Laya, the scope detector and the recalibration once; `analyze` is safe for concurrent calls (one at a
    time on the GPU)."""

    def __init__(self, root: Path):
        root = Path(root)
        print("Loading Laya...")
        t0 = time.time()
        self.model = inf.LayaInference(inf.find_checkpoint(root), batch_size=8)
        self.model.predict([inf.build_text("", "", "")])  # warm-up (GPU kernels and allocator)
        print(f"Laya ready in {time.time() - t0:.1f}s | device: {self.model.device}")
        # Recalibration by the amount of text written (recalibration.py): moderates the output while the case is
        # incomplete. Without the file, the original Laya output is used.
        f = root / "data" / "recalibration.json"
        self.recal = Recalibrator.load(f) if f.exists() else None
        print("Recalibration:", f"on ({f.name})" if self.recal else "off (optional: notebook, part 6)")
        # Out-of-scope detector (scope.py). Without it the server refuses to start: any text would get a probability,
        # which is exactly the error it prevents.
        f = root / "data" / "scope.pt"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}: build the out-of-scope detector (notebook, part 5)")
        self.scope = ScopeDetector.load(f, device=self.model.agent.device)
        print(f"Scope detector: distance <= {self.scope.limit:.1f} and >= {self.scope.min_terms} clinical term(s)")
        self._lock = threading.Lock()

    def understanding(self, n_tok: int, pe: str, lab: str) -> float:
        """Case understanding, from 1% to 100% (see the constants above)."""
        size = min(max((n_tok - MIN_CONTEXT_TOKENS) / (FULL_CASE_TOKENS - MIN_CONTEXT_TOKENS), 0.0), 1.0)
        full = {}
        for section, text in (("pe", pe), ("lab", lab)):
            cleaned = inf.clean_section(text, section)
            n = 0 if cleaned == inf.EMPTY_SECTION[section] else self.model.count_tokens(cleaned)
            full[section] = min(n / FULL_SECTION_TOKENS, 1.0)
        return max(MIN_UNDERSTANDING, size * (1 + full["pe"] + full["lab"]) / 3)

    def analyze(self, hpi: str, pe: str, lab: str) -> dict:
        """Builds the text in the training format and, with minimum context, runs Laya."""
        hpi, pe, lab = hpi or "", pe or "", lab or ""
        text = inf.build_text(hpi, pe, lab)
        out = {"model_text": text, "warnings": inf.audit_text("\n".join([hpi, pe, lab])),
               "min_tokens": MIN_CONTEXT_TOKENS, "recalibrated": self.recal is not None}
        if not any(s.strip() for s in (hpi, pe, lab)):
            return {**out, "state": "empty", "record_tokens": 0}

        with self._lock:
            # Counting tokens only uses the tokenizer (no GPU); Laya stays off until the minimum context
            bud = self.model.budget(text)["diagnosis"]
            n_tok = bud["record_tokens_used"]
            if n_tok < MIN_CONTEXT_TOKENS:
                return {**out, "state": "waiting", "record_tokens": n_tok}
            t0 = time.time()
            scope = self.scope.evaluate(self.model.embeddings([text]), [text])[0]
            if scope["out_of_scope"]:
                return {**out, "state": "out_of_scope", "record_tokens": n_tok, "scope": scope}
            P, p_surg = self.model.predict([text])
            ms = (time.time() - t0) * 1000
            und = self.understanding(n_tok, pe, lab)

        original = dict(zip(inf.CLASSES, map(float, P[0])))
        probs = dict(zip(inf.CLASSES, map(float, self.recal.predict(P, [n_tok], [text])[0]))) if self.recal \
            else original
        return {**out, "state": "result", "record_tokens": n_tok, "scope": scope,
                "probabilities": probs, "original": original, "understanding": und,
                "evidence": {c: p * und for c, p in probs.items()},
                "leader": max(probs, key=probs.get), "original_leader": max(original, key=original.get),
                "p_surgical": float(p_surg[0]), "budget": bud, "ms": ms}


# ==============================================================================
# EXAMPLE CASES (hand-written, fictional). The last one starts as cholecystitis (history and exam) and should turn
# into pancreatitis when the lipase appears in the labs.
# ==============================================================================
EXAMPLES = [
    ["19-year-old woman with no past medical history presents with 18 hours of abdominal pain. The pain started around "
     "the umbilicus and overnight moved to the right lower quadrant, where it is now sharp and constant. Associated "
     "with loss of appetite, nausea and two episodes of non-bloody vomiting. Reports subjective fevers. The pain "
     "worsens with walking and coughing. Last menstrual period two weeks ago, denies vaginal discharge. No urinary "
     "symptoms, no diarrhea.",
     "T 38.0 C, HR 102, BP 118/72, RR 16. Uncomfortable, lying still. Abdomen soft, non-distended, focal tenderness in "
     "the right lower quadrant with rebound and voluntary guarding. Positive Rovsing and psoas signs. No "
     "costovertebral angle tenderness.",
     "White Blood Cells: 14.6 K/uL, Neutrophils: 84.0 %, Hemoglobin: 12.9 g/dL, Lipase: 28.0 IU/L, "
     "Alanine Aminotransferase (ALT): 18.0 IU/L, Bilirubin, Total: 0.5 mg/dL, Creatinine: 0.7 mg/dL"],
    ["58-year-old woman with obesity and type 2 diabetes presents with 2 days of constant right upper quadrant and "
     "epigastric pain that began a few hours after a fried dinner. The pain radiates to the right shoulder blade and "
     "did not improve with antacids. Associated with nausea, one episode of vomiting and chills. She reports similar "
     "but shorter episodes after fatty meals over the past 6 months. No jaundice, no change in stool or urine color.",
     "T 38.3 C, HR 98, BP 142/86. Mild distress. Abdomen soft, tender in the right upper quadrant with inspiratory "
     "arrest on deep palpation (positive Murphy's sign). No rebound. Bowel sounds present.",
     "White Blood Cells: 13.8 K/uL, Alkaline Phosphatase: 165.0 IU/L, Alanine Aminotransferase (ALT): 74.0 IU/L, "
     "Aspartate Aminotransferase (AST): 61.0 IU/L, Bilirubin, Total: 1.5 mg/dL, Lipase: 52.0 IU/L, "
     "Glucose: 212.0 mg/dL"],
    ["44-year-old man with daily alcohol use, about 8 beers a day for years, presents with sudden severe epigastric "
     "pain that started this morning after a weekend of heavy drinking. The pain is constant and boring, radiates "
     "straight through to the back and improves slightly when leaning forward. Associated with persistent nausea and "
     "repeated vomiting; unable to keep fluids down. No prior similar episodes. Denies black stools or vomiting blood.",
     "T 37.8 C, HR 116, BP 102/64, RR 22. Diaphoretic and ill-appearing. Abdomen mildly distended with marked "
     "epigastric tenderness and voluntary guarding, no rebound. Hypoactive bowel sounds.",
     "Lipase: 2450.0 IU/L, White Blood Cells: 15.9 K/uL, Hematocrit: 49.0 %, Urea Nitrogen: 28.0 mg/dL, "
     "Calcium, Total: 7.9 mg/dL, Triglycerides: 210.0 mg/dL, Alanine Aminotransferase (ALT): 45.0 IU/L, "
     "Bilirubin, Total: 1.0 mg/dL"],
    ["71-year-old man with hypertension and chronic constipation presents with 3 days of progressive left lower "
     "quadrant pain, constant and cramping. Associated with low-grade fevers, decreased appetite and no bowel movement "
     "for 2 days. A similar but milder episode last year resolved with oral antibiotics. Denies blood in the stool, "
     "vomiting or urinary symptoms. A colonoscopy 5 years ago showed multiple small outpouchings in the sigmoid colon.",
     "T 38.1 C, HR 94, BP 148/88. Abdomen soft, mildly distended, tender in the left lower quadrant with localized "
     "guarding, no rebound. No palpable mass. Bowel sounds present.",
     "White Blood Cells: 14.2 K/uL, Neutrophils: 82.0 %, C-Reactive Protein: 104.0 mg/L, Lipase: 30.0 IU/L, "
     "Creatinine: 1.1 mg/dL, Hemoglobin: 13.6 g/dL"],
    ["52-year-old woman with known gallstones presents with 10 hours of upper abdominal "
     "pain that began after a fatty dinner. The pain is severe and associated with nausea and several episodes of "
     "vomiting. She had similar but shorter episodes after meals over the past few months. No fevers at home. "
     "Denies alcohol use. No prior abdominal surgery.",
     "T 37.6 C, HR 108, BP 128/80, RR 20. Uncomfortable, sitting up and leaning forward. Upper abdominal "
     "tenderness with voluntary guarding. Murphy's sign equivocal. The pain radiates straight to the back. "
     "Mild scleral icterus. Hypoactive bowel sounds.",
     "White Blood Cells: 13.9 K/uL, Alanine Aminotransferase (ALT): 196.0 IU/L, Aspartate Aminotransferase (AST): "
     "152.0 IU/L, Alkaline Phosphatase: 188.0 IU/L, Bilirubin, Total: 2.6 mg/dL, Calcium, Total: 8.4 mg/dL, "
     "Lipase: 2280.0 IU/L"],
]
EXAMPLE_LABELS = ["Appendicitis", "Cholecystitis", "Pancreatitis", "Diverticulitis", "Biliary → pancreatic"]
