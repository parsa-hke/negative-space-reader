"""experiments/induce.py candidate (3.4 read-back), coverage (3.5) and freeze (3.6) stages, with a fake teacher."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from data.schema import Document, Section, write_jsonl
from experiments import induce
from experiments.config import load_config
from models.teacher import PROMPTS, Teacher
from packs.schema_induction import coverage_output_model, render_coverage_prompt, schema_fingerprint

TYPES = [{"id": "retention_period", "name": "retention_period", "definition": "States how long data is kept.",
          "positive_examples": ["We keep data for a year.", "Data is kept for 30 days."],
          "near_miss": "We delete data on request.", "severity_weight": 1.5, "section_hint": "middle",
          "label_source": ["Data Retention"], "source_clusters": ["c000"]},
         {"id": "contact_channel", "name": "contact_channel", "definition": "Tells readers how to contact the author.",
          "positive_examples": ["Email us.", "Write to us."], "near_miss": "We may email you.", "severity_weight": 1.0,
          "section_hint": "end", "label_source": ["Other :: Other Type = Privacy contact information"],
          "source_clusters": ["c001"]}]


class FakeClassifier(Teacher):
    """First pass: 'keep' -> retention_period, except backups (a miss: "none"); 'contact' -> contact_channel;
    'odd' -> none. Second look: a statement about keeping things -> retention_period when offered, else none."""

    backend = "fake"

    def __init__(self):
        super().__init__("fake-classifier", max_retries=0)
        self.verify_prompts = []

    def _complete(self, conversations, *, json_schema, max_tokens):
        outs = []
        for conv in conversations:
            p = conv[-1]["content"]
            if "Candidate types" in p:
                self.verify_prompts.append(p)
                statement = p.split("Statement: ", 1)[1].split("\n", 1)[0]
                fits = "keeps" in statement and "retention_period:" in p
                outs.append(json.dumps({"type": "retention_period" if fits else "none"}))
                self._served()
                continue
            text = p.split('Section text:\n"""\n', 1)[1].lower()
            a = []
            if "backups" in text:
                a.append({"claim": "The company keeps backups for a while.", "type": "none"})
            elif "keep" in text:
                a.append({"claim": "The company keeps data for a period.", "type": "retention_period"})
            if "contact" in text:
                a.append({"claim": "The company gives a contact channel.", "type": "contact_channel"})
            if "odd" in text:
                a.append({"claim": "Something no type covers.", "type": "none"})
            outs.append(json.dumps({"assertions": a}))
            self._served()
        return outs


class FakeEmbedder:
    def __init__(self, model_id, **kw):
        self.model_id = model_id

    def encode(self, texts, show_progress=False, prompt=None):
        out = []
        for t in texts:
            v = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).normal(0, 1, 8)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


def test_coverage_model_constrains_type_ids():
    M = coverage_output_model(["retention_period", "contact_channel"])
    assert M.model_validate({"assertions": [{"claim": "c", "type": "none"}]})
    for bad in ("made_up", "background_or_filler"):
        with pytest.raises(Exception):
            M.model_validate({"assertions": [{"claim": "c", "type": bad}]})
    p = render_coverage_prompt(Section(doc_id="d", idx=0, heading=None, text="We keep data."),
                               {"document_kind": "policy", "author_role": "the company"}, TYPES, max_assertions=9)
    assert "retention_period: States how long data is kept." in p and "We keep data." in p and "{{" not in p


def test_schema_fingerprint_ignores_provenance():
    a = {"genre": "privacy", "types": TYPES, "seed_mapping": {}, "pruned_clusters": [], "provenance": {"date": "x"}}
    b = {**a, "provenance": {"date": "y", "candidate_run": "other"}, "status": "candidate"}
    assert schema_fingerprint(a) == schema_fingerprint(b)
    c = {**a, "types": [{**TYPES[0], "definition": "changed"}, TYPES[1]]}
    assert schema_fingerprint(c) != schema_fingerprint(a)


@pytest.fixture
def env(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    texts = ["We keep your data for two years.", "Contact us at the address below.", "We keep logs. An odd clause.",
             "Contact the office.", "We keep backups."]
    secs, docs = [], []
    for d in range(10):
        docs.append(Document(id=f"privacy:{d}", genre="privacy", source="s", entity_id=f"e{d}", date="2019-01-01"))
        secs.append(Section(doc_id=f"privacy:{d}", idx=0, heading=None, text=texts[d % 5] * 3))
    write_jsonl(processed / "privacy" / "sections.jsonl", secs)
    write_jsonl(processed / "privacy" / "documents.jsonl", docs)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "privacy.json").write_text(json.dumps(
        {"doc_ids": {"train": [], "dev": [f"privacy:{d}" for d in range(8)], "test": ["privacy:8", "privacy:9"]}}))
    (tmp_path / "sample_privacy.json").write_text(json.dumps(["privacy:0", "privacy:1"]))
    cfg = load_config("configs/induce.yaml")
    cfg["paths"].update(data_processed=str(processed), splits=str(tmp_path / "splits"), packs=str(tmp_path / "packs"))
    cfg["run"]["seed"] = 0
    cfg["pack"] = "privacy"
    cfg["induce"]["coverage"].update(schema=str(tmp_path / "packs" / "{genre}" / "schema_candidate.yaml"),
                                     exclude_sample=str(tmp_path / "sample_{genre}.json"))
    cfg["schema"].update(coverage_check_sections=6, min_types=2)
    candidate = {"genre": "privacy", "status": "candidate", "types": TYPES, "seed_mapping": {"s1": {"types": ["retention_period"]}},
                 "pruned_clusters": [], "provenance": {"workbook": "w.xlsx"}}
    (tmp_path / "packs" / "privacy").mkdir(parents=True)
    (tmp_path / "packs" / "privacy" / "schema_candidate.yaml").write_text(yaml.safe_dump(candidate))
    teacher = FakeClassifier()
    monkeypatch.setattr("models.teacher.load_teacher", lambda *a, **k: teacher)
    monkeypatch.setattr("engine.embed.Embedder", FakeEmbedder)
    return cfg, tmp_path, teacher


def test_coverage_stage_second_look_and_rates(env):
    cfg, tmp, teacher = env
    cfg["induce"]["stage"] = "coverage"
    (tmp / "cov").mkdir()
    PROMPTS.reset_usage()
    m = induce.main(cfg, tmp / "cov")
    rows = [json.loads(l) for l in open(tmp / "cov" / "coverage_privacy.jsonl")]
    used = {r["doc_id"] for r in rows}
    assert len(rows) == 6 and used <= {f"privacy:{d}" for d in range(2, 8)}  # dev, minus the extraction sample
    assert m["privacy/sections"] == 6 and m["privacy/types"] == 2
    lower = [r["text"].lower() for r in rows]
    odd, backups = sum("odd" in t for t in lower), sum("backups" in t for t in lower)
    assert backups and m["privacy/recovered_statements"] == backups  # the first pass missed them; the second look found them
    assert m["privacy/first_pass_sections_with_unmapped"] == odd + backups
    assert m["privacy/sections_with_unmapped"] == odd and m["privacy/unmapped_section_rate"] == round(odd / 6, 3)
    assert m["privacy/first_pass_unmapped_section_rate"] == round((odd + backups) / 6, 3)
    assert m["privacy/passes"] == (odd / 6 < cfg["schema"]["max_unmapped_fraction"])
    assert len(teacher.verify_prompts) == odd + backups and "retention_period:" in teacher.verify_prompts[0]
    rec = next(a for r in rows for a in r["assertions"] if "backups" in a["claim"])
    assert rec["first_type"] == "none" and rec["type"] == "retention_period"
    assert PROMPTS.usage()["induce_coverage"] == PROMPTS.get("induce_coverage").tag
    assert PROMPTS.usage()["induce_coverage_verify"] == PROMPTS.get("induce_coverage_verify").tag
    assert (tmp / "cov" / "unmapped_privacy.csv").exists() and (tmp / "cov" / "schema_checked_privacy.yaml").exists()


def test_freeze_needs_sign_off_and_a_passing_coverage_run_and_never_overwrites(env):
    cfg, tmp, _ = env
    cfg["induce"]["stage"] = "coverage"
    (tmp / "cov").mkdir()
    m = {"metrics": induce.main(cfg, tmp / "cov")}
    (tmp / "cov" / "metrics.json").write_text(json.dumps(m))  # experiments/run.py writes this for real runs
    cfg["induce"]["stage"] = "freeze"
    cfg["induce"]["freeze"].update(coverage_run=str(tmp / "cov"), signed_off_by=None)
    (tmp / "f1").mkdir()
    with pytest.raises(ValueError, match="sign-off"):
        induce.main(cfg, tmp / "f1")
    cfg["induce"]["freeze"]["signed_off_by"] = "Reviewer, 2026-10-13"
    if not m["metrics"]["privacy/passes"]:
        with pytest.raises(ValueError, match="coverage"):
            induce.main(cfg, tmp / "f1")
        cfg["schema"]["max_unmapped_fraction"] = 1.0
        m["metrics"]["privacy/passes"] = True
        (tmp / "cov" / "metrics.json").write_text(json.dumps(m))
    # a candidate whose content changed after its coverage run cannot be frozen on that run; provenance may change
    cand = tmp / "packs" / "privacy" / "schema_candidate.yaml"
    original = cand.read_text()
    cand.write_text(original.replace("States how long data is kept.", "States how long data is kept, and why."))
    with pytest.raises(ValueError, match="changed since"):
        induce.main(cfg, tmp / "f1")
    cand.write_text(original.replace("w.xlsx", "rebuilt.xlsx"))
    out = induce.main(cfg, tmp / "f1")
    schema = yaml.safe_load((tmp / "packs" / "privacy" / "schema.yaml").read_text())
    assert schema["version"] == 1 and schema["status"] == "frozen" and len(schema["types"]) == 2
    assert schema["frozen"]["signed_off_by"] == "Reviewer, 2026-10-13" and schema["frozen"]["coverage_run"] == "cov"
    assert out["privacy/types"] == ["retention_period", "contact_channel"]
    (tmp / "f2").mkdir()
    with pytest.raises(FileExistsError):
        induce.main(cfg, tmp / "f2")
