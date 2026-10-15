"""Re-linking extracted claims after a corpus rebuild split long sections (fix of 2026-10-15): claim ids stay, the
section index follows the claim's verbatim span into the chunk that now holds it."""

from data.schema import Section
from packs.schema_induction import align_sections, relink_claims


def secs(doc, texts):
    return [Section(doc_id=doc, idx=i, heading="H", text=t) for i, t in enumerate(texts)]


OLD = secs("d", ["Abstract text here.", "First part of results.\nSecond part of results.\nThird part.", "Conclusion."])
NEW = secs("d", ["Abstract text here.", "First part of results.", "Second part of results.\nThird part.", "Conclusion."])


def claim(idx, span, k=0, grounded=True):
    return {"claim_id": f"d#{idx}#{k}", "doc_id": "d", "section_idx": idx, "span": span, "span_grounded": grounded}


def test_align_sections_maps_each_old_section_to_its_chunks():
    assert align_sections(OLD, NEW) == {0: [0], 1: [1, 2], 2: [3]}


def test_align_sections_refuses_a_changed_text():
    changed = secs("d", ["Abstract text here.", "Something else entirely.", "Conclusion."])
    try:
        align_sections(OLD, changed)
    except ValueError as e:
        assert "d" in str(e)
    else:
        raise AssertionError("a changed section text must not be aligned silently")


def test_relink_claims_follows_the_span():
    claims = [claim(0, "Abstract text here."), claim(1, "second part of RESULTS."), claim(1, "Third part.", k=1),
              claim(1, "First part of results.", k=2), claim(1, "results. Second part", k=3),
              claim(1, "not in the section at all", k=4, grounded=False), claim(2, "Conclusion.")]
    out, counts = relink_claims(claims, {"d": OLD}, {"d": NEW})
    assert [c["claim_id"] for c in out] == [c["claim_id"] for c in claims]  # ids and order unchanged
    assert [c["section_idx"] for c in out] == [0, 2, 2, 1, 1, 1, 3]
    assert [c["section_idx_unsplit"] for c in out] == [0, 1, 1, 1, 1, 1, 2]
    assert [c["relink"] for c in out] == ["same_section", "span", "span", "span", "span_across_chunks",
                                          "first_chunk", "same_section"]
    assert counts == {"same_section": 2, "span": 3, "span_across_chunks": 1, "first_chunk": 1}
    assert claims[1]["section_idx"] == 1  # input rows are not modified


def test_relink_stage_end_to_end(tmp_path):
    import json

    from data.schema import write_jsonl
    from experiments import induce
    from experiments.config import load_config

    old_path, new_dir = tmp_path / "old_sections.jsonl", tmp_path / "processed" / "papers"
    new_dir.mkdir(parents=True)
    write_jsonl(old_path, OLD)
    write_jsonl(new_dir / "sections.jsonl", NEW + secs("other", ["Not a claim document."]))
    write_jsonl(tmp_path / "claims.jsonl", [claim(1, "Third part."), claim(2, "Conclusion.", k=1)])
    cfg = load_config("configs/induce.yaml", ["smoke"])
    cfg["pack"], cfg["paths"]["data_processed"] = "papers", str(tmp_path / "processed")
    cfg["induce"].update(stage="relink", relink={"old_sections": {"papers": str(old_path)}})
    cfg["induce"]["cluster"]["claims"]["papers"] = str(tmp_path / "claims.jsonl")
    metrics = induce.main(cfg, tmp_path)
    rows = [json.loads(line) for line in open(tmp_path / "claims_papers.jsonl")]
    assert [r["section_idx"] for r in rows] == [2, 3]
    assert metrics["papers/claims_index_changed"] == 2 and metrics["papers/relink_span"] == 1
