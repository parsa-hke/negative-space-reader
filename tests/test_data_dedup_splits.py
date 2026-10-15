"""data/dedup.py, data/splits.py, data/revisions.py (plan §2.6, §2.6b, §2.7)."""

import json

import pytest

from data.dedup import Version, collapse_chain, exact_key, near_duplicate_clusters, signature
from data.revisions import chain_pairs
from data.schema import Document, ParsedDocument, Section, with_char_spans
from data.splits import (assign_split, load_manifest, manifest_hash, resolve_cross_split_duplicates,
                         write_manifest)

BASE = " ".join(f"clause{i} says the company handles personal data under rule {i} with care." for i in range(120))
EDITED = BASE.replace("rule 7 ", "rule seven ")
ADDED = BASE + " " + " ".join(f"new clause{i} on data retention periods and deletion schedule {i}." for i in range(25))
OTHER = " ".join(f"unrelated text{i} about cooking recipes and garden plants {i}." for i in range(120))


def test_exact_key_ignores_format():
    assert exact_key("We **keep** data.\n\n") == exact_key("we keep   DATA")


def test_near_duplicate_clusters():
    sigs = {k: signature(t) for k, t in {"a": BASE, "b": EDITED, "c": OTHER}.items()}
    cl = near_duplicate_clusters(sigs, threshold=0.85)
    assert cl["a"] == cl["b"] != cl["c"]


def test_collapse_chain_merges_near_identical_consecutive_versions():
    chain = collapse_chain([Version("d3", "2020", ADDED), Version("d1", "2018", BASE), Version("d2", "2019", EDITED)])
    assert [(v.doc_id, n) for v, n, _ in chain] == [("d1", 2), ("d3", 1)]
    assert 0.5 < chain[1][2] < 0.95


def pdoc(doc_id, date, text, entity="e.com"):
    d = Document(id=doc_id, genre="privacy", source="t", entity_id=entity, date=date)
    return ParsedDocument(d, with_char_spans([Section(doc_id=doc_id, idx=0, heading=None, text=text)]))


def test_chain_pairs_material_changes_only():
    pairs = list(chain_pairs([pdoc("a", "2018-01-01", BASE), pdoc("b", "2019-01-01", EDITED),
                              pdoc("c", "2020-01-01", ADDED), pdoc("d", "2021-01-01", OTHER)]))
    # a->b cosmetic (collapsed), (a|b)->c material, c->d a different document (above max_change)
    assert [(p.old_doc_id, p.new_doc_id) for p in pairs] == [("a", "c")]
    assert pairs[0].old_date == "2018-01-01" and 0.05 <= pairs[0].change_ratio <= 0.9


def test_assign_split_is_deterministic_and_respects_fractions():
    fr = {"train": 0.8, "dev": 0.1, "test": 0.1}
    splits = [assign_split(f"site{i}.com", seed=1234, fractions=fr) for i in range(5000)]
    assert splits == [assign_split(f"site{i}.com", seed=1234, fractions=fr) for i in range(5000)]
    assert abs(splits.count("train") / 5000 - 0.8) < 0.03 and abs(splits.count("test") / 5000 - 0.1) < 0.02
    assert assign_split("site3.com", seed=1234, fractions=fr, forced={"site3.com": "dev"}) == "dev"
    assert splits != [assign_split(f"site{i}.com", seed=99, fractions=fr) for i in range(5000)]


def test_cross_split_duplicates_never_leave_a_test_twin():
    cluster = {"t1": "g", "tr1": "g", "d1": "g", "tr2": "h", "d2": "h", "tr3": "i"}
    split = {"t1": "test", "tr1": "train", "d1": "dev", "tr2": "train", "d2": "dev", "tr3": "train"}
    assert resolve_cross_split_duplicates(cluster, split) == {"tr1", "d1", "tr2"}


def test_manifest_seals_test_and_rejects_entity_overlap(tmp_path):
    doc_split = {"a1": "train", "b1": "test", "c1": "dev"}
    entity = {"a1": "a", "b1": "b", "c1": "c"}
    summary = write_manifest(tmp_path / "m.json", genre="privacy", seed=1, fractions={}, doc_split=doc_split,
                             entity_of=entity)
    assert summary["test_sha256"] == manifest_hash(["b1"])
    assert load_manifest(tmp_path / "m.json") == doc_split
    assert json.loads((tmp_path / "m.json").read_text())["documents"] == {"train": 1, "dev": 1, "test": 1}
    with pytest.raises(ValueError, match="test and another split"):
        write_manifest(tmp_path / "bad.json", genre="privacy", seed=1, fractions={},
                       doc_split={"a1": "train", "a2": "test"}, entity_of={"a1": "a", "a2": "a"})
