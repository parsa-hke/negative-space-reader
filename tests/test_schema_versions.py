"""packs/schema_versions.py: a new pack version that merges types, and v1 labels read through it."""

import pytest

from packs.schema_versions import merge_types, project_labels, projection

V1 = {"genre": "g", "version": 1, "status": "frozen",
      "types": [{"id": "a", "definition": "A.", "source_clusters": ["c1"], "severity_weight": 1.0},
                {"id": "a2", "definition": "A again.", "source_clusters": ["c2"], "severity_weight": 1.5},
                {"id": "b", "definition": "B.", "source_clusters": ["c3"], "severity_weight": 1.0},
                {"id": "c", "definition": "C.", "source_clusters": ["c4"], "severity_weight": 1.0}],
      "seed_mapping": {"s1": {"types": ["a2", "c"]}, "s2": {"exclude": "post-dates the corpus"}}}

MERGES = [{"id": "a", "from": ["a", "a2"], "card": {"definition": "A, merged.", "positive_examples": ["x", "y"],
                                                     "near_miss": "z", "section_hint": "early"}},
          {"id": "bc", "from": ["b", "c"], "card": {"definition": "B or C.", "positive_examples": ["p", "q"],
                                                     "near_miss": "r", "section_hint": "late"}}]


def test_merge_types_builds_the_next_version():
    v2 = merge_types(V1, MERGES, reason="neighbours", decided_by="P", date="2026-10-18", v1_file="h/v1.yaml", v1_sha1="abc")
    assert v2["version"] == 2 and v2["status"] == "frozen" and [t["id"] for t in v2["types"]] == ["a", "bc"]
    a, bc = v2["types"]
    assert a["merged_from"] == ["a", "a2"] and a["source_clusters"] == ["c1", "c2"] and a["severity_weight"] == 1.5
    assert a["definition"] == "A, merged." and bc["merged_from"] == ["b", "c"]
    assert v2["seed_mapping"]["s1"]["types"] == ["a", "bc"] and v2["seed_mapping"]["s2"] == {"exclude": "post-dates the corpus"}
    assert v2["derived_from"] == {"version": 1, "file": "h/v1.yaml", "sha1": "abc"}
    assert v2["changes"][0]["merged"] == ["a", "a2"] and v2["frozen"]["decided_by"] == "P"
    assert V1["types"][0]["definition"] == "A."  # the input is not modified


def test_merge_types_refuses_unknown_or_overlapping_ids():
    with pytest.raises(ValueError, match="unknown"):
        merge_types(V1, [{"id": "x", "from": ["a", "zz"], "card": {}}], reason="", decided_by="", date="", v1_file="", v1_sha1="")
    with pytest.raises(ValueError, match="more than once"):
        merge_types(V1, [MERGES[0], {"id": "y", "from": ["a2", "b"], "card": {}}], reason="", decided_by="", date="",
                    v1_file="", v1_sha1="")


def test_v1_labels_are_read_as_v2():
    v2 = merge_types(V1, MERGES, reason="", decided_by="", date="", v1_file="", v1_sha1="")
    proj = projection(v2)
    assert proj == {"a": "a", "a2": "a", "b": "bc", "c": "bc"}
    row = {"labels": {"a": 0, "a2": 1, "b": 0, "c": 0},
           "found_by": {"a2": "second_look"}, "evidence": {"a2": "span"}, "grounded": {"a2": True}}
    out = project_labels(row, proj)
    assert out["labels"] == {"a": 1, "bc": 0} and out["found_by"] == {"a": "second_look"}
    both = project_labels({"labels": {"a": 1, "a2": 1, "b": 0, "c": 0}, "found_by": {"a": "second_look", "a2": "first"}}, proj)
    assert both["found_by"] == {"a": "first"}  # found by the first pass if any merged type was
    assert projection(V1) == {}  # a schema without merges reads labels unchanged
