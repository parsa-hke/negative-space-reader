"""The labelled data of presence detection (plan 4.3-4.6), assembled once per genre for every stage that needs it.

PresenceData(cfg, genre) holds: the current schema and its projection of older label versions; the sections of the
teacher set, the reference set and (genre with annotations) every annotated section outside the reference documents;
their stored embeddings (presence uses the pack embedder's section vectors, plan §0.2 #7); teacher labels, reference
labels (silver, plan 4.2) and annotation labels, all read in the current schema version; and the reference split
(fit / eval, by document). Training rows per type follow presence.train_labels: `plan` = annotations for types whose
label_source is a rule, teacher labels otherwise (§0.2 #7); `teacher` = teacher labels for every type.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import Section, read_jsonl
from engine.classifier import reference_split
from packs.presence_labels import teacher_types
from packs.schema_versions import project_labels, projection

Key = tuple[str, int]


def _path(p: str | Path) -> Path:
    from experiments.distil import _path as p_

    return p_(p)


def _full_rows(path: Path, proj: dict[str, str]) -> dict[Key, dict[str, Any]]:
    """Label rows (labels, evidence, grounded, found_by) read in the current schema version; failed rows left out."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("failed") or not r.get("labels"):
            continue
        out[(r["doc_id"], r["section_idx"])] = project_labels(r, proj)
    return out


def _rows(path: Path, proj: dict[str, str]) -> dict[Key, dict[str, int]]:
    return {k: r["labels"] for k, r in _full_rows(path, proj).items()}


class PresenceData:
    def __init__(self, cfg: dict[str, Any], genre: str, *, need_vectors: bool = True):
        from experiments.distil import load_schema

        pc = cfg["presence"]
        self.genre, self.cfg = genre, cfg
        self.schema = load_schema(cfg, genre)
        self.types = [t["id"] for t in self.schema["types"]]
        self.proj = projection(self.schema)
        lab = pc["labels"]
        self.teacher_full = _full_rows(_path(lab["teacher"].format(genre=genre)), self.proj)
        self.teacher = {k: r["labels"] for k, r in self.teacher_full.items()}
        self.reference = _rows(_path(lab["reference"].format(genre=genre)), self.proj)
        ann_path = (lab.get("annotations") or {}).get(genre)
        self.annotations = _rows(_path(ann_path), self.proj) if ann_path else {}
        self.rule_types = [t for t in self.types if t not in teacher_types(self.schema["types"])] if self.annotations else []
        ref_docs = {d for d, _ in self.reference}
        self.annotation_train = {k: v for k, v in self.annotations.items() if k[0] not in ref_docs}
        teacher_train = {k: v for k, v in self.teacher.items() if k[0] not in ref_docs}
        if set(teacher_train) != {k for k in self.teacher if k not in self.reference}:
            raise ValueError("teacher-set and reference documents overlap")
        self.teacher_train = teacher_train
        self.split = reference_split(sorted(self.reference), seed=pc.get("reference_split_seed", 1234))
        needed = set(self.teacher) | set(self.reference) | set(self.annotation_train)
        docs = {d for d, _ in needed}
        self.sections: dict[Key, Section] = {}
        for s in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section):
            if s.doc_id in docs and (s.doc_id, s.idx) in needed:
                self.sections[(s.doc_id, s.idx)] = s
        missing = needed - set(self.sections)
        if missing:
            raise ValueError(f"{genre}: {len(missing)} labelled sections are not in the corpus, e.g. {sorted(missing)[:2]}")
        self.vectors: dict[Key, np.ndarray] = {}
        if need_vectors:
            self._load_vectors(cfg)

    def _load_vectors(self, cfg: dict[str, Any]) -> None:
        from engine.embed import EmbeddingStore, embedding_slug

        store = EmbeddingStore.load(_path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"])
                                    / self.genre / "sections")
        keys = sorted(self.sections)
        found = store.lookup([self.sections[k].text for k in keys])
        missing = [k for k, v in zip(keys, found) if v is None]
        if missing:
            raise ValueError(f"{self.genre}: {len(missing)} labelled sections have no stored embedding (re-embed first)")
        self.vectors = dict(zip(keys, found))

    # ---------------------------------------------------------------- matrices
    def X(self, keys: list[Key]) -> np.ndarray:
        return np.stack([self.vectors[k] for k in keys]).astype(np.float32)

    def Y(self, keys: list[Key], source: dict[Key, dict[str, int]]) -> np.ndarray:
        return np.array([[int(source[k].get(t, 0)) for t in self.types] for k in keys], dtype=int)

    def reference_keys(self, half: str) -> list[Key]:
        return sorted(k for k, h in self.split.items() if h == half)

    def train_rows(self, t: str, mode: str) -> tuple[list[Key], np.ndarray]:
        """Training keys and labels for one type: annotations (rule types under `plan`) or teacher labels."""
        if mode == "plan" and t in self.rule_types:
            keys = sorted(self.annotation_train)
            return keys, np.array([self.annotation_train[k].get(t, 0) for k in keys], dtype=int)
        keys = sorted(self.teacher_train)
        return keys, np.array([self.teacher_train[k].get(t, 0) for k in keys], dtype=int)

    def describe(self) -> dict[str, Any]:
        halves = defaultdict(int)
        for h in self.split.values():
            halves[h] += 1
        return {"schema_version": self.schema["version"], "types": len(self.types),
                "teacher_train_sections": len(self.teacher_train), "annotation_train_sections": len(self.annotation_train),
                "rule_types": len(self.rule_types), "reference_fit": halves["fit"], "reference_eval": halves["eval"],
                "reference_fit_docs": len({d for (d, _), h in self.split.items() if h == "fit"}),
                "reference_eval_docs": len({d for (d, _), h in self.split.items() if h == "eval"})}
