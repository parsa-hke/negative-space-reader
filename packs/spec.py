"""GenrePack: everything analyse() needs for one genre, loaded once from the pack folder (plan §0.2 #3, §0.4, §5.6).

A pack is data, not code: the frozen schema, the presence model (linear heads + prototypes + cascade.json), the
document-level presence model (doc_presence.json), the index of the indexed corpus (document vectors + payloads,
section offsets, best section per type), the document profiles, and the models named in the config (the embedder
must be the one the index was built with; load() checks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GenrePack:
    genre: str
    cfg: dict[str, Any]
    schema: dict[str, Any]
    types: list[str]
    heads: Any
    cascade: Any
    scorer: Any
    doc_model: Any
    index: Any
    profiles: Any
    best: dict[str, np.ndarray]
    section_texts: Any
    prototypes: dict[str, list[dict[str, Any]]]
    embedder: Any
    info: dict[str, Any] = field(default_factory=dict)
    evidence_store: Any = None

    @property
    def cards(self) -> dict[str, dict[str, Any]]:
        return {t["id"]: t for t in self.schema["types"]}

    def weights(self) -> np.ndarray:
        if not (self.cfg.get("rank") or {}).get("use_severity_weights", True):
            return np.ones(len(self.types))
        default = self.cfg["schema"].get("severity_weight_default", 1.0)
        return np.array([float(self.cards[t].get("severity_weight") or default) for t in self.types])

    @classmethod
    def load(cls, cfg: dict[str, Any], genre: str, *, embedder: Any = None, colbert: Any = None,
             device: str | None = None) -> "GenrePack":
        import yaml

        from engine.classifier import LinearHeads
        from engine.doc_presence import DocPresence
        from engine.embed import Embedder
        from engine.evidence import EvidenceStore
        from engine.expectation import ProfileStore
        from engine.index import load_index
        from engine.presence import Cascade, PrototypeScorer
        from experiments.distil import load_colbert
        from experiments.run import resolve_device
        from packs.builder import SectionTexts, _path, load_best_sections

        pack = _path(cfg["paths"]["packs"]) / genre
        schema = yaml.safe_load((pack / "schema.yaml").read_text(encoding="utf-8"))
        types = [t["id"] for t in schema["types"]]
        mode = cfg["presence"].get("train_labels", "teacher")
        heads, cascade = LinearHeads.load(pack / "classifier" / f"linear_{mode}"), Cascade.load(pack / "cascade.json")
        if heads.types != types or cascade.types != types:
            raise ValueError(f"{genre}: presence model types differ from the schema")
        index = load_index(pack / "index", backend=(cfg.get("index") or {}).get("backend"))
        ecfg = cfg["models"]["embedder"]
        if index.meta.get("embedder") != ecfg["id"]:
            raise ValueError(f"{genre}: the index was built with {index.meta.get('embedder')}, the config's embedder "
                             f"is {ecfg['id']}; uploads must be embedded with the index's model")
        device = device or resolve_device(cfg["run"].get("device", "auto"))
        if embedder is None:
            embedder = Embedder(ecfg["id"], devices=[device], dtype=ecfg.get("dtype", "float16"),
                                max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                                normalize=True, revision=ecfg.get("revision"))
        colbert = colbert or load_colbert(cfg)
        scorer = PrototypeScorer.from_pack(colbert, pack / "prototypes", len(types),
                                           min_sentence_chars=cfg["segment"]["min_sentence_chars"])
        doc_model = DocPresence.load(pack / "doc_presence.json") if (pack / "doc_presence.json").exists() else None
        if doc_model is not None and doc_model.T != len(types):
            raise ValueError(f"{genre}: doc_presence.json has {doc_model.T} types, the schema {len(types)}")
        manifest = pack / "profiles" / "manifest.json"
        prof_version = json.loads(manifest.read_text()).get("version", 1) if manifest.exists() else 1
        protos = json.loads((pack / "prototypes" / "prototypes.json").read_text(encoding="utf-8"))["prototypes"]
        sections_path = _path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl"
        store = (EvidenceStore.load(pack / "evidence", dict(zip(types, map(float, cascade.threshold))))
                 if any((pack / "evidence").glob("part-*.parquet")) else None)
        return cls(genre=genre, cfg=cfg, schema=schema, types=types, heads=heads, cascade=cascade, scorer=scorer,
                   doc_model=doc_model, index=index, profiles=ProfileStore.load(pack / "profiles", types),
                   best=load_best_sections(pack / "index"), section_texts=SectionTexts.load(pack / "index", sections_path),
                   prototypes=protos, embedder=embedder, evidence_store=store,
                   info={"schema_version": schema.get("version"), "embedder": ecfg["id"], "device": device,
                         "late_interaction": cfg["models"]["late_interaction"]["id"], "profiles_version": prof_version,
                         "presence": "document model" if doc_model is not None else "best section (v1)",
                         "evidence": "stored sentences" if store is not None else "scored at query time",
                         "indexed_documents": len(index), "tier": cfg.get("tier", "lean")})
