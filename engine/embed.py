"""Embedding models behind one interface, and an on-disk store keyed by text hash (plan §5.1).

    emb = Embedder("Qwen/Qwen3-Embedding-0.6B", devices=["cuda:0", "cuda:1"], dtype="float16")
    vectors = emb.encode(texts)                         # L2-normalised float32 rows
    EmbeddingStore.save(path, ids, texts, vectors, meta) # float16 matrix + ids + sha1 per row
    store = EmbeddingStore.load(path); store.lookup(texts) -> rows (or None when a text was never embedded)

Several devices -> one worker process per device (sentence-transformers multi-process pool). Documents are encoded
without an instruction prompt (Qwen3-Embedding uses prompts on the query side only).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embedding_slug(ecfg: dict[str, Any]) -> str:
    """Folder name for a models.embedder config under paths.embeddings: model id, plus the truncated dimension."""
    return ecfg["id"].replace("/", "__") + (f"__d{ecfg['truncate_dim']}" if ecfg.get("truncate_dim") else "")


class Embedder:
    def __init__(self, model_id: str, *, devices: Sequence[str] = ("cpu",), dtype: str = "float16",
                 max_length: int = 1024, batch_size: int = 32, normalize: bool = True,
                 truncate_dim: int | None = None, revision: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id, self.devices, self.batch_size = model_id, list(devices), batch_size
        self.normalize, self.truncate_dim = normalize, truncate_dim
        torch_dtype = getattr(torch, dtype) if self.devices[0] != "cpu" else torch.float32
        self.model = SentenceTransformer(model_id, device=self.devices[0], revision=revision,
                                         model_kwargs={"dtype": torch_dtype},
                                         processor_kwargs={"padding_side": "left"}, truncate_dim=truncate_dim)
        self.model.max_seq_length = max_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], *, show_progress: bool = False, prompt: str | None = None) -> np.ndarray:
        """prompt: text prepended to every input (an instruction for instruction-aware models), or None."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(self.devices) > 1:
            pool = self.model.start_multi_process_pool(target_devices=self.devices)
            try:
                out = self.model.encode(list(texts), pool=pool, batch_size=self.batch_size, prompt=prompt,
                                        normalize_embeddings=self.normalize, show_progress_bar=show_progress,
                                        chunk_size=max(1000, self.batch_size * 20))
            finally:
                self.model.stop_multi_process_pool(pool)
        else:
            out = self.model.encode(list(texts), batch_size=self.batch_size, prompt=prompt, normalize_embeddings=self.normalize,
                                    show_progress_bar=show_progress, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)


@dataclass
class EmbeddingStore:
    ids: list[Any]
    sha1: list[str]
    vectors: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def save(path: str | Path, ids: list[Any], texts: Sequence[str], vectors: np.ndarray, meta: dict[str, Any]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.f16.npy", vectors.astype(np.float16))
        with open(path / "rows.jsonl", "w", encoding="utf-8") as f:
            for i, t in zip(ids, texts):
                f.write(json.dumps({"id": i, "sha1": text_sha1(t)}) + "\n")
        (path / "meta.json").write_text(json.dumps({**meta, "rows": len(ids), "dim": int(vectors.shape[1])},
                                                   indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStore":
        path = Path(path)
        rows = [json.loads(l) for l in open(path / "rows.jsonl", encoding="utf-8")]
        return cls(ids=[r["id"] for r in rows], sha1=[r["sha1"] for r in rows],
                   vectors=np.load(path / "vectors.f16.npy").astype(np.float32),
                   meta=json.loads((path / "meta.json").read_text(encoding="utf-8")))

    def lookup(self, texts: Sequence[str]) -> list[np.ndarray | None]:
        index = {h: i for i, h in enumerate(self.sha1)}
        return [self.vectors[index[h]] if (h := text_sha1(t)) in index else None for t in texts]


def embed_and_store(embedder: Embedder, ids: list[Any], texts: list[str], path: str | Path,
                    extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    vectors = embedder.encode(texts, show_progress=True)
    seconds = round(time.perf_counter() - t0, 1)
    meta = {"model": embedder.model_id, "devices": embedder.devices, "normalized": embedder.normalize,
            "truncate_dim": embedder.truncate_dim, "seconds": seconds, **(extra_meta or {})}
    EmbeddingStore.save(path, ids, texts, vectors, meta)
    return {"rows": len(ids), "dim": int(vectors.shape[1]), "seconds": seconds,
            "rows_per_second": round(len(ids) / max(seconds, 1e-6), 1)}
