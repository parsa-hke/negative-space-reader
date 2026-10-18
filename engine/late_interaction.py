"""Late-interaction matching (plan §0.2 #8 step 1): ColBERT token embeddings and MaxSim, for prototype matching.

The lean model, answerdotai/answerai-colbert-small-v1 (Apache-2.0), is a ColBERT checkpoint (colbert-ai format,
checked on the hub 2026-10-18): a 384-wide BERT plus `linear.weight` [96, 384]; queries start with the [Q] marker
([unused0]) and are at most 32 tokens, documents start with [D] ([unused1]), are at most 300 tokens and drop
punctuation tokens; token vectors are L2-normalised. It is loaded here with `transformers` (no extra library).
One difference from ColBERT's own query encoding, on purpose: queries are not padded with [MASK] tokens ("query
augmentation"), so a prototype matches on its own words only.

Use: prototypes are encoded as queries, section sentences as documents. maxsim(q, d) = mean over query tokens of the
best-matching document token (in [-1, 1]); a section's score for a type = the best maxsim over its sentences and the
type's prototypes (section_scores; score_matrix does it for many types at once).
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Sequence

import numpy as np


class ColBERT:
    query_maxlen, doc_maxlen, dim = 32, 300, 96

    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu", batch_size: int = 64):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoTokenizer, BertModel

        path = Path(snapshot_download(model_id, revision=revision,
                                      allow_patterns=["config.json", "model.safetensors", "tokenizer*",
                                                     "special_tokens_map.json", "vocab.txt", "artifact.metadata"]))
        self.model_id, self.revision, self.device, self.batch_size = model_id, path.name, device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.bert = BertModel.from_pretrained(path, add_pooling_layer=False).to(device).eval()
        with safe_open(str(path / "model.safetensors"), "pt") as f:
            self.linear = f.get_tensor("linear.weight").to(device=device, dtype=torch.float32)  # [96, 384]
        tok = self.tokenizer
        self.q_marker, self.d_marker = tok.convert_tokens_to_ids("[unused0]"), tok.convert_tokens_to_ids("[unused1]")
        self.skip = {tok.convert_tokens_to_ids(c) for c in string.punctuation} - {tok.unk_token_id}

    def encode(self, texts: Sequence[str], *, mode: str) -> list[np.ndarray]:
        """Token embeddings [n_tokens, 96] (float16 rows, unit norm) for each text; mode 'query' or 'doc'."""
        import torch

        maxlen = self.query_maxlen if mode == "query" else self.doc_maxlen
        marker = self.q_marker if mode == "query" else self.d_marker
        tok = self.tokenizer
        body = tok(list(texts), add_special_tokens=False, truncation=True, max_length=maxlen - 3)["input_ids"]
        ids = [[tok.cls_token_id, marker] + b + [tok.sep_token_id] for b in body]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out: list[np.ndarray | None] = [None] * len(ids)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                width = max(len(ids[i]) for i in batch)
                inp = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(batch), width), dtype=torch.long)
                for r, i in enumerate(batch):
                    inp[r, :len(ids[i])] = torch.tensor(ids[i])
                    att[r, :len(ids[i])] = 1
                h = self.bert(input_ids=inp.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
                v = torch.nn.functional.normalize(h.float() @ self.linear.T, dim=-1).cpu().numpy()
                for r, i in enumerate(batch):
                    n = len(ids[i])
                    keep = [p for p in range(n) if mode == "query" or ids[i][p] not in self.skip] or [0]
                    out[i] = v[r, keep].astype(np.float16)
        return out  # type: ignore[return-value]


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q.astype(np.float32) @ d.astype(np.float32).T).max(axis=1).mean())


def score_matrix(sentences: Sequence[np.ndarray], protos: np.ndarray, proto_lengths: np.ndarray,
                 proto_types: np.ndarray, n_types: int, *, device: str = "cpu") -> np.ndarray:
    """Best maxsim per type for one section: sentences = doc token embeddings per sentence; protos = all prototypes'
    query tokens stacked [Nq, 96], proto_lengths = tokens per prototype, proto_types = type index per prototype.
    Returns [n_types] (-1 for a type with no prototype)."""
    import torch

    out = np.full(n_types, -1.0, dtype=np.float32)
    if not len(sentences) or not len(protos):
        return out
    D = torch.from_numpy(np.concatenate(sentences).astype(np.float32)).to(device)
    Q = protos if isinstance(protos, torch.Tensor) else torch.from_numpy(protos.astype(np.float32)).to(device)
    S = Q @ D.T  # [Nq, L]
    bounds = np.cumsum([0] + [len(s) for s in sentences])
    per_sent = torch.stack([S[:, a:b].max(dim=1).values for a, b in zip(bounds[:-1], bounds[1:])], dim=1)  # [Nq, n_s]
    rows = torch.from_numpy(np.repeat(np.arange(len(proto_lengths)), proto_lengths)).to(device)
    sums = torch.zeros((len(proto_lengths), per_sent.shape[1]), device=device).index_add_(0, rows, per_sent)
    proto_best = (sums / torch.from_numpy(proto_lengths.astype(np.float32)).to(device)[:, None]).max(dim=1).values
    best = torch.full((n_types,), -1.0, device=device).scatter_reduce(
        0, torch.from_numpy(proto_types.astype(np.int64)).to(device), proto_best, reduce="amax")
    return best.cpu().numpy()


def section_scores(sentences: Sequence[np.ndarray], protos: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """The same as score_matrix, keyed by type id (convenience for small uses and tests)."""
    names = list(protos)
    flat = [(j, p) for j, n in enumerate(names) for p in protos[n]]
    Q = np.concatenate([p for _, p in flat])
    lengths = np.array([len(p) for _, p in flat])
    types = np.array([j for j, _ in flat])
    s = score_matrix(sentences, Q, lengths, types, len(names))
    return dict(zip(names, map(float, s)))
