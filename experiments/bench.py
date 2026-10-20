"""OmitBench synthetic and real runs (plan §6.1-6.3, §9.3). Stages (`bench.stage`):

  probs   every test section's and sentence's presence probabilities -> <run>/<genre>/{section_probs,sentence_probs}
          (a GPU pass: Colab job bench_probs_v1; the Mac for a limited smoke run). Never written into a pack.
  synth   OmitBench-synthetic instances (6.1, bench/synth.py) -> <run>/<genre>/{instances.jsonl,edited.jsonl,arrays.npz}
          and the per-type acceptance table
  eval    baselines (a) genre-uniform and (b) level-1 cohort prevalence (6.3, bench/metrics.py) -> bench_synth_<genre>.csv

Tables go to bench.tables_dir (reports/tables for the recorded runs) or, when it is null, to the run folder, so that a
smoke run never overwrites a reported table.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any


def _path(p: str | Path) -> Path:
    from experiments.distil import _path as p_

    return p_(p)


def _genres(cfg: dict[str, Any]) -> list[str]:
    return [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])


def run_probs(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import numpy as np

    import engine.embed as E
    from bench.probs import bench_order, build_sentence_probs, check_outside_packs
    from experiments.distil import _line_ranges, _presence_model, load_colbert
    from packs.profiles import build_section_probs

    if (cfg["presence"].get("cascade") or {}).get("reranker"):
        raise NotImplementedError("bench probabilities with the reranker are not built in the lean tier")
    bc = cfg["bench"]
    pc = bc.get("probs") or {}
    colbert = load_colbert(cfg)
    ecfg = cfg["models"]["embedder"]
    embedder = E.Embedder(ecfg["id"], devices=[colbert.device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                          normalize=True, revision=ecfg.get("revision"))
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        out = Path(run_dir) / genre
        check_outside_packs(out, _path(cfg["paths"]["packs"]))
        pm = _presence_model(cfg, genre, colbert, ["test"], pc.get("limit"), allow_test=True,
                             order=lambda ids: bench_order(ids, seed=bc["seed"]))
        docs, types = pm["docs"], pm["types"]
        chunk = pc.get("chunk_documents", 250)
        log = lambda m, s: print(f"[bench probs {genre} {s}] {m}", flush=True)  # noqa: E731
        sec = build_section_probs(docs, lambda d: pm["sections"][d], pm["score"], types, out / "section_probs",
                                  chunk_documents=chunk, log=lambda m: log(m, "sections"))
        path = _path(cfg["paths"]["data_processed"]) / genre / "sentences.jsonl"
        ranges = _line_ranges(path, {d["id"] for d in docs})
        fh = open(path, "rb")

        def sentences_of(doc_id: str) -> list[tuple[int, int, str]]:
            rows = []
            for a, b in ranges.get(doc_id, []):
                fh.seek(a)
                rows += [json.loads(line) for line in fh.read(b - a).splitlines() if line.strip()]
            return [(int(r["section_idx"]), int(r["idx"]), r["text"]) for r in rows]

        def score(texts: list[str]) -> Any:
            clf = pm["heads"].predict_proba(np.asarray(embedder.encode(texts)))
            return pm["model"].predict(clf, pm["scorer"].score(texts), np.full(clf.shape, np.nan))

        sent = build_sentence_probs(docs, sentences_of, score, types, out / "sentence_probs", chunk_documents=chunk,
                                    log=lambda m: log(m, "sentences"))
        fh.close()
        (out / "documents.json").write_text(json.dumps([d["id"] for d in docs]) + "\n")
        metrics.update({f"{genre}/documents": len(docs), **{f"{genre}/sections_{k}": v for k, v in sec.items()},
                        **{f"{genre}/sentences_{k}": v for k, v in sent.items()}})
    return metrics


def load_bench_docs(cfg: dict[str, Any], genre: str, probs: Path) -> tuple[list[Any], list[str], dict[str, Any]]:
    """The test documents scored by the probs stage, in the bench order, as BenchDocs (stored sections with their
    headings, sentences re-split exactly as sentences.jsonl was built, stored probabilities and embeddings)."""
    from collections import defaultdict

    import numpy as np

    from bench.synth import BenchDoc, BenchSection
    from data.schema import Document, Section, read_jsonl
    from data.segment import split_sentences
    from engine.embed import EmbeddingStore, embedding_slug, text_sha1
    from experiments.distil import load_schema
    from packs.profiles import read_profiles

    types = [t["id"] for t in load_schema(cfg, genre)["types"]]
    cols = [f"prob_{t}" for t in types]
    order = json.loads((probs / "documents.json").read_text())
    keep = set(order)
    sp, st = read_profiles(probs / "section_probs"), read_profiles(probs / "sentence_probs")
    sec_p = {(d, int(i)): r for d, i, r in zip(sp.doc_id, sp.section_idx, sp[cols].to_numpy(np.float32))}
    sent_rows: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for d, i, j, r in zip(st.doc_id, st.section_idx, st.sentence_idx, st[cols].to_numpy(np.float32)):
        sent_rows[(d, int(i))].append((int(j), r))
    processed = _path(cfg["paths"]["data_processed"]) / genre
    meta = {d.id: d for d in read_jsonl(processed / "documents.jsonl", Document) if d.id in keep}
    secs: dict[str, list[Any]] = defaultdict(list)
    for s in read_jsonl(processed / "sections.jsonl", Section):
        if s.doc_id in keep:
            secs[s.doc_id].append(s)
    emb = _path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"]) / genre
    S, T = EmbeddingStore.load(emb / "sections"), EmbeddingStore.load(emb / "titles")
    row_of, title_row = {h: i for i, h in enumerate(S.sha1)}, {d: i for i, d in enumerate(T.ids)}
    min_chars = cfg["segment"].get("min_sentence_chars", 15)
    docs, skipped = [], defaultdict(int)
    for d in order:
        out = []
        for s in sorted(secs[d], key=lambda s: s.idx):
            sents = split_sentences(s.text, min_chars=min_chars)
            rows = sorted(sent_rows.get((d, s.idx), []), key=lambda x: x[0])
            if len(rows) != len(sents) or (d, s.idx) not in sec_p or text_sha1(s.text) not in row_of:
                skipped["sections_without_stored_values"] += 1
                out = None
                break
            out.append(BenchSection(idx=s.idx, heading=s.heading, text=s.text, sentences=sents,
                                    S=np.stack([r for _, r in rows]) if rows else np.zeros((0, len(types)), np.float32),
                                    P=sec_p[(d, s.idx)], V=S.vectors[row_of[text_sha1(s.text)]]))
        if not out:
            skipped["documents_skipped"] += 1
            continue
        m = meta[d]
        tv = T.vectors[title_row[d]] if d in title_row and m.title else None
        docs.append(BenchDoc(id=d, entity=m.entity_id, title=m.title, category=m.category, date=m.date,
                             title_vector=tv, sections=out))
    return docs, types, dict(skipped)


def run_synth(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """OmitBench-synthetic instances (plan 6.1): bench.instances_per_regime verified deletions per regime and genre
    -> <run>/<genre>/instances.jsonl (ground truth, what was deleted, presence before and after), edited.jsonl (the
    edited documents' sections) and arrays.npz (per instance: document presence after the edit, the edited document's
    vector and its section probabilities), so that the evaluation needs no model."""
    import time
    from collections import Counter

    import numpy as np

    import engine.embed as E
    from bench.synth import build_bench
    from engine.classifier import LinearHeads
    from engine.doc_presence import DocPresence
    from engine.presence import Cascade, PrototypeScorer
    from experiments.distil import load_colbert

    bc = cfg["bench"]
    sc = bc.get("synth") or {}
    colbert = load_colbert(cfg)
    ecfg = cfg["models"]["embedder"]
    embedder = E.Embedder(ecfg["id"], devices=[colbert.device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024),
                          batch_size=ecfg.get("query_batch_size", ecfg.get("batch_size", 32)), normalize=True,
                          revision=ecfg.get("revision"))
    mode = cfg["presence"].get("train_labels", "teacher")
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        t0 = time.perf_counter()
        packs = _path(cfg["paths"]["packs"]) / genre
        probs = _path(sc["probs"].format(genre=genre))
        docs, types, skipped = load_bench_docs(cfg, genre, probs)
        heads, cascade = LinearHeads.load(packs / "classifier" / f"linear_{mode}"), Cascade.load(packs / "cascade.json")
        scorer = PrototypeScorer.from_pack(colbert, packs / "prototypes", len(types),
                                           min_sentence_chars=cfg["segment"]["min_sentence_chars"])
        doc_model = DocPresence.load(packs / "doc_presence.json")
        calls = Counter()

        def rescore(texts: list[str]) -> tuple[Any, Any]:
            calls["sections_rescored"] += len(texts)
            V = np.asarray(embedder.encode(texts))
            clf = heads.predict_proba(V)
            return cascade.predict(clf, scorer.score(texts), np.full(clf.shape, np.nan)), V

        out, stats = build_bench(
            docs, bc["regimes"], bc["instances_per_regime"], seed=bc["seed"], types=types,
            thresholds=np.asarray(cascade.threshold), rescore=rescore, doc_prob=doc_model.predict_proba,
            min_section_chars=sc.get("min_section_chars") or cfg["segment"]["min_section_chars"],
            max_removed_share=sc.get("max_removed_share", 0.5), sentence_rounds=sc.get("sentence_rounds", 8),
            sentences_per_round=sc.get("sentences_per_round", 5), min_carrier_prob=sc.get("min_carrier_prob", 0.02),
            section_rounds=sc.get("section_rounds", 5), attempts_per_document=sc.get("attempts_per_document", 3),
            title_abstract_weight=ecfg.get("title_abstract_weight", 0.5),
            log=lambda m: print(f"[bench synth {genre}] {m}", flush=True))
        folder = Path(run_dir) / genre
        folder.mkdir(parents=True, exist_ok=True)
        flat = [i for r in bc["regimes"] for i in out[r]]
        arrays = {"P", "p_doc", "doc_vector", "sections"}
        with open(folder / "instances.jsonl", "w", encoding="utf-8") as f:
            for i in flat:
                f.write(json.dumps({k: v for k, v in i.items() if k not in arrays}, ensure_ascii=False) + "\n")
        with open(folder / "edited.jsonl", "w", encoding="utf-8") as f:
            for i in flat:
                f.write(json.dumps({"instance_id": i["instance_id"], "title": i["title"], "sections": i["sections"]},
                                   ensure_ascii=False) + "\n")
        offsets = np.cumsum([0] + [len(i["P"]) for i in flat])
        np.savez_compressed(folder / "arrays.npz", instance_id=np.array([i["instance_id"] for i in flat]),
                            p_doc=np.stack([i["p_doc"] for i in flat]) if flat else np.zeros((0, len(types))),
                            doc_vector=np.stack([i["doc_vector"] for i in flat]) if flat else np.zeros((0, 1)),
                            P=np.concatenate([i["P"] for i in flat]) if flat else np.zeros((0, len(types))),
                            P_offsets=offsets, types=np.array(types))
        from experiments.distil import _write_table

        accept = []
        for r in bc["regimes"]:
            by = stats[r + "/by_type"]
            for t in types:
                c = {reason: n_ for key, n_ in by.items() for tt, reason in [key.split("|")] if tt == t}
                tried = sum(c.values())
                accept.append({"regime": r, "type_id": t, "draws": tried, "accepted": c.get("ok", 0),
                               "acceptance": round(c.get("ok", 0) / tried, 3) if tried else None,
                               **{f"rejected_{k}": c.get(k, 0) for k in ("still_present", "too_much_removed", "empty")}})
        tables = _path(bc["tables_dir"]) if bc.get("tables_dir") else Path(run_dir)
        _write_table(tables / f"bench_synth_acceptance_{genre}.csv", accept)
        metrics[f"{genre}/documents_available"] = len(docs)
        metrics.update({f"{genre}/{k}": v for k, v in skipped.items()})
        for r in bc["regimes"]:
            inst = out[r]
            metrics.update({f"{genre}/{r}/instances": len(inst), f"{genre}/{r}/outcomes": dict(stats[r])})
            if inst:
                metrics.update({
                    f"{genre}/{r}/ground_truth_mean": round(float(np.mean([len(i["ground_truth"]) for i in inst])), 3),
                    f"{genre}/{r}/with_collateral": round(float(np.mean([bool(i["collateral"]) for i in inst])), 3),
                    f"{genre}/{r}/escalated": round(float(np.mean([i["rounds"] > 0 for i in inst])), 3),
                    f"{genre}/{r}/escalated_to_sections": round(float(np.mean([i["escalation"]["section_rounds"] > 0
                                                                                for i in inst])), 3),
                    f"{genre}/{r}/removed_share_median": round(float(np.median([i["removed_share"] for i in inst])), 4),
                    f"{genre}/{r}/deleted_sentences_mean": round(float(np.mean([len(i["deleted"]) for i in inst])), 2),
                    f"{genre}/{r}/deleted_sections_mean": round(float(np.mean([len(i["deleted_sections"]) for i in inst])), 2)})
        metrics[f"{genre}/sections_rescored"] = calls["sections_rescored"]
        metrics[f"{genre}/seconds"] = round(time.perf_counter() - t0, 1)
    return metrics


SYSTEMS = ("genre_uniform", "level1", "absence_only", "random")  # (a), (b) of §0.2 #17, then two reference rows


def _expectations(system: str, *, uniform: Any, cohort: Any, n_types: int, rng: Any) -> tuple[Any, Any]:
    """(expected, p_present multiplier) per system: genre_uniform = corpus prevalence, level1 = cohort prevalence;
    the reference rows rank by (1 - P(present)) alone (absence_only) or at random."""
    import numpy as np

    if system == "genre_uniform":
        return uniform, True
    if system == "level1":
        return cohort, True
    if system == "absence_only":
        return np.ones(n_types), True
    if system == "random":
        return rng.uniform(size=n_types), False
    raise ValueError(f"unknown system {system!r}")


def _ranked(types: list[str], expected: Any, p_doc: Any, bits: Any, weights: Any, use_presence: bool) -> list[str]:
    import numpy as np

    from engine.rank import rank_absences

    p = p_doc if use_presence else np.zeros(len(types))
    return [r["type"] for r in rank_absences(types, expected, p, bits, weights=weights, prevalence=expected, top_k=None)]


def run_eval(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Baselines (a) genre-uniform and (b) level-1 cohort prevalence, plus reference rows, on OmitBench-synthetic
    (plan 6.3) -> per-instance results in the run folder, bench_synth_<genre>.csv (every metric with a bootstrap CI
    over documents, paired differences level1 - genre_uniform) and the gate check of 6.6 (paired recall@5 difference
    on single_type, CI above zero). False alarms (§0.2 #15) and the calibration of P(expected): on the untouched
    whole labelled dev documents of decision record 006, with out-of-fold document presence (cross-fitted on the two
    halves, so that no document is scored by a model fitted on its own labels)."""
    import zlib

    import numpy as np

    from bench.metrics import bootstrap, bootstrap_mean, ece, false_alarm_rate, instance_metrics, paired_bootstrap
    from engine.cohort import CohortRetriever
    from engine.doc_presence import DocPresence
    from engine.expectation import ProfileStore, level1
    from engine.index import load_index
    from experiments.distil import _write_table, labelled_documents, load_schema
    from packs.builder import document_vectors

    bc = cfg["bench"]
    ec = bc.get("eval") or {}
    ks, ndcg_k, fa_k = bc.get("recall_ks", [1, 5, 10]), bc.get("ndcg_k", 10), bc.get("false_alarm_top", 5)
    boot = dict(n_boot=bc.get("bootstrap_samples", 1000), ci=bc.get("ci", 0.95), seed=bc["seed"])
    use = (cfg.get("expectation") or {}).get("profile", "bit")
    z = (cfg.get("expectation") or {}).get("wilson_z", 1.96)
    systems = list(ec.get("systems") or SYSTEMS)
    tables = _path(bc["tables_dir"]) if bc.get("tables_dir") else Path(run_dir)
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        packs = _path(cfg["paths"]["packs"]) / genre
        schema = load_schema(cfg, genre)
        types = [t["id"] for t in schema["types"]]
        cards = {t["id"]: t for t in schema["types"]}
        w = (np.array([float(cards[t].get("severity_weight") or cfg["schema"].get("severity_weight_default", 1.0))
                       for t in types]) if (cfg.get("rank") or {}).get("use_severity_weights", True) else np.ones(len(types)))
        profiles = ProfileStore.load(packs / "profiles", types)
        uniform = (profiles.bits if use == "bit" else profiles.probs).mean(0).astype(float)
        retriever = CohortRetriever.from_config(load_index(packs / "index", backend=(cfg.get("index") or {}).get("backend")),
                                                cfg.get("cohort") or {})

        def cohort_expectation(vec: Any, entity: Any, category: Any) -> tuple[Any, dict[str, Any]]:
            coh = retriever.retrieve(np.asarray(vec, dtype=np.float32), entity_id=entity, category=category)
            e = level1(profiles, [h.id for h in coh.peers], z=z, use=use)
            return e["prevalence"], {"n": e["n"], "fallback": coh.fallback, "mean_similarity": coh.stats["mean_similarity"]}

        # ---- synthetic instances
        folder = _path(ec["synth"].format(genre=genre))
        inst = [json.loads(l) for l in open(folder / "instances.jsonl", encoding="utf-8")]
        arr = np.load(folder / "arrays.npz")
        if list(arr["types"]) != types or [str(x) for x in arr["instance_id"]] != [i["instance_id"] for i in inst]:
            raise ValueError(f"{genre}: {folder} does not match the schema or its instances")
        rows, cohort_stats = [], []
        for k, i in enumerate(inst):
            p_doc = arr["p_doc"][k].astype(float)
            bits = (p_doc >= 0.5).astype(int)
            cexp, cs = cohort_expectation(arr["doc_vector"][k], i.get("entity_id"), i.get("category"))
            cohort_stats.append(cs)
            gt = set(i["ground_truth"])
            rng = np.random.default_rng([bc["seed"], zlib.crc32(i["instance_id"].encode())])
            for s in systems:
                e, with_p = _expectations(s, uniform=uniform, cohort=cexp, n_types=len(types), rng=rng)
                ranked = _ranked(types, e, p_doc, bits, w, with_p)
                rows.append({"instance_id": i["instance_id"], "regime": i["regime"], "system": s,
                             "candidates": len(ranked), "ground_truth": len(gt),
                             **instance_metrics(ranked, gt, ks=ks, ndcg_k=ndcg_k), "top5": "|".join(ranked[:5])})
        _write_table(Path(run_dir) / genre / "per_instance.csv", rows)
        names = [f"recall@{k}" for k in ks] + ["mrr", f"ndcg@{ndcg_k}"]
        table = []
        for regime in bc["regimes"]:
            for s in systems:
                sel = [r for r in rows if r["regime"] == regime and r["system"] == s]
                if not sel:
                    continue
                for m in names:
                    v, lo, hi = bootstrap_mean([r[m] for r in sel], **boot)
                    table.append({"genre": genre, "regime": regime, "system": s, "metric": m, "value": round(v, 4),
                                  "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(sel)})
                    metrics[f"{genre}/{regime}/{s}/{m}"] = round(v, 4)
                table.append({"genre": genre, "regime": regime, "system": s, "metric": "candidates",
                              "value": round(float(np.mean([r["candidates"] for r in sel])), 2), "ci_low": None,
                              "ci_high": None, "n": len(sel)})
            a = {r["instance_id"]: r for r in rows if r["regime"] == regime and r["system"] == "level1"}
            b = {r["instance_id"]: r for r in rows if r["regime"] == regime and r["system"] == "genre_uniform"}
            ids = sorted(a.keys() & b.keys())
            for m in names if ids else []:
                d = paired_bootstrap([a[x][m] for x in ids], [b[x][m] for x in ids], **boot)
                table.append({"genre": genre, "regime": regime, "system": "level1 - genre_uniform", "metric": m,
                              "value": round(d["diff"], 4), "ci_low": round(d["ci_low"], 4),
                              "ci_high": round(d["ci_high"], 4), "n": d["n"]})
                if m == "recall@5":
                    metrics[f"{genre}/{regime}/recall@5_diff"] = {k2: round(v2, 4) for k2, v2 in d.items()}
        # ---- untouched labelled documents: false alarms and calibration of P(expected)
        L = labelled_documents(cfg, genre)
        dep = DocPresence.load(packs / "doc_presence.json")
        oof: dict[str, Any] = {}
        for a_ids, b_ids in ((L["fit"], L["eval"]), (L["eval"], L["fit"])):
            ix = [L["documents"].index(d) for d in a_ids]
            m = DocPresence(dep.features).fit([L["by_doc"][d] for d in a_ids], L["Y"][ix], C_grid=(dep.C,),
                                              seed=cfg["run"].get("seed", 0))
            oof.update({d: m.predict_proba(L["by_doc"][d]) for d in b_ids})
        ids, X, payloads, _ = document_vectors(cfg, genre, ["dev"])
        vec = {d: (X[k], payloads[k]) for k, d in enumerate(ids)}
        docs = [d for d in L["documents"] if d in vec]
        Y = np.stack([L["Y"][L["documents"].index(d)] for d in docs])
        fa: dict[str, list[float]] = {s: [] for s in systems}
        E: dict[str, list[Any]] = {"genre_uniform": [], "level1": []}
        for n_doc, d in enumerate(docs):
            p_doc = oof[d]
            bits = (p_doc >= 0.5).astype(int)
            cexp, _ = cohort_expectation(vec[d][0], vec[d][1].get("entity_id"), vec[d][1].get("category"))
            E["genre_uniform"].append(uniform)
            E["level1"].append(cexp)
            present = {t for t, y in zip(types, Y[n_doc]) if y}
            rng = np.random.default_rng([bc["seed"], zlib.crc32(d.encode())])
            for s in systems:
                e, with_p = _expectations(s, uniform=uniform, cohort=cexp, n_types=len(types), rng=rng)
                v = false_alarm_rate(_ranked(types, e, p_doc, bits, w, with_p), present, fa_k)
                fa[s].append(np.nan if v is None else v)
        for s in systems:
            x = np.array(fa[s])
            x = x[~np.isnan(x)]
            v, lo, hi = bootstrap_mean(x, **boot)
            table.append({"genre": genre, "regime": "untouched_labelled", "system": s, "metric": f"false_alarm@{fa_k}",
                          "value": round(v, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(x)})
            metrics[f"{genre}/false_alarm@{fa_k}/{s}"] = round(v, 4)
        for s, M in E.items():
            M = np.stack(M)
            v, lo, hi = bootstrap(lambda idx: ece(M[idx], Y[idx], bins=bc.get("ece_bins", 15)), len(docs), **boot)
            table.append({"genre": genre, "regime": "untouched_labelled", "system": s, "metric": "ece_expected",
                          "value": round(v, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(docs)})
            metrics[f"{genre}/ece_expected/{s}"] = round(v, 4)
        _write_table(tables / f"bench_synth_{genre}.csv", table)
        gate = metrics.get(f"{genre}/single_type/recall@5_diff")
        metrics[f"{genre}/gate_6_6"] = bool(gate and gate["ci_low"] > 0)
        metrics.update({f"{genre}/instances": len(inst), f"{genre}/labelled_documents": len(docs),
                        f"{genre}/cohort_fallback_share": round(float(np.mean([c["fallback"] for c in cohort_stats])), 4)
                        if cohort_stats else None})
    return metrics


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = (cfg.get("bench") or {}).get("stage")
    if stage == "probs":
        return run_probs(cfg, run_dir)
    if stage == "synth":
        return run_synth(cfg, run_dir)
    if stage == "eval":
        return run_eval(cfg, run_dir)
    raise ValueError(f"unknown bench.stage {stage!r} (probs | synth | eval)")
