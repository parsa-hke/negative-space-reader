"""OmitBench synthetic and real runs (plan §6.1-6.3, §9.3). Stages (`bench.stage`):

  probs   every test section's and sentence's presence probabilities -> <run>/<genre>/{section_probs,sentence_probs}
          (a GPU pass: Colab job bench_probs_v1; the Mac for a limited smoke run). Never written into a pack.
  synth   OmitBench-synthetic instances (6.1, bench/synth.py) -> <run>/<genre>/{instances.jsonl,edited.jsonl,arrays.npz}
          and the per-type acceptance table
  eval    baselines (a) genre-uniform and (b) level-1 cohort prevalence (6.3, bench/metrics.py) -> bench_synth_<genre>.csv
  revision_probs  section probabilities and document vectors of every revision version (6.3b; GPU or the Mac)
  revision        OmitBench-revision instances (bench/revision.py) and (a), (b) on them -> revision_bench_<genre>.csv,
                  bench_revision_<genre>.csv
  diagnose        cohort diagnosis after gate 6.6 (decision record 007) -> cohort_diagnosis_<genre>.csv

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


def _summarise(rows: list[dict[str, Any]], genre: str, regimes: list[str], systems: list[str], names: list[str],
               boot: dict[str, Any], metrics: dict[str, Any], groups: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per (regime, system): every metric's mean with a bootstrap CI and the mean candidate count; per regime the paired
    difference level1 - genre_uniform. groups (instance id -> group, e.g. entity) resamples whole groups."""
    import numpy as np

    from bench.metrics import bootstrap_mean, paired_bootstrap

    table = []
    for regime in regimes:
        for s in systems:
            sel = [r for r in rows if r["regime"] == regime and r["system"] == s]
            if not sel:
                continue
            g = [groups[r["instance_id"]] for r in sel] if groups else None
            for m in names:
                v, lo, hi = bootstrap_mean([r[m] for r in sel], groups=g, **boot)
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
            d = paired_bootstrap([a[x][m] for x in ids], [b[x][m] for x in ids],
                                 groups=[groups[x] for x in ids] if groups else None, **boot)
            table.append({"genre": genre, "regime": regime, "system": "level1 - genre_uniform", "metric": m,
                          "value": round(d["diff"], 4), "ci_low": round(d["ci_low"], 4),
                          "ci_high": round(d["ci_high"], 4), "n": d["n"]})
            if m == "recall@5":
                metrics[f"{genre}/{regime}/recall@5_diff"] = {k2: round(v2, 4) for k2, v2 in d.items()}
    return table


def run_eval(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Baselines (a) genre-uniform and (b) level-1 cohort prevalence, plus reference rows, on OmitBench-synthetic
    (plan 6.3) -> per-instance results in the run folder, bench_synth_<genre>.csv (every metric with a bootstrap CI
    over documents, paired differences level1 - genre_uniform) and the gate check of 6.6 (paired recall@5 difference
    on single_type, CI above zero). False alarms (§0.2 #15) and the calibration of P(expected): on the untouched
    whole labelled dev documents of decision record 006, with out-of-fold document presence (cross-fitted on the two
    halves, so that no document is scored by a model fitted on its own labels)."""
    import zlib

    import numpy as np

    from bench.metrics import bootstrap, bootstrap_mean, ece, false_alarm_rate, instance_metrics
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
        table = _summarise(rows, genre, bc["regimes"], systems, names, boot, metrics)
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
        p_after = [v for i in inst for t, v in i["p_after"].items() if t in i["targets"]]
        if p_after:
            table.append({"genre": genre, "regime": "all", "system": "instances", "metric": "target_p_present_after_median",
                          "value": round(float(np.median(p_after)), 4), "ci_low": None, "ci_high": None, "n": len(p_after)})
        _write_table(tables / f"bench_synth_{genre}.csv", table)
        acc = folder.parent / f"bench_synth_acceptance_{genre}.csv"
        if acc.exists() and tables != folder.parent:  # the synth run's per-type acceptance, published with the results
            (tables / acc.name).write_text(acc.read_text(encoding="utf-8"), encoding="utf-8")
        gate = metrics.get(f"{genre}/single_type/recall@5_diff")
        metrics[f"{genre}/gate_6_6"] = bool(gate and gate["ci_low"] > 0)
        metrics.update({f"{genre}/instances": len(inst), f"{genre}/labelled_documents": len(docs),
                        f"{genre}/cohort_fallback_share": round(float(np.mean([c["fallback"] for c in cohort_stats])), 4)
                        if cohort_stats else None})
    return metrics


def run_revision_probs(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Every revision version's section probabilities and document vector (plan 6.3b): the versions of the pairs in
    data/processed/<genre>/revisions.jsonl live in data/processed/<genre>/revisions/ (test entities, never indexed).
    Sections already in the embedding store reuse their vectors; the rest are embedded here. -> <run>/<genre>/
    section_probs/, doc_vectors.npz."""
    from collections import defaultdict

    import numpy as np

    import engine.embed as E
    from bench.probs import check_outside_packs
    from data.schema import Document, Section, read_jsonl
    from engine.classifier import LinearHeads
    from engine.embed import EmbeddingStore, document_vector, embedding_slug, text_sha1
    from engine.presence import Cascade, PrototypeScorer
    from experiments.distil import load_colbert, load_schema
    from packs.profiles import build_section_probs

    rc = cfg["bench"].get("revision") or {}
    colbert = load_colbert(cfg)
    ecfg = cfg["models"]["embedder"]
    embedder = E.Embedder(ecfg["id"], devices=[colbert.device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                          normalize=True, revision=ecfg.get("revision"))
    mode = cfg["presence"].get("train_labels", "teacher")
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        out = Path(run_dir) / genre
        packs = _path(cfg["paths"]["packs"]) / genre
        check_outside_packs(out, packs.parent)
        types = [t["id"] for t in load_schema(cfg, genre)["types"]]
        heads, cascade = LinearHeads.load(packs / "classifier" / f"linear_{mode}"), Cascade.load(packs / "cascade.json")
        scorer = PrototypeScorer.from_pack(colbert, packs / "prototypes", len(types),
                                           min_sentence_chars=cfg["segment"]["min_sentence_chars"])
        rev = _path(cfg["paths"]["data_processed"]) / genre / "revisions"
        docs = [d for d in read_jsonl(rev / "documents.jsonl", Document)][: rc.get("limit") or None]
        keep = {d.id for d in docs}
        sections: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for s in read_jsonl(rev / "sections.jsonl", Section):
            if s.doc_id in keep:
                sections[s.doc_id].append((s.idx, s.text))
        store = EmbeddingStore.load(_path(cfg["paths"]["embeddings"]) / embedding_slug(ecfg) / genre / "sections")
        row_of = {h: i for i, h in enumerate(store.sha1)}
        vecs: dict[str, np.ndarray] = {}
        counts = {"reused": 0, "embedded": 0}

        def score(texts: list[str]) -> Any:
            missing = [t for t in dict.fromkeys(texts) if text_sha1(t) not in row_of and text_sha1(t) not in vecs]
            if missing:
                for t, v in zip(missing, np.asarray(embedder.encode(missing))):
                    vecs[text_sha1(t)] = v.astype(np.float16)
            counts["embedded"] += len(missing)
            counts["reused"] += sum(text_sha1(t) in row_of for t in texts)
            V = np.stack([store.vectors[row_of[h]] if (h := text_sha1(t)) in row_of else vecs[h].astype(np.float32)
                          for t in texts])
            for t, v in zip(texts, V):
                vecs.setdefault(text_sha1(t), v.astype(np.float16))
            clf = heads.predict_proba(V)
            return cascade.predict(clf, scorer.score(texts), np.full(clf.shape, np.nan))

        res = build_section_probs([{"id": d.id, "split": "revision"} for d in docs], lambda d: sections[d], score, types,
                                  out / "section_probs", chunk_documents=rc.get("chunk_documents", 250),
                                  log=lambda m: print(f"[revision probs {genre}] {m}", flush=True))
        titled = [d for d in docs if d.title]
        T = dict(zip([d.id for d in titled], np.asarray(embedder.encode([d.title for d in titled])))) if titled else {}
        w = ecfg.get("title_abstract_weight", 0.5)
        ids, X = [], []
        for d in docs:
            if not sections[d.id]:
                continue
            V = np.stack([vecs[text_sha1(t)].astype(np.float32) if text_sha1(t) in vecs else store.vectors[row_of[text_sha1(t)]]
                          for _, t in sorted(sections[d.id])])
            ids.append(d.id)
            X.append(document_vector(V, T.get(d.id), title_abstract_weight=w))
        np.savez_compressed(out / "doc_vectors.npz", ids=np.array(ids), X=np.asarray(X, dtype=np.float32))
        metrics.update({f"{genre}/documents": len(docs), f"{genre}/sections_reused": counts["reused"],
                        f"{genre}/sections_embedded": counts["embedded"], **{f"{genre}/{k}": v for k, v in res.items()}})
    return metrics


def run_revision(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """OmitBench-revision (plan 6.3b): pairs whose newer version adds a type that at least min_added_prevalence of the
    older version's peers dated before it already state (time-windowed cohort, §0.2 #16, #19); instance = the older
    version with those types as ground truth. Also the system-neutral variant (every added type). Baselines (a) and
    (b) and the reference rows on both, the corpus prevalence restricted to the same date window for (a); bootstrap over
    entities. -> revision_bench_<genre>.csv (counts per year, prevalence of added types) and
    bench_revision_<genre>.csv (metrics)."""
    import datetime as dt
    import zlib

    import numpy as np

    from bench.metrics import instance_metrics
    from bench.revision import revision_instances
    from engine.cohort import CohortRetriever
    from engine.doc_presence import DocPresence
    from engine.expectation import ProfileStore, level1
    from engine.index import load_index
    from experiments.distil import _write_table, load_schema
    from packs.profiles import read_profiles

    bc = cfg["bench"]
    rc = bc.get("revision") or {}
    ks, ndcg_k = bc.get("recall_ks", [1, 5, 10]), bc.get("ndcg_k", 10)
    boot = dict(n_boot=bc.get("bootstrap_samples", 1000), ci=bc.get("ci", 0.95), seed=bc["seed"])
    use = (cfg.get("expectation") or {}).get("profile", "bit")
    z = (cfg.get("expectation") or {}).get("wilson_z", 1.96)
    min_prev = (bc.get("real") or {}).get("min_added_prevalence", 0.5)
    systems = list((bc.get("eval") or {}).get("systems") or SYSTEMS)
    tables = _path(bc["tables_dir"]) if bc.get("tables_dir") else Path(run_dir)
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        packs = _path(cfg["paths"]["packs"]) / genre
        types = [t["id"] for t in load_schema(cfg, genre)["types"]]
        cols = [f"prob_{t}" for t in types]
        probs = _path(rc["probs"].format(genre=genre))
        sp = read_profiles(probs / "section_probs")
        doc_model = DocPresence.load(packs / "doc_presence.json")
        p_doc = {d: doc_model.predict_proba(g.sort_values("section_idx")[cols].to_numpy(np.float64))
                 for d, g in sp.groupby("doc_id")}
        z_ = np.load(probs / "doc_vectors.npz")
        vec = dict(zip([str(x) for x in z_["ids"]], z_["X"]))
        meta = {json.loads(l)["id"]: json.loads(l) for l in
                open(_path(cfg["paths"]["data_processed"]) / genre / "revisions" / "documents.jsonl", encoding="utf-8")}
        pairs = [json.loads(l) for l in open(_path(cfg["paths"]["data_processed"]) / genre / "revisions.jsonl", encoding="utf-8")]
        pairs = [p for p in pairs if p["old_doc_id"] in vec]
        index = load_index(packs / "index", backend=(cfg.get("index") or {}).get("backend"))
        retriever = CohortRetriever.from_config(index, cfg.get("cohort") or {})
        profiles = ProfileStore.load(packs / "profiles", types)
        M = (profiles.bits if use == "bit" else profiles.probs).astype(float)
        dates = np.array([index.payloads[index.pos[d]].get("date") or "" for d in profiles.doc_ids])
        order = np.argsort(dates, kind="stable")
        cum = np.vstack([np.zeros(len(types)), np.cumsum(M[order], axis=0)])
        sorted_dates = dates[order]

        def before(date: str | None) -> str | None:
            return (dt.date.fromisoformat(date[:10]) - dt.timedelta(days=1)).isoformat() if date else None

        def uniform_before(date: str | None) -> Any:
            n = int(np.searchsorted(sorted_dates, date[:10], side="left")) if date else len(sorted_dates)
            return cum[n] / n if n else cum[-1] / len(sorted_dates)

        cache: dict[str, tuple[Any, dict[str, Any]]] = {}

        def cohort(pr: dict[str, Any]) -> Any:
            d = pr["old_doc_id"]
            if d not in cache:
                coh = retriever.retrieve(vec[d], entity_id=pr["entity_id"], category=meta.get(d, {}).get("category"),
                                         date_window=(None, before(pr.get("old_date"))))
                e = level1(profiles, [h.id for h in coh.peers], z=z, use=use)
                cache[d] = (e["prevalence"], {"n": e["n"], "fallback": coh.fallback})
            return cache[d][0]

        sets = {"revision": min_prev, "revision_all_added": None}
        rows, counts, per_year = [], {}, []
        for name, mp in sets.items():
            inst, stats = revision_instances(pairs, p_doc, cohort, types, min_prevalence=mp)
            counts[name] = stats
            years: dict[Any, int] = {}
            for i in inst:
                years[i["year"]] = years.get(i["year"], 0) + 1
            per_year += [{"genre": genre, "set": name, "year": y, "instances": n} for y, n in sorted(years.items(), key=lambda x: str(x[0]))]
            for i in inst:
                p = p_doc[i["doc_id"]]
                bits = (p >= 0.5).astype(int)
                gt = set(i["ground_truth"])
                rng = np.random.default_rng([bc["seed"], zlib.crc32(i["instance_id"].encode())])
                for s in systems:
                    e, with_p = _expectations(s, uniform=uniform_before(i["old_date"]), cohort=cohort(
                        {"old_doc_id": i["doc_id"], "entity_id": i["entity_id"], "old_date": i["old_date"]}),
                        n_types=len(types), rng=rng)
                    ranked = _ranked(types, e, p, bits, np.ones(len(types)), with_p)
                    rows.append({"instance_id": i["instance_id"], "entity_id": i["entity_id"], "regime": name,
                                 "system": s, "candidates": len(ranked), "ground_truth": len(gt),
                                 **instance_metrics(ranked, gt, ks=ks, ndcg_k=ndcg_k), "top5": "|".join(ranked[:5])})
            prev = np.array([v for i in inst for t, v in i["added_prevalence"].items() if t in i["ground_truth"]])
            if len(prev):
                metrics[f"{genre}/{name}/ground_truth_prevalence_quantiles"] = np.round(np.quantile(prev, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()
            metrics[f"{genre}/{name}/counts"] = stats
        groups = {r["instance_id"]: r["entity_id"] for r in rows}
        names = [f"recall@{k}" for k in ks] + ["mrr", f"ndcg@{ndcg_k}"]
        table = _summarise(rows, genre, list(sets), systems, names, boot, metrics, groups=groups)
        _write_table(tables / f"bench_revision_{genre}.csv", table)
        if rows:
            _write_table(Path(run_dir) / genre / "per_instance.csv", rows)
        added_all = [v for i in revision_instances(pairs, p_doc, cohort, types, min_prevalence=None)[0]
                     for v in i["added_prevalence"].values()]
        bins = np.histogram(added_all, bins=10, range=(0, 1))[0] if added_all else np.zeros(10, int)
        _write_table(tables / f"revision_bench_{genre}.csv",
                     [{"genre": genre, "kind": "pairs", "set": n, "key": k, "value": v} for n, st in counts.items() for k, v in st.items()]
                     + [{"genre": genre, "kind": "instances_per_year", "set": r["set"], "key": r["year"], "value": r["instances"]} for r in per_year]
                     + [{"genre": genre, "kind": "added_type_prevalence_hist", "set": "all_added", "key": f"{k / 10:.1f}-{(k + 1) / 10:.1f}",
                         "value": int(c)} for k, c in enumerate(bins)])
        metrics[f"{genre}/cohort_fallback_share"] = round(float(np.mean([c[1]["fallback"] for c in cache.values()])), 4) if cache else None
    return metrics


def one_per_entity(peers: list[str], entity_of: dict[str, Any], k: int) -> list[str]:
    """The first k peers with at most one document per entity (the nearest snapshot of each site)."""
    seen, out = set(), []
    for p in peers:
        e = entity_of.get(p, p)
        if e not in seen:
            seen.add(e)
            out.append(p)
            if len(out) == k:
                break
    return out


def run_diagnose(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Cohort diagnosis after gate 6.6 (decision record 007). For each cohort variant in bench.diagnose.variants
    (title/abstract weight of the document vector, category filter, k, one snapshot per entity, bits or mean
    probability), two measurements, with bootstrap CIs:
      presence   how well the cohort's prevalence predicts which types an untouched test document states (its
                 document-level presence bits): Brier score and within-document AUC, against genre prevalence;
      leak       recall@5 on OmitBench-synthetic when the cohort is retrieved with the edited document's vector (what
                 the system sees) and with the original document's vector (before the deletion), each paired with
                 genre prevalence, and their difference: how much of the cohort's gain the deletion itself removes.
    Variants with another title weight use document vectors rebuilt from the stored section and title vectors (for
    the edited documents, the section mean recovered from the stored vector). -> cohort_diagnosis_<genre>.csv."""
    import copy

    import numpy as np

    from bench.metrics import bootstrap, bootstrap_mean, paired_bootstrap, recall_at_k, within_document_auc
    from engine.cohort import CohortRetriever
    from engine.doc_presence import DocPresence
    from engine.embed import EmbeddingStore, document_vector, embedding_slug, section_mean_vector
    from engine.expectation import ProfileStore, level1
    from engine.index import load_index, make_index
    from experiments.distil import _write_table, load_schema
    from packs.builder import document_vectors
    from packs.profiles import read_profiles

    bc = cfg["bench"]
    dc = bc.get("diagnose") or {}
    boot = dict(n_boot=bc.get("bootstrap_samples", 1000), ci=bc.get("ci", 0.95), seed=bc["seed"])
    w0 = cfg["models"]["embedder"].get("title_abstract_weight", 0.5)
    tables = _path(bc["tables_dir"]) if bc.get("tables_dir") else Path(run_dir)
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        packs = _path(cfg["paths"]["packs"]) / genre
        types = [t["id"] for t in load_schema(cfg, genre)["types"]]
        cols = [f"prob_{t}" for t in types]
        profiles = ProfileStore.load(packs / "profiles", types)
        index = load_index(packs / "index", backend=(cfg.get("index") or {}).get("backend"))
        entity_of = {d: index.payloads[index.pos[d]].get("entity_id") for d in index.ids}
        emb = _path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"]) / genre
        T = EmbeddingStore.load(emb / "titles")
        title_of = {d: T.vectors[i] for i, d in enumerate(T.ids)}
        indexes: dict[float, Any] = {w0: index}
        test_vec: dict[float, dict[str, Any]] = {}
        payload_of: dict[str, Any] = {}

        def vectors(w: float) -> None:
            if w in test_vec:
                return
            c = copy.deepcopy(cfg)
            c["models"]["embedder"]["title_abstract_weight"] = w
            ids, X, pay, _ = document_vectors(c, genre, ["test"])
            test_vec[w] = dict(zip(ids, X))
            payload_of.update(dict(zip(ids, pay)))
            if w not in indexes:
                tr, Xtr, ptr, _ = document_vectors(c, genre, list((cfg.get("index") or {}).get("splits", ["train"])))
                ix = make_index((cfg.get("index") or {}).get("backend", "faiss"), Xtr.shape[1])
                ix.add(tr, Xtr, ptr)
                indexes[w] = ix

        def expectation(var: dict[str, Any], vec: Any, entity: Any, category: Any) -> Any:
            k = var.get("k", (cfg.get("cohort") or {}).get("k", 200))
            dedup = var.get("one_per_entity", False)
            r = CohortRetriever(indexes[var.get("title_abstract_weight", w0)], k=k * (3 if dedup else 1),
                                category_filter=var.get("category_filter", True))
            peers = [h.id for h in r.retrieve(np.asarray(vec, dtype=np.float32), entity_id=entity, category=category).peers]
            if dedup:
                peers = one_per_entity(peers, entity_of, k)
            return level1(profiles, peers, use=var.get("profile", "bit"))["prevalence"]

        def uniform(var: dict[str, Any]) -> Any:
            return (profiles.bits if var.get("profile", "bit") == "bit" else profiles.probs).mean(0).astype(float)

        # untouched test documents: their own document-level presence
        sp = read_profiles(_path(dc["probs"].format(genre=genre)) / "section_probs")
        doc_model = DocPresence.load(packs / "doc_presence.json")
        P = {d: doc_model.predict_proba(g.sort_values("section_idx")[cols].to_numpy(np.float64)) for d, g in sp.groupby("doc_id")}
        vectors(w0)
        pool = sorted(d for d in P if d in test_vec[w0])
        rng = np.random.default_rng(bc["seed"])
        docs = [pool[i] for i in sorted(rng.choice(len(pool), size=min(dc.get("presence_documents", 800), len(pool)), replace=False))]
        Y = np.stack([(P[d] >= 0.5).astype(int) for d in docs])
        inst = [json.loads(l) for l in open(_path(dc["synth"].format(genre=genre)) / "instances.jsonl", encoding="utf-8")]
        arr = np.load(_path(dc["synth"].format(genre=genre)) / "arrays.npz")
        regimes = dc.get("leak_regimes", ["single_type", "whole_section"])
        table = []

        def brier_auc(E: Any) -> tuple[Any, Any]:
            return (bootstrap(lambda ix: float(((E[ix] - Y[ix]) ** 2).mean()), len(docs), **boot),
                    bootstrap(lambda ix: within_document_auc(E[ix], Y[ix]), len(docs), **boot))

        for name, var in [("genre_uniform", {})] + list(dc["variants"].items()):
            w = var.get("title_abstract_weight", w0)
            vectors(w)
            if name == "genre_uniform":
                E = np.tile(uniform({}), (len(docs), 1))
            else:
                E = np.stack([expectation(var, test_vec[w][d], payload_of[d].get("entity_id"), payload_of[d].get("category"))
                              for d in docs])
            (b, blo, bhi), (a, alo, ahi) = brier_auc(E)
            row = {"genre": genre, "variant": name, "title_abstract_weight": w, "category_filter": var.get("category_filter", True),
                   "k": var.get("k", (cfg.get("cohort") or {}).get("k", 200)), "one_per_entity": var.get("one_per_entity", False),
                   "profile": var.get("profile", "bit"), "post_hoc": bool(var.get("post_hoc", False)),
                   "presence_documents": len(docs), "brier": round(b, 4), "brier_ci": f"[{blo:.4f}, {bhi:.4f}]",
                   "within_doc_auc": round(a, 4), "within_doc_auc_ci": f"[{alo:.4f}, {ahi:.4f}]"}
            metrics[f"{genre}/{name}/within_doc_auc"] = round(a, 4)
            if name != "genre_uniform":
                for regime in regimes:
                    rec: dict[str, list[float]] = {"uniform": [], "edited": [], "original": []}
                    u = uniform(var)
                    for k_, i in enumerate(inst):
                        if i["regime"] != regime:
                            continue
                        p = arr["p_doc"][k_].astype(float)
                        bits = (p >= 0.5).astype(int)
                        gt = set(i["ground_truth"])
                        tv = title_of.get(i["doc_id"]) if i.get("title") else None
                        edited = arr["doc_vector"][k_]
                        if w != w0:  # the edited document's vector at weight w, from its recovered section mean
                            edited = document_vector(section_mean_vector(edited, tv, title_abstract_weight=w0)[None], tv,
                                                     title_abstract_weight=w)
                        for src, vec in (("edited", edited), ("original", test_vec[w][i["doc_id"]])):
                            e = expectation(var, vec, i.get("entity_id"), i.get("category"))
                            rec[src].append(recall_at_k(_ranked(types, e, p, bits, np.ones(len(types)), True), gt, 5))
                        rec["uniform"].append(recall_at_k(_ranked(types, u, p, bits, np.ones(len(types)), True), gt, 5))
                    de, do = paired_bootstrap(rec["edited"], rec["uniform"], **boot), paired_bootstrap(rec["original"], rec["uniform"], **boot)
                    leak = paired_bootstrap(rec["original"], rec["edited"], **boot)
                    row.update({f"{regime}/uniform_recall@5": round(float(np.mean(rec["uniform"])), 4),
                                f"{regime}/edited_minus_uniform": f"{de['diff']:+.4f} [{de['ci_low']:+.4f}, {de['ci_high']:+.4f}]",
                                f"{regime}/original_minus_uniform": f"{do['diff']:+.4f} [{do['ci_low']:+.4f}, {do['ci_high']:+.4f}]",
                                f"{regime}/leak_original_minus_edited": f"{leak['diff']:+.4f} [{leak['ci_low']:+.4f}, {leak['ci_high']:+.4f}]"})
                    metrics[f"{genre}/{name}/{regime}/edited_minus_uniform"] = round(de["diff"], 4)
                    metrics[f"{genre}/{name}/{regime}/original_minus_uniform"] = round(do["diff"], 4)
            print(f"[diagnose {genre}] {name}: brier {b:.4f} within-doc AUC {a:.4f}", flush=True)
            table.append(row)
        keys = list(dict.fromkeys(k for r in table for k in r))
        _write_table(tables / f"cohort_diagnosis_{genre}.csv", [{k: r.get(k) for k in keys} for r in table])
    return metrics


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = (cfg.get("bench") or {}).get("stage")
    if stage == "probs":
        return run_probs(cfg, run_dir)
    if stage == "synth":
        return run_synth(cfg, run_dir)
    if stage == "eval":
        return run_eval(cfg, run_dir)
    if stage == "revision_probs":
        return run_revision_probs(cfg, run_dir)
    if stage == "revision":
        return run_revision(cfg, run_dir)
    if stage == "diagnose":
        return run_diagnose(cfg, run_dir)
    raise ValueError(f"unknown bench.stage {stage!r} (probs | synth | eval | revision_probs | revision | diagnose)")
