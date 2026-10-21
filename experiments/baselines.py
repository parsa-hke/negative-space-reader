"""Frontier-model baselines (d)-(f) (plan §0.2 #17; §7.3, §9.3).

Baseline (e), candidate list (7.3, `baselines.stage: candidate`): a seeded sample of `llm_gap_instances`
OmitBench-synthetic instances per genre (`baselines.regimes` of the run `baselines.synth_run`), each edited document
written as plain Markdown into prompts/llm_gap_candidate.md with the genre's frozen catalogue; the model (the
`models.teacher` route: configs/llm_gap_anthropic.yaml for Claude Opus 5.5 by Message Batches, configs/openrouter.yaml
for the open teacher) returns at most `top_k` absent type ids, most conspicuous first. Scored like the system
(bench/metrics.py) and paired, instance by instance, with the level-1 and genre-uniform rankings recorded in
`baselines.compare_to` (bench_eval_v2): MRR is cut at top_k for every system, since the model ranks only top_k.

    python -m experiments.run --config configs/baselines.yaml --overlay llm_gap_anthropic --name llm_gap_dry \
        --set baselines.dry_run=true --set 'run.seeds=[0]'

dry_run renders every prompt, writes <run>/<genre>/requests.jsonl and an estimate of tokens and dollars, and calls
no model. Baselines (d) and (f): Phase 9.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Literal


def plain_markdown(title: str | None, sections: list[dict[str, Any]]) -> str:
    """An edited document as the model reads it: title, then each section under its heading (no markers)."""
    out = [f"# {title}"] if title else []
    for s in sections:
        out.append((f"## {s['heading']}\n" if s.get("heading") else "") + s["text"].strip())
    return "\n\n".join(out)


def sample_instances(folder: Path, n: int, regimes: list[str], seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in open(folder / "instances.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["regime"] in regimes]
    rows.sort(key=lambda r: (r["regime"], r["doc_id"]))
    return random.Random(seed).sample(rows, min(n, len(rows)))


def answer_model(types: list[str]) -> Any:
    from pydantic import BaseModel, create_model

    return create_model("AbsentTypes", __base__=BaseModel, absent_types=(list[Literal[tuple(types)]], ...))  # type: ignore[valid-type]


def _route(teacher: Any, cfg: dict[str, Any]) -> str:
    tcfg = cfg["models"]["teacher"]
    return getattr(teacher, "route", None) or f"{tcfg.get('backend')}:{tcfg.get('id')}"


def _mean_ci(x: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    from bench.metrics import bootstrap_mean

    return bootstrap_mean(x, n_boot=n_boot, seed=seed)


def _answer(teacher: Any, prompts: list[str], model: Any, max_tokens: int) -> list[Any]:
    """All answers in one call; if some item fails (e.g. a prompt beyond the route's context), each prompt alone, the
    failed ones None (scored as an empty ranking and counted as unanswered; prompts are never truncated). Answers
    already received come back from the teacher's cache."""
    from models.teacher import StructuredOutputError

    try:
        return teacher.generate_structured_batch(prompts, model, max_tokens=max_tokens)
    except StructuredOutputError as e:
        print(f"[baselines] {e}; answering one prompt at a time", flush=True)
    out = []
    for p in prompts:
        try:
            out.append(teacher.generate_structured_batch([p], model, max_tokens=max_tokens)[0])
        except StructuredOutputError:
            out.append(None)
    return out


def run_candidate(cfg: dict[str, Any], run_dir: Path, *, teacher: Any = None) -> dict[str, Any]:
    import yaml

    from bench.metrics import instance_metrics, paired_bootstrap
    from models.teacher import PROMPTS, load_teacher
    from packs.builder import _path

    b = cfg["baselines"]
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    top_k, n, dry = int(b.get("top_k", 10)), int(b["llm_gap_instances"]), bool(b.get("dry_run", False))
    regimes, seed = list(b.get("regimes") or ["single_type"]), int(b.get("sample_seed", 2026))
    n_boot = cfg["bench"].get("bootstrap_samples", 1000)
    prompt = PROMPTS.get("llm_gap_candidate")
    run_dir = Path(run_dir)
    m: dict[str, Any] = {"dry_run": dry, "prompt": prompt.tag}
    if not dry and teacher is None:
        teacher = load_teacher(cfg["models"]["teacher"])
    system = None if dry else _route(teacher, cfg)
    for g in genres:
        out = run_dir / g
        out.mkdir(parents=True, exist_ok=True)
        folder = Path(str(b["synth_run"]).format(genre=g))
        insts = sample_instances(folder, n, regimes, seed)
        want = {f"{r['regime']}:{r['doc_id']}" for r in insts}
        edited = {}
        for line in open(folder / "edited.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["instance_id"] in want:
                edited[r["instance_id"]] = r
        schema = yaml.safe_load((_path(cfg["paths"]["packs"]) / g / "schema.yaml").read_text(encoding="utf-8"))
        types = [t["id"] for t in schema["types"]]
        info = cfg["genre_info"][g]
        catalogue = "\n".join(f"{t['id']}: {t.get('definition', '').strip()}" for t in schema["types"])
        ids = [f"{r['regime']}:{r['doc_id']}" for r in insts]
        prompts = [prompt.render(document_kind=info["document_kind"], author_role=info["author_role"],
                                 type_catalogue=catalogue, top_k=top_k,
                                 document=plain_markdown(edited[i].get("title"), edited[i]["sections"])) for i in ids]
        chars = sum(len(p) for p in prompts)
        est = b.get("estimate") or {}
        tok_in = chars / float(est.get("chars_per_token", 3.5))
        tok_out = len(prompts) * float(est.get("output_tokens", 2500))
        usd = (tok_in * est.get("usd_per_mtok_input", 4.0) + tok_out * est.get("usd_per_mtok_output", 20.0)) / 1e6
        usd *= float(est.get("batch_discount", 0.5))
        m.update({f"{g}/instances": len(ids), f"{g}/prompt_chars": chars, f"{g}/estimated_input_tokens": round(tok_in),
                  f"{g}/estimated_output_tokens": round(tok_out), f"{g}/estimated_usd": round(usd, 2)})
        if dry:
            with open(out / "requests.jsonl", "w", encoding="utf-8") as f:
                for i, p in zip(ids, prompts):
                    f.write(json.dumps({"instance_id": i, "prompt": p}, ensure_ascii=False) + "\n")
            continue
        answers = _answer(teacher, prompts, answer_model(types), int(b.get("max_tokens", 16000)))
        gt = {f"{r['regime']}:{r['doc_id']}": set(r["ground_truth"]) for r in insts}
        rows, invalid = [], 0
        for i, a in zip(ids, answers):
            ranked: list[str] = []
            for t in (a.absent_types if a is not None else []):
                if t not in types:
                    invalid += 1
                elif t not in ranked:
                    ranked.append(t)
            ranked = ranked[:top_k]
            im = instance_metrics(ranked, gt[i], ks=(1, 5, 10))
            rows.append({"instance_id": i, "regime": i.split(":", 1)[0], "system": system, "ground_truth": len(gt[i]),
                         "answered": a is not None, **{k: im[k] for k in ("recall@1", "recall@5", "recall@10", "ndcg@10")},
                         "mrr@10": im["mrr"], "rank_first": im["rank_first"] or "", "ranked": "|".join(ranked)})
        with open(out / "per_instance.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        m[f"{g}/invalid_ids"] = invalid
        m[f"{g}/unanswered"] = sum(not r["answered"] for r in rows)
        # the recorded systems on the same instances, MRR cut at top_k like the model's
        rec: dict[str, dict[str, dict[str, float]]] = {}
        for r in csv.DictReader(open(Path(b["compare_to"]) / g / "per_instance.csv", encoding="utf-8")):
            if r["instance_id"] in gt:
                rf = int(r["rank_first"]) if r["rank_first"] else None
                rec.setdefault(r["system"], {})[r["instance_id"]] = {
                    **{k: float(r[k]) for k in ("recall@1", "recall@5", "recall@10", "ndcg@10")},
                    "mrr@10": 1 / rf if rf and rf <= top_k else 0.0}
        metrics = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
        llm = {r["instance_id"]: {k: float(r[k]) for k in metrics} for r in rows}
        table = []
        for name, vals in [(system, llm)] + [(s, rec[s]) for s in ("level1", "genre_uniform") if s in rec]:
            row: dict[str, Any] = {"genre": g, "system": name, "n": len(ids)}
            for k in metrics:
                mean, lo, hi = _mean_ci([vals[i][k] for i in ids], n_boot, seed)
                row.update({k: round(mean, 3), f"{k}_ci": f"[{lo:.3f}, {hi:.3f}]"})
            table.append(row)
            m[f"{g}/{name}/recall@5"] = row["recall@5"]
        for s in ("level1", "genre_uniform"):
            if s in rec:
                row = {"genre": g, "system": f"{system} - {s}", "n": len(ids)}
                for k in metrics:
                    d = paired_bootstrap([llm[i][k] for i in ids], [rec[s][i][k] for i in ids], n_boot=n_boot, seed=seed)
                    row.update({k: round(d["diff"], 3), f"{k}_ci": f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]"})
                table.append(row)
        for folder_out in [out] + ([_path(b["tables_dir"])] if b.get("tables_dir") else []):
            folder_out.mkdir(parents=True, exist_ok=True)
            path = folder_out / f"llm_gap_candidate_{g}.csv"
            rows_out = list(table)
            if path.exists():  # keep the other models' rows (one recorded run per model); this run replaces its own
                ours = {r["system"] for r in table}
                kept = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["system"] not in ours]
                rows_out = [r for r in kept if " - " not in r["system"]] + [r for r in table if " - " not in r["system"]] \
                    + [r for r in kept if " - " in r["system"]] + [r for r in table if " - " in r["system"]]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(rows_out)
    return m


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = (cfg.get("baselines") or {}).get("stage", "candidate")
    if stage != "candidate":
        raise ValueError(f"baselines.stage {stage!r}: only candidate (baseline (e), plan 7.3) is implemented")
    return run_candidate(cfg, run_dir)
