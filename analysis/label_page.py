"""Data for the guided hand-labelling page (plan task 4.2): the human sanity sets (packs/<genre>/label_sets.json,
key "human"), the frozen schemas, and for each section the types closest to it by wording (cosine of the pack
embedder's section vector and the type cards; a reading aid, never the teacher's answer: the page is blind to the
teacher so that teacher accuracy can be measured against it).

    .venv/bin/python -m analysis.label_page --out reports/human_labels/page/label_data.json

Every schema type is judged on the page in both genres. (Plan §4.2 had privacy derive OPP-mapped types from
OPP-115; Parsa chose on 2026-10-15 to hand-label all 58, because the calibration showed several mappings are loose.)
The page stores answers in its own database (collections labels_<genre>); Claude reads them back with ArtifactData.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from data.schema import Document, Section, read_jsonl
from packs.schema_induction import type_card

REPO = Path(__file__).resolve().parent.parent
GENRES = {"privacy": "Privacy policies", "papers": "Research papers"}

# Reading order on the page only (no effect on labels): families of related types, so neighbours sit together.
FAMILIES = {
    "papers": [
        ("Framing", ["motivates_problem", "task_description", "reviews_prior_work",
                     "states_limitations_of_existing_approaches", "proposes_method",
                     "states_method_advantages", "claims_broader_applicability"]),
        ("Theory", ["defines_notation", "assumption_and_distribution", "mdp_component_definition",
                    "states_theoretical_result", "proves_or_derives"]),
        ("Method", ["describes_model_architecture", "describes_training_objective", "describes_training_procedure",
                    "introduces_regularization", "describes_algorithm_steps", "describes_method_details",
                    "fine_tunes_pretrained_model", "describes_data_augmentation", "gives_illustrative_example"]),
        ("Data", ["describes_datasets", "reports_dataset_statistics", "describes_data_split", "describes_preprocessing",
                  "generates_synthetic_data", "describes_annotation_process"]),
        ("Setup", ["describes_evaluation_metrics", "uses_baselines", "specifies_hyperparameters",
                   "specifies_compute_resources", "names_software_and_libraries", "reports_computational_cost"]),
        ("Results", ["reports_experimental_results", "compares_with_other_methods", "reports_ablation_or_improvement",
                     "reports_variance_or_significance", "visualizes_results", "reports_observations"]),
        ("Closing", ["states_own_limitations", "suggests_future_work", "discusses_societal_impact_or_misuse",
                     "ethics_and_declarations", "code_and_data_availability", "discloses_funding",
                     "acknowledges_contributors", "refers_to_appendix"]),
    ],
    "privacy": [
        ("Collection", ["data_collection_methods", "data_collection_contests_surveys",
                        "states_information_required_to_register", "collects_usage_data_automatically",
                        "defines_personal_information", "information_provision_is_optional"]),
        ("Cookies and tracking", ["uses_cookies", "uses_web_beacons", "tracking_data_not_linked_to_identity",
                                  "third_party_advertising_and_cookies", "disclaims_control_of_third_party_cookies",
                                  "browser_cookie_controls", "warns_disabling_cookies_limits_site",
                                  "do_not_track_response"]),
        ("Use", ["data_use_for_services", "uses_data_to_improve_services", "uses_data_to_personalize",
                 "sends_marketing_communications"]),
        ("Sharing and transfer", ["does_not_sell_or_share", "shares_with_service_providers",
                                  "shares_with_affiliates_and_partners", "shares_aggregated_data",
                                  "discloses_when_legally_required", "transfers_data_in_business_transfer",
                                  "transfers_data_internationally", "discloses_public_areas"]),
        ("Other sites", ["provides_third_party_links", "third_party_sites_and_their_policies",
                         "scope_excludes_third_party_sites"]),
        ("Security", ["states_security_measures", "encrypts_data_in_transit", "stores_data_securely",
                      "limits_internal_access", "does_not_store_payment_cards", "security_not_guaranteed",
                      "advises_users_on_security"]),
        ("Choices and rights", ["opt_out_of_communications", "opt_out_of_tracking_or_ads", "user_data_control",
                                "deletion_on_request", "disclose_data_rights", "california_disclosure_rights"]),
        ("Retention", ["states_retention_period"]),
        ("Children", ["no_knowing_collection_from_children", "states_minimum_age", "deletes_children_data_if_found"]),
        ("About the policy and the service", ["states_policy_scope_and_commitment", "policy_coverage",
                                             "company_identification_and_service_definition",
                                             "reserves_right_to_change_policy", "states_how_changes_are_notified",
                                             "states_last_updated_date", "continued_use_means_acceptance",
                                             "consent_by_use", "content_and_usage_restrictions",
                                             "complies_with_laws_and_programs", "governing_law_and_disputes",
                                             "provides_privacy_contact"]),
    ],
}


def plain(type_id: str) -> str:
    s = type_id.replace("_", " ")
    return s[0].upper() + s[1:]


def section_vectors(genre: str, sections: list[Section], stores: list[Path], embedder_id: str) -> np.ndarray:
    from engine.embed import Embedder, EmbeddingStore

    found: list = [None] * len(sections)
    for path in stores:
        if (path / "rows.jsonl").exists():
            got = EmbeddingStore.load(path).lookup([s.text for s in sections])
            found = [f if f is not None else g for f, g in zip(found, got)]
    missing = [i for i, v in enumerate(found) if v is None]
    if missing:
        print(f"[{genre}] embedding {len(missing)} sections not in a store (CPU)")
        emb = Embedder(embedder_id, devices=["cpu"], max_length=1024, batch_size=8)
        for i, v in zip(missing, emb.encode([sections[i].text for i in missing])):
            found[i] = v
    X = np.asarray(found, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def genre_data(genre: str, embedder_id: str, top_k: int) -> dict:
    from engine.embed import Embedder

    schema = yaml.safe_load((REPO / "packs" / genre / "schema.yaml").read_text(encoding="utf-8"))
    sets = json.loads((REPO / "packs" / genre / "label_sets.json").read_text(encoding="utf-8"))
    refs = sets["human"]
    # Every type is judged by hand in both genres (Parsa, 2026-10-15): the calibration showed the OPP-115 -> schema
    # mapping is loose for several types, so OPP-derived labels are measured against the human set as well.
    judged = [t["id"] for t in schema["types"]]
    order = [tid for _, ids in FAMILIES[genre] for tid in ids]
    if sorted(order) != sorted(judged):
        raise ValueError(f"{genre}: page families do not cover the judged types exactly: "
                         f"{sorted(set(order) ^ set(judged))}")
    by_id = {t["id"]: t for t in schema["types"]}
    want = {(r["doc_id"], r["section_idx"]): r["text_sha1"] for r in refs}
    docs = {d for d, _ in want}
    processed = REPO / "data" / "processed" / genre
    all_secs: dict[str, list[Section]] = defaultdict(list)
    for s in read_jsonl(processed / "sections.jsonl", Section):
        if s.doc_id in docs:
            all_secs[s.doc_id].append(s)
    titles = {d.id: d.title for d in read_jsonl(processed / "documents.jsonl", Document) if d.id in docs}
    sections = []
    for r in refs:
        s = next(x for x in all_secs[r["doc_id"]] if x.idx == r["section_idx"])
        if hashlib.sha1(s.text.encode("utf-8")).hexdigest() != r["text_sha1"]:
            raise ValueError(f"{genre}: section {r['doc_id']}#{r['section_idx']} changed since it was selected")
        sections.append(s)
    slug = embedder_id.replace("/", "__")
    stores = [REPO / "data" / "embeddings" / slug / genre / "sections",
              REPO / "data" / "embeddings" / slug / genre / "_sections_unsplit_2026-10-15"]
    S = section_vectors(genre, sections, stores, embedder_id)
    cards = [by_id[tid] for tid in judged]
    T = Embedder(embedder_id, devices=["cpu"], max_length=512, batch_size=16).encode([type_card(t) for t in cards])
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    sims = S @ T.T
    out_sections = []
    for s, row in zip(sections, sims):
        n = len(all_secs[s.doc_id])
        out_sections.append({
            "key": f"{s.doc_id}#{s.idx}", "doc_id": s.doc_id, "idx": s.idx, "of": n, "heading": s.heading or "",
            "title": titles.get(s.doc_id) or "", "text": s.text,
            "previous_heading": next((x.heading for x in all_secs[s.doc_id] if x.idx == s.idx - 1), None) or "",
            "closest": [cards[j]["id"] for j in np.argsort(-row)[:top_k]],
        })
    return {
        "genre": genre, "label": GENRES[genre], "schema_version": schema["version"], "set_source": sets["source"],
        "judged_types": len(judged), "annotation_types": len(schema["types"]) - len(judged),
        "families": [{"name": name, "types": ids} for name, ids in FAMILIES[genre]],
        "types": {tid: {"id": tid, "name": plain(tid), "definition": by_id[tid]["definition"],
                        "examples": by_id[tid].get("positive_examples", []), "near_miss": by_id[tid].get("near_miss", ""),
                        "where": by_id[tid].get("section_hint", "")} for tid in judged},
        "sections": out_sections,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/human_labels/page/label_data.json")
    ap.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    data = {"version": 1, "genres": {g: genre_data(g, args.embedder, args.top_k) for g in GENRES}}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data["data_sha1"] = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB), data {data['data_sha1']}")


if __name__ == "__main__":
    main()
