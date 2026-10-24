"""Assemble the class report (plan 7.4) from its source: reports/report_source.md -> reports/report.md.

    python -m analysis.report                      # default paths
    python -m analysis.report --check              # fail if reports/report.md is not what the source produces
    python -m analysis.report --words              # prose word count per section (tables, captions, figures,
                                                   # references and the GenAI acknowledgement excluded)

The source carries three kinds of placeholder, so that no result number is typed by hand:
  {{v:key|fmt}}          a value from reports/tables/report_fragments/values.json (written by analysis/tables.py);
                         fmt is f3, s3, ci, sci, pct0 or empty (ints get thousands separators)
  <!-- include: name --> the fragment reports/tables/report_fragments/<name>.md, inserted verbatim
  [@key; @key2]          citations, numbered by first appearance; the IEEE list below replaces {{references}}
A missing key, fragment or reference raises, so a stale table cannot silently leave a number out.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REFS = {
    "truong": 'T. H. Truong, T. Baldwin, K. Verspoor, and T. Cohn, "Language models are not naysayers: An analysis of language models on negation benchmarks," in *Proc. 12th Joint Conf. on Lexical and Computational Semantics (\\*SEM 2023)*, Toronto, Canada, 2023, pp. 101–114, doi: 10.18653/v1/2023.starsem-1.10.',
    "hossain": 'M. M. Hossain, D. Chinnappa, and E. Blanco, "An analysis of negation in natural language understanding corpora," in *Proc. 60th Annu. Meeting of the Association for Computational Linguistics (Short Papers)*, Dublin, Ireland, 2022, pp. 716–723, doi: 10.18653/v1/2022.acl-short.81.',
    "weller": 'O. Weller, D. Lawrie, and B. Van Durme, "NevIR: Negation in neural information retrieval," in *Proc. 18th Conf. of the European Chapter of the Association for Computational Linguistics (Long Papers)*, St. Julian\'s, Malta, 2024, pp. 2274–2287.',
    "wilson1927": 'E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927, doi: 10.1080/01621459.1927.10502953.',
    "amos": 'R. Amos, G. Acar, E. Lucherini, M. Kshirsagar, A. Narayanan, and J. Mayer, "Privacy policies over time: Curation and analysis of a million-document dataset," in *Proc. Web Conf. 2021 (WWW \'21)*, Ljubljana, Slovenia, 2021, pp. 2165–2176, doi: 10.1145/3442381.3450048.',
    "wilson2016": 'S. Wilson et al., "The creation and analysis of a website privacy policy corpus," in *Proc. 54th Annu. Meeting of the Association for Computational Linguistics (Long Papers)*, Berlin, Germany, 2016, pp. 1330–1340, doi: 10.18653/v1/P16-1126.',
    "saier": 'T. Saier, J. Krause, and M. Färber, "unarXive 2022: All arXiv publications pre-processed for NLP, including structured full-text and citation network," in *Proc. 2023 ACM/IEEE Joint Conf. on Digital Libraries (JCDL)*, Santa Fe, NM, USA, 2023, pp. 66–70.',
    "qwen3emb": 'Y. Zhang et al., "Qwen3 Embedding: Advancing text embedding and reranking through foundation models," arXiv:2506.05176, 2025.',
    "umap": 'L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," arXiv:1802.03426, 2018.',
    "campello": 'R. J. G. B. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on hierarchical density estimates," in *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, Gold Coast, Australia, 2013, pp. 160–172, doi: 10.1007/978-3-642-37456-2_14.',
    "mcinnes2017": 'L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," *J. Open Source Softw.*, vol. 2, no. 11, p. 205, 2017, doi: 10.21105/joss.00205.',
    "neurips": 'NeurIPS, "NeurIPS paper checklist guidelines," 2026. [Online]. Available: https://neurips.cc/public/guides/PaperChecklist (accessed Oct. 13, 2026).',
    "aclchecklist": 'ACL Rolling Review, "Responsible NLP research checklist," 2024. [Online]. Available: https://aclrollingreview.org/responsibleNLPresearch/ (accessed Oct. 13, 2026).',
    "khattab": 'O. Khattab and M. Zaharia, "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT," in *Proc. 43rd Int. ACM SIGIR Conf. on Research and Development in Information Retrieval*, 2020, pp. 39–48, doi: 10.1145/3397271.3401075.',
    "colbertsmall": 'Answer.AI, "answerai-colbert-small-v1," model card, Hugging Face, 2024. [Online]. Available: https://huggingface.co/answerdotai/answerai-colbert-small-v1',
    "clavie": 'B. Clavié, "JaColBERTv2.5: Optimising multi-vector retrievers to create state-of-the-art Japanese retrievers with constrained resources," arXiv:2407.20750, 2024.',
    "bgem3": 'J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, "BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation," arXiv:2402.03216, 2024.',
    "faiss": 'M. Douze et al., "The Faiss library," arXiv:2401.08281, 2024.',
    "gradio": 'A. Abid, A. Abdalla, A. Abid, D. Khan, A. Alfozan, and J. Zou, "Gradio: Hassle-free sharing and testing of ML models in the wild," arXiv:1906.02569, 2019.',
    "wolf": 'T. Wolf et al., "Transformers: State-of-the-art natural language processing," in *Proc. 2020 Conf. on Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45, doi: 10.18653/v1/2020.emnlp-demos.6.',
    "reimers": 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. 2019 Conf. on Empirical Methods in Natural Language Processing and 9th Int. Joint Conf. on Natural Language Processing (EMNLP-IJCNLP)*, Hong Kong, China, 2019, pp. 3982–3992, doi: 10.18653/v1/D19-1410.',
    "qwen25": 'Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.',
    "llamacpp": 'G. Gerganov and contributors, "llama.cpp: LLM inference in C/C++," 2023. [Online]. Available: https://github.com/ggml-org/llama.cpp',
    "guo": 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. on Machine Learning*, PMLR 70, Sydney, Australia, 2017, pp. 1321–1330.',
    "naeini": 'M. P. Naeini, G. F. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in *Proc. 29th AAAI Conf. on Artificial Intelligence*, Austin, TX, USA, 2015, pp. 2901–2907, doi: 10.1609/aaai.v29i1.9602.',
    "efron": 'B. Efron, "Bootstrap methods: Another look at the jackknife," *Ann. Statist.*, vol. 7, no. 1, pp. 1–26, 1979, doi: 10.1214/aos/1176344552.',
    "harkous": 'H. Harkous, K. Fawaz, R. Lebret, F. Schaub, K. G. Shin, and K. Aberer, "Polisis: Automated analysis and presentation of privacy policies using deep learning," in *Proc. 27th USENIX Security Symp.*, Baltimore, MD, USA, 2018, pp. 531–548.',
    "pineau": 'J. Pineau et al., "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)," *J. Mach. Learn. Res.*, vol. 22, no. 164, pp. 1–20, 2021.',
    "magnusson": 'I. Magnusson, N. A. Smith, and J. Dodge, "Reproducibility in NLP: What have we learned from the checklist?," in *Findings of the Association for Computational Linguistics: ACL 2023*, Toronto, Canada, 2023, pp. 12789–12811, doi: 10.18653/v1/2023.findings-acl.809.',
    "lee": 'J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh, "Set Transformer: A framework for attention-based permutation-invariant neural networks," in *Proc. 36th Int. Conf. on Machine Learning*, PMLR 97, Long Beach, CA, USA, 2019, pp. 3744–3753.',
}


def fmt(v, f: str) -> str:
    if f == "f3":
        return f"{float(v):.3f}"
    if f == "s3":
        return f"{float(v):+.3f}"
    if f == "ci":
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    if f == "sci":
        return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
    if f == "pct0":
        return f"{100 * float(v):.0f}%"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def build(source: str, values: dict, fragments: Path, refs: dict | None = None) -> str:
    refs = refs or REFS

    def sub_value(m):
        key, f = m.group(1), (m.group(2) or "")
        if key not in values:
            raise KeyError(f"values.json has no key {key!r}")
        return fmt(values[key], f)

    text = re.sub(r"\{\{v:([^}|]+)(?:\|([a-z0-9]+))?\}\}", sub_value, source)

    def sub_include(m):
        p = fragments / f"{m.group(1)}.md"
        if not p.is_file():
            raise FileNotFoundError(p)
        return f"<!-- include: {m.group(1)}.md -->\n\n{p.read_text(encoding='utf-8').strip()}\n\n"

    text = re.sub(r"<!-- include: (\w+) -->\s*", sub_include, text)
    order: list[str] = []

    def sub_cite(m):
        nums = []
        for k in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if k not in refs:
                raise KeyError(f"no reference {k!r}")
            if k not in order:
                order.append(k)
            nums.append(str(order.index(k) + 1))
        return "[" + ", ".join(nums) + "]"

    text = re.sub(r"\[(@[^\]]+)\]", sub_cite, text)
    if "{{references}}" not in text:
        raise ValueError("source has no {{references}} marker")
    return text.replace("{{references}}", "\n".join(f"[{i + 1}] {refs[k]}" for i, k in enumerate(order)))


def word_counts(report: str) -> dict[str, int]:
    """Prose words per section of an assembled report: table rows, figure lines, captions and HTML comments do not
    count; the GenAI acknowledgement and everything after it (the references) are left out. The title block counts
    as the first section ('')."""
    text = re.sub(r"<!--.*?-->", "", report, flags=re.S).split("## GenAI acknowledgement")[0]
    parts = re.split(r"^(## .*)$", text, flags=re.M)

    def count(block: str) -> int:
        return sum(len(line.lstrip("#").split()) for line in block.splitlines()
                   if line.strip() and not line.lstrip().startswith(("|", "!["))
                   and not re.match(r"^\*(Figure|Table) \d", line.strip()))

    out = {"": count(parts[0])}
    for i in range(1, len(parts), 2):
        out[parts[i][3:].strip()] = count(parts[i + 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="reports/report_source.md")
    ap.add_argument("--fragments", default="reports/tables/report_fragments")
    ap.add_argument("--out", default="reports/report.md")
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from what the source produces")
    ap.add_argument("--words", action="store_true", help="print the prose word count per section of --out and exit")
    a = ap.parse_args()
    if a.words:
        counts = word_counts(Path(a.out).read_text(encoding="utf-8"))
        for name, n in counts.items():
            print(f"{n:6,d}  {name or '(title block)'}")
        body = sum(n for k, n in counts.items() if k and not k.startswith("Abstract"))
        print(f"{sum(counts.values()):6,d}  total (title block and abstract included)\n{body:6,d}  sections 1 onward")
        return
    fragments = Path(a.fragments)
    values = json.loads((fragments / "values.json").read_text(encoding="utf-8"))
    text = build(Path(a.source).read_text(encoding="utf-8"), values, fragments)
    out = Path(a.out)
    if a.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print("up to date" if same else "STALE: rerun python -m analysis.report")
        raise SystemExit(0 if same else 1)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
