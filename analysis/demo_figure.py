"""The demo figure of the report (Section 4): the document panel of the Gradio demo with its first ghost paragraph,
rendered from a recorded analysis and the corpus, never from a live call, and screenshotted with headless Chrome.

    python -m analysis.demo_figure --report runs/space_cpu_timing_privacy_s0/report_privacy_pl_36309.json \\
        --sections data/processed/privacy/sections.jsonl --out reports/figures/demo_ghost.png

The HTML is the app's own renderer (app/gradio_app.py document_html) with the app's stylesheet, cut to the section
the first ghost follows, so the figure shows what a user sees without inventing anything. Chrome (macOS path below, or
--chrome) must exist to write the PNG; build_html() alone needs nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH = 860


def _sections(path: Path, doc_id: str) -> list[tuple[str | None, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if f'"{doc_id}"' in line[:60]:
                r = json.loads(line)
                if r["doc_id"] == doc_id:
                    rows.append(r)
    rows.sort(key=lambda r: r["idx"])
    return [(r.get("heading"), r["text"]) for r in rows]


def build_html(report: dict, sections: list[tuple[str | None, str]], *, ghosts: int = 1, max_chars: int = 700) -> str:
    """The title, the section the first ghost follows (shortened with an ellipsis), then the ghost, so that one
    ghost paragraph fits one figure; a note says how many sections sit above."""
    from app.gradio_app import css, document_html
    from engine.explain import place_ghosts

    absences = report["absences"][:ghosts]
    placed = place_ghosts(absences, [h for h, _ in sections])
    last = max(placed) if placed else len(sections) - 1
    h, t = sections[last]  # the section the first ghost follows, shortened, then the ghost: one figure, one ghost
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] + " …"
    lead = f"<p class='nsr-small'>… {last} section{'s' if last != 1 else ''} above …</p>" if last else ""
    body = document_html([(h, t)], absences, title=report.get("title")).replace("</h2>", "</h2>" + lead, 1)
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + css() +
            f"\nbody {{ margin: 0; padding: 22px 26px; background: var(--nsr-void); width: {WIDTH - 52}px; }}"
            "\n.nsr-ghost { animation: none !important; opacity: 1 !important; transform: none !important; }"
            "\n.nsr-bar i { animation: none !important; }"
            "</style></head><body>" + body + "</body></html>")


def screenshot(html: str, out: Path, *, chrome: str = CHROME, height: int = 1400) -> Path:
    """Render with headless Chrome and crop the PNG to its content."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "page.html"
        page.write_text(html, encoding="utf-8")
        png = Path(d) / "shot.png"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={WIDTH},{height}", "--virtual-time-budget=6000", f"--screenshot={png}",
                        page.as_uri()], check=True, capture_output=True, timeout=120)
        from PIL import Image, ImageChops

        im = Image.open(png).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))
        box = ImageChops.difference(im, bg).getbbox()
        if box:
            pad = 2 * 18
            im = im.crop((0, 0, im.width, min(im.height, box[3] + pad)))
        im.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="runs/space_cpu_timing_privacy_s0/report_privacy_pl_36309.json")
    ap.add_argument("--sections", default="data/processed/privacy/sections.jsonl")
    ap.add_argument("--doc-id", default="privacy_pl:36309")
    ap.add_argument("--out", default="reports/figures/demo_ghost.png")
    ap.add_argument("--chrome", default=CHROME)
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    html = build_html(report, _sections(Path(a.sections), a.doc_id))
    print(screenshot(html, Path(a.out), chrome=a.chrome))


if __name__ == "__main__":
    main()
