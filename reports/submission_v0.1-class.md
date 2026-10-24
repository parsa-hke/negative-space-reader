# 7.7 — `v0.1-class`: tag and submission list

Written 2026-10-25. Plan §0.3 (class deliverables) and Appendix C (brief and rubric). **GATE: submitted.**

## What is submitted

| Item | Where | State on 2026-10-25 |
|---|---|---|
| Technical report, 3,000–4,000 words | `reports/report.docx` (A4, Calibri, 16 tables, 4 figures, 30 IEEE references), built from `reports/report.md` | 3,989 words all-in (title block and abstract included; `python -m analysis.report --words`); AI-tell checker zero errors. **Parsa:** open it once in Word, confirm the GenAI acknowledgement, add the Anthropic references for Claude Opus 5.5 / Haiku 5.5 if a vendor page exists. |
| Working prototype | https://huggingface.co/spaces/parsa-hke/negative-space-reader (public, `cpu-basic`, sleeps after 48 h idle) | Live since 2026-10-21; results identical to the Mac. Wake it before the demo. Re-uploaded only on Parsa's say-so. |
| Demonstration | 3–5 minute walkthrough video (`reports/demo_video_plan.md`) | **Not recorded yet (7.2).** |
| Repository with README and commented code | https://github.com/parsa-hke/negative-space-reader at tag `v0.1-class` | **The repository is private.** Before submitting: make it public, or add the marker as a collaborator, or attach the archive `git archive v0.1-class` produces. README rewritten 2026-10-25 (7.6). |

## Tag `v0.1-class` (after the last commit Parsa approves)

The package version is bumped to 0.1.0 in `pyproject.toml` for this tag. Authored and pushed by Parsa only.

```bash
git status --short
```

```bash
.venv/bin/python -m pytest -q
```

```bash
git tag -a v0.1-class -m "v0.1-class: MAAI 902 Assessment 2 submission (Phases 1-7, lean tier): both packs, level-1 expectation, OmitBench-synthetic and -revision, frontier comparison, Gradio demo on a CPU Space, report"
```

```bash
git push origin main v0.1-class
```

Optional before the tag, when no job writes into the hub: `python -m experiments.hub_sync push --bundles runs`
(the two `space_cpu_timing_<genre>_s0` runs are the only files the hub lacks).

## Checklist, in order

1. Parsa's final read of `reports/report.md`; edits in `reports/report_source.md`; `python -m analysis.report`;
   `python -m analysis.report --check`; the checker; `--words`; rebuild the .docx (session converter or pandoc).
2. Record the video per `reports/demo_video_plan.md`; export; upload per the subject outline; link it in the README.
3. Fresh-clone smoke passes (7.6; recorded in CHANGELOG 2026-10-25).
4. Commit everything (CHANGELOG, HANDOFF, plan ticks for 7.2, 7.4–7.7), tag, push.
5. Repository visible to the marker (public or collaborator); `LICENSE` and the README status line checked; the Space
   public and warm (restart it, run one paper so the 36-second cold call is already paid), then submit: .docx, repository link at the
   tag, Space link, video link.
6. Tick 7.7 in the plan with the submission date. GATE passed.
