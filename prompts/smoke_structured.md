---
id: smoke_structured
version: 1
description: >
  Smoke-test prompt for the teacher round trip (plan §1.7). Not used for any measured result.
  Asks for a fixed, easily checked extraction so the structured-output path can be verified on
  the 1.5B stand-in and on the full teacher.
---
Read the passage and return a JSON object with these fields:
- "topic": one or two words naming what the passage is about
- "mentions_retention": true if the passage says how long data is kept, otherwise false
- "retention_days": the retention period in days as an integer, or null if none is stated

Passage:
{{text}}
