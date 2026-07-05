# Project Memory

## Collaboration Loop

The current workflow is:

1. Codex writes or updates code, benchmark scripts, and local documentation.
2. The user pushes the work to GitHub.
3. The user pulls the latest code on an AutoDL RTX 3090 machine and runs the
   requested validation commands there.
4. The user sends the runtime output back to Codex.
5. Codex analyzes the results, records appropriate validation notes in
   `benchmarks/results/*.md`, and prepares a concise report for GPT.
6. The user forwards Codex's report to GPT, then shares GPT's reply back with
   Codex before the next implementation step.

## Operating Principles

- Codex may recommend the next direction, but reports should clearly separate:
  - measured evidence;
  - Codex's interpretation;
  - recommended next steps;
  - open questions for GPT.
- Do not claim benchmark success until the user has run the relevant command on
  the AutoDL RTX 3090 environment and pasted the result back.
- Prefer phase-style artifacts under `benchmarks/results/` so GPT can review a
  stable summary rather than raw logs only.
- Keep the project direction profile-driven: write small validation scripts,
  run them on 3090, analyze evidence, then decide whether to implement.
