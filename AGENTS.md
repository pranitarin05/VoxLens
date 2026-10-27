# AGENTS.md: rules for working in this repo

Project: VoC Intelligence Platform. The spec is docs/PRD.md. Read it before any task.

## Working rules
1. Do only the phase I ask for. Never start the next phase.
2. Before writing code, state your plan in under 15 lines and list every assumption. Ask when something is ambiguous.
3. Never invent numbers. Every metric, latency, cost, row count or benchmark you report must come from a command you ran or a file a script produced. If you could not run something, say so.
4. Never guess a library API. Check the installed version and read its documentation or source.
5. Before adding a dependency, check its latest stable version and its compatibility with Python 3.11, then pin it exactly.
6. Never commit secrets, data files, model weights or .env. Use .env.example for variable names only.
7. Never execute SQL written by an LLM. Use parameterised queries and whitelisted templates.
8. Every LLM call path must respect MAX_LLM_SPEND_USD and use the disk cache.
9. Write tests together with the code. Run lint and tests before saying a task is done and show the output.
10. Keep functions small, typed and documented. No hidden global state. Seeds and paths come from config.
11. Make every pipeline step idempotent: re-running must not duplicate data.
12. When a command fails, show the real error and fix the cause. Do not silence errors, skip tests or weaken assertions to make something pass.
13. Do not modify files unrelated to the current task. End every task with a summary of changed files.
14. Record important decisions as short notes in docs/decisions/ (and append to decision.md).

## Why each rule exists
Rules 3 and 4 stop invented results and invented APIs (the two most damaging agent habits); 5 prevents version conflicts; 6 and 7 are security basics; 8 protects your money; 9 and 12 stop fake 'done' claims; 1 and 13 prevent scope creep and surprise edits.

## Project-specific notes
- Companion docs: flow.md (how the system flows), decision.md (append-only decision log).
- Python: 3.11.9 (last Windows-installer release of 3.11; 3.11 is security-fixes-only per PEP 664).
- LLM: Groq, two keys in .env (primary + backup fallback).
- Scale engine: Dask (not Spark).
- Training: local RTX 3050 4GB only; no Colab/cloud.
