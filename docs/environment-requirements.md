# Environment requirements for analysis paths

Environment validation guidance for analysis paths. Verify any environment with:

```bash
python scripts/verify_env.py           # SKIPPED tolerated (core-only installs)
python scripts/verify_env.py --strict  # SKIPPED fails (full installs, release gate)
```

Exit code is 0 only when nothing FAILED. Each check is independent, so one absent
optional dependency never masks another failure.

## Why this matters beyond packaging hygiene

When a specialist agent errors on a missing dependency, the supervisor writes
`execute_custom_code` to work around it — and sometimes reports success for work that
never ran. Environment faults therefore surface as *routing* and *escalation* problems,
and pollute evaluation of those paths. Validate the environment before interpreting
analysis outcomes.

## Numba threading layer

Numba prefers `tbb` > `omp` > `workqueue`. Only `tbb` is fork-safe and composes under
nested parallelism, which is relevant because Lobster runs tool calls concurrently via
`asyncio.to_thread`.

The available layer can vary by platform. Report the selected layer rather than forcing
one, and validate the analysis paths in the target environment.

Consequences:

- **Do not** declare `tbb` as a dependency where it cannot resolve on arm64.
- **Do not** set `NUMBA_THREADING_LAYER=tbb` globally. Where the library is absent this
  converts a working fallback into a hard error.
- **Do not** pin the layer differently per platform without validating that CI matches
  production behavior.

Operators who need a fork-safe layer can opt in when it is available:

```bash
pip install tbb && export NUMBA_THREADING_LAYER=tbb
python scripts/verify_env.py   # confirms the layer actually in use
```

**Open risk.** Validate that the selected layer is safe for the application's concurrent
access pattern. If nested-parallelism faults appear, build a reproducer before changing
packaging.

## Survival analysis

`scikit-survival` was declared **only** in `lobster-ml`'s optional `survival` extra,
which no root extra pulled in — not even `lobster-ai[full]`. So `lobster-ml` installed,
`survival_analysis_expert` shipped and advertised survival tools, and those tools failed
at call time.

The dependency was moved into `lobster-ml`'s **required** dependencies: an agent that
ships must be able to run. The `survival` extra is retained as an empty alias so existing
`lobster-ml[survival]` pins still resolve.

`lifelines` backs the separate proteomics survival path and is correctly declared in
`lobster-proteomics`.

## Static image export (kaleido)

`kaleido` is declared in core, but PNG export also needs a working Chrome/Chromium.
On a machine without Chrome, `to_image()` fails — `verify_env.py` reports this as a
FAILED check rather than letting it surface later as an agent error mid-analysis.

## Environment checks

`verify_env.py` exercises the analysis dependencies and reports any unavailable paths.
Use `--strict` when skipped optional checks should fail validation.

## CI

- `ci-basic.yml` — runs `tests/unit/test_verify_env.py`, the script's reporting
  contract (stdlib-only, no scientific stack needed).
- `ci-testing.yml` (integration job, installs `[dev,full]`) — runs
  `scripts/verify_env.py` in report-only mode while the CI environment is being validated.
