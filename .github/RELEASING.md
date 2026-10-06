# Releasing Lobster AI

Release tags publish immutable packages. Obtain explicit release approval and merge
release changes through the protected PR process before tagging. A package release
does **not** deploy Omics-OS Cloud.

## Preparation

1. Set the same `X.Y.Z` in `lobster/version.py`, root
   `tool.bumpversion.current_version`, and all twelve package manifests. Do not
   change dependency constraints as part of a version-only release.
2. Verify source versions:
   `uv run --no-project --python 3.12 python scripts/verify_release_artifacts.py source X.Y.Z`.
3. Check that the version is unused on both registries. Verify trusted publishers
   for all thirteen distributions, including pending publishers for new projects.
   The workflow identities remain `publish-packages.yml` and `publish-tui.yml`,
   with their existing `testpypi-{distribution}` / `pypi-{distribution}` environments.
4. Require green PR CI, release verifier tests, actionlint, and artifact checks.
   Local cross-compilation validates binary architecture, not foreign-platform
   runtime behavior. Registry installation verification runs on Linux.

## TestPyPI first

Push the approved commit's annotated `vX.Y.Z-test` tag. Both publisher workflows
build and verify the artifacts, publish to TestPyPI, and compare registry hashes.
The package suite waits for all four TUI wheels and a successful TUI workflow.
Verification installs all thirteen exact first-party wheel URLs/hashes from
TestPyPI; third-party dependencies resolve from production PyPI only.

Both workflows must succeed, including the suite's installation/summary jobs.
Experimental packages retain that label, but are not omitted from complete-release
verification. No GitHub Release is created for the test tag.

## Production approval and promotion

After successful TestPyPI verification, obtain explicit production confirmation.
Check that the successful test runs' build artifacts are still available (retained
for 30 days). The production tag `vX.Y.Z` must point to the **same commit** as
`vX.Y.Z-test`. Production downloads the successful test runs' artifacts instead of
rebuilding them, verifies them against TestPyPI, publishes them, and verifies the
production registry hashes and exact installed versions.

Only the package suite creates the GitHub Release, after successful production
verification. Its core archives and four raw TUI binaries are checked against the
published wheels. TUI no longer independently creates a GitHub Release.

## Failure and retry discipline

- A green build alone is not a completed release. Inspect both workflows and every
  publish/verification job; record partial publication honestly.
- Prefer **rerun failed jobs** for transient publisher or installation failures;
  this retains successful build artifacts. Configure a missing trusted publisher
  before retrying its failed publication job.
- Do not blindly rerun all jobs or dispatch a fresh TestPyPI build for an already
  uploaded version. Rebuilds can produce different bytes for immutable filenames.
  `skip-existing` is not proof of equivalence: registry hash verification must pass.
- If a failed build already uploaded an artifact, inspect that run's artifact
  inventory before recovery. Do not delete artifacts needed by successful uploads.
- Expired/missing test artifacts fail closed. Do not substitute an unverified
  production rebuild; recover the original verified artifacts or prepare a new
  version through TestPyPI.
- Installation-pin evidence uses attempt-qualified artifact names so failed-job
  retries do not collide with an earlier verification attempt.
- If GitHub release creation/upload fails after packages publish, inspect the
  existing release and its assets before retrying. Download existing assets and
  compare SHA256 with the verified artifacts. Upload only missing matching assets;
  never overwrite differing assets or delete immutable package versions to hide
  a partial release. A simple rerun intentionally does not clobber an existing
  GitHub Release.
- Manual workflow dispatch must select the corresponding **existing tag**, not
  `main`, and the target registry must match that tag. Production cannot bypass
  successful test-run and same-commit checks.
