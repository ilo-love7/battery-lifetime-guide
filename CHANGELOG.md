# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-23

Functional additions on top of v1.0.0: a reusable protocol-drift diagnostic, a second
CI platform, and documentation entry points. **Nothing in this release changes the
reproduction results, and no negative result was removed or softened** — the crash
scenes stay exactly where they were, and the new tooling is framed as a way to test
whether two protocols are comparable at all, not as a new headline number.

### Added

- **`scripts/protocol_drift.py` — reusable protocol-drift diagnostics** for the
  repository's central finding, protocol-level non-stationarity ("the same early
  fingerprint, a different future"):
  - `ks_drift`: two-sample Kolmogorov–Smirnov test (statistic, p-value, median shift)
    plus a verdict that states, in both English and Chinese, what the result does
    **not** prove.
  - `quantile_alignment`: quantile table with the fixed schema
    `quantile, reference, target, delta`. Deliberately no ratio column — these
    features cross zero, so a quantile ratio would be silently meaningless.
  - `band_audit` / `feature_band_audit`: is each observation inside a reference
    P05–P95 band?
  - `ks_matrix` / `cross_feature_ks`: every pairwise KS comparison across a set of
    protocols.
  - CLI: `--demo`, `--batch-drift`, `--evidence`, `--json`.
  - Offline by construction: it reads only the CSV artifacts committed under
    `results/`, so it needs none of the multi-GB MIT/NASA downloads. From committed
    artifacts alone it re-derives the zero-shot MIT→NASA elastic-net error multiplier
    **×74.8–125.3** (the unrounded form of the ×75–125 quoted in the README; RF
    ×2.25–3.14), **18/18** NASA cell × feature observations outside the MIT dev
    P05–P95 band, the E0 → E1 elastic-net geometric-mean blow-up
    **×120.42 → ×246.11** ("matching the voltage window makes it worse"), and
    cross-batch drift with **6/6** features significant (KS).
- **`tests/test_protocol_drift.py` — known-answer controls** so the diagnostic cannot
  quietly stop working: synthetic same-distribution samples must not be flagged
  (swept over 400 seeded trials; empirical flag rate 0.028 against α = 0.05), a
  deliberate +1 σ shift must be flagged, the quantile-table and band-audit schemas are
  pinned, and the committed-artifact numbers above are re-asserted in CI.
- **`tests/test_console_encoding.py`** — guards the Windows console / UTF-8 path.
- **Second CI leg `python-tests-windows`** (`.github/workflows/ci.yml`,
  `runs-on: windows-latest`) running the same commands as the existing
  `ubuntu-latest` job, which is left unchanged.
- **README sections** (English and Chinese): `## What you can do with this guide` /
  `## 用途 / 你能用它做什么`, `## Checking whether two protocols are comparable` /
  `## 判断两条协议到底可不可比`, and the `## Related repositories` / `## 关联仓库`
  cross-link table.

### Changed

- The test suite grew from **6 to 16** tests (`tests/test_protocol_drift.py` 5,
  `tests/test_console_encoding.py` 5, plus the 6 that already existed); the totals
  quoted in both READMEs now match the measured count.

### Fixed

- **Windows console encoding**: `scripts/project_config.enable_utf8_stdout()` switches
  stdout/stderr to UTF-8 so the Chinese and box-drawing output (e.g. the tutorial
  decision tree's `├─ 是 →`) no longer kills the route on `windows-latest`.
  `scripts/run_project.py` additionally sets `PYTHONIOENCODING=utf-8` for child
  scripts without overriding a value the user already set, and
  `scripts/render_github_banner.py` / `scripts/step5d_validate_km.py` call the guard
  explicitly because they do not import `project_config`.

## [1.0.0] - 2026-10-22

Initial release: the from-scratch Severson et al. 2019 reproduction guide, with the
English README as the landing page and the full Chinese guide in `README.zh-CN.md`
(13 documents), the frozen `scripts/step1`–`step6` routes, and the committed
`results/` snapshot — including the documented negative results and the debugging log.

[1.1.0]: https://github.com/ilo-love7/battery-lifetime-guide/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ilo-love7/battery-lifetime-guide/releases/tag/v1.0.0