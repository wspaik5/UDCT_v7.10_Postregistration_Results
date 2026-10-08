# UDCT v7.10 Post-registration reproduction

Won Shik Paik, Auckland, New Zealand - 8 October 2026.

This reproduces the numerical calculations reported in
`UDCT_v7_10_Postregistration_Results_and_Interpretation_2026-10-08.pdf`.
The theory under test is the specified GQUMOND example implementation, not a newly validated UDCT gravitational theory.

## Files

- `UDCT_v7_10_Postregistration_run.py`: the executed, pre-execution amended v7.10 runner, renamed for convenient distribution. Its bytes are unchanged.
- `UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py`: required frozen dependency. Keep this exact filename beside the runner. The runner verifies its full SHA-256 before loading it.
- `requirements.txt`: NumPy/SciPy versions used in the reported execution.
- `results.json` and `independent_audit.json`: original full-run outputs and independent outcome audit, included in the ZIP package.

The dependency supplies fixed stellar inputs, Galactic-field routines and integration methods. Loading it does not rerun the historical v7.06 hypothesis.

## Installation

The reported execution used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 on Linux 6.18.44 x86_64 with glibc 2.39. Other compatible environments may work but were not separately validated. Results contain the actual execution environment.

From the folder containing the files:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the self-tests first

```bash
python UDCT_v7_10_Postregistration_run.py --selftest
```

Expect `self-test: ALL PASSED` and exit code 0. These checks evaluate analytic identities, softened-profile geometry, limiting behavior and synthetic decision logic. They do not themselves evaluate registered virial or pressure outputs.

## Reproduce the complete registered calculation

```bash
python UDCT_v7_10_Postregistration_run.py > reproduced_results.json
```

With no options the runner evaluates BASE, A, B and C on the prescribed grids. It automatically runs the self-tests, verifies the dependency, archives case summaries, evaluates reproduction gate R and reports G/J/K. Its internal sequence computes the variant grid before evaluating gate R at the end; no variant conclusion is valid unless gate R passes.

The full set consists of 282 common cases on each of smoke/default/refined, plus three refined-only C cases: **849 registered case-grid evaluations**. Two cases trigger the prescribed 5-rh follow-up on all three grids: **six additional evaluations**. The global decision covers 216 A/B GQUMOND cases and 30 relevant controls.

The runner prints JSON to standard output; redirect it as shown. Runtime and memory depend on the machine. Do not change the physical inputs or grids to obtain a preferred outcome.

## Inspect the complete result

```bash
python -c "import json; r=json.load(open('reproduced_results.json')); print('complete:',r['complete_registered_set']); print('gate R:',r['gate_R']); print({k:r[k]['outcome'] for k in ('G','J','K')})"
```

Expected reported results:

- `complete_registered_set`: `true`
- `gate_R.passed`: `true`
- Global outcome **G2**: negative work for Hydrus I, A-f1/2, 10 pc, both Plummer and tapered cusp; all relevant controls positive.
- Function outcome **J1**: every tested function variant retains at least one resolved negative-pressure Hydrus I cusp case.
- Profile outcome **K2**: at 10 pc negative pressure remains at the largest sampled softening, epsilon/rh = 0.3.

Refined full-boundary V/VN is approximately -7.009036 for the negative Plummer case and -5.557278 for the negative cusp case. All three grids agree on their negative signs. Follow-up values are approximately -8.100853 and -6.104543; they must not replace the registered full-boundary values.

Original gate-R maximum relative differences were 1.209428877047003e-7 for baseline virials and 1.1087361757589065e-6 for baseline pressure minima, within fixed limits of 1e-5 and 1e-3. References are rounded published values, so exact zero error is not expected.

## Optional partial runs

```bash
# BASE reproduction only (no complete-set classification)
python UDCT_v7_10_Postregistration_run.py --parts BASE > baseline_results.json

# Small-grid check (partial; cannot establish G/J/K)
python UDCT_v7_10_Postregistration_run.py --grids smoke > smoke_results.json

# List all supported options
python UDCT_v7_10_Postregistration_run.py --help
```

Partial runs do not evaluate a complete-set gate/outcome. Do not present them as full confirmation. Part C is defined only on the refined grid.

## Binding interpretation

- Negative GQUMOND virial work requires strictly negative values on all three grids and a strictly positive paired control.
- Exact-zero GQUMOND work is a boundary case, not a negative detection. Missing/nonfinite primary values invalidate a substantive claim.
- Negative pressure requires a strictly negative minimum on both default and refined meshes. Exact-zero pressure is nonnegative for the diagnostic, not evidence of equilibrium.
- A minimum at the lowest sampled radius is boundary limited; sign agreement does not certify its depth or inner extent.
- Sampled band endpoints are not root-fitted boundaries.
- The inherited formal aperture RMS uses sqrt(max(aperture second moment, 0)). Zero in an inadmissible case must not be interpreted as a physical zero dispersion. All negative-pressure aperture outputs are formal.
- No observed-dispersion likelihood, self-consistent distribution function, coupled dynamical evolution, lensing or universal MOND verdict is tested.

## Provenance and hashes

Original reviewed preregistration runner:

```text
ac0c3359ecc7f2d04bdcc81f8922c13cb3d28c56cb09162d6a4216622d54929d
```

Executed runner (pre-execution amendment dated 5 October 2026), distributed here without byte changes:

```text
a4d1a07ebcaa06348be3022d0f1f6ca463347e00194f7fde59f2bd9dbcd521b7
```

Frozen v7.06 dependency:

```text
a64440e1350ae8bf10eb3188ec34829b6d586abf275de5cdc841a50ad3478284
```

The amendment corrects reporting/decision logic, adds finite-data and zero handling, checks the full dependency hash and suppresses claims after a failed reproduction gate. Comparison with the original runner found no changes to the physical kernels, geometry, meshes, case enumeration or integral routines. No new physical or numerical amendment was made during the 8 October execution. Renaming the runner changes its recorded command path, not its content hash or physics; the original result retains its original command and start time. The source docstring's “not yet run” statement is a preserved pre-execution status, superseded by the results record.

Execution start: 2026-10-08 04:34:57.757663 UTC, or 17:34:57 Auckland. Numerical execution and reporting were assisted by OpenAI ChatGPT/Codex under the author's direction. Original protocol design used Anthropic Claude. This is an independent research record, not peer review.

## Records

- Reviewed v7.10 preregistration: original draft 4 October 2026, reviewed 5 October 2026.
- v7.09 post-registration: https://doi.org/10.5281/zenodo.23130843
- v7.09 preregistration: https://doi.org/10.5281/zenodo.23120660
- v7.08 prior-art correction: https://doi.org/10.5281/zenodo.23117541
- v7.06 sign audit: https://doi.org/10.5281/zenodo.23039754

No unverified v7.10 DOI or repository URL is assigned in this README.
