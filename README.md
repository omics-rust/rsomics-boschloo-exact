# rsomics-boschloo-exact

Boschloo's 2×2 unconditional exact test — a value-exact, faster
`scipy.stats.boschloo_exact`. Boschloo's test uses Fisher's exact test p-value
as its ordering statistic and is uniformly more powerful than Fisher's exact
test.

## Install

```sh
cargo install rsomics-boschloo-exact
```

## Usage

```sh
# Two-sided test on the table [[a, b], [c, d]]
rsomics-boschloo-exact --table 7,12,8,3
# -> 0.06406796601699073	0.06821830932319428   (statistic<TAB>p)

# One-sided
rsomics-boschloo-exact --table 2,7,8,2 --alternative less
rsomics-boschloo-exact --table 2,7,8,2 --alternative greater

# Many tables, one a,b,c,d per line
rsomics-boschloo-exact --batch tables.txt --alternative two-sided
```

The **statistic** is Fisher's one-sided exact p-value of the observed table; the
**p-value** is the maximum, over a nuisance success probability `π ∈ [0, 1]`, of
the total binomial mass of every table at least as extreme (in the Fisher-p
ordering) as the observed one. A degenerate column (both its cells zero) yields
`NaN	NaN`, matching SciPy. `--n` mirrors SciPy's nuisance-grid resolution; the
maximisation is solved to full precision regardless.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.boschloo_exact` based on:

- R. D. Boschloo, "Raised conditional level of significance for the 2×2-table
  when testing the equality of two probabilities," *Statistica Neerlandica*
  24(1):1–9, 1970. doi:10.1111/j.1467-9574.1970.tb00104.x
- The SciPy reference implementation (`scipy/stats/_hypotests.py`,
  `boschloo_exact` + `_get_binomial_log_p_value_with_nuisance_param`), BSD-3,
  read to match the Fisher-p statistic, the nuisance objective, the
  `(1 + 1e-13)` extremity guard, and the two-sided `2 × min` rule exactly.

The Fisher-p statistic is computed from the hypergeometric CDF in log space
(`lgamma` combinations), and the nuisance maximisation reproduces SciPy's
log-sum-exp objective. Validated value-exact (statistic and p within ~1e-12
relative error) against `scipy.stats.boschloo_exact` 1.17.1 over a broad set of
2×2 tables; goldens are frozen in `tests/golden/`.

License: MIT OR Apache-2.0.
Upstream credit: SciPy (https://scipy.org, BSD-3-Clause).
