//! Boschloo's 2x2 unconditional exact test — value-exact `scipy.stats.boschloo_exact`.
//!
//! The statistic is the one-sided Fisher exact p-value of the observed table,
//! read off the hypergeometric CDF over the unconditional sample space. The
//! reported p-value is the maximum over a nuisance parameter `pi in [0, 1]` of
//! the total binomial probability of every table at least as extreme (in the
//! Fisher-p ordering) as the observed one:
//!
//! `p = max_pi  sum_{Fisher_p(X) <= Fisher_p(obs)} C(n1,x1) C(n2,x2) pi^(x1+x2) (1-pi)^(N-x1-x2)`
//!
//! Everything is carried in log space (`lgamma` combinations, log-sum-exp
//! reduction) exactly as SciPy does. The nuisance maximisation scans a grid for
//! candidate basins and golden-section refines each — the objective is
//! multimodal, so refining only the global seed minimum would be unsafe. The one
//! case the maximiser cannot resolve is a flat objective identically equal to 1
//! (the observed table least extreme, so every table is in-index); that is
//! detected up front and returns exactly 1.

use libm::lgamma;
use rsomics_common::{Result, RsomicsError};

use crate::hypergeom;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BoschlooResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// `ln(e^a + e^b)`, stable for very negative inputs and with `-inf` identity.
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// `ln C(n, k)` for every `k` in `0..=n`, matching SciPy's
/// `_compute_log_combinations`.
fn log_combinations(n: u64) -> Vec<f64> {
    let nf = n as f64;
    let g: Vec<f64> = (0..=n).map(|k| lgamma(k as f64 + 1.0)).collect();
    let ln_n_fact = lgamma(nf + 1.0);
    (0..=n as usize)
        .map(|k| ln_n_fact - g[k] - g[n as usize - k])
        .collect()
}

/// The per-table sample space, collapsed by `s = x1 + x2`. Every table sharing
/// a value of `s` carries the same `pi^s (1-pi)^(N-s)` factor, so the inner sum
/// is grouped: `log_coeff[s]` is the log-sum-exp of the in-index tables'
/// log-combinations at that `s`. The nuisance objective then runs over `N + 1`
/// terms instead of the full `(n1+1)(n2+1)` grid — log-space throughout, exactly
/// as SciPy computes it.
struct SampleSpace {
    n: f64,
    /// `(s, log_coeff[s])` for each `s` that has at least one in-index table.
    terms: Vec<(u32, f64)>,
}

/// Negative log p-value at one nuisance value — SciPy's
/// `_get_binomial_log_p_value_with_nuisance_param`, grouped by `s`. The
/// log-sum-exp is centred on the running max for precision.
fn neg_log_pvalue(nuisance: f64, sp: &SampleSpace) -> f64 {
    let n = sp.n;
    let log_nuisance = nuisance.ln();
    let log_one_minus = (1.0 - nuisance).ln();

    let term = |&(s, log_coeff): &(u32, f64)| -> f64 {
        let s_f = s as f64;
        let pow_lo = if s == 0 { 0.0 } else { log_nuisance * s_f };
        let pow_hi = if s_f == n {
            0.0
        } else {
            log_one_minus * (n - s_f)
        };
        log_coeff + pow_lo + pow_hi
    };

    let max_value = sp.terms.iter().map(term).fold(f64::NEG_INFINITY, f64::max);
    if max_value == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    let acc: f64 = sp.terms.iter().map(|t| (term(t) - max_value).exp()).sum();
    if acc > 0.0 {
        -(max_value + acc.ln())
    } else {
        f64::INFINITY
    }
}

const GOLDEN_RESID: f64 = 0.381_966_011_250_105_15; // 2 - phi

/// Golden-section minimisation of `neg_log_pvalue` on a unimodal `[lo, hi]`
/// bracket, tightened to the floating-point limit.
fn golden_min(lo: f64, hi: f64, sp: &SampleSpace) -> f64 {
    let mut a = lo;
    let mut b = hi;
    let mut c = a + GOLDEN_RESID * (b - a);
    let mut d = b - GOLDEN_RESID * (b - a);
    let mut fc = neg_log_pvalue(c, sp);
    let mut fd = neg_log_pvalue(d, sp);
    for _ in 0..200 {
        if (b - a).abs() <= 1e-15 * (b.abs() + a.abs()) + 1e-300 {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = a + GOLDEN_RESID * (b - a);
            fc = neg_log_pvalue(c, sp);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - GOLDEN_RESID * (b - a);
            fd = neg_log_pvalue(d, sp);
        }
    }
    neg_log_pvalue(0.5 * (a + b), sp)
}

/// Seed points scanned across `[0, 1]` to locate candidate basins. The nuisance
/// objective is multimodal, so refining only the global seed minimum is unsafe —
/// the global maximum may sit in a basin the grid samples at a non-minimal point.
/// Refining every grid-local minimum makes the result insensitive to the grid
/// density, so a coarse grid suffices.
const SEED_POINTS: usize = 257;

/// Margin (in negative-log-p units) within which a grid-local minimum is worth
/// refining. The global maximum cannot sit in a basin whose grid floor lies more
/// than this above the best seed, so candidates far above it are skipped — which
/// also discards the flat-plateau noise minima that appear where p ≈ 1.
const REFINE_MARGIN: f64 = 1e-6;

/// Maximise the p-value over the nuisance parameter: scan the seed grid, then
/// golden-section refine the bracket around each promising grid-local minimum
/// and keep the best. Returns the p-value (not its negative log).
fn maximise(sp: &SampleSpace) -> f64 {
    let last = SEED_POINTS - 1;
    let at = |i: usize| i as f64 / last as f64;

    let mut fvals = Vec::with_capacity(SEED_POINTS);
    let mut grid_min = f64::INFINITY;
    for i in 0..SEED_POINTS {
        let f = neg_log_pvalue(at(i), sp);
        grid_min = grid_min.min(f);
        fvals.push(f);
    }
    let cutoff = grid_min + REFINE_MARGIN;

    let mut best = grid_min;
    for i in 0..SEED_POINTS {
        let is_local_min =
            (i == 0 || fvals[i] <= fvals[i - 1]) && (i == last || fvals[i] <= fvals[i + 1]);
        if !is_local_min || fvals[i] > cutoff {
            continue;
        }
        let lo = at(i.saturating_sub(1));
        let hi = at((i + 1).min(last));
        best = best.min(golden_min(lo, hi, sp));
    }
    (-best).exp().clamp(0.0, 1.0)
}

/// One-sided Boschloo: returns `(fisher_statistic, pvalue)` for `Less`/`Greater`.
fn one_sided(a: u64, b: u64, c: u64, d: u64, alt: Alternative) -> (f64, f64) {
    let total_col_1 = a + c;
    let total_col_2 = b + d;
    let total = total_col_1 + total_col_2;

    // Fisher one-sided p for every table in the sample space, stored row-major
    // over (x2 in 0..=tc2, x1 in 0..=tc1) so index [x1][x2] reads cleanly below.
    // The Fisher p of a cell is a hypergeometric CDF whose distribution depends
    // only on s = x1 + x2; every cell sharing an s draws from one CDF row, so the
    // rows are built once per s with a single cumulative sum (O(N^2) overall).
    let nrow = total_col_2 as usize + 1; // x2 axis
    let ncol = total_col_1 as usize + 1; // x1 axis
    let mut pvalues = vec![0.0f64; nrow * ncol];
    let mut row = vec![0.0f64; total as usize + 1];
    for s in 0..=total {
        let x1_lo = s.saturating_sub(total_col_2);
        let x1_hi = s.min(total_col_1);
        match alt {
            Alternative::Less => {
                hypergeom::cdf_row(x1_hi as i64, total, s, total_col_1, &mut row);
                for x1 in x1_lo..=x1_hi {
                    let x2 = s - x1;
                    pvalues[x2 as usize * ncol + x1 as usize] = row[x1 as usize];
                }
            }
            Alternative::Greater => {
                let x2_hi = s.min(total_col_2);
                hypergeom::cdf_row(x2_hi as i64, total, s, total_col_2, &mut row);
                for x1 in x1_lo..=x1_hi {
                    let x2 = s - x1;
                    pvalues[x2 as usize * ncol + x1 as usize] = row[x2 as usize];
                }
            }
            Alternative::TwoSided => unreachable!(),
        }
    }

    let fisher_stat = pvalues[b as usize * ncol + a as usize];
    let threshold = fisher_stat * (1.0 + 1e-13);

    let c1 = log_combinations(total_col_1);
    let c2 = log_combinations(total_col_2);

    // Group the in-index tables by s = x1 + x2 via per-s log-sum-exp of their
    // log-combination, so the nuisance objective costs O(N) per evaluation.
    let mut log_coeff = vec![f64::NEG_INFINITY; total as usize + 1];
    let mut in_index = 0usize;
    for x2 in 0..=total_col_2 as usize {
        for x1 in 0..=total_col_1 as usize {
            if pvalues[x2 * ncol + x1] <= threshold {
                in_index += 1;
                let s = x1 + x2;
                log_coeff[s] = log_add_exp(log_coeff[s], c1[x1] + c2[x2]);
            }
        }
    }

    // When the observed table is the least extreme in the tested tail, every
    // table enters the index set. The grouped coefficients then reduce (by
    // Vandermonde) to C(N, s), so the nuisance objective is the full binomial
    // sum_s C(N,s) pi^s (1-pi)^(N-s) = (pi + (1-pi))^N = 1 for every pi. The true
    // maximum is exactly 1; the grid/refine maximiser only reaches a noise floor
    // just below it on this flat objective, so short-circuit to the exact value.
    if in_index == nrow * ncol {
        return (fisher_stat, 1.0);
    }
    let terms: Vec<(u32, f64)> = log_coeff
        .iter()
        .enumerate()
        .filter(|&(_, &lc)| lc != f64::NEG_INFINITY)
        .map(|(s, &lc)| (s as u32, lc))
        .collect();

    let sp = SampleSpace {
        n: total as f64,
        terms,
    };
    (fisher_stat, maximise(&sp))
}

/// Boschloo's unconditional exact test on the 2x2 table `[[a, b], [c, d]]`.
///
/// Mirrors `scipy.stats.boschloo_exact`: a degenerate column (its two cells
/// both zero) yields `(NaN, NaN)`; the two-sided p-value is twice the smaller
/// one-sided p, clipped to 1, reported with that side's statistic.
pub fn boschloo(a: u64, b: u64, c: u64, d: u64, alt: Alternative) -> Result<BoschlooResult> {
    if a + c == 0 || b + d == 0 {
        return Ok(BoschlooResult {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }

    match alt {
        Alternative::Less | Alternative::Greater => {
            let (statistic, pvalue) = one_sided(a, b, c, d, alt);
            Ok(BoschlooResult { statistic, pvalue })
        }
        Alternative::TwoSided => {
            let (s_less, p_less) = one_sided(a, b, c, d, Alternative::Less);
            let (s_greater, p_greater) = one_sided(a, b, c, d, Alternative::Greater);
            let (statistic, pmin) = if p_less < p_greater {
                (s_less, p_less)
            } else {
                (s_greater, p_greater)
            };
            Ok(BoschlooResult {
                statistic,
                pvalue: (2.0 * pmin).clamp(0.0, 1.0),
            })
        }
    }
}

/// Parse a `a,b,c,d` table spec into nonnegative counts.
pub fn parse_table(spec: &str) -> Result<(u64, u64, u64, u64)> {
    let v: Vec<&str> = spec.split(',').collect();
    if v.len() != 4 {
        return Err(RsomicsError::InvalidInput(format!(
            "--table needs 4 comma-separated counts a,b,c,d, got {:?}",
            spec
        )));
    }
    let n: Vec<u64> = v
        .iter()
        .map(|s| {
            s.trim().parse::<u64>().map_err(|_| {
                RsomicsError::InvalidInput(format!(
                    "table cell {:?} is not a nonnegative integer",
                    s
                ))
            })
        })
        .collect::<Result<_>>()?;
    Ok((n[0], n[1], n[2], n[3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn matches_scipy_two_sided() {
        // scipy.stats.boschloo_exact([[7,12],[8,3]]) -> stat 0.06406796601699151, p 0.0682183093231944
        let r = boschloo(7, 12, 8, 3, Alternative::TwoSided).unwrap();
        assert!(rel(r.statistic, 0.064_067_966_016_991_51) <= 1e-11);
        assert!(rel(r.pvalue, 0.068_218_309_323_194_4) <= 1e-11);
    }

    #[test]
    fn matches_scipy_less() {
        let r = boschloo(2, 7, 8, 2, Alternative::Less).unwrap();
        assert!(rel(r.statistic, 0.018_521_725_952_066_51) <= 1e-11);
        assert!(rel(r.pvalue, 0.009_886_140_844_640_604) <= 1e-11);
    }

    #[test]
    fn degenerate_column_is_nan() {
        let r = boschloo(0, 5, 0, 4, Alternative::TwoSided).unwrap();
        assert!(r.statistic.is_nan() && r.pvalue.is_nan());
    }

    // Zero-cell tables where the observed table is the least extreme in the
    // tested tail: every table enters the index set, the nuisance objective is
    // identically 1, so both the Fisher statistic and the p-value are exactly 1.
    // scipy.stats.boschloo_exact returns 1.0 here; the grid maximiser used to
    // report a ~0.995 noise floor instead.
    #[test]
    fn least_extreme_zero_cell_is_exactly_one() {
        for &(a, b, c, d) in &[(450, 1, 450, 0), (500, 1, 500, 0), (300, 2, 300, 0)] {
            let r = boschloo(a, b, c, d, Alternative::Greater).unwrap();
            assert_eq!(r.statistic, 1.0, "statistic for {a},{b},{c},{d}");
            assert_eq!(r.pvalue, 1.0, "pvalue for {a},{b},{c},{d}");
        }
    }

    // The exact-1 short-circuit fires only on the genuinely-1 side, so the
    // two-sided value (twice the smaller one-sided p) is untouched.
    #[test]
    fn two_sided_unaffected_by_short_circuit() {
        // scipy.stats.boschloo_exact([[300,2],[300,0]], 'two-sided') -> p 0.3005...
        let r = boschloo(300, 2, 300, 0, Alternative::TwoSided).unwrap();
        let g = boschloo(300, 2, 300, 0, Alternative::Greater).unwrap();
        let l = boschloo(300, 2, 300, 0, Alternative::Less).unwrap();
        let expected = (2.0 * g.pvalue.min(l.pvalue)).clamp(0.0, 1.0);
        assert!((r.pvalue - expected).abs() <= 1e-15);
        assert!(r.pvalue < 1.0, "two-sided should not be forced to 1");
    }
}
