//! Log-space hypergeometric machinery for Fisher's exact one-sided p-value.
//!
//! `hypergeom.cdf(k, M, n, N)` with the SciPy parameterisation: `M` is the
//! population size, `n` the number of "successes" in the population, `N` the
//! number drawn. The probability mass is
//! `C(n, k) C(M-n, N-k) / C(M, N)`, summed over the lower tail to give the CDF.
//! All combinations are evaluated through `lgamma` so large tables stay stable.

use libm::lgamma;

/// `ln C(a, b)`, with the out-of-support cases (`b < 0`, `b > a`) returning
/// `-inf` so their mass is zero.
fn ln_binom(a: u64, b: i64) -> f64 {
    if b < 0 || b as u64 > a {
        return f64::NEG_INFINITY;
    }
    let a = a as f64;
    let b = b as f64;
    lgamma(a + 1.0) - lgamma(b + 1.0) - lgamma(a - b + 1.0)
}

/// `ln P(X = k)` for `X ~ Hypergeom(M, n, N)` (SciPy convention).
fn ln_pmf(k: i64, m: u64, n: u64, big_n: u64) -> f64 {
    ln_binom(n, k) + ln_binom(m - n, big_n as i64 - k) - ln_binom(m, big_n as i64)
}

/// Lower-tail CDF `P(X <= k)` at every `k` in `0..=k_max` for fixed `(M, n, N)`,
/// built by one ascending cumulative sum of the PMF. The support runs from
/// `max(0, N - (M - n))` to `min(n, N)`; the ascending summation order
/// reproduces `scipy.stats.hypergeom.cdf` to ~1e-13. `out[k]` is `P(X <= k)`.
///
/// At and beyond the top of the support the CDF is exactly 1 — the whole
/// distribution's mass — so it is set to `1.0` there rather than left as the
/// forward sum's `1 - O(eps)`. This matches `scipy.stats.hypergeom.cdf` bit for
/// bit at the boundary, which Boschloo's index-set threshold relies on: when the
/// observed table is the least extreme, its statistic must read as exactly 1 so
/// every tied top-of-support table stays inside the index set.
pub fn cdf_row(k_max: i64, m: u64, n: u64, big_n: u64, out: &mut [f64]) {
    let support_lo = ((big_n as i64) - (m as i64 - n as i64)).max(0);
    let support_hi = (n as i64).min(big_n as i64);
    let mut acc = 0.0;
    for (k, slot) in out.iter_mut().enumerate().take(k_max as usize + 1) {
        let k = k as i64;
        if (support_lo..=support_hi).contains(&k) {
            acc += ln_pmf(k, m, n, big_n).exp();
        }
        *slot = if k >= support_hi { 1.0 } else { acc.min(1.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cdf(k: i64, m: u64, n: u64, big_n: u64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        let mut row = vec![0.0; k as usize + 1];
        cdf_row(k, m, n, big_n, &mut row);
        row[k as usize]
    }

    #[test]
    fn cdf_edges() {
        // Full support sums to 1.
        assert!((cdf(10, 20, 10, 10) - 1.0).abs() < 1e-12);
        // Below support is 0.
        assert_eq!(cdf(-1, 20, 10, 10), 0.0);
    }

    #[test]
    fn cdf_known_value() {
        // hypergeom.cdf(2, 10, 5, 4) == 0.738095... from SciPy.
        assert!((cdf(2, 10, 5, 4) - 0.7380952380952381).abs() < 1e-12);
    }
}
