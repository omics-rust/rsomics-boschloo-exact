//! Boschloo's 2x2 unconditional exact test — a value-exact, faster
//! `scipy.stats.boschloo_exact`.
//!
//! The statistic is Fisher's one-sided exact p-value of the observed table; the
//! reported p-value is the maximum, over a nuisance success probability, of the
//! total binomial mass of every table at least as extreme. Boschloo's test is
//! uniformly more powerful than Fisher's exact test.

mod boschloo;
mod hypergeom;

pub use boschloo::{Alternative, BoschlooResult, boschloo, parse_table};
