use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_boschloo_exact::{Alternative, boschloo};
use std::hint::black_box;

fn bench_boschloo(c: &mut Criterion) {
    c.bench_function("boschloo_two_sided_small", |b| {
        b.iter(|| {
            black_box(
                boschloo(
                    black_box(7),
                    black_box(12),
                    black_box(8),
                    black_box(3),
                    Alternative::TwoSided,
                )
                .unwrap(),
            )
        });
    });
    c.bench_function("boschloo_two_sided_mid", |b| {
        b.iter(|| {
            black_box(
                boschloo(
                    black_box(20),
                    black_box(14),
                    black_box(12),
                    black_box(18),
                    Alternative::TwoSided,
                )
                .unwrap(),
            )
        });
    });
    c.bench_function("boschloo_two_sided_large", |b| {
        b.iter(|| {
            black_box(
                boschloo(
                    black_box(40),
                    black_box(35),
                    black_box(28),
                    black_box(50),
                    Alternative::TwoSided,
                )
                .unwrap(),
            )
        });
    });
}

criterion_group!(benches, bench_boschloo);
criterion_main!(benches);
