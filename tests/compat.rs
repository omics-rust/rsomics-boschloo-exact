//! Compat against committed `scipy.stats.boschloo_exact` goldens. Each row of
//! `tests/golden/expected.tsv` names a 2x2 table + alternative and the
//! statistic / p SciPy 1.17.1 produced; we run the binary and assert value-exact
//! agreement. No SciPy at test time — the goldens are frozen.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-boschloo-exact"))
}

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn run(table: &str, alt: &str) -> (f64, f64) {
    let out = Command::new(bin())
        .args(["--table", table])
        .args(["--alternative", alt])
        .output()
        .expect("run binary");
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = String::from_utf8(out.stdout).unwrap();
    let f: Vec<f64> = line
        .trim()
        .split('\t')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(f.len(), 2, "expected statistic,p, got {line:?}");
    (f[0], f[1])
}

fn rel(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn matches_scipy_goldens() {
    let expected = std::fs::read_to_string(golden("expected.tsv")).unwrap();
    let mut checked = 0;
    for line in expected.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (table, alt) = (c[0], c[1]);
        let stat: f64 = c[2].parse().unwrap();
        let p: f64 = c[3].parse().unwrap();
        let (gs, gp) = run(table, alt);
        assert!(
            rel(gs, stat) <= 1e-11,
            "{table}/{alt} statistic: got {gs}, want {stat}, rel {:e}",
            rel(gs, stat)
        );
        assert!(
            rel(gp, p) <= 1e-11,
            "{table}/{alt} p: got {gp}, want {p}, rel {:e}",
            rel(gp, p)
        );
        checked += 1;
    }
    assert!(checked >= 60, "expected many golden rows, ran {checked}");
}

#[test]
fn batch_matches_single() {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    writeln!(f, "7,12,8,3\n2,7,8,2\n20,14,12,18").unwrap();
    let out = Command::new(bin())
        .args(["--batch", f.path().to_str().unwrap()])
        .args(["--alternative", "two-sided"])
        .output()
        .expect("run batch");
    assert!(out.status.success());
    let lines: Vec<&str> = std::str::from_utf8(&out.stdout).unwrap().lines().collect();
    assert_eq!(lines.len(), 3, "batch should emit one line per table");
    for (table, line) in [
        ("7,12,8,3", lines[0]),
        ("2,7,8,2", lines[1]),
        ("20,14,12,18", lines[2]),
    ] {
        let (ss, sp) = run(table, "two-sided");
        let b: Vec<f64> = line.split('\t').map(|s| s.parse().unwrap()).collect();
        assert!(rel(b[0], ss) <= 1e-14 && rel(b[1], sp) <= 1e-14, "{table}");
    }
}

#[test]
fn json_envelope_smoke() {
    let out = Command::new(bin())
        .args(["--table", "7,12,8,3"])
        .arg("--json")
        .output()
        .expect("run binary");
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("\"statistic\""), "json missing statistic: {s}");
    assert!(s.contains("\"pvalue\""), "json missing pvalue: {s}");
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help did not exit 0");
}
