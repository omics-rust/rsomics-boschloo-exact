use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta, run};

use rsomics_boschloo_exact::{Alternative, BoschlooResult, boschloo, parse_table};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AlternativeArg {
    TwoSided,
    Less,
    Greater,
}

impl From<AlternativeArg> for Alternative {
    fn from(a: AlternativeArg) -> Self {
        match a {
            AlternativeArg::TwoSided => Alternative::TwoSided,
            AlternativeArg::Less => Alternative::Less,
            AlternativeArg::Greater => Alternative::Greater,
        }
    }
}

/// Boschloo's 2x2 unconditional exact test — value-exact `scipy.stats.boschloo_exact`.
///
/// Pass a single table with `--table a,b,c,d` (row-major `[[a, b], [c, d]]`) or
/// a file of one `a,b,c,d` table per line with `--batch`. Output is
/// `statistic<TAB>p`, the statistic being Fisher's one-sided exact p-value of
/// the observed table. `--alternative` selects the tail; `--n` is the grid
/// resolution SciPy exposes (kept for parity — the maximisation is solved to
/// full precision regardless).
#[derive(Parser, Debug)]
#[command(name = "rsomics-boschloo-exact", version, about, long_about = None)]
pub struct Cli {
    /// 2x2 table as `a,b,c,d` (row-major `[[a, b], [c, d]]`).
    #[arg(long, value_name = "A,B,C,D", required_unless_present = "batch")]
    pub table: Option<String>,

    /// File with one `a,b,c,d` table per line; emits one result line each.
    #[arg(long, value_name = "FILE", conflicts_with = "table")]
    pub batch: Option<PathBuf>,

    /// Which tail to test.
    #[arg(long, value_enum, default_value_t = AlternativeArg::TwoSided)]
    pub alternative: AlternativeArg,

    /// Nuisance-grid resolution (SciPy parity; default 32).
    #[arg(long, default_value_t = 32)]
    pub n: u32,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        let alt: Alternative = self.alternative.into();
        run(&common, META, || {
            if let Some(path) = &self.batch {
                run_batch(path, alt)?;
                Ok(None)
            } else {
                let spec = self.table.as_ref().expect("clap requires table or batch");
                let (a, b, c, d) = parse_table(spec)?;
                let res = boschloo(a, b, c, d, alt)?;
                if !common.json {
                    println!("{}\t{}", res.statistic, res.pvalue);
                }
                Ok(Some(res))
            }
        })
    }
}

/// Stream a whitespace/newline-delimited file of `a,b,c,d` tables, writing one
/// `statistic<TAB>p` line each through a buffered writer.
fn run_batch(path: &PathBuf, alt: Alternative) -> Result<()> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut line_no = 0usize;
    for line in bytes.split(|&b| b == b'\n') {
        let line = trim_ascii(line);
        if line.is_empty() {
            continue;
        }
        line_no += 1;
        let spec = std::str::from_utf8(line)
            .map_err(|_| RsomicsError::InvalidInput(format!("line {line_no}: not UTF-8")))?;
        let (a, b, c, d) = parse_table(spec)?;
        let res: BoschlooResult = boschloo(a, b, c, d, alt)?;
        let mut sbuf = ryu::Buffer::new();
        let mut pbuf = ryu::Buffer::new();
        out.write_all(sbuf.format(res.statistic).as_bytes())?;
        out.write_all(b"\t")?;
        out.write_all(pbuf.format(res.pvalue).as_bytes())?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = s {
        if last.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
