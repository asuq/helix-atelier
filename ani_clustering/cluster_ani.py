#!/usr/bin/env python3
"""
Cluster genomes from a PHYLIP lower-triangular ANI matrix (from FastANI)
using complete linkage, then choose one representative per cluster

Required packages: numpy, pandas, scipy

Core pipeline
-------------
1) Read matrix: parse PHYLIP lower‑triangular ANI; enforce strict structure.
2) Validate tables:
   - input_list.tsv: Accession, Path (1:1, paths must exist).
   - metadata.csv : required fields present & parseable (see REQUIRED_NONEMPTY_COLS).
   - Name identity: every matrix name is present in both TSV and CSV (hard fail).
   - Cross‑file equality for Assembly_Name / Organism_Name, allowing space↔underscore
     normalization (differences beyond that -> hard fail).
3) Cluster: complete‑linkage on distance d = 1 − ANI/100; cut at (1 − threshold)
   so all pairs in a cluster have ANI ≥ threshold and non‑NA (post‑checked).
4) Score candidates, per cluster, with channel‑specific transforms,
   winsorization (5–95%), and min–max normalization within cluster:
   - A: Assembly level (rank/3; Complete Genome>Chromosome>Scaffold>Contig).
   - B: BUSCO   (Complete − 1·Missing) %, winsorized -> min–max.
   - Q: CheckM2 (Complete − 5·Contamination) %, winsorized -> min–max (Gcode‑matched).
   - N: N50, log10(x+1) -> winsorized -> min–max.
   - S: Scaffolds, log10(x+1) -> winsorized -> min–max -> invert (fewer is better).
   - C: ANI centrality within cluster:
           - for each genome i, compute mean ANI to all other members;
           - if max(mean_i) − min(mean_i) < 0.05, treat the cluster as
             homogeneous and set C_i = 0.5 for all members;
           - otherwise rescale the mean ANI values so the minimum becomes 0
             and the maximum becomes 1;
           singleton clusters get C=1.
   Composite score S = wA·A + wB·B + wQ·Q + wN·N + wS·S + wC·C.
   Weight presets chosen by --score-profile {default,isolate,mag}.
5) Select representative: pick highest score (ties if |Δ| ≤ ε). Tie‑cascade:
   assembly rank (higher) -> BUSCO C (higher) -> BUSCO M (lower) ->
   CheckM2 contamination (lower) -> CheckM2 completeness (higher) -> Scaffolds (fewer) ->
   N50 (higher) -> lexicographically smallest Accession.
6) Soft screen (warn only): list all offenders per cluster where
   Completeness < 90 or Contamination > 5; never hard‑fail here.
7) Outputs:
   - cluster.tsv:  Accession, Cluster_ID, Is_Representative, ANI_to_Representative, Score, Path
   - representatives.tsv: Cluster_ID, Representative_Accession, Organism_Name,
     CheckM2_Completeness, CheckM2_Contamination,
     BUSCO, Assembly_Level, N50, Cluster_Size
     (Organism_Name has whitespace collapsed to underscores).

Strict validation:
  - PHYLIP names == Accession set in matrix (hard fail if any matrix name missing in TSV/CSV).
  - PHYLIP structure enforced (row i has i values; tokens are float or literal uppercase 'NA').
  - 'NA' allowed only in the ANI matrix; not allowed in required CSV fields.
  - Gcode must be 4 or 11 (hard fail).
  - Assembly_Level normalized case-insensitively to exactly:
        {'Complete Genome','Chromosome','Scaffold','Contig'} (no synonyms).
  - Required CSV fields must be present & parseable; missing/unparsable => hard fail.
  - Accession <-> Path must be 1:1 and Path must exist on disk.

Threading:
  - --threads is an upper bound; actual worker count is capped by hardware/scheduler limits
    (affinity/cgroups/cgroups v1/v2, SLURM/PBS/SGE). That cap is applied to:
      (a) BLAS/OpenMP/numexpr via env vars (set before importing numpy/scipy/pandas)
      (b) Python ThreadPoolExecutors.

References:
  - FastANI: Jain et al., Nat Commun (2018), doi:10.1038/s41467-018-07641-9
  - Hierarchical clustering: Murtagh & Contreras, WIREs DMKD (2012), doi:10.1002/widm.53
  - CheckM2: Chklovski et al., Nat Methods (2023), doi:10.1038/s41592-023-01940-w
  - BUSCO v5: Manni et al., Mol Biol Evol (2021), doi:10.1093/molbev/msab199
  - QUAST/N50: Gurevich et al., Bioinformatics (2013), doi:10.1093/bioinformatics/btt086

Author: Akito Shima (asuq)
Email: asuq.4096@gmail.com
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


# --------------------------------------------------------------------------- #
# CLI, logging, and fatal helper
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cluster genomes from an ANI PHYLIP matrix using complete linkage and select "
            "representatives by a composite quality score with deterministic tie-break."
        )
    )
    p.add_argument(
        "-a",
        "--ani-matrix",
        required=True,
        type=Path,
        help="PHYLIP lower-triangular ANI matrix (output from fastANI --matrix).",
    )
    p.add_argument(
        "-l",
        "--input-list",
        required=True,
        type=Path,
        help="TSV with: Accession, Assembly_Name, Organism_Name, Path.",
    )
    p.add_argument(
        "-m",
        "--metadata",
        required=True,
        type=Path,
        help="CSV with full metadata columns.",
    )
    p.add_argument(
        "-t",
        "--threshold",
        required=False,
        type=float,
        default=0.95,
        help="ANI threshold as a fraction in (0,1). Default: 0.95 (i.e., 95%%).",
    )
    p.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for cluster.tsv and representatives.tsv.",
    )
    p.add_argument(
        "-p",
        "--threads",
        required=False,
        type=int,
        default=8,
        help="Upper bound on concurrency; default 8. Capped by hardware/scheduler.",
    )
    p.add_argument(
        "--score-profile",
        choices=["default", "isolate", "mag"],
        default="default",
        help="Weighting profile for representative scoring (default, isolate, mag).",
    )

    # Mutually exclusive logging controls
    log_group = p.add_mutually_exclusive_group()
    log_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show ERROR, WARNING and CRITICAL only.",
    )
    log_group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Verbose debug logging (include per-candidate scoring details).",
    )

    return p.parse_args()


def configure_logging(quiet: bool, debug: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def die(msg: str) -> NoReturn:
    logging.critical(msg)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Scheduler/hardware-aware thread capping
# --------------------------------------------------------------------------- #
def _available_cpus() -> int:
    """
    CPUs effectively available to this process.
    On Linux, sched_getaffinity(0) reflects cpusets/cgroup placement.
    """
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return os.cpu_count() or 1


def _cgroup_cpu_cap() -> int | None:
    """
    Best-effort CPU cap from cgroups (v2: cpu.max; v1: cpu.cfs_quota_us / cpu.cfs_period_us).
    Returns None if not detectable.
    """
    # cgroups v2
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.is_file():
            txt = cpu_max.read_text().strip().split()
            if len(txt) >= 2 and txt[0] != "max":
                quota = int(txt[0])
                period = int(txt[1])
                if quota > 0 and period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    # cgroups v1
    try:
        qf = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        pf = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if qf.is_file() and pf.is_file():
            quota = int(qf.read_text().strip())
            period = int(pf.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except Exception:
        pass
    return None


def resolve_thread_cap(requested: int) -> int:
    """
    Conservative upper bound on threads: min(requested, scheduler cap, available CPUs),
    also honoring common scheduler env vars.
    """
    candidates: list[int] = [max(1, requested), _available_cpus()]
    for var in ("NSLOTS", "PBS_NUM_PPN", "PBS_NP", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(var)
        if v and v.isdigit():
            candidates.append(int(v))
    # cgroups cap
    cg = _cgroup_cpu_cap()
    if cg is not None:
        candidates.append(cg)
    return max(1, min(candidates))


def set_thread_envs(n_threads: int) -> None:
    """
    Cap math-library threads. If an env var is already set to a positive int, keep min(existing, cap);
    else set to cap. Must be called BEFORE importing numpy/scipy/pandas.
    """

    def cap(existing: str | None, limit: int) -> str:
        try:
            if existing is not None:
                v = int(existing)
                if v > 0:
                    return str(min(v, limit))
        except Exception:
            pass
        return str(limit)

    keys = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ]
    for k in keys:
        os.environ[k] = cap(os.environ.get(k), n_threads)


# --------------------------------------------------------------------------- #
# Parsing helpers and types
# --------------------------------------------------------------------------- #
REQUIRED_NONEMPTY_COLS: set[str] = {
    "Accession",
    "Gcode",
    "N50",
    "Assembly_Level",
    "BUSCO_bacillota_odb12",
    "Scaffolds",
    "Genome_Size",
    "Completeness_gcode4",
    "Completeness_gcode11",
    "Contamination_gcode4",
    "Contamination_gcode11",
}

BUSCO_RE = re.compile(r"C:(?P<C>\d+(?:\.\d+)?)%.*?M:(?P<M>\d+(?:\.\d+)?)%", re.IGNORECASE)

ASSEMBLY_LEVEL_MAP: dict[str, str] = {
    "complete genome": "Complete Genome",
    "chromosome": "Chromosome",
    "scaffold": "Scaffold",
    "contig": "Contig",
}

ASSEMBLY_RANK: dict[str, int] = {
    "Contig": 0,
    "Scaffold": 1,
    "Chromosome": 2,
    "Complete Genome": 3,
}


def normalize_assembly_level(raw: str, *, acc: str) -> str:
    key = raw.strip().casefold()
    if key not in ASSEMBLY_LEVEL_MAP:
        die(f"Unknown Assembly_Level for accession '{acc}': '{raw}'")
    return ASSEMBLY_LEVEL_MAP[key]


def parse_int_like(x: Any, field: str, acc: str) -> int:
    """
    Accept integers or strings that represent integers.
    Float-like string only if exactly integral (e.g., '1234.0').
    """
    try:
        xs = str(x).strip()
        if xs.upper() == "NA" or xs == "":
            die(f"Required integer field '{field}' is empty or 'NA' for accession '{acc}'")
        if re.fullmatch(r"[+-]?\d+", xs):
            return int(xs)
        xf = float(xs)
        if xf.is_integer():
            return int(xf)
        die(f"Non-integer value for '{field}' in accession '{acc}': '{x}'")
    except Exception:
        die(f"Unparsable integer for '{field}' in accession '{acc}': '{x}'")


def parse_float_like(x: Any, field: str, acc: str) -> float:
    try:
        xs = str(x).strip()
        if xs.upper() == "NA" or xs == "":
            die(f"Required float field '{field}' is empty or 'NA' for accession '{acc}'")
        return float(xs)
    except Exception:
        die(f"Unparsable float for '{field}' in accession '{acc}': '{x}'")


def parse_busco(busco_str: str, acc: str) -> tuple[float, float]:
    """
    Extract C and M percentages from a BUSCO string like:
    'C:94.0%[S:93.5%,D:0.5%],F:1.5%,M:4.5%,n:201'
    """
    if not isinstance(busco_str, str) or not busco_str.strip():
        die(f"Missing BUSCO string for accession '{acc}'")
    m = BUSCO_RE.search(busco_str)
    if not m:
        die(f"Unparsable BUSCO string for accession '{acc}': '{busco_str}'")
    return float(m.group("C")), float(m.group("M"))


def _canon_space_underscore(s: str) -> str:
    """
    Normalize by treating sequences of spaces/underscores as a single space.
    """
    return re.sub(r"[\s_]+", " ", s.strip())


@dataclass(slots=True)
class Genome:
    Accession: str
    Organism_Name: str
    Gcode: int
    CheckM2_Completeness: float
    CheckM2_Contamination: float
    N50: int
    Scaffolds: int
    Genome_Size: int
    BUSCO_str: str
    BUSCO_C: float
    BUSCO_M: float
    Assembly_Level: str
    Assembly_Rank: int
    Path: str
    Score: float | None = None


# --------------------------------------------------------------------------- #
# PHYLIP parsing + matrix builder
# --------------------------------------------------------------------------- #
def load_phylip_lower_triangular(path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load PHYLIP lower-triangular ANI matrix.
    Returns (names, rows), where rows[i] is a list of exactly i values (strings),
    each value either a float-string or literal uppercase 'NA'.

    Robust to spaces in names: for row i, split from the RIGHT into (i values + 1) parts.
    """
    if not path.is_file():
        die(f"ANI matrix file not found: {path}")

    lines: list[str] = []
    with path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                lines.append(s)

    # File format check
    if not lines:
        die("ANI matrix: empty file or missing taxon count.")
    try:
        n = int(lines[0])
    except Exception:
        die(f"First line must be integer taxon count; got: '{lines[0]}'")

    if n == 0:
        die(f"ANI matrix does not contain any sample")
    elif n == 1:
        die(f"ANI matrix only contains one sample")

    if len(lines) - 1 != n:
        die(f"Expected {n} matrix rows, found {len(lines) - 1}.")

    # Matrix parsing
    names: list[str] = []
    rows: list[list[str]] = []
    for i in range(1, n + 1):  # n sample lines starting from index 1
        raw = lines[i]
        expected = i - 1
        # Split from the right: (expected values) + 1 part for the name
        parts = raw.rsplit(maxsplit=expected)
        if len(parts) != expected + 1:
            die(f"Row {i}: expected {expected} values; got {len(parts) - 1}. Raw: '{raw}'")
        name = parts[0].strip()
        vals = [p.strip() for p in parts[1:]]
        # Value validation
        for j, val in enumerate(vals, start=1):
            if val == "NA":
                continue
            try:
                v = float(val)
            except Exception:
                die(f"Non-numeric/non-NA token at row '{name}', column {j}: '{val}'")
            if not (0.0 <= v <= 100.0):
                die(f"ANI value out of range [0,100] at row '{name}', column {j}: {v}")
        names.append(name)
        rows.append(vals)

    return names, rows


def load_matrix(path: Path) -> tuple[list[str], "np.ndarray", dict[str, int]]:
    """
    Convert a PHYLIP lower-triangular ANI file into a full symmetric matrix
    and return:
        - names: list of accession names in matrix order
        - ani:   full symmetric ANI matrix (n x n), float64, 'NA' -> np.nan
        - name_to_idx: mapping from name -> row/column index
    """
    import numpy as np  # local import for type name

    names, rows = load_phylip_lower_triangular(path)
    n = len(names)
    logging.info("Loaded ANI matrix with %d taxa.", n)

    ani = np.full((n, n), np.nan, dtype=np.float64)
    name_to_idx: dict[str, int] = {}
    for i, name in enumerate(names):
        if name in name_to_idx:
            die(f"Duplicate name in ANI matrix: '{name}'")
        name_to_idx[name] = i

    for i in range(n):
        ani[i, i] = 100.0
        vals = rows[i]
        for j in range(i):
            val = vals[j]
            if val == "NA":
                continue
            v = float(val)
            ani[i, j] = v
            ani[j, i] = v

    return names, ani, name_to_idx


# --------------------------------------------------------------------------- #
# TSV/CSV loading + structural checks
# --------------------------------------------------------------------------- #
def load_and_check_tables(
    input_list: Path,
    metadata: Path,
    matrix_names: list[str],
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Load input_list.tsv and metadata CSV, perform structural checks:

      - Keep 'NA' as literal strings (not NaN)
      - TSV: required columns 'Accession' and 'Path'
      - TSV: Accession and Path must be unique (1:1 mapping)
      - TSV: every Path must exist on disk
      - CSV: required metadata columns must be present
      - Identity/coverage: every matrix accession MUST exist in both TSV and CSV (hard fail).
        TSV/CSV MAY contain extras not in the matrix (warn and ignore).
      - Cross-file equality checks for names apply only to accessions present in the matrix.
      - Assembly_Name and Organism_Name equality tolerates space↔underscore substitutions; other differences hard-fail.

    Returns:
        (tsv_df, csv_df) with 'Accession' as a column.
    """
    import pandas as pd  # local import for type name

    if not input_list.is_file():
        die(f"Input TSV not found: {input_list}")
    if not metadata.is_file():
        die(f"Metadata CSV not found: {metadata}")

    # Keep 'NA' as literal strings, not NaN
    tsv_df = pd.read_csv(input_list, sep="\t", dtype=str, keep_default_na=False).fillna("")
    csv_df = pd.read_csv(metadata, dtype=str, keep_default_na=False).fillna("")

    # TSV: required cols and bijection Accession <-> Path
    for col in ("Accession", "Path"):
        if col not in tsv_df.columns:
            die(f"input_list.tsv missing required column: '{col}'")

    dup_acc = tsv_df["Accession"][tsv_df["Accession"].duplicated()].unique().tolist()
    dup_path = tsv_df["Path"][tsv_df["Path"].duplicated()].unique().tolist()
    if dup_acc:
        die(f"Duplicate Accession(s) in TSV (first few): {dup_acc[:10]}")
    if dup_path:
        die(f"Duplicate Path(s) in TSV (first few): {dup_path[:5]}")

    # Path existence
    for acc, p in zip(tsv_df["Accession"], tsv_df["Path"], strict=True):
        p_str = str(p).strip()
        if not p_str or p_str.upper() == "NA":
            die(f"Empty or 'NA' Path for Accession '{acc}' in input_list.tsv")
        if not Path(p_str).exists():
            die(f"Path does not exist for Accession '{acc}': {p_str}")

    # CSV: presence of columns
    if "Accession" not in csv_df.columns:
        die("metadata CSV missing 'Accession' column.")
    missing_cols = [c for c in REQUIRED_NONEMPTY_COLS if c not in csv_df.columns]
    if missing_cols:
        die(f"Metadata CSV missing required columns: {missing_cols}")

    # Identity / coverage: matrix ⊆ TSV ∩ CSV (extras warn)
    mat_acc = set(matrix_names)
    tsv_acc = set(tsv_df["Accession"])
    csv_acc = set(csv_df["Accession"])

    missing_in_tsv = sorted(mat_acc - tsv_acc)
    missing_in_csv = sorted(mat_acc - csv_acc)
    if missing_in_tsv or missing_in_csv:

        def head(xs: list[str], n: int = 20) -> list[str]:
            return xs[:n]

        die(
            "Matrix accessions must exist in both TSV and CSV.\n"
            f"  In matrix not in TSV (first 20): {head(missing_in_tsv)}\n"
            f"  In matrix not in CSV (first 20): {head(missing_in_csv)}"
        )

    extras_tsv = sorted(tsv_acc - mat_acc)
    extras_csv = sorted(csv_acc - mat_acc)
    if extras_tsv:
        logging.warning(
            "Ignoring %d TSV accession(s) not present in the ANI matrix (first 20): %s",
            len(extras_tsv),
            extras_tsv[:20],
        )
    if extras_csv:
        logging.warning(
            "Ignoring %d CSV accession(s) not present in the ANI matrix (first 20): %s",
            len(extras_csv),
            extras_csv[:20],
        )

    # Assembly_Name equality (tolerate spaces<->underscores)
    if "Assembly_Name" in tsv_df.columns and "Assembly_Name" in csv_df.columns:
        t_map = tsv_df.set_index("Accession")["Assembly_Name"].to_dict()
        c_map = csv_df.set_index("Accession")["Assembly_Name"].to_dict()
        bad: list[tuple[str, str, str]] = []
        ignored_count = 0
        for acc in matrix_names:
            a_raw = (t_map.get(acc, "") or "").strip()
            b_raw = (c_map.get(acc, "") or "").strip()
            if not a_raw or not b_raw:
                continue
            if a_raw == b_raw:
                continue
            if _canon_space_underscore(a_raw) == _canon_space_underscore(b_raw):
                logging.info(
                    "Assembly_Name differs only by space/underscore; ignoring (Acc=%s): TSV='%s' CSV='%s'",
                    acc,
                    a_raw,
                    b_raw,
                )
                ignored_count += 1
            else:
                bad.append((acc, a_raw, b_raw))
        if bad:
            die(f"Assembly_Name mismatch between TSV and CSV (first 10): {bad[:10]}")
        if ignored_count:
            logging.info(
                "Ignored %d Assembly_Name difference(s) due to space/underscore normalization.",
                ignored_count,
            )

    # Organism_Name equality (tolerate spaces<->underscores)
    if "Organism_Name" in tsv_df.columns and "Organism_Name" in csv_df.columns:
        t_map = tsv_df.set_index("Accession")["Organism_Name"].to_dict()
        c_map = csv_df.set_index("Accession")["Organism_Name"].to_dict()
        bad: list[tuple[str, str, str]] = []
        ignored_count = 0
        for acc in matrix_names:
            a_raw = (t_map.get(acc, "") or "").strip()
            b_raw = (c_map.get(acc, "") or "").strip()
            if not a_raw or not b_raw:
                continue
            if a_raw == b_raw:
                continue
            if _canon_space_underscore(a_raw) == _canon_space_underscore(b_raw):
                logging.info(
                    "Organism_Name differs only by space/underscore; ignoring (Acc=%s): TSV='%s' CSV='%s'",
                    acc,
                    a_raw,
                    b_raw,
                )
                ignored_count += 1
            else:
                bad.append((acc, a_raw, b_raw))
        if bad:
            die(f"Organism_Name mismatch between TSV and CSV (first 10): {bad[:10]}")
        if ignored_count:
            logging.info(
                "Ignored %d Organism_Name difference(s) due to space/underscore normalization.",
                ignored_count,
            )

    return tsv_df, csv_df


# --------------------------------------------------------------------------- #
# Genome metadata building
# --------------------------------------------------------------------------- #
def build_genome_metadata(
    names: list[str],
    tsv: "pd.DataFrame",
    csv_df: "pd.DataFrame",
) -> dict[str, Genome]:
    """
    Construct per-genome metadata records used for ranking and filters.
    For each accession:
      - Enforces required fields to be non-empty/non-'NA'
      - Normalizes Assembly_Level
      - Validates Gcode and selects appropriate CheckM2 Completeness/Contamination fields
      - Parses N50, Scaffolds, Genome_Size, BUSCO, and Path
    """
    csv_idx = csv_df.set_index("Accession", drop=False)
    tsv_idx = tsv.set_index("Accession", drop=False)

    meta: dict[str, Genome] = {}
    for acc in names:
        if acc not in csv_idx.index or acc not in tsv_idx.index:
            die(f"Accession '{acc}' missing from TSV or CSV after alignment.")

        row = csv_idx.loc[acc].to_dict()

        # Required non-empty columns (not 'NA')
        for col in REQUIRED_NONEMPTY_COLS:
            val = str(row.get(col, "")).strip()
            if val == "" or val.upper() == "NA":
                die(f"Required column '{col}' empty or 'NA' for accession '{acc}'")

        # Normalize Assembly_Level
        asm_level = normalize_assembly_level(str(row["Assembly_Level"]), acc=acc)

        # Gcode & metrics chosen by Gcode
        gcode = parse_int_like(row["Gcode"], "Gcode", acc)
        if gcode not in (4, 11):
            die(f"Gcode must be 4 or 11 for accession '{acc}', got: {gcode}")
        comp_col = "Completeness_gcode4" if gcode == 4 else "Completeness_gcode11"
        cont_col = "Contamination_gcode4" if gcode == 4 else "Contamination_gcode11"

        # Ensure chosen columns exist and non-empty/non-'NA'
        for col in (comp_col, cont_col):
            if col not in csv_idx.columns:
                die(f"Metadata CSV missing required column '{col}' for accession '{acc}'")
            val = str(row.get(col, "")).strip()
            if val == "" or val.upper() == "NA":
                die(f"Required column '{col}' empty or 'NA' for accession '{acc}'")

        org_name = str(csv_idx.loc[acc].get("Organism_Name", "") or "")
        checkm2 = parse_float_like(row[comp_col], comp_col, acc)
        contam = parse_float_like(row[cont_col], cont_col, acc)
        n50 = parse_int_like(row["N50"], "N50", acc)
        scaffolds = parse_int_like(row["Scaffolds"], "Scaffolds", acc)
        genome_size = parse_int_like(row["Genome_Size"], "Genome_Size", acc)
        busco_str = str(row["BUSCO_bacillota_odb12"])
        busco_c, busco_m = parse_busco(busco_str, acc)
        path = str(tsv_idx.loc[acc]["Path"])

        meta[acc] = Genome(
            Accession=acc,
            Organism_Name=org_name,
            Gcode=gcode,
            CheckM2_Completeness=checkm2,
            CheckM2_Contamination=contam,
            N50=n50,
            Scaffolds=scaffolds,
            Genome_Size=genome_size,
            BUSCO_str=busco_str,
            BUSCO_C=busco_c,
            BUSCO_M=busco_m,
            Assembly_Level=asm_level,
            Assembly_Rank=ASSEMBLY_RANK[asm_level],
            Path=path,
        )

    return meta


# --------------------------------------------------------------------------- #
# Clustering + post-check
# --------------------------------------------------------------------------- #
def cluster_complete_linkage(
    ani: "np.ndarray",
    names: list[str],
    threshold: float,
) -> dict[int, list[int]]:
    """
    Complete-linkage clustering on an ANI matrix.
    Distances: d = 1 - ANI/100, with NA treated as max distance 1.0.
    Cut at distance 1 - threshold (so all pairs in a cluster have ANI ≥ threshold).
    """
    # local import for type name
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Convert ANI matrix to distance matrix, fill diagonal with 0 (identical),
    # and generate condensed distance vector
    dist = np.where(np.isnan(ani), 1.0, 1.0 - (ani / 100.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    cut_distance = 1.0 - threshold
    z = linkage(condensed, method="complete")
    labels = fcluster(
        z, t=cut_distance, criterion="distance"
    )  # output: sample_idx, cluster_num

    # group row indices by their label
    clusters: defaultdict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)  # clusters: cluster_num = list of sample_idx that belong to

    logging.info(
        "Formed %d complete-linkage clusters at ANI ≥ %.2f%%.", len(clusters), threshold * 100.0
    )

    # Post-check: every pair in each cluster must be finite ANI ≥ threshold
    # Checked by the values in the matrix (e.g. ani[0, 1] = ANI distance between sample idx 0 and 1)
    for lab, idxs in clusters.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                val = ani[a, b]
                if np.isnan(val) or val < (threshold * 100.0):
                    die(
                        f"Post-check failed: cluster label {lab} contains pair "
                        f"({names[a]}, {names[b]}) with ANI={val} "
                        f"(must be non-NA and ≥ {threshold*100:.2f})."
                    )

    return clusters


# --------------------------------------------------------------------------- #
# Representative selection (scoring + tie-break) + cluster IDs
# --------------------------------------------------------------------------- #
def select_representative_for_indices(
    idxs: list[int],
    names: list[str],
    meta: dict[str, Genome],
    ani: "np.ndarray",
    score_profile: str,
) -> tuple[int, str, list[str]]:
    """
    Select one representative by a composite score.

    Scoring components in [0,1]:
      A: Assembly rank / 3
      Q: CheckM2: (completeness - 5*contamination)/100 -> winsorize 5-95% -> min-max
      B: BUSCO:   (BUSCO_C - 1*BUSCO_M)/100 -> winsorize 5-95% -> min-max
      N: N50:     log10(x+1) -> winsorize 5–95% if n>=8 -> min–max
      S: Scaffolds: log10(x+1) -> winsorize 5–95% if n>=8 -> min–max -> invert
      C: ANI centrality within cluster:
           - for each genome i, compute mean ANI to all other members;
           - if max(mean_i) − min(mean_i) < 0.05, treat the cluster as
             homogeneous and set C_i = 0.5 for all members;
           - otherwise rescale the mean ANI values so the minimum becomes 0
             and the maximum becomes 1;
           singleton clusters get C=1.

    Ties (|difference of score| ≤ _EPS) are broken by:
      1) higher assembly rank,
      2) BUSCO: higher C, then lower M,
      3) CheckM2: lower contamination, then higher completeness
      4) fewer Scaffolds,
      5) higher N50,
      6) lexicographically smallest Accession.

    Weights by profile (sum is not required to be 1):
      - default: A=3.0, Q=2.0, B=1.5, N=0.75, S=0.50, C=0.50
      - isolate: A=3.5, Q=1.5, B=1.5, N=1.00, S=0.25, C=0.50
      - mag    : A=1.0, Q=3.0, B=2.0, N=0.50, S=1.50, C=0.50
    """
    import numpy as np  # local import for type name

    _EPS = 1e-6  # tie tolerance for scores

    dbg: list[str] = []
    accs = [names[i] for i in idxs]
    infos = [meta[a] for a in accs]
    n = len(infos)

    ## Helpers ================================================================
    def winsorize(arr: list[float]) -> np.ndarray:
        """winsorize within cluster at 5–95% only if cluster is reasonably sized"""
        if n < 8:  # guard for tiny clusters
            return np.asarray(arr, dtype=float)
        a = np.asarray(arr, dtype=float)
        ql, qh = np.quantile(a, [0.05, 0.95])
        return np.clip(a, ql, qh)

    def minmax_norm(arr: "np.ndarray") -> "np.ndarray":
        """normalise value with minimum and maximum values"""
        a = arr.astype(float, copy=False)
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi > lo + 1e-12:
            return (a - lo) / (hi - lo)
        return np.full_like(a, 0.5, dtype=float)

    def ani_centrality_within_cluster(
        idxs: list[int],
        ani: "np.ndarray",
        homogeneity_delta: float = 0.05,
    ) -> "np.ndarray":
        """
        Compute ANI centrality within a cluster.

        For each genome i in the cluster:
        - Compute m_i = mean ANI from i to all other members.
        - If max(m_i) - min(m_i) < homogeneity_delta (in % ANI),
            treat the cluster as homogeneous and return 0.5 for all members.
        - Otherwise, linearly scale m_i so that min(m_i) -> 0 and max(m_i) -> 1.
        """
        n = len(idxs)
        if n <= 1:
            return np.ones(n, dtype=float)

        means: list[float] = []
        for i in idxs:
            others = [ani[i, j] for j in idxs if j != i]
            mean_ani = float(np.mean(others)) if others else 100.0
            means.append(mean_ani)

        m = np.asarray(means, dtype=float)
        m_min = float(m.min())
        m_max = float(m.max())
        spread = m_max - m_min

        # Homogeneous cluster: centrality isn't meaningful -> constant 0.5
        if spread < homogeneity_delta:
            return np.full_like(m, 0.5, dtype=float)

        # Heterogeneous cluster: scale to [0, 1]
        return (m - m_min) / spread

    ## Scoring Components =====================================================
    # Assembly
    A_vals = np.array([g.Assembly_Rank / 3.0 for g in infos], dtype=float)

    # BUSCO
    B_raw = np.array([(g.BUSCO_C - 1.0 * g.BUSCO_M) / 100.0 for g in infos], dtype=float)
    B_vals = minmax_norm(winsorize(B_raw))

    # CheckM2
    Q_raw = np.array(
        [(g.CheckM2_Completeness - 5.0 * g.CheckM2_Contamination) / 100.0 for g in infos],
        dtype=float,
    )
    Q_vals = minmax_norm(winsorize(Q_raw))

    # N50
    n50_log = np.log10(np.array([g.N50 for g in infos], dtype=float) + 1.0)
    N_vals = minmax_norm(winsorize(n50_log))

    # Scaffolds
    cont_log = np.log10(np.array([g.Scaffolds for g in infos], dtype=float) + 1.0)
    S_vals = 1.0 - minmax_norm(winsorize(cont_log))

    # ANI centrality
    C_vals = ani_centrality_within_cluster(idxs, ani, homogeneity_delta=0.05)

    ## Weights ================================================================
    if score_profile == "isolate":
        wA, wQ, wB, wN, wS, wC = 3.5, 1.5, 1.5, 1.00, 0.25, 0.50
    elif score_profile == "mag":
        wA, wQ, wB, wN, wS, wC = 1.0, 3.0, 2.0, 0.50, 1.50, 0.50
    else:  # default
        wA, wQ, wB, wN, wS, wC = 3.0, 2.0, 1.5, 0.75, 0.50, 0.50

    ## Score per candidate ====================================================

    # Compute normalized total score and collect component breakdowns for DEBUG
    scored: list[tuple[str, float, dict[str, float], Genome]] = []
    for i_loc, g in enumerate(infos):
        comps = {
            "A": float(A_vals[i_loc]),
            "Q": float(Q_vals[i_loc]),
            "B": float(B_vals[i_loc]),
            "N": float(N_vals[i_loc]),
            "S": float(S_vals[i_loc]),
            "C": float(C_vals[i_loc]),
            "Q_raw": float(Q_raw[i_loc]),
            "B_raw": float(B_raw[i_loc]),
        }
        total = (
            wA * comps["A"]
            + wQ * comps["Q"]
            + wB * comps["B"]
            + wN * comps["N"]
            + wS * comps["S"]
            + wC * comps["C"]
        )

        g.Score = float(total)

        scored.append((g.Accession, float(total), comps, g))

    # Emit per-candidate debug lines
    dbg.append(
        f"Cluster size={n}; profile={score_profile}; weights:"
        f" A={wA}, Q={wQ}, B={wB}, N={wN}, S={wS}, C={wC}"
    )
    for acc, total, c, g in sorted(scored, key=lambda x: (-x[1], x[0])):
        dbg.append(
            "  %s: score=%.6f | A=%.3f Q=%.3f(q_raw=%.3f) B=%.3f(b_raw=%.3f)"
            " N=%.3f S=%.3f C=%.3f | asm=%s(rank=%d) busco(C=%.2f,M=%.2f)"
            " checkm2=%.2f contam=%.2f scaffolds=%d n50=%d"
            % (
                acc,
                total,
                c["A"],
                c["Q"],
                c["Q_raw"],
                c["B"],
                c["B_raw"],
                c["N"],
                c["S"],
                c["C"],
                g.Assembly_Level,
                g.Assembly_Rank,
                g.BUSCO_C,
                g.BUSCO_M,
                g.CheckM2_Completeness,
                g.CheckM2_Contamination,
                g.Scaffolds,
                g.N50,
            )
        )

    # Primary selection by score
    scored.sort(key=lambda x: (-x[1], x[0]))
    top_score = scored[0][1]
    tied = [acc for acc, s, _, _ in scored if abs(s - top_score) <= _EPS]
    if len(tied) == 1:
        dbg.append(f"Chosen by score alone: {tied[0]} (score={top_score:.6f})")
        return n, tied[0], dbg

    ## Tie cascade ============================================================
    acc2gen = {g.Accession: g for _, _, _, g in scored}

    def reduce_tie(
        acc_list: list[str], key_fn, desc: str, prefer_lower: bool = False
    ) -> list[str]:
        if len(acc_list) <= 1:
            return acc_list
        vals = [key_fn(acc2gen[a]) for a in acc_list]
        best = (min if prefer_lower else max)(vals)
        winners = [a for a, v in zip(acc_list, vals, strict=True) if v == best]
        if len(winners) < len(acc_list):
            dbg.append(f"Tie-break by {desc}: {len(acc_list)} -> {len(winners)} (best={best}).")
        return winners

    # 1) higher assembly rank
    # 2) BUSCO: higher C, then lower M
    # 3) CheckM2: lower contamination, then higher completeness
    # 4) fewer Scaffolds
    # 5) higher N50
    # 6) lexicographically smallest accession
    tied = reduce_tie(tied, lambda g: g.Assembly_Rank, "Assembly rank (higher)")
    tied = reduce_tie(tied, lambda g: g.BUSCO_C, "BUSCO C (higher)")
    tied = reduce_tie(tied, lambda g: g.BUSCO_M, "BUSCO M (lower)", True)
    tied = reduce_tie(tied, lambda g: g.CheckM2_Contamination, "Contamination (lower)", True)
    tied = reduce_tie(tied, lambda g: g.CheckM2_Completeness, "CheckM2 (higher)")
    tied = reduce_tie(tied, lambda g: g.Scaffolds, "Scaffolds (fewer)", True)
    tied = reduce_tie(tied, lambda g: g.N50, "N50 (higher)")

    rep_acc = sorted(tied)[0]
    if len(tied) > 1:
        dbg.append(
            f"Final tie among {len(tied)} candidate(s); picking lexicographically smallest: {rep_acc}"
        )
    else:
        dbg.append(f"Chosen by tie cascade: {rep_acc}")

    return n, rep_acc, dbg


def select_representatives_for_clusters(
    clusters: dict[int, list[int]],
    names: list[str],
    meta: dict[str, Genome],
    ani: "np.ndarray",
    score_profile: str,
    threads: int,
) -> list[tuple[int, str, int, list[int], list[str]]]:
    """
    Select representatives for all clusters in parallel.

    Returns:
        A list of tuples:
          (cluster_size, rep_acc, original_label, idxs, debug_lines)
    """
    results: list[tuple[int, str, int, list[int], list[str]]] = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        fut_to_lab = {
            ex.submit(
                select_representative_for_indices,
                idxs,
                names,
                meta,
                ani,
                score_profile,
            ): lab
            for lab, idxs in clusters.items()
        }
        for fut in as_completed(fut_to_lab):
            lab = fut_to_lab[fut]
            try:
                size, rep_acc, dbg_lines = fut.result()
                results.append((size, rep_acc, lab, clusters[lab], dbg_lines))
            except Exception as e:
                die(f"Error selecting representative for cluster {lab}: {e}")
    return results


def assign_cluster_ids(
    results: list[tuple[int, str, int, list[int], list[str]]],
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """
    Assign stable Cluster_IDs (C000001, C000002, ...) to clusters.
    Clusters are ordered by:
      - size (descending)
      - representative accession (lex ascending)
    """
    results.sort(key=lambda x: (-x[0], x[1]))
    rep_by_cid: dict[str, str] = {}  # cluster idx = representative sample idx
    idxs_by_cid: dict[str, list[int]] = {}  # cluster idx = samples idx in cluster
    for i, (size, rep_acc, _lab, idxs, dbg_lines) in enumerate(results, start=1):
        cid = f"C{i:06d}"
        rep_by_cid[cid] = rep_acc
        idxs_by_cid[cid] = idxs
        for line in dbg_lines:
            logging.debug("%s (%s): %s", cid, rep_acc, line)
    return rep_by_cid, idxs_by_cid


def warn_gcode_mixture(
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    meta: dict[str, Genome],
) -> None:
    """
    Emit warnings for clusters that mix Gcode 4 and 11 genomes.
    """
    for cid, idxs in idxs_by_cid.items():
        g4 = g11 = 0
        for idx in idxs:
            g = meta[names[idx]].Gcode
            if g == 4:
                g4 += 1
            elif g == 11:
                g11 += 1
        if g4 > 0 and g11 > 0:
            logging.warning("Gcode mixture in %s: gcode4=%d, gcode11=%d", cid, g4, g11)


def warn_soft_screen_offenders(
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    meta: dict[str, Genome],
) -> None:
    """
    List all soft-screen offenders per cluster (warn only).
    Offender: Completeness < 90 or Contamination > 5.
    """
    for cid, idxs in idxs_by_cid.items():
        offenders: list[str] = []
        for idx in idxs:
            g = meta[names[idx]]
            reasons: list[str] = []
            if g.CheckM2_Completeness < 90:
                reasons.append(f"Completeness={g.CheckM2_Completeness:.2f}")
            if g.CheckM2_Contamination > 5:
                reasons.append(f"Contam={g.CheckM2_Contamination:.2f}%")
            if reasons:
                offenders.append(f"{g.Accession}[{';'.join(reasons)}]")
        if offenders:
            logging.warning("Soft-screen offenders in %s: %s", cid, ", ".join(offenders))


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def build_cluster_rows(
    rep_by_cid: dict[str, str],
    idxs_by_cid: dict[str, list[int]],
    names: list[str],
    ani: "np.ndarray",
    name_to_idx: dict[str, int],
    meta: dict[str, Genome],
) -> list[tuple[str, str, str, str, str, str]]:
    """
    Construct rows for cluster.tsv.
    Each row has:
        Accession, Cluster_ID, Is_Representative, ANI_to_Representative, Score, Path
    """
    import numpy as np  # local import for type name

    rows: list[tuple[str, str, str, str, str, str]] = []
    for cid, idxs in idxs_by_cid.items():
        rep_acc = rep_by_cid[cid]
        rep_idx = name_to_idx[rep_acc]
        for idx in sorted(idxs, key=lambda k: names[k]):
            acc = names[idx]
            path = meta[acc].Path
            score = meta[acc].Score

            is_rep = "yes" if acc == rep_acc else "no"
            if acc == rep_acc:
                ani_to_rep = "100.0000"
            else:
                v = ani[idx, rep_idx]
                if np.isnan(v):
                    die(
                        f"Internal error: NA ANI inside complete-link cluster {cid} "
                        f"between '{acc}' and representative '{rep_acc}'"
                    )
                ani_to_rep = f"{v:.4f}"

            if score is None:
                die(
                    f"Internal error: composite score not set for accession '{acc}' "
                    f"in cluster {cid}"
                )
            score_str = f"{score:.6f}"

            rows.append((acc, cid, is_rep, ani_to_rep, score_str, path))
    return rows


def build_representative_rows(
    rep_by_cid: dict[str, str],
    idxs_by_cid: dict[str, list[int]],
    meta: dict[str, Genome],
) -> list[tuple[str, str, str, str, str, str, str, str, str]]:
    """
    Construct rows for representatives.tsv.
    Each row has:
        Cluster_ID, Representative_Accession, Organism_Name,
        CheckM2_Completeness, CheckM2_Contamination
        BUSCO, Assembly_Level, N50, Cluster_Size
    """
    rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    for cid, idxs in idxs_by_cid.items():
        rep_acc = rep_by_cid[cid]
        g = meta[rep_acc]
        org_out = re.sub(r"\s+", "_", (g.Organism_Name or "").strip())
        rows.append(
            (
                cid,
                rep_acc,
                org_out,
                f"{g.CheckM2_Completeness:.2f}",
                f"{g.CheckM2_Contamination:.2f}",
                g.BUSCO_str,
                g.Assembly_Level,
                f"{g.N50:d}",
                str(len(idxs)),
            )
        )
    return rows


def write_outputs(
    outdir: Path,
    cluster_rows: list[tuple[str, str, str, str, str, str]],
    reps_rows: list[tuple[str, str, str, str, str, str, str, str, str]],
) -> None:
    """
    Write cluster.tsv and representatives.tsv to the output directory.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cluster_tsv = outdir / "cluster.tsv"
    reps_tsv = outdir / "representatives.tsv"

    with cluster_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "Accession",
                "Cluster_ID",
                "Is_Representative",
                "ANI_to_Representative",
                "Score",
                "Path",
            ]
        )
        w.writerows(cluster_rows)

    with reps_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "Cluster_ID",
                "Representative_Accession",
                "Organism_Name",
                "CheckM2_Completeness",
                "CheckM2_Contamination",
                "BUSCO",
                "Assembly_Level",
                "N50",
                "Cluster_Size",
            ]
        )
        w.writerows(reps_rows)

    logging.info("Wrote: %s", cluster_tsv)
    logging.info("Wrote: %s", reps_tsv)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(args: argparse.Namespace, threads: int) -> None:
    """
    Run the full clustering and representative selection pipeline.

    Steps:
      1) Load ANI matrix.
      2) Load and validate TSV/CSV tables.
      3) Build per-genome metadata.
      4) Complete-linkage clustering.
      5) Parallel representative selection.
      6) Assign cluster IDs.
      7) Warn on Gcode mixtures.
      8) Build output rows.
      9) Write output files.
    """
    if not (0.0 < float(args.threshold) < 1.0):
        die(f"--threshold must be a fraction in (0,1). Got: {args.threshold}")
    threshold = float(args.threshold)

    # 1) ANI matrix
    names, ani, name_to_idx = load_matrix(args.ani_matrix)

    # 2) Tables + structural checks
    tsv, csv_df = load_and_check_tables(
        input_list=args.input_list,
        metadata=args.metadata,
        matrix_names=names,
    )

    # 3) Build per-genome metadata
    meta = build_genome_metadata(names, tsv, csv_df)

    # 4) Complete-linkage clustering
    clusters = cluster_complete_linkage(ani, names, threshold)

    # 5) Representative selection (parallel)
    results = select_representatives_for_clusters(
        clusters=clusters,
        names=names,
        meta=meta,
        ani=ani,
        score_profile=args.score_profile,
        threads=threads,
    )

    # 6) Assign cluster IDs
    rep_by_cid, idxs_by_cid = assign_cluster_ids(results)

    # 7) Warnings
    warn_gcode_mixture(idxs_by_cid, names, meta)
    warn_soft_screen_offenders(idxs_by_cid, names, meta)

    # 8) Build output rows
    cluster_rows = build_cluster_rows(
        rep_by_cid=rep_by_cid,
        idxs_by_cid=idxs_by_cid,
        names=names,
        ani=ani,
        name_to_idx=name_to_idx,
        meta=meta,
    )
    reps_rows = build_representative_rows(
        rep_by_cid=rep_by_cid,
        idxs_by_cid=idxs_by_cid,
        meta=meta,
    )

    # 9) Write outputs
    write_outputs(args.outdir, cluster_rows, reps_rows)


def main() -> None:
    """
    Entry point: parse arguments, configure logging and threading, then run pipeline.
    """
    args = parse_args()
    configure_logging(args.quiet, args.debug)

    # Threads: respect both user request and hardware/scheduler limits
    requested = max(1, int(args.threads))
    threads = resolve_thread_cap(requested)
    if threads < requested:
        logging.info(
            "Capping --threads from %d to %d based on hardware/scheduler limits.",
            requested,
            threads,
        )

    # Cap BLAS/OpenMP/numexpr before heavy imports:
    set_thread_envs(threads)

    # Heavy imports (fail fast if libraries are missing)
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    run_pipeline(args, threads)


if __name__ == "__main__":
    main()
