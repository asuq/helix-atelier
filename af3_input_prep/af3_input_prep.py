#!/usr/bin/env python3
"""
Prepare AlphaFold 3 input JSON files from small TSV specifications.

TSV schema
----------
Required columns:
    entity_type, id, source

Supported entity_type values:
    protein        source is a FASTA/FAA file with exactly one record
    ligand_ccd     source is a CCD code, such as F43, COM, PQQ, or ATP
    ligand_smiles  source is a SMILES string; local style only
    ion            source is an ion CCD code, such as MG, ZN, CA, or NI

Optional columns:
    stoichiometry, chain_ids, description

Examples
--------
    af3_input_prep single anme2d_mcr.tsv --style local --outdir af3_inputs
    af3_input_prep batch anme1.tsv anme2d.tsv --style server --outdir af3_inputs --prefix methane

Limitations
-----------
This first version intentionally does not implement PTMs, custom CCD entries,
covalent bonds, RNA/DNA entities, MSA paths, or structural templates.

Local AlphaFold 3 JSON is written as one job object per file. AlphaFold Server
JSON is written as a top-level list, even for a single job, and batch server
mode combines all jobs into one upload JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from Bio import SeqIO


LOCAL_DIALECT = "alphafold3"
LOCAL_VERSION = 4
SERVER_DIALECT = "alphafoldserver"
SERVER_VERSION = 1

REQUIRED_COLUMNS = {"entity_type", "id", "source"}
SUPPORTED_ENTITY_TYPES = {"protein", "ligand_ccd", "ligand_smiles", "ion"}
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
NONCANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
MAX_UINT32 = 2**32 - 1

MANIFEST_COLUMNS = [
    "job_name",
    "input_tsv",
    "output_json",
    "style",
    "status",
    "message",
]


class Af3InputPrepError(Exception):
    """Expected user-facing error raised by this utility."""


@dataclass(frozen=True)
class InputRow:
    entity_type: str
    id: str
    source: str
    stoichiometry: str
    chain_ids: str
    description: str
    line_number: int


@dataclass(frozen=True)
class PreparedEntity:
    entity_type: str
    ids: str | list[str]
    count: int
    source: str
    sequence: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class PreparedJob:
    name: str
    model_seeds: list[int]
    entities: list[PreparedEntity]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AlphaFold 3 input JSON files from TSV specifications.",
        epilog=(
            "TSV columns: required entity_type, id, source; optional "
            "stoichiometry, chain_ids, description. Supported entity_type values: "
            "protein, ligand_ccd, ligand_smiles, ion. Server style supports "
            "proteinChain, CCD ligand, and ion entities; SMILES ligands are "
            "local-style only in this utility."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single",
        help="Convert one TSV into one AF3 prediction job JSON.",
        description="Convert one TSV into one AF3 prediction job JSON.",
    )
    single.add_argument("input_tsv", type=Path, help="Input TSV specification.")
    add_common_options(single, prefix_required=False)

    batch = subparsers.add_parser(
        "batch",
        help="Convert one or more TSVs into batch AF3 input JSON.",
        description=(
            "Convert one or more TSVs. Server style writes one combined top-level "
            "list. Local style writes one JSON per TSV plus a manifest TSV."
        ),
    )
    batch.add_argument("input_tsvs", nargs="+", type=Path, help="Input TSV specification(s).")
    add_common_options(batch, prefix_required=False)

    return parser.parse_args(argv)


def add_common_options(parser: argparse.ArgumentParser, *, prefix_required: bool) -> None:
    parser.add_argument("--style", required=True, choices=["local", "server"], help="Output JSON dialect.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--prefix",
        required=prefix_required,
        help=(
            "Optional job name or prefix. For single, defaults to the input TSV stem. "
            "For batch, prepends this value to each TSV stem when supplied."
        ),
    )
    parser.add_argument(
        "--seeds",
        help=(
            "Comma-separated positive integer seeds. Defaults: local=1, server=auto. "
            "Use 'auto' only with server style."
        ),
    )
    parser.add_argument(
        "--fasta-root",
        type=Path,
        help="Resolve relative protein FASTA paths relative to this directory.",
    )
    parser.add_argument(
        "--allow-noncanonical-aa",
        action="store_true",
        help="Allow B, X, Z, J, U, and O in protein sequences.",
    )
    parser.add_argument(
        "--preserve-ligand-case",
        action="store_true",
        help="Do not uppercase CCD and ion codes.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation. Default: 2.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout messages.")


def read_tsv(path: Path) -> list[InputRow]:
    if not path.is_file():
        raise Af3InputPrepError(f"TSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise Af3InputPrepError(f"{path}: missing header row")

        fieldnames = {field.strip() for field in reader.fieldnames if field is not None}
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise Af3InputPrepError(
                f"{path}: missing required column(s): {', '.join(missing)}"
            )

        rows: list[InputRow] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw_row.values()):
                continue

            rows.append(
                InputRow(
                    entity_type=(raw_row.get("entity_type") or "").strip().lower(),
                    id=(raw_row.get("id") or "").strip(),
                    source=(raw_row.get("source") or "").strip(),
                    stoichiometry=(raw_row.get("stoichiometry") or "").strip(),
                    chain_ids=(raw_row.get("chain_ids") or "").strip(),
                    description=(raw_row.get("description") or "").strip(),
                    line_number=line_number,
                )
            )

    if not rows:
        raise Af3InputPrepError(f"{path}: contains no entity rows")

    return rows


def normalise_row(
    row: InputRow,
    tsv_path: Path,
    fasta_root: Path | None,
    options: argparse.Namespace,
) -> PreparedEntity:
    if row.entity_type not in SUPPORTED_ENTITY_TYPES:
        valid = ", ".join(sorted(SUPPORTED_ENTITY_TYPES))
        raise Af3InputPrepError(
            f"{tsv_path}:{row.line_number}: unknown entity_type '{row.entity_type}'. "
            f"Valid values: {valid}"
        )
    if not row.id:
        raise Af3InputPrepError(f"{tsv_path}:{row.line_number}: id must not be empty")
    if not row.source:
        raise Af3InputPrepError(f"{tsv_path}:{row.line_number}: source must not be empty")

    stoichiometry = parse_stoichiometry(row.stoichiometry, tsv_path, row.line_number)
    chain_ids = parse_chain_ids(row.chain_ids)
    if chain_ids is not None and stoichiometry is not None and len(chain_ids) != stoichiometry:
        raise Af3InputPrepError(
            f"{tsv_path}:{row.line_number}: chain_ids count ({len(chain_ids)}) does not "
            f"match stoichiometry ({stoichiometry})"
        )

    effective_stoichiometry = len(chain_ids) if chain_ids is not None else (stoichiometry or 1)
    ids = expand_ids(row.id, effective_stoichiometry, chain_ids)
    description = row.description or None

    if row.entity_type == "protein":
        fasta_path = resolve_fasta_path(row.source, tsv_path, fasta_root)
        sequence = read_single_record_fasta(
            fasta_path,
            allow_noncanonical=options.allow_noncanonical_aa,
        )
        return PreparedEntity(
            entity_type=row.entity_type,
            ids=ids,
            count=effective_stoichiometry,
            source=str(fasta_path),
            sequence=sequence,
            description=description,
        )

    source = row.source.strip()
    if row.entity_type in {"ligand_ccd", "ion"} and not options.preserve_ligand_case:
        source = source.upper()

    return PreparedEntity(
        entity_type=row.entity_type,
        ids=ids,
        count=effective_stoichiometry,
        source=source,
        description=description,
    )


def parse_stoichiometry(value: str, tsv_path: Path, line_number: int) -> int | None:
    if not value:
        return None
    try:
        stoichiometry = int(value)
    except ValueError as exc:
        raise Af3InputPrepError(
            f"{tsv_path}:{line_number}: stoichiometry must be a positive integer"
        ) from exc
    if stoichiometry <= 0:
        raise Af3InputPrepError(
            f"{tsv_path}:{line_number}: stoichiometry must be a positive integer"
        )
    return stoichiometry


def parse_chain_ids(value: str) -> list[str] | None:
    if not value.strip():
        return None
    chain_ids = [part.strip() for part in value.split(",")]
    if any(not chain_id for chain_id in chain_ids):
        raise Af3InputPrepError("chain_ids must be a comma-separated list of non-empty IDs")
    return chain_ids


def expand_ids(
    base_id: str,
    stoichiometry: int,
    chain_ids: list[str] | None,
) -> str | list[str]:
    if stoichiometry <= 0:
        raise Af3InputPrepError("stoichiometry must be a positive integer")
    ids = chain_ids if chain_ids is not None else (
        [base_id] if stoichiometry == 1 else [f"{base_id}{idx}" for idx in range(1, stoichiometry + 1)]
    )
    return ids[0] if len(ids) == 1 else list(ids)


def resolve_fasta_path(source: str, tsv_path: Path, fasta_root: Path | None) -> Path:
    raw_path = Path(source).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path
    else:
        base = fasta_root.expanduser() if fasta_root is not None else tsv_path.parent
        resolved = base / raw_path

    if not resolved.is_file():
        raise Af3InputPrepError(f"Protein FASTA file not found: {resolved}")
    return resolved


def read_single_record_fasta(path: Path, allow_noncanonical: bool) -> str:
    if not path.is_file():
        raise Af3InputPrepError(f"Protein FASTA file not found: {path}")

    has_header = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                has_header = True
                break
    if not has_header:
        raise Af3InputPrepError(
            f"{path}: FASTA file contains no records or lacks a header line starting with '>'"
        )

    try:
        records = list(SeqIO.parse(str(path), "fasta"))
    except Exception as exc:  # BioPython parser errors should be user-facing here.
        raise Af3InputPrepError(f"{path}: failed to parse FASTA: {exc}") from exc

    if not records:
        raise Af3InputPrepError(
            f"{path}: FASTA file contains no records or lacks a header line starting with '>'"
        )
    if len(records) > 1:
        raise Af3InputPrepError(
            f"{path}: expected exactly one FASTA record, found {len(records)}"
        )

    sequence = "".join(str(records[0].seq).split()).upper()
    if not sequence:
        raise Af3InputPrepError(f"{path}: FASTA record has an empty sequence")

    allowed = NONCANONICAL_AA if allow_noncanonical else CANONICAL_AA
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        chars = ", ".join(invalid)
        raise Af3InputPrepError(f"{path}: invalid protein sequence character(s): {chars}")

    return sequence


def validate_unique_ids(entities: Sequence[PreparedEntity]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for entity in entities:
        for chain_id in flatten_ids(entity.ids):
            if chain_id in seen:
                duplicates.append(chain_id)
            else:
                seen[chain_id] = entity.entity_type
    if duplicates:
        unique_duplicates = ", ".join(sorted(set(duplicates)))
        raise Af3InputPrepError(f"Duplicate expanded entity ID(s): {unique_duplicates}")


def flatten_ids(ids: str | Sequence[str]) -> list[str]:
    if isinstance(ids, str):
        return [ids]
    return list(ids)


def parse_seeds(seed_text: str | None, style: str) -> list[int]:
    if seed_text is None:
        return [1] if style == "local" else []

    seed_text = seed_text.strip()
    if seed_text.lower() == "auto":
        if style != "server":
            raise Af3InputPrepError("--seeds auto is only allowed with --style server")
        return []

    parts = [part.strip() for part in seed_text.split(",")]
    if any(not part for part in parts):
        raise Af3InputPrepError("--seeds must be a comma-separated list of positive integers")

    seeds: list[int] = []
    for part in parts:
        try:
            seed = int(part)
        except ValueError as exc:
            raise Af3InputPrepError(
                "--seeds must be a comma-separated list of positive integers"
            ) from exc
        if seed <= 0 or seed > MAX_UINT32:
            raise Af3InputPrepError(
                f"--seeds values must be positive uint32 integers (1..{MAX_UINT32})"
            )
        seeds.append(seed)

    if style == "local" and not seeds:
        raise Af3InputPrepError("local style requires at least one model seed")
    return seeds


def build_job_from_tsv(
    path: Path,
    job_name: str,
    style: str,
    options: argparse.Namespace,
) -> dict[str, Any]:
    rows = read_tsv(path)
    entities = [normalise_row(row, path, options.fasta_root, options) for row in rows]
    validate_unique_ids(entities)

    prepared_job = PreparedJob(
        name=job_name,
        model_seeds=parse_seeds(options.seeds, style),
        entities=entities,
    )
    if style == "local":
        return build_local_job(prepared_job)
    if style == "server":
        return build_server_job(prepared_job)
    raise Af3InputPrepError(f"Unsupported style: {style}")


def build_local_job(prepared_job: PreparedJob) -> dict[str, Any]:
    if not prepared_job.model_seeds:
        raise Af3InputPrepError("local style requires at least one model seed")

    sequences: list[dict[str, Any]] = []
    for entity in prepared_job.entities:
        if entity.entity_type == "protein":
            if entity.sequence is None:
                raise Af3InputPrepError("internal error: protein entity missing sequence")
            payload: dict[str, Any] = {"id": entity.ids, "sequence": entity.sequence}
            add_description(payload, entity.description)
            sequences.append({"protein": payload})
        elif entity.entity_type == "ligand_ccd":
            payload = {"id": entity.ids, "ccdCodes": [entity.source]}
            add_description(payload, entity.description)
            sequences.append({"ligand": payload})
        elif entity.entity_type == "ligand_smiles":
            payload = {"id": entity.ids, "smiles": entity.source}
            add_description(payload, entity.description)
            sequences.append({"ligand": payload})
        elif entity.entity_type == "ion":
            payload = {"id": entity.ids, "ccdCodes": [strip_ccd_prefix(entity.source)]}
            add_description(payload, entity.description)
            sequences.append({"ligand": payload})
        else:
            raise Af3InputPrepError(f"Unsupported entity_type: {entity.entity_type}")

    return {
        "name": prepared_job.name,
        "modelSeeds": prepared_job.model_seeds,
        "sequences": sequences,
        "dialect": LOCAL_DIALECT,
        "version": LOCAL_VERSION,
    }


def build_server_job(prepared_job: PreparedJob) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    for entity in prepared_job.entities:
        if entity.entity_type == "protein":
            if entity.sequence is None:
                raise Af3InputPrepError("internal error: protein entity missing sequence")
            sequences.append(
                {"proteinChain": {"sequence": entity.sequence, "count": entity.count}}
            )
        elif entity.entity_type == "ligand_ccd":
            sequences.append(
                {
                    "ligand": {
                        "ligand": format_server_ligand_code(entity.source),
                        "count": entity.count,
                    }
                }
            )
        elif entity.entity_type == "ligand_smiles":
            raise Af3InputPrepError(
                "SMILES ligand rows are not supported in server-style JSON by this "
                "script/schema. Use local style or convert the ligand to a "
                "CCD-supported code if possible."
            )
        elif entity.entity_type == "ion":
            sequences.append(
                {"ion": {"ion": strip_ccd_prefix(entity.source), "count": entity.count}}
            )
        else:
            raise Af3InputPrepError(f"Unsupported entity_type: {entity.entity_type}")

    return {
        "name": prepared_job.name,
        "modelSeeds": [str(seed) for seed in prepared_job.model_seeds],
        "sequences": sequences,
        "dialect": SERVER_DIALECT,
        "version": SERVER_VERSION,
    }


def add_description(payload: dict[str, Any], description: str | None) -> None:
    if description:
        payload["description"] = description


def strip_ccd_prefix(code: str) -> str:
    return re.sub(r"^CCD_", "", code.strip(), flags=re.IGNORECASE)


def format_server_ligand_code(code: str) -> str:
    return f"CCD_{strip_ccd_prefix(code)}"


def sanitise_name(value: str, *, label: str) -> str:
    replaced = "".join(
        char if char.isalnum() or char in "_-." else "_" for char in value.strip()
    )
    collapsed = re.sub(r"_+", "_", replaced).strip("_")
    if not collapsed:
        raise Af3InputPrepError(f"{label} is empty after sanitisation")
    return collapsed


def write_json(path: Path, obj: Any, indent: int, force: bool) -> None:
    ensure_can_write(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def write_manifest(path: Path, rows: Sequence[dict[str, str]], force: bool) -> None:
    ensure_can_write(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise Af3InputPrepError(f"Output already exists, use --force to overwrite: {path}")


def preflight_outputs(paths: Sequence[Path], force: bool) -> None:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for path in paths:
        if path in seen:
            duplicates.append(path)
        seen.add(path)
        ensure_can_write(path, force)
    if duplicates:
        duplicate_text = ", ".join(str(path) for path in sorted(set(duplicates)))
        raise Af3InputPrepError(f"Duplicate output path(s): {duplicate_text}")


def validate_runtime_options(args: argparse.Namespace) -> None:
    if args.indent < 0:
        raise Af3InputPrepError("--indent must be zero or a positive integer")


def run_single(args: argparse.Namespace) -> None:
    validate_runtime_options(args)
    job_name = args.prefix if args.prefix else args.input_tsv.stem
    output_name = sanitise_name(job_name, label="single job name")
    output_path = args.outdir / f"{output_name}.json"
    preflight_outputs([output_path], args.force)

    job = build_job_from_tsv(args.input_tsv, job_name, args.style, args)
    output_obj: Any = job if args.style == "local" else [job]
    write_json(output_path, output_obj, args.indent, args.force)

    if not args.quiet:
        print(f"Created {output_path}")


def run_batch(args: argparse.Namespace) -> None:
    validate_runtime_options(args)
    prefix = sanitise_name(args.prefix, label="--prefix") if args.prefix else None

    job_names = [batch_job_name(path, prefix) for path in args.input_tsvs]
    validate_unique_job_names(job_names)

    if args.style == "server":
        output_path = args.outdir / (f"{prefix}_batch.json" if prefix else "batch.json")
        preflight_outputs([output_path], args.force)
        jobs = [
            build_job_from_tsv(path, job_name, args.style, args)
            for path, job_name in zip(args.input_tsvs, job_names, strict=True)
        ]
        write_json(output_path, jobs, args.indent, args.force)
        if not args.quiet:
            print(f"Created {output_path}")
        return

    output_paths = [args.outdir / f"{job_name}.json" for job_name in job_names]
    manifest_path = args.outdir / (
        f"{prefix}_batch_manifest.tsv" if prefix else "batch_manifest.tsv"
    )
    preflight_outputs([*output_paths, manifest_path], args.force)

    jobs = [
        build_job_from_tsv(path, job_name, args.style, args)
        for path, job_name in zip(args.input_tsvs, job_names, strict=True)
    ]
    manifest_rows: list[dict[str, str]] = []
    for input_tsv, output_path, job_name, job in zip(
        args.input_tsvs, output_paths, job_names, jobs, strict=True
    ):
        write_json(output_path, job, args.indent, args.force)
        manifest_rows.append(
            {
                "job_name": job_name,
                "input_tsv": str(input_tsv),
                "output_json": str(output_path),
                "style": args.style,
                "status": "created",
                "message": "",
            }
        )

    write_manifest(manifest_path, manifest_rows, args.force)
    if not args.quiet:
        print(f"Created {len(output_paths)} JSON file(s) and {manifest_path}")


def batch_job_name(path: Path, prefix: str | None) -> str:
    stem = sanitise_name(path.stem, label=f"TSV stem for {path}")
    return f"{prefix}_{stem}" if prefix else stem


def validate_unique_job_names(job_names: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in job_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise Af3InputPrepError(f"Duplicate batch job name(s): {duplicate_text}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "single":
            run_single(args)
        elif args.command == "batch":
            run_batch(args)
        else:
            raise Af3InputPrepError(f"Unsupported command: {args.command}")
    except Af3InputPrepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
