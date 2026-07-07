#!/usr/bin/env python3
"""
Create ligand/cofactor inventory tables from AlphaFill mmCIF outputs.

This script recursively scans AlphaFill-enriched CIF files and extracts
non-protein ligand/cofactor instances from the _atom_site table.

It writes three TSV files:
    alphafill_ligand_inventory.tsv
    alphafill_model_summary.tsv
    alphafill_target_summary.tsv

No external Python packages are required.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
}

WATER_CODES = {"HOH", "WAT", "DOD", "H2O"}

COMMON_ION_CODES = {
    "NA",
    "K",
    "MG",
    "CA",
    "ZN",
    "FE",
    "MN",
    "CO",
    "NI",
    "CU",
    "CL",
    "BR",
    "IOD",
    "CD",
    "HG",
    "SR",
    "BA",
    "CS",
    "RB",
}

COMMON_BUFFER_OR_ADDITIVE_CODES = {
    "SO4",
    "PO4",
    "NO3",
    "ACT",
    "ACY",
    "ACE",
    "GOL",
    "EDO",
    "PEG",
    "PGE",
    "PG4",
    "MPD",
    "TRS",
    "MES",
    "HEP",
    "MSE",
    "DMS",
    "DMSO",
    "ETX",
}

DEFAULT_CLOSE_MCR_DONORS = {"9QR1", "9QQT", "9QR3", "9QM5"}
PDB_ID_RE = re.compile(r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")

DIRECT_CLASS_BY_COMP_ID = {
    "F43": "F430",
    "COM": "CoM",
    "TP7": "CoB",
    "TPZ": "CoB_like",
    "TXZ": "CoB_like",
    "F42": "F420",
    "F40": "F420_like",
    "F4N": "F420_like",
    "FMN": "FMN",
    "FAD": "FAD",
    "MFN": "MFR",
    "SF4": "FeS_cluster",
    "FES": "FeS_cluster",
    "F3S": "FeS_cluster",
    "FS4": "FeS_cluster",
    "B12": "corrinoid",
    "CNC": "corrinoid",
    "CBL": "corrinoid",
    "CBA": "corrinoid",
    "COB": "corrinoid",
    "HEM": "heme",
    "HEC": "heme",
    "HEA": "heme",
    "HEB": "heme",
}

TARGET_EXPECTED_CLASSES = {
    "MCR": {
        "yes": {"F430", "CoM", "CoB", "CoB_like", "F430_like"},
        "possible": {"ion", "metal", "FeS_cluster"},
    },
    "Ftr": {
        "yes": {"H4MPT", "MFR", "formyl_MFR"},
        "possible": {"ion", "metal"},
    },
    "Mch": {
        "yes": {"H4MPT", "methenyl_H4MPT"},
        "possible": {"MFR", "ion", "metal"},
    },
    "Mer": {
        "yes": {"F420", "F420_like"},
        "possible": {"H4MPT", "FMN", "FAD", "ion", "metal"},
    },
    "Mtd": {
        "yes": {"F420", "F420_like"},
        "possible": {"H4MPT", "FMN", "FAD", "ion", "metal"},
    },
    "Mtr": {
        "yes": {"corrinoid", "H4MPT", "CoM"},
        "possible": {"ion", "metal", "Na_site", "F420", "F420_like"},
    },
    "unknown": {
        "yes": set(),
        "possible": {
            "F430",
            "CoM",
            "CoB",
            "CoB_like",
            "F420",
            "F420_like",
            "H4MPT",
            "MFR",
            "corrinoid",
            "FMN",
            "FAD",
            "FeS_cluster",
            "heme",
        },
    },
}


class AlphaFillInventoryError(Exception):
    """Expected user-facing error raised by this utility."""


@dataclass
class LigandInstance:
    target: str
    model_index: str
    cif_path: str
    cif_name: str
    parent_dir: str
    ligand_comp_id: str
    ligand_name: str
    ligand_entity_id: str
    ligand_instance_id: str
    label_asym_id: str
    auth_asym_id: str
    label_seq_id: str
    auth_seq_id: str
    atom_count: int
    element_counts: str
    biological_class: str
    expected_for_target: str
    inspection_priority: str
    ignored_reason: str
    notes: str


@dataclass
class ModelSummary:
    target: str
    model_index: str
    cif_path: str
    cif_name: str
    parent_dir: str
    total_ligand_instances: int
    high_priority_instances: int
    medium_priority_instances: int
    low_priority_instances: int
    ignored_instances: int
    ligand_comp_ids: str
    high_priority_ligands: str
    expected_ligands_found: str
    f43_count: int
    com_count: int
    tp7_count: int
    txz_tpz_count: int
    f420_like_count: int
    fmn_fad_count: int
    ion_count: int
    buffer_additive_count: int
    preliminary_verdict: str
    json_found: str
    json_transplant_records: int
    donor_pdb_ids: str
    high_priority_donor_pdb_ids: str
    mcr_f43_donor_pdb_ids: str
    mcr_com_donor_pdb_ids: str
    mcr_cob_like_donor_pdb_ids: str
    has_close_anme2d_mcr_donor: str
    close_anme2d_mcr_donors: str


@dataclass
class TargetSummary:
    target: str
    n_models: int
    models_with_high_priority: int
    models_with_expected_ligands: int
    best_model_index: str
    best_model_high_priority_instances: int
    observed_ligand_comp_ids: str
    observed_high_priority_ligands: str
    preliminary_target_verdict: str
    donor_pdb_ids: str
    high_priority_donor_pdb_ids: str
    models_with_json: int
    models_with_close_anme2d_mcr_donor: int
    target_donor_interpretation: str


@dataclass
class AlphaFillJsonTransplant:
    target: str
    model_index: str
    cif_name: str
    parent_dir: str
    ligand_comp_id: str
    donor_compound_id: str
    biological_class: str
    expected_for_target: str
    inspection_priority: str
    donor_pdb_id: str
    donor_asym_id: str
    donor_auth_asym_id: str
    donor_auth_seq_id: str
    donor_label_asym_id: str
    donor_label_seq_id: str
    recipient_asym_id: str
    recipient_auth_asym_id: str
    recipient_auth_seq_id: str
    recipient_label_asym_id: str
    recipient_label_seq_id: str
    analogue_id: str
    local_rmsd: str
    global_rmsd: str
    clash: str
    clash_count: str
    sequence_identity: str
    alignment_length: str
    coverage: str
    json_record_path: str
    notes: str
    cif_path: str
    json_path: str
    raw_json_path: str


@dataclass
class AlphaFillDonorSummary:
    target: str
    model_index: str
    ligand_comp_id: str
    biological_class: str
    donor_pdb_id: str
    n_transplants: int
    min_local_rmsd: str
    median_local_rmsd: str
    max_local_rmsd: str
    n_clashing: int
    recipient_sites: str
    donor_sites: str
    inspection_priority: str
    expected_for_target: str


TABLE_FIELD_ORDER: dict[type[object], list[str]] = {
    LigandInstance: [
        "target",
        "model_index",
        "cif_name",
        "parent_dir",
        "ligand_comp_id",
        "ligand_name",
        "ligand_entity_id",
        "ligand_instance_id",
        "label_asym_id",
        "auth_asym_id",
        "label_seq_id",
        "auth_seq_id",
        "atom_count",
        "element_counts",
        "biological_class",
        "expected_for_target",
        "inspection_priority",
        "ignored_reason",
        "notes",
        "cif_path",
    ],
    ModelSummary: [
        "target",
        "model_index",
        "cif_name",
        "parent_dir",
        "total_ligand_instances",
        "high_priority_instances",
        "medium_priority_instances",
        "low_priority_instances",
        "ignored_instances",
        "ligand_comp_ids",
        "high_priority_ligands",
        "expected_ligands_found",
        "f43_count",
        "com_count",
        "tp7_count",
        "txz_tpz_count",
        "f420_like_count",
        "fmn_fad_count",
        "ion_count",
        "buffer_additive_count",
        "preliminary_verdict",
        "json_found",
        "json_transplant_records",
        "donor_pdb_ids",
        "high_priority_donor_pdb_ids",
        "mcr_f43_donor_pdb_ids",
        "mcr_com_donor_pdb_ids",
        "mcr_cob_like_donor_pdb_ids",
        "has_close_anme2d_mcr_donor",
        "close_anme2d_mcr_donors",
        "cif_path",
    ],
    TargetSummary: [
        "target",
        "n_models",
        "models_with_high_priority",
        "models_with_expected_ligands",
        "best_model_index",
        "best_model_high_priority_instances",
        "observed_ligand_comp_ids",
        "observed_high_priority_ligands",
        "preliminary_target_verdict",
        "donor_pdb_ids",
        "high_priority_donor_pdb_ids",
        "models_with_json",
        "models_with_close_anme2d_mcr_donor",
        "target_donor_interpretation",
    ],
    AlphaFillJsonTransplant: [
        "target",
        "model_index",
        "cif_name",
        "parent_dir",
        "ligand_comp_id",
        "donor_compound_id",
        "biological_class",
        "expected_for_target",
        "inspection_priority",
        "donor_pdb_id",
        "donor_asym_id",
        "donor_auth_asym_id",
        "donor_auth_seq_id",
        "donor_label_asym_id",
        "donor_label_seq_id",
        "recipient_asym_id",
        "recipient_auth_asym_id",
        "recipient_auth_seq_id",
        "recipient_label_asym_id",
        "recipient_label_seq_id",
        "analogue_id",
        "local_rmsd",
        "global_rmsd",
        "clash",
        "clash_count",
        "sequence_identity",
        "alignment_length",
        "coverage",
        "json_record_path",
        "notes",
        "cif_path",
        "json_path",
        "raw_json_path",
    ],
    AlphaFillDonorSummary: [
        "target",
        "model_index",
        "ligand_comp_id",
        "biological_class",
        "donor_pdb_id",
        "n_transplants",
        "min_local_rmsd",
        "median_local_rmsd",
        "max_local_rmsd",
        "n_clashing",
        "recipient_sites",
        "donor_sites",
        "inspection_priority",
        "expected_for_target",
    ],
}


def iter_cif_tokens(path: Path) -> Iterable[str]:
    """
    Tokenise mmCIF content.

    Handles whitespace-separated tokens, quoted tokens, comments, and
    semicolon-delimited multiline text fields.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise AlphaFillInventoryError(f"Failed to read CIF file {path}: {exc}") from exc

    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i]

        if line.startswith(";"):
            text_lines = [line[1:]]
            i += 1
            while i < n_lines and not lines[i].startswith(";"):
                text_lines.append(lines[i])
                i += 1
            if i < n_lines and lines[i].startswith(";"):
                i += 1
            yield "\n".join(text_lines)
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        for token in tokenise_cif_line(line, path, i + 1):
            yield token

        i += 1


def tokenise_cif_line(line: str, path: Path, line_number: int) -> list[str]:
    """
    Tokenise one non-multiline CIF line.

    CIF quotes delimit a value only when they begin a token. Apostrophes inside
    bare values, such as ADENOSINE-5'-DIPHOSPHATE or atom names like C5', are
    ordinary characters.
    """
    tokens: list[str] = []
    i = 0
    n_chars = len(line)

    while i < n_chars:
        while i < n_chars and line[i].isspace():
            i += 1

        if i >= n_chars or line[i] == "#":
            break

        if line[i] in {"'", '"'}:
            quote = line[i]
            i += 1
            start = i
            value_parts: list[str] = []

            while i < n_chars:
                if line[i] == quote and (i + 1 == n_chars or line[i + 1].isspace()):
                    value_parts.append(line[start:i])
                    i += 1
                    break
                i += 1
            else:
                raise AlphaFillInventoryError(
                    f"{path}:{line_number}: quoted CIF value is missing closing {quote}"
                )

            tokens.append("".join(value_parts))
            continue

        start = i
        while i < n_chars and not line[i].isspace():
            i += 1
        tokens.append(line[start:i])

    return tokens


def category_from_tag(tag: str) -> str:
    """Return category name from a tag such as _atom_site.label_comp_id."""
    if not tag.startswith("_"):
        return ""
    body = tag[1:]
    if "." in body:
        return body.split(".", 1)[0]
    return body


def parse_cif_categories(path: Path, wanted_categories: set[str]) -> dict[str, list[dict[str, str]]]:
    """
    Parse selected categories from a CIF file.

    This is a pragmatic parser for AF3/AlphaFill outputs. It is not intended to
    be a complete mmCIF validator.
    """
    tokens = list(iter_cif_tokens(path))
    categories: dict[str, list[dict[str, str]]] = {cat: [] for cat in wanted_categories}

    i = 0
    n_tokens = len(tokens)
    while i < n_tokens:
        token = tokens[i]

        if token == "loop_":
            i += 1
            tags: list[str] = []
            while i < n_tokens and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1
            if not tags:
                continue

            loop_categories = {category_from_tag(tag) for tag in tags}
            keep_loop = len(loop_categories) == 1 and next(iter(loop_categories)) in wanted_categories
            category = next(iter(loop_categories)) if len(loop_categories) == 1 else ""
            width = len(tags)
            row_values: list[str] = []

            while i < n_tokens:
                next_token = tokens[i]
                if (
                    next_token == "loop_"
                    or next_token.startswith("data_")
                    or next_token.startswith("save_")
                ):
                    break
                if next_token.startswith("_") and len(row_values) % width == 0:
                    break
                row_values.append(next_token)
                i += 1

            if keep_loop:
                clean_tags = [tag.split(".", 1)[1] if "." in tag else tag for tag in tags]
                if len(row_values) % width != 0:
                    print(
                        f"[WARNING] Incomplete loop rows for category {category} in {path}",
                        file=sys.stderr,
                    )
                for j in range(0, len(row_values) - width + 1, width):
                    values = row_values[j : j + width]
                    categories[category].append(dict(zip(clean_tags, values, strict=True)))
            continue

        if token.startswith("_"):
            category = category_from_tag(token)
            if category in wanted_categories:
                item_name = token.split(".", 1)[1] if "." in token else token
                if i + 1 < n_tokens:
                    value = tokens[i + 1]
                    categories[category].append({item_name: value})
                    i += 2
                    continue

        i += 1

    return categories


def first_nonempty(*values: str | None) -> str:
    """Return first value that is not empty, '.', '?', or None."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in {".", "?"}:
            return text
    return ""


def normalise_comp_id(comp_id: str) -> str:
    return comp_id.strip().upper()


def infer_target_from_path(path: Path) -> str:
    """Infer biological target/enzyme from path or filename."""
    text = str(path).lower()
    if "mcr_core" in text or re.search(r"(^|[_/-])mcr([_/-]|$)", text):
        return "MCR"
    if re.search(r"(^|[_/-])ftr([_/-]|$)", text):
        return "Ftr"
    if re.search(r"(^|[_/-])mch([_/-]|$)", text):
        return "Mch"
    if re.search(r"(^|[_/-])mer([_/-]|$)", text):
        return "Mer"
    if re.search(r"(^|[_/-])mtd([_/-]|$)", text):
        return "Mtd"
    if re.search(r"(^|[_/-])mtr([_/-]|$)", text):
        return "Mtr"
    return "unknown"


def infer_model_index(path: Path) -> str:
    """Infer AF3 model index from filename."""
    match = re.search(r"_model_(\d+)", path.name)
    if match:
        return match.group(1)
    return ""


def build_entity_map(entity_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map entity.id to the entity row."""
    entity_map: dict[str, dict[str, str]] = {}
    for row in entity_rows:
        entity_id = first_nonempty(row.get("id"))
        if entity_id:
            entity_map[entity_id] = row
    return entity_map


def build_struct_asym_map(struct_asym_rows: list[dict[str, str]]) -> dict[str, str]:
    """Map struct_asym.id to entity_id."""
    asym_to_entity: dict[str, str] = {}
    for row in struct_asym_rows:
        asym_id = first_nonempty(row.get("id"))
        entity_id = first_nonempty(row.get("entity_id"))
        if asym_id and entity_id:
            asym_to_entity[asym_id] = entity_id
    return asym_to_entity


def build_chem_comp_map(chem_comp_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map chem_comp.id to the chem_comp row."""
    chem_map: dict[str, dict[str, str]] = {}
    for row in chem_comp_rows:
        comp_id = normalise_comp_id(first_nonempty(row.get("id")))
        if comp_id:
            chem_map[comp_id] = row
    return chem_map


def get_ligand_name(
    comp_id: str,
    entity_id: str,
    entity_map: dict[str, dict[str, str]],
    chem_map: dict[str, dict[str, str]],
) -> str:
    """Get best ligand name or description."""
    chem_row = chem_map.get(comp_id, {})
    entity_row = entity_map.get(entity_id, {})
    return first_nonempty(
        chem_row.get("name"),
        entity_row.get("pdbx_description"),
        chem_row.get("pdbx_synonyms"),
    )


def classify_by_name(name: str) -> str:
    """Classify ligand/cofactor using descriptive names."""
    text = name.lower()
    if not text:
        return "unknown"
    if "factor 430" in text or "f430" in text:
        return "F430"
    if "coenzyme m" in text or "thioethanesulfonic" in text:
        return "CoM"
    if "coenzyme b" in text:
        return "CoB"
    if "phosphono" in text and "sulfanyl" in text and "threonine" in text:
        return "CoB_like"
    if "f420" in text or "coenzyme f420" in text:
        return "F420"
    if "methanopterin" in text or "h4mpt" in text or "tetrahydromethanopterin" in text:
        if "methenyl" in text:
            return "methenyl_H4MPT"
        return "H4MPT"
    if "methanofuran" in text:
        if "formyl" in text:
            return "formyl_MFR"
        return "MFR"
    if "cobalamin" in text or "corrinoid" in text or "cobamide" in text:
        return "corrinoid"
    if "flavin mononucleotide" in text:
        return "FMN"
    if "flavin adenine dinucleotide" in text:
        return "FAD"
    if "heme" in text or "haem" in text:
        return "heme"
    if "iron-sulfur" in text or "iron sulfur" in text:
        return "FeS_cluster"
    if "potassium ion" in text or "sodium ion" in text or "magnesium ion" in text:
        return "ion"
    if "calcium ion" in text or "zinc ion" in text or "nickel ion" in text:
        return "metal"
    if "ethoxyethanol" in text:
        return "buffer_or_additive"
    return "unknown"


def classify_ligand(comp_id: str, ligand_name: str) -> str:
    """Classify ligand/cofactor class."""
    comp_id = normalise_comp_id(comp_id)
    if comp_id in WATER_CODES:
        return "water"
    if comp_id in DIRECT_CLASS_BY_COMP_ID:
        return DIRECT_CLASS_BY_COMP_ID[comp_id]
    if comp_id in COMMON_ION_CODES:
        if comp_id in {"ZN", "FE", "NI", "MN", "CO", "CU", "MO", "W"}:
            return "metal"
        return "ion"
    if comp_id in COMMON_BUFFER_OR_ADDITIVE_CODES:
        return "buffer_or_additive"

    by_name = classify_by_name(ligand_name)
    if by_name != "unknown":
        return by_name
    return "unknown"


def expected_and_priority(target: str, biological_class: str, comp_id: str) -> tuple[str, str, str]:
    """Return expected_for_target, inspection_priority, and ignored_reason."""
    target_rules = TARGET_EXPECTED_CLASSES.get(target, TARGET_EXPECTED_CLASSES["unknown"])

    if biological_class == "water":
        return "no", "ignore", "water"
    if biological_class == "buffer_or_additive":
        return "no", "ignore", "common_buffer_or_additive"
    if biological_class in {"ion", "metal"}:
        if biological_class in target_rules["yes"]:
            return "yes", "high", ""
        if biological_class in target_rules["possible"]:
            return "possible", "low", ""
        return "no", "low", ""
    if biological_class in target_rules["yes"]:
        return "yes", "high", ""
    if biological_class in target_rules["possible"]:
        return "possible", "medium", ""
    if target in {"Mer", "Mtd"} and biological_class in {"FMN", "FAD"}:
        return "possible", "medium", "check_if_F420_false_positive"
    if target == "MCR" and comp_id in {"FMN", "FAD"}:
        return "no", "low", "not_expected_for_MCR"
    return "unknown", "low", ""


def find_matching_json(cif_path: Path, json_pattern: str | None = None) -> Path | None:
    """Find the most likely AlphaFill JSON metadata file for a CIF file."""
    exact = cif_path.with_suffix(".json")
    if exact.is_file():
        return exact

    pattern = json_pattern or "*.json"
    candidates = sorted(path for path in cif_path.parent.glob(pattern) if path.is_file())
    if not candidates:
        return None

    cif_stem = cif_path.stem
    base_stems = [cif_stem]
    if cif_stem.endswith("_alphafill"):
        base_stems.append(cif_stem.removesuffix("_alphafill"))

    scored: list[tuple[int, str, Path]] = []
    for candidate in candidates:
        score = max(shared_prefix_score(base_stem, candidate.stem) for base_stem in base_stems)
        if score > 0:
            scored.append((-score, candidate.name, candidate))

    if not scored:
        return None
    return sorted(scored)[0][2]


def shared_prefix_score(left: str, right: str) -> int:
    """Score a plausible same-directory CIF/JSON stem match."""
    if left == right:
        return 10_000 + len(left)
    if left.startswith(right) or right.startswith(left):
        return 5_000 + min(len(left), len(right))

    limit = min(len(left), len(right))
    i = 0
    while i < limit and left[i] == right[i]:
        i += 1

    if i >= max(8, min(len(left), len(right)) // 2):
        return i
    return 0


def parse_close_mcr_donors(value: str) -> set[str]:
    """Parse comma-separated close MCR donor PDB IDs."""
    donors = {normalise_pdb_id(part) for part in value.split(",") if part.strip()}
    return {donor for donor in donors if donor}


def normalise_pdb_id(value: str) -> str:
    """Return a normalised PDB ID if value contains one."""
    match = PDB_ID_RE.search(value.strip())
    return match.group(1).upper() if match else ""


def extract_pdb_id_from_record(record: dict[str, Any]) -> str:
    """Extract the first plausible PDB ID from known donor/hit fields."""
    keys = (
        "pdb_id",
        "pdb",
        "entry_id",
        "source_pdb_id",
        "donor_pdb_id",
        "hit_pdb_id",
        "target",
        "hit",
        "id",
        "name",
        "defline",
        "description",
    )
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            pdb_id = normalise_pdb_id(value)
            if pdb_id:
                return pdb_id
    return ""


def parse_alphafill_json(
    path: Path,
    *,
    target: str = "",
    model_index: str = "",
    cif_path: Path | None = None,
) -> list[AlphaFillJsonTransplant]:
    """Parse AlphaFill JSON metadata into transplant donor records."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AlphaFillInventoryError(f"Failed to parse JSON file {path}: {exc}") from exc
    except OSError as exc:
        raise AlphaFillInventoryError(f"Failed to read JSON file {path}: {exc}") from exc

    raw_json_path = ""
    if isinstance(payload, dict):
        raw_json_path = stringify_json_scalar(payload.get("file"))

    rows: list[AlphaFillJsonTransplant] = []
    base_context = {
        "donor_pdb_id": "",
        "donor_asym_id": "",
        "donor_auth_asym_id": "",
        "donor_auth_seq_id": "",
        "donor_label_asym_id": "",
        "donor_label_seq_id": "",
        "sequence_identity": "",
        "alignment_length": "",
        "coverage": "",
        "global_rmsd": "",
    }
    walk_json_for_transplants(
        payload,
        record_path="",
        context=base_context,
        rows=rows,
        json_path=path,
        target=target,
        model_index=model_index,
        cif_path=cif_path,
        raw_json_path=raw_json_path,
    )
    return rows


def walk_json_for_transplants(
    node: Any,
    *,
    record_path: str,
    context: dict[str, str],
    rows: list[AlphaFillJsonTransplant],
    json_path: Path,
    target: str,
    model_index: str,
    cif_path: Path | None,
    raw_json_path: str,
) -> None:
    """Recursively walk JSON and extract likely transplant records."""
    if isinstance(node, dict):
        next_context = derive_json_context(node, context)
        if is_likely_transplant_record(node):
            rows.append(
                build_json_transplant_row(
                    node,
                    context=next_context,
                    record_path=record_path,
                    json_path=json_path,
                    target=target,
                    model_index=model_index,
                    cif_path=cif_path,
                    raw_json_path=raw_json_path,
                )
            )

        for key, value in node.items():
            child_path = key if not record_path else f"{record_path}.{key}"
            walk_json_for_transplants(
                value,
                record_path=child_path,
                context=next_context,
                rows=rows,
                json_path=json_path,
                target=target,
                model_index=model_index,
                cif_path=cif_path,
                raw_json_path=raw_json_path,
            )
        return

    if isinstance(node, list):
        for index, value in enumerate(node):
            child_path = f"{record_path}[{index}]" if record_path else f"[{index}]"
            walk_json_for_transplants(
                value,
                record_path=child_path,
                context=context,
                rows=rows,
                json_path=json_path,
                target=target,
                model_index=model_index,
                cif_path=cif_path,
                raw_json_path=raw_json_path,
            )


def derive_json_context(record: dict[str, Any], context: dict[str, str]) -> dict[str, str]:
    """Carry donor and alignment metadata from parent JSON dictionaries."""
    updated = dict(context)
    updated["donor_pdb_id"] = first_nonempty(extract_pdb_id_from_record(record), context.get("donor_pdb_id"))
    updated["donor_asym_id"] = first_nonempty(
        stringify_json_scalar(record.get("pdb_asym_id")),
        stringify_json_scalar(record.get("donor_asym_id")),
        context.get("donor_asym_id"),
    )
    updated["donor_auth_asym_id"] = first_nonempty(
        stringify_json_scalar(record.get("pdb_auth_asym_id")),
        stringify_json_scalar(record.get("donor_auth_asym_id")),
        context.get("donor_auth_asym_id"),
    )
    updated["donor_auth_seq_id"] = first_nonempty(
        stringify_json_scalar(record.get("pdb_auth_seq_id")),
        stringify_json_scalar(record.get("donor_auth_seq_id")),
        context.get("donor_auth_seq_id"),
    )
    updated["donor_label_asym_id"] = first_nonempty(
        stringify_json_scalar(record.get("pdb_label_asym_id")),
        stringify_json_scalar(record.get("donor_label_asym_id")),
        context.get("donor_label_asym_id"),
    )
    updated["donor_label_seq_id"] = first_nonempty(
        stringify_json_scalar(record.get("pdb_label_seq_id")),
        stringify_json_scalar(record.get("donor_label_seq_id")),
        context.get("donor_label_seq_id"),
    )
    updated["global_rmsd"] = first_nonempty(
        stringify_json_scalar(record.get("global_rmsd")),
        stringify_json_scalar(record.get("rmsd")),
        context.get("global_rmsd"),
    )

    alignment = record.get("alignment")
    if isinstance(alignment, dict):
        updated["sequence_identity"] = first_nonempty(
            stringify_json_scalar(alignment.get("sequence_identity")),
            stringify_json_scalar(alignment.get("identity")),
            context.get("sequence_identity"),
        )
        updated["alignment_length"] = first_nonempty(
            stringify_json_scalar(alignment.get("alignment_length")),
            stringify_json_scalar(alignment.get("length")),
            context.get("alignment_length"),
        )
        updated["coverage"] = first_nonempty(
            stringify_json_scalar(alignment.get("coverage")),
            context.get("coverage"),
        )

    updated["sequence_identity"] = first_nonempty(
        stringify_json_scalar(record.get("sequence_identity")),
        stringify_json_scalar(record.get("identity")),
        updated.get("sequence_identity"),
    )
    updated["alignment_length"] = first_nonempty(
        stringify_json_scalar(record.get("alignment_length")),
        updated.get("alignment_length"),
    )
    updated["coverage"] = first_nonempty(
        stringify_json_scalar(record.get("coverage")),
        updated.get("coverage"),
    )
    return updated


def is_likely_transplant_record(record: dict[str, Any]) -> bool:
    """Return true for likely AlphaFill transplant/ligand records."""
    has_ligand = any(
        first_nonempty(stringify_json_scalar(record.get(key)))
        for key in ("compound_id", "analogue_id", "ligand_id", "comp_id")
    )
    if not has_ligand:
        return False
    return any(
        key in record
        for key in (
            "local_rmsd",
            "pdb_asym_id",
            "pdb_auth_seq_id",
            "pdb_label_seq_id",
            "asym_id",
            "auth_seq_id",
            "label_seq_id",
            "clash",
            "clashes",
            "clash_count",
        )
    )


def build_json_transplant_row(
    record: dict[str, Any],
    *,
    context: dict[str, str],
    record_path: str,
    json_path: Path,
    target: str,
    model_index: str,
    cif_path: Path | None,
    raw_json_path: str,
) -> AlphaFillJsonTransplant:
    """Build one JSON transplant row from a likely transplant dictionary."""
    donor_compound_id = normalise_comp_id(
        first_nonempty(
            stringify_json_scalar(record.get("compound_id")),
            stringify_json_scalar(record.get("ligand_id")),
            stringify_json_scalar(record.get("comp_id")),
        )
    )
    analogue_id = normalise_comp_id(stringify_json_scalar(record.get("analogue_id")))
    ligand_comp_id = normalise_comp_id(first_nonempty(analogue_id, donor_compound_id))
    biological_class = classify_ligand(ligand_comp_id, "")
    expected, priority, ignored_reason = expected_and_priority(target, biological_class, ligand_comp_id)
    clash_value, clash_count = extract_clash_fields(record)

    notes = []
    if ignored_reason:
        notes.append(ignored_reason)
    if donor_compound_id and analogue_id and donor_compound_id != analogue_id:
        notes.append(f"donor_compound_id={donor_compound_id}")

    return AlphaFillJsonTransplant(
        target=target,
        model_index=model_index,
        cif_name=cif_path.name if cif_path is not None else "",
        parent_dir=cif_path.parent.name if cif_path is not None else "",
        ligand_comp_id=ligand_comp_id,
        donor_compound_id=donor_compound_id,
        biological_class=biological_class,
        expected_for_target=expected,
        inspection_priority=priority,
        donor_pdb_id=first_nonempty(extract_pdb_id_from_record(record), context.get("donor_pdb_id")),
        donor_asym_id=first_nonempty(
            stringify_json_scalar(record.get("pdb_asym_id")),
            stringify_json_scalar(record.get("donor_asym_id")),
            context.get("donor_asym_id"),
        ),
        donor_auth_asym_id=first_nonempty(
            stringify_json_scalar(record.get("pdb_auth_asym_id")),
            stringify_json_scalar(record.get("donor_auth_asym_id")),
            context.get("donor_auth_asym_id"),
        ),
        donor_auth_seq_id=first_nonempty(
            stringify_json_scalar(record.get("pdb_auth_seq_id")),
            stringify_json_scalar(record.get("donor_auth_seq_id")),
            context.get("donor_auth_seq_id"),
        ),
        donor_label_asym_id=first_nonempty(
            stringify_json_scalar(record.get("pdb_label_asym_id")),
            stringify_json_scalar(record.get("donor_label_asym_id")),
            context.get("donor_label_asym_id"),
        ),
        donor_label_seq_id=first_nonempty(
            stringify_json_scalar(record.get("pdb_label_seq_id")),
            stringify_json_scalar(record.get("donor_label_seq_id")),
            context.get("donor_label_seq_id"),
        ),
        recipient_asym_id=first_nonempty(
            stringify_json_scalar(record.get("asym_id")),
            stringify_json_scalar(record.get("recipient_asym_id")),
            stringify_json_scalar(record.get("label_asym_id")),
        ),
        recipient_auth_asym_id=first_nonempty(
            stringify_json_scalar(record.get("auth_asym_id")),
            stringify_json_scalar(record.get("recipient_auth_asym_id")),
        ),
        recipient_auth_seq_id=first_nonempty(
            stringify_json_scalar(record.get("auth_seq_id")),
            stringify_json_scalar(record.get("recipient_auth_seq_id")),
        ),
        recipient_label_asym_id=first_nonempty(
            stringify_json_scalar(record.get("label_asym_id")),
            stringify_json_scalar(record.get("recipient_label_asym_id")),
        ),
        recipient_label_seq_id=first_nonempty(
            stringify_json_scalar(record.get("label_seq_id")),
            stringify_json_scalar(record.get("recipient_label_seq_id")),
        ),
        analogue_id=analogue_id,
        local_rmsd=first_nonempty(
            stringify_json_scalar(record.get("local_rmsd")),
            stringify_json_scalar(record.get("rmsd")),
        ),
        global_rmsd=context.get("global_rmsd", ""),
        clash=clash_value,
        clash_count=clash_count,
        sequence_identity=context.get("sequence_identity", ""),
        alignment_length=context.get("alignment_length", ""),
        coverage=context.get("coverage", ""),
        json_record_path=record_path,
        notes=";".join(notes),
        cif_path=str(cif_path) if cif_path is not None else "",
        json_path=str(json_path),
        raw_json_path=raw_json_path,
    )


def stringify_json_scalar(value: Any) -> str:
    """Convert scalar JSON values to compact TSV text."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    return str(value).strip()


def extract_clash_fields(record: dict[str, Any]) -> tuple[str, str]:
    """Extract compact clash score/value and clash count."""
    clash = record.get("clash", record.get("clashes"))
    count = record.get("clash_count")

    if isinstance(clash, dict):
        count = clash.get("clash_count", count)
        return stringify_json_scalar(clash.get("score", "yes")), stringify_json_scalar(count)

    return stringify_json_scalar(clash), stringify_json_scalar(count)


def create_donor_summaries(transplants: list[AlphaFillJsonTransplant]) -> list[AlphaFillDonorSummary]:
    """Create one row per target/model/ligand/donor combination."""
    grouped: dict[tuple[str, str, str, str], list[AlphaFillJsonTransplant]] = collections.defaultdict(list)
    for row in transplants:
        grouped[(row.target, row.model_index, row.ligand_comp_id, row.donor_pdb_id)].append(row)

    summaries: list[AlphaFillDonorSummary] = []
    for (target, model_index, ligand_comp_id, donor_pdb_id), rows in sorted(grouped.items()):
        rmsds = [value for value in (parse_float(row.local_rmsd) for row in rows) if value is not None]
        priority = best_priority(row.inspection_priority for row in rows)
        expected = best_expected(row.expected_for_target for row in rows)
        summaries.append(
            AlphaFillDonorSummary(
                target=target,
                model_index=model_index,
                ligand_comp_id=ligand_comp_id,
                biological_class=first_nonempty(*(row.biological_class for row in rows)),
                donor_pdb_id=donor_pdb_id,
                n_transplants=len(rows),
                min_local_rmsd=format_float(min(rmsds)) if rmsds else "",
                median_local_rmsd=format_float(median(rmsds)) if rmsds else "",
                max_local_rmsd=format_float(max(rmsds)) if rmsds else "",
                n_clashing=sum(1 for row in rows if transplant_is_clashing(row)),
                recipient_sites=compact_unique(recipient_site_id(row) for row in rows),
                donor_sites=compact_unique(donor_site_id(row) for row in rows),
                inspection_priority=priority,
                expected_for_target=expected,
            )
        )
    return summaries


def parse_float(value: str) -> float | None:
    """Parse float text, returning None for empty or non-numeric values."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_float(value: float) -> str:
    """Format numeric metrics compactly for TSV output."""
    return f"{value:.6g}"


def transplant_is_clashing(row: AlphaFillJsonTransplant) -> bool:
    """Return true if a JSON transplant record reports one or more clashes."""
    count = parse_float(row.clash_count)
    if count is not None:
        return count > 0
    clash = row.clash.strip().lower()
    return bool(clash and clash not in {"0", "0.0", "no", "false"})


def recipient_site_id(row: AlphaFillJsonTransplant) -> str:
    """Create a compact recipient site identifier."""
    return "|".join(
        part if part else "."
        for part in [
            row.recipient_asym_id,
            first_nonempty(row.recipient_auth_asym_id, row.recipient_label_asym_id),
            first_nonempty(row.recipient_auth_seq_id, row.recipient_label_seq_id),
        ]
    )


def donor_site_id(row: AlphaFillJsonTransplant) -> str:
    """Create a compact donor site identifier."""
    return "|".join(
        part if part else "."
        for part in [
            row.donor_pdb_id,
            row.donor_asym_id,
            first_nonempty(row.donor_auth_asym_id, row.donor_label_asym_id),
            first_nonempty(row.donor_auth_seq_id, row.donor_label_seq_id),
        ]
    )


def compact_unique(items: Iterable[str]) -> str:
    """Return semicolon-separated unique non-empty values."""
    values: set[str] = set()
    for item in items:
        for part in str(item).split(";"):
            part = part.strip()
            if part and part.strip(".|"):
                values.add(part)
    return ";".join(sorted(values))


def best_priority(priorities: Iterable[str]) -> str:
    """Return the highest inspection priority represented."""
    rank = {"high": 4, "medium": 3, "low": 2, "ignore": 1, "": 0}
    values = list(priorities)
    if not values:
        return ""
    return max(values, key=lambda value: rank.get(value, 0))


def best_expected(values: Iterable[str]) -> str:
    """Return the strongest expected-for-target value represented."""
    rank = {"yes": 4, "possible": 3, "unknown": 2, "no": 1, "": 0}
    value_list = list(values)
    if not value_list:
        return ""
    return max(value_list, key=lambda value: rank.get(value, 0))


def parse_ligand_instances_from_cif(path: Path) -> list[LigandInstance]:
    """Parse one AlphaFill CIF and return ligand instances."""
    wanted = {"entity", "struct_asym", "chem_comp", "atom_site"}
    categories = parse_cif_categories(path, wanted)

    entity_map = build_entity_map(categories.get("entity", []))
    asym_to_entity = build_struct_asym_map(categories.get("struct_asym", []))
    chem_map = build_chem_comp_map(categories.get("chem_comp", []))
    target = infer_target_from_path(path)
    model_index = infer_model_index(path)

    grouped_atoms: dict[tuple[str, str, str, str, str], collections.Counter[str]] = {}
    grouped_meta: dict[tuple[str, str, str, str, str], dict[str, str]] = {}

    for row in categories.get("atom_site", []):
        group_pdb = first_nonempty(row.get("group_PDB")).upper()
        label_comp_id = normalise_comp_id(
            first_nonempty(row.get("label_comp_id"), row.get("auth_comp_id"))
        )
        if not label_comp_id or label_comp_id in STANDARD_AA:
            continue
        if group_pdb not in {"HETATM", ""} and label_comp_id in STANDARD_AA:
            continue
        if label_comp_id in WATER_CODES:
            continue

        label_asym_id = first_nonempty(row.get("label_asym_id"))
        auth_asym_id = first_nonempty(row.get("auth_asym_id"))
        label_entity_id = first_nonempty(row.get("label_entity_id"))
        label_seq_id = first_nonempty(row.get("label_seq_id"))
        auth_seq_id = first_nonempty(row.get("auth_seq_id"))
        if not label_entity_id and label_asym_id in asym_to_entity:
            label_entity_id = asym_to_entity[label_asym_id]

        residue_number = first_nonempty(auth_seq_id, label_seq_id, ".")
        instance_key = (
            label_comp_id,
            label_asym_id,
            auth_asym_id,
            residue_number,
            label_entity_id,
        )
        element = first_nonempty(row.get("type_symbol")).upper()
        grouped_atoms.setdefault(instance_key, collections.Counter())
        grouped_atoms[instance_key][element] += 1
        grouped_meta.setdefault(
            instance_key,
            {
                "label_seq_id": label_seq_id,
                "auth_seq_id": auth_seq_id,
            },
        )

    instances: list[LigandInstance] = []
    for key, element_counter in sorted(grouped_atoms.items()):
        comp_id, label_asym_id, auth_asym_id, residue_number, entity_id = key
        meta = grouped_meta[key]
        ligand_name = get_ligand_name(comp_id, entity_id, entity_map, chem_map)
        biological_class = classify_ligand(comp_id, ligand_name)
        expected, priority, ignored_reason = expected_and_priority(
            target,
            biological_class,
            comp_id,
        )
        element_counts = ",".join(
            f"{element}:{count}" for element, count in sorted(element_counter.items()) if element
        )
        ligand_instance_id = "|".join(
            part if part else "."
            for part in [comp_id, label_asym_id, auth_asym_id, residue_number, entity_id]
        )
        instances.append(
            LigandInstance(
                target=target,
                model_index=model_index,
                cif_path=str(path),
                cif_name=path.name,
                parent_dir=path.parent.name,
                ligand_comp_id=comp_id,
                ligand_name=ligand_name,
                ligand_entity_id=entity_id,
                ligand_instance_id=ligand_instance_id,
                label_asym_id=label_asym_id,
                auth_asym_id=auth_asym_id,
                label_seq_id=meta.get("label_seq_id", ""),
                auth_seq_id=meta.get("auth_seq_id", ""),
                atom_count=sum(element_counter.values()),
                element_counts=element_counts,
                biological_class=biological_class,
                expected_for_target=expected,
                inspection_priority=priority,
                ignored_reason=ignored_reason,
                notes="",
            )
        )

    return instances


def compact_counts(items: Iterable[str]) -> str:
    """Return compact count string such as A:2;B:3."""
    counter = collections.Counter(item for item in items if item)
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def create_model_summary(
    path: Path,
    instances: list[LigandInstance],
    json_transplants: list[AlphaFillJsonTransplant] | None = None,
    *,
    json_found: bool = False,
    close_mcr_donors: set[str] | None = None,
) -> ModelSummary:
    """Create one row per CIF/model."""
    target = infer_target_from_path(path)
    model_index = infer_model_index(path)
    json_transplants = json_transplants or []
    close_mcr_donors = close_mcr_donors or set()
    priority_counter = collections.Counter(instance.inspection_priority for instance in instances)
    ligand_comp_ids = compact_counts(instance.ligand_comp_id for instance in instances)
    high_priority_ligands = compact_counts(
        instance.ligand_comp_id for instance in instances if instance.inspection_priority == "high"
    )
    expected_found = any(instance.expected_for_target == "yes" for instance in instances)
    comp_counter = collections.Counter(instance.ligand_comp_id for instance in instances)
    f420_like_count = sum(
        count
        for comp_id, count in comp_counter.items()
        if classify_ligand(comp_id, "") in {"F420", "F420_like"}
    )
    fmn_fad_count = comp_counter.get("FMN", 0) + comp_counter.get("FAD", 0)
    ion_count = sum(count for comp_id, count in comp_counter.items() if comp_id in COMMON_ION_CODES)
    buffer_count = sum(
        count for comp_id, count in comp_counter.items() if comp_id in COMMON_BUFFER_OR_ADDITIVE_CODES
    )
    high_count = priority_counter.get("high", 0)
    medium_count = priority_counter.get("medium", 0)

    if high_count > 0:
        verdict = "inspect_high_priority"
    elif medium_count > 0:
        verdict = "inspect_medium_priority"
    elif instances:
        verdict = "low_priority_only"
    else:
        verdict = "no_ligands_detected"

    donor_ids = compact_unique(row.donor_pdb_id for row in json_transplants)
    high_priority_donor_ids = compact_unique(
        row.donor_pdb_id for row in json_transplants if row.inspection_priority == "high"
    )
    mcr_f43_donors = compact_unique(
        row.donor_pdb_id
        for row in json_transplants
        if target == "MCR" and row.ligand_comp_id == "F43"
    )
    mcr_com_donors = compact_unique(
        row.donor_pdb_id
        for row in json_transplants
        if target == "MCR" and row.ligand_comp_id == "COM"
    )
    mcr_cob_like_donors = compact_unique(
        row.donor_pdb_id
        for row in json_transplants
        if target == "MCR"
        and (
            row.ligand_comp_id in {"TP7", "TPZ", "TXZ"}
            or row.biological_class in {"CoB", "CoB_like"}
        )
    )
    close_donors = compact_unique(
        row.donor_pdb_id for row in json_transplants if row.donor_pdb_id in close_mcr_donors
    )

    return ModelSummary(
        target=target,
        model_index=model_index,
        cif_path=str(path),
        cif_name=path.name,
        parent_dir=path.parent.name,
        total_ligand_instances=len(instances),
        high_priority_instances=high_count,
        medium_priority_instances=medium_count,
        low_priority_instances=priority_counter.get("low", 0),
        ignored_instances=priority_counter.get("ignore", 0),
        ligand_comp_ids=ligand_comp_ids,
        high_priority_ligands=high_priority_ligands,
        expected_ligands_found="yes" if expected_found else "no",
        f43_count=comp_counter.get("F43", 0),
        com_count=comp_counter.get("COM", 0),
        tp7_count=comp_counter.get("TP7", 0),
        txz_tpz_count=comp_counter.get("TXZ", 0) + comp_counter.get("TPZ", 0),
        f420_like_count=f420_like_count,
        fmn_fad_count=fmn_fad_count,
        ion_count=ion_count,
        buffer_additive_count=buffer_count,
        preliminary_verdict=verdict,
        json_found="yes" if json_found else "no",
        json_transplant_records=len(json_transplants),
        donor_pdb_ids=donor_ids,
        high_priority_donor_pdb_ids=high_priority_donor_ids,
        mcr_f43_donor_pdb_ids=mcr_f43_donors,
        mcr_com_donor_pdb_ids=mcr_com_donors,
        mcr_cob_like_donor_pdb_ids=mcr_cob_like_donors,
        has_close_anme2d_mcr_donor="yes" if close_donors else "no",
        close_anme2d_mcr_donors=close_donors,
    )


def create_target_summaries(
    model_summaries: list[ModelSummary],
    json_transplants: list[AlphaFillJsonTransplant] | None = None,
) -> list[TargetSummary]:
    """Create one row per target."""
    by_target: dict[str, list[ModelSummary]] = collections.defaultdict(list)
    for summary in model_summaries:
        by_target[summary.target].append(summary)

    transplants_by_target: dict[str, list[AlphaFillJsonTransplant]] = collections.defaultdict(list)
    for row in json_transplants or []:
        transplants_by_target[row.target].append(row)

    target_summaries: list[TargetSummary] = []
    for target, rows in sorted(by_target.items()):
        rows_sorted = sorted(rows, key=lambda row: row.model_index)
        models_with_high = sum(1 for row in rows_sorted if row.high_priority_instances > 0)
        models_with_expected = sum(1 for row in rows_sorted if row.expected_ligands_found == "yes")
        best = max(
            rows_sorted,
            key=lambda row: (
                row.high_priority_instances,
                row.medium_priority_instances,
                -int(row.model_index) if row.model_index.isdigit() else 0,
            ),
        )
        observed_comp_counter: collections.Counter[str] = collections.Counter()
        observed_high_counter: collections.Counter[str] = collections.Counter()

        for row in rows_sorted:
            update_counter_from_compact_counts(observed_comp_counter, row.ligand_comp_ids)
            update_counter_from_compact_counts(observed_high_counter, row.high_priority_ligands)

        observed_comp_ids = ";".join(
            f"{key}:{observed_comp_counter[key]}" for key in sorted(observed_comp_counter)
        )
        observed_high = ";".join(
            f"{key}:{observed_high_counter[key]}" for key in sorted(observed_high_counter)
        )
        donor_pdb_ids = compact_unique(row.donor_pdb_ids for row in rows_sorted)
        high_priority_donor_pdb_ids = compact_unique(row.high_priority_donor_pdb_ids for row in rows_sorted)
        models_with_json = sum(1 for row in rows_sorted if row.json_found == "yes")
        models_with_close = sum(1 for row in rows_sorted if row.has_close_anme2d_mcr_donor == "yes")

        if models_with_high == len(rows_sorted) and rows_sorted:
            verdict = "robust_high_priority_transfer"
        elif models_with_high > 0:
            verdict = "model_dependent_high_priority_transfer"
        elif any(row.medium_priority_instances > 0 for row in rows_sorted):
            verdict = "medium_priority_only"
        elif any(row.total_ligand_instances > 0 for row in rows_sorted):
            verdict = "low_priority_or_irrelevant_only"
        else:
            verdict = "no_ligands_detected"

        target_summaries.append(
            TargetSummary(
                target=target,
                n_models=len(rows_sorted),
                models_with_high_priority=models_with_high,
                models_with_expected_ligands=models_with_expected,
                best_model_index=best.model_index,
                best_model_high_priority_instances=best.high_priority_instances,
                observed_ligand_comp_ids=observed_comp_ids,
                observed_high_priority_ligands=observed_high,
                preliminary_target_verdict=verdict,
                donor_pdb_ids=donor_pdb_ids,
                high_priority_donor_pdb_ids=high_priority_donor_pdb_ids,
                models_with_json=models_with_json,
                models_with_close_anme2d_mcr_donor=models_with_close,
                target_donor_interpretation=target_donor_interpretation(
                    target,
                    rows_sorted,
                    transplants_by_target.get(target, []),
                ),
            )
        )

    return target_summaries


def target_donor_interpretation(
    target: str,
    model_summaries: list[ModelSummary],
    transplants: list[AlphaFillJsonTransplant],
) -> str:
    """Return a compact donor interpretation flag for the target."""
    if target == "MCR":
        if any(row.has_close_anme2d_mcr_donor == "yes" for row in model_summaries):
            return "close_template_dependent_or_needs_exclusion_control"
        if any(
            row.ligand_comp_id in {"F43", "COM", "TP7", "TPZ", "TXZ"}
            or row.biological_class in {"F430", "CoM", "CoB", "CoB_like"}
            for row in transplants
        ):
            return "broad_mcr_family_transfer_candidate"
        return ""

    if target in {"Mer", "Mtd"}:
        if any(row.ligand_comp_id == "F42" for row in transplants):
            return "f420_like_donor_supported"
        if any(row.ligand_comp_id in {"FMN", "FAD"} for row in transplants):
            return "possible_F420_false_positive"
        return ""

    if target == "Ftr" and any(row.ligand_comp_id == "MFN" for row in transplants):
        return "mfr_donor_supported"

    if target == "Mtr" and any(row.ligand_comp_id == "B12" for row in transplants):
        return "corrinoid_donor_supported"

    return ""


def update_counter_from_compact_counts(counter: collections.Counter[str], text: str) -> None:
    """Add A:2;B:3 style counts into a Counter."""
    for token in text.split(";"):
        if not token:
            continue
        comp_id, count_text = token.split(":", 1)
        counter[comp_id] += int(count_text)


def dataclass_fieldnames(row_type: type[object]) -> list[str]:
    """Return explicit TSV field names for a dataclass row type."""
    if row_type in TABLE_FIELD_ORDER:
        fieldnames = TABLE_FIELD_ORDER[row_type]
    else:
        fieldnames = [field.name for field in fields(row_type)]

    dataclass_names = {field.name for field in fields(row_type)}
    missing = sorted(dataclass_names - set(fieldnames))
    extra = sorted(set(fieldnames) - dataclass_names)
    if missing or extra:
        raise AlphaFillInventoryError(
            f"Internal field-order mismatch for {row_type.__name__}: "
            f"missing={missing}, extra={extra}"
        )
    return fieldnames


def write_tsv(path: Path, rows: list[object], row_type: type[object]) -> None:
    """Write dataclass rows to TSV, including a header for empty tables."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = dataclass_fieldnames(row_type)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    except OSError as exc:
        raise AlphaFillInventoryError(f"Failed to write TSV file {path}: {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ligand/cofactor inventory from AlphaFill CIF outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing AlphaFill CIF outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for TSV summary tables.",
    )
    parser.add_argument(
        "--pattern",
        default="*_alphafill.cif",
        help="Glob pattern for AlphaFill CIF files. Default: *_alphafill.cif",
    )
    parser.add_argument(
        "--json-pattern",
        default="*_alphafill.json",
        help="Glob pattern for same-directory AlphaFill JSON fallback search. Default: *_alphafill.json",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable AlphaFill JSON metadata parsing.",
    )
    parser.add_argument(
        "--warn-missing-json",
        action="store_true",
        help="Print warnings when a CIF has no matching JSON metadata file.",
    )
    parser.add_argument(
        "--close-mcr-donors",
        default=",".join(sorted(DEFAULT_CLOSE_MCR_DONORS)),
        help="Comma-separated PDB IDs treated as close ANME-2d MCR donors.",
    )
    parser.add_argument(
        "--top-model-only",
        action="store_true",
        help="Only parse files containing _model_0 in the filename.",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Exit non-zero if no ligand instances are found at all.",
    )
    return parser.parse_args(argv)


def run_inventory(args: argparse.Namespace) -> int:
    input_root: Path = args.input_root
    outdir: Path = args.outdir
    close_mcr_donors = parse_close_mcr_donors(args.close_mcr_donors)

    if not input_root.is_dir():
        raise AlphaFillInventoryError(f"Input root is not a directory: {input_root}")

    cif_paths = sorted(input_root.rglob(args.pattern))
    if args.top_model_only:
        cif_paths = [path for path in cif_paths if "_model_0" in path.name]
    if not cif_paths:
        raise AlphaFillInventoryError(
            f"No CIF files found under {input_root} using pattern {args.pattern}"
        )

    all_instances: list[LigandInstance] = []
    all_json_transplants: list[AlphaFillJsonTransplant] = []
    model_summaries: list[ModelSummary] = []

    print(f"[INFO] Found CIF files: {len(cif_paths)}")
    for cif_path in cif_paths:
        print(f"[INFO] Parsing: {cif_path}")
        instances = parse_ligand_instances_from_cif(cif_path)
        target = infer_target_from_path(cif_path)
        model_index = infer_model_index(cif_path)
        json_transplants: list[AlphaFillJsonTransplant] = []
        json_found = False

        if not args.no_json:
            json_path = find_matching_json(cif_path, args.json_pattern)
            if json_path is None:
                if args.warn_missing_json:
                    print(f"[WARNING] No matching JSON metadata for {cif_path}", file=sys.stderr)
            else:
                json_found = True
                json_transplants = parse_alphafill_json(
                    json_path,
                    target=target,
                    model_index=model_index,
                    cif_path=cif_path,
                )
                all_json_transplants.extend(json_transplants)

        all_instances.extend(instances)
        model_summaries.append(
            create_model_summary(
                cif_path,
                instances,
                json_transplants,
                json_found=json_found,
                close_mcr_donors=close_mcr_donors,
            )
        )
        print(
            f"[INFO]   ligand instances: {len(instances)}; "
            f"json transplants: {len(json_transplants)}"
        )

    donor_summaries = create_donor_summaries(all_json_transplants)
    target_summaries = create_target_summaries(model_summaries, all_json_transplants)
    inventory_tsv = outdir / "alphafill_ligand_inventory.tsv"
    model_summary_tsv = outdir / "alphafill_model_summary.tsv"
    target_summary_tsv = outdir / "alphafill_target_summary.tsv"
    json_transplants_tsv = outdir / "alphafill_json_transplants.tsv"
    donor_summary_tsv = outdir / "alphafill_donor_summary.tsv"

    write_tsv(inventory_tsv, all_instances, LigandInstance)
    write_tsv(model_summary_tsv, model_summaries, ModelSummary)
    write_tsv(target_summary_tsv, target_summaries, TargetSummary)
    write_tsv(json_transplants_tsv, all_json_transplants, AlphaFillJsonTransplant)
    write_tsv(donor_summary_tsv, donor_summaries, AlphaFillDonorSummary)

    print("[INFO] Wrote:")
    print(f"[INFO]   {inventory_tsv}")
    print(f"[INFO]   {model_summary_tsv}")
    print(f"[INFO]   {target_summary_tsv}")
    print(f"[INFO]   {json_transplants_tsv}")
    print(f"[INFO]   {donor_summary_tsv}")

    if args.fail_on_empty and not all_instances:
        raise AlphaFillInventoryError("No ligand instances detected in any CIF.")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_inventory(args)
    except AlphaFillInventoryError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
