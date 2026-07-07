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
import re
import shlex
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence


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

        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        for token in lexer:
            yield token

        i += 1


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


def create_model_summary(path: Path, instances: list[LigandInstance]) -> ModelSummary:
    """Create one row per CIF/model."""
    target = infer_target_from_path(path)
    model_index = infer_model_index(path)
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
    )


def create_target_summaries(model_summaries: list[ModelSummary]) -> list[TargetSummary]:
    """Create one row per target."""
    by_target: dict[str, list[ModelSummary]] = collections.defaultdict(list)
    for summary in model_summaries:
        by_target[summary.target].append(summary)

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
            )
        )

    return target_summaries


def update_counter_from_compact_counts(counter: collections.Counter[str], text: str) -> None:
    """Add A:2;B:3 style counts into a Counter."""
    for token in text.split(";"):
        if not token:
            continue
        comp_id, count_text = token.split(":", 1)
        counter[comp_id] += int(count_text)


def dataclass_fieldnames(row_type: type[object]) -> list[str]:
    """Return dataclass field names in definition order."""
    return [field.name for field in fields(row_type)]


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
    model_summaries: list[ModelSummary] = []

    print(f"[INFO] Found CIF files: {len(cif_paths)}")
    for cif_path in cif_paths:
        print(f"[INFO] Parsing: {cif_path}")
        instances = parse_ligand_instances_from_cif(cif_path)
        all_instances.extend(instances)
        model_summaries.append(create_model_summary(cif_path, instances))
        print(f"[INFO]   ligand instances: {len(instances)}")

    target_summaries = create_target_summaries(model_summaries)
    inventory_tsv = outdir / "alphafill_ligand_inventory.tsv"
    model_summary_tsv = outdir / "alphafill_model_summary.tsv"
    target_summary_tsv = outdir / "alphafill_target_summary.tsv"

    write_tsv(inventory_tsv, all_instances, LigandInstance)
    write_tsv(model_summary_tsv, model_summaries, ModelSummary)
    write_tsv(target_summary_tsv, target_summaries, TargetSummary)

    print("[INFO] Wrote:")
    print(f"[INFO]   {inventory_tsv}")
    print(f"[INFO]   {model_summary_tsv}")
    print(f"[INFO]   {target_summary_tsv}")

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
