from __future__ import annotations

import csv
from pathlib import Path

import alphafill_ligand_inventory as inv


ATOM_SITE_HEADER = """\
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
"""


def write_cif(
    path: Path,
    comp_id: str = "F43",
    chem_name: str = '"factor 430"',
    target_dir: str | None = "mcr_core",
    atom_rows: list[str] | None = None,
) -> Path:
    if target_dir:
        path = path / target_dir / "sample_model_0_alphafill.cif"
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_rows = atom_rows or [
        "ATOM C CA ALA A 1 1 A 1",
        f"HETATM C C1 {comp_id} B 2 . B 101",
        f"HETATM N N1 {comp_id} B 2 . B 101",
    ]
    content = f"""\
data_sample
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer protein
2 non-polymer {chem_name}
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
loop_
_chem_comp.id
_chem_comp.name
ALA alanine
{comp_id} {chem_name}
#
{ATOM_SITE_HEADER}{chr(10).join(atom_rows)}
#
"""
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_parse_minimal_alphafill_cif_extracts_ligand_instance(tmp_path: Path) -> None:
    cif = write_cif(tmp_path)

    instances = inv.parse_ligand_instances_from_cif(cif)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.target == "MCR"
    assert instance.model_index == "0"
    assert instance.ligand_comp_id == "F43"
    assert instance.ligand_name == "factor 430"
    assert instance.atom_count == 2
    assert instance.element_counts == "C:1,N:1"
    assert instance.biological_class == "F430"
    assert instance.expected_for_target == "yes"
    assert instance.inspection_priority == "high"


def test_classifies_project_relevant_ligands() -> None:
    assert inv.classify_ligand("F43", "") == "F430"
    assert inv.classify_ligand("COM", "") == "CoM"
    assert inv.classify_ligand("TP7", "") == "CoB"
    assert inv.classify_ligand("FMN", "") == "FMN"
    assert inv.classify_ligand("FAD", "") == "FAD"
    assert inv.classify_ligand("NA", "") == "ion"
    assert inv.classify_ligand("ZN", "") == "metal"
    assert inv.classify_ligand("ETX", "") == "buffer_or_additive"


def test_infers_target_and_model_index_from_path() -> None:
    assert inv.infer_target_from_path(Path("outputs/mcr_core/job_model_0_alphafill.cif")) == "MCR"
    assert inv.infer_target_from_path(Path("outputs/mer/job_model_0_alphafill.cif")) == "Mer"
    assert inv.infer_target_from_path(Path("outputs/unknown/job_model_0_alphafill.cif")) == "unknown"
    assert inv.infer_model_index(Path("outputs/mer/job_model_3_alphafill.cif")) == "3"
    assert inv.infer_model_index(Path("outputs/mer/job_alphafill.cif")) == ""


def test_cli_writes_three_summary_tables(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    outdir = tmp_path / "out"
    write_cif(input_root)

    status = inv.main(["--input-root", str(input_root), "--outdir", str(outdir)])

    assert status == 0
    inventory = read_tsv(outdir / "alphafill_ligand_inventory.tsv")
    model_summary = read_tsv(outdir / "alphafill_model_summary.tsv")
    target_summary = read_tsv(outdir / "alphafill_target_summary.tsv")
    assert inventory[0]["ligand_comp_id"] == "F43"
    assert model_summary[0]["f43_count"] == "1"
    assert model_summary[0]["preliminary_verdict"] == "inspect_high_priority"
    assert target_summary[0]["target"] == "MCR"
    assert target_summary[0]["preliminary_target_verdict"] == "robust_high_priority_transfer"


def test_top_model_only_filters_nonzero_models(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    outdir = tmp_path / "out"
    write_cif(input_root / "mcr_core" / "sample_model_0_alphafill.cif", target_dir=None)
    write_cif(input_root / "mcr_core" / "sample_model_1_alphafill.cif", comp_id="COM", target_dir=None)

    status = inv.main(
        [
            "--input-root",
            str(input_root),
            "--outdir",
            str(outdir),
            "--top-model-only",
        ]
    )

    assert status == 0
    model_summary = read_tsv(outdir / "alphafill_model_summary.tsv")
    assert len(model_summary) == 1
    assert model_summary[0]["model_index"] == "0"
    assert model_summary[0]["ligand_comp_ids"] == "F43:1"


def test_cli_reports_missing_input_root_and_no_cif_matches(tmp_path: Path) -> None:
    assert inv.main(["--input-root", str(tmp_path / "missing"), "--outdir", str(tmp_path / "out")]) == 1

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert inv.main(["--input-root", str(empty_root), "--outdir", str(tmp_path / "out")]) == 1


def test_fail_on_empty_when_only_protein_atoms_are_present(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    outdir = tmp_path / "out"
    write_cif(
        input_root,
        atom_rows=[
            "ATOM C CA ALA A 1 1 A 1",
            "ATOM C CB ALA A 1 1 A 1",
        ],
    )

    status = inv.main(
        [
            "--input-root",
            str(input_root),
            "--outdir",
            str(outdir),
            "--fail-on-empty",
        ]
    )

    assert status == 1
    assert read_tsv(outdir / "alphafill_ligand_inventory.tsv") == []
    assert read_tsv(outdir / "alphafill_model_summary.tsv")[0]["preliminary_verdict"] == "no_ligands_detected"


def test_semicolon_multiline_and_quoted_names_are_tokenised(tmp_path: Path) -> None:
    cif = tmp_path / "mer" / "sample_model_0_alphafill.cif"
    cif.parent.mkdir(parents=True)
    cif.write_text(
        f"""\
data_sample
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer protein
2 non-polymer "unknown entity"
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
loop_
_chem_comp.id
_chem_comp.name
ABC "coenzyme M"
F42
;coenzyme F420
reduced form
;
#
{ATOM_SITE_HEADER}ATOM C CA ALA A 1 1 A 1
HETATM C C1 ABC B 2 . B 101
HETATM C C1 F42 C 2 . C 102
#
""",
        encoding="utf-8",
        newline="\n",
    )

    instances = inv.parse_ligand_instances_from_cif(cif)

    classes_by_comp = {instance.ligand_comp_id: instance.biological_class for instance in instances}
    names_by_comp = {instance.ligand_comp_id: instance.ligand_name for instance in instances}
    assert classes_by_comp == {"ABC": "CoM", "F42": "F420"}
    assert names_by_comp["ABC"] == "coenzyme M"
    assert names_by_comp["F42"] == "coenzyme F420\nreduced form"
