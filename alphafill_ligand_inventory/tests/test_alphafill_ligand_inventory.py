from __future__ import annotations

import csv
import json
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


def read_header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split("\t")


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def alphafill_json_payload(
    *,
    pdb_id: str = "9QR1",
    transplants: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "alphafill_version": "2.3.0",
        "file": "/remote/model.cif",
        "hits": [
            {
                "alignment": {"identity": 0.75, "length": 100, "coverage": 0.8},
                "global_rmsd": 0.5,
                "pdb_asym_id": "A",
                "pdb_id": pdb_id,
                "transplants": transplants
                or [
                    {
                        "compound_id": "M43",
                        "analogue_id": "F43",
                        "asym_id": "G",
                        "clash": {"clash_count": 2, "score": 0.1},
                        "local_rmsd": 0.2,
                        "pdb_asym_id": "X",
                        "pdb_auth_asym_id": "A",
                        "pdb_auth_seq_id": "1001",
                    }
                ],
            }
        ],
    }


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
    assert inv.classify_ligand("MFN", "") == "MFR"
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
    assert model_summary[0]["json_found"] == "no"
    assert model_summary[0]["preliminary_verdict"] == "inspect_high_priority"
    assert target_summary[0]["target"] == "MCR"
    assert target_summary[0]["preliminary_target_verdict"] == "robust_high_priority_transfer"
    assert (outdir / "alphafill_json_transplants.tsv").is_file()
    assert (outdir / "alphafill_donor_summary.tsv").is_file()
    assert read_header(outdir / "alphafill_ligand_inventory.tsv")[-1] == "cif_path"
    assert read_header(outdir / "alphafill_model_summary.tsv")[-1] == "cif_path"


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


def test_unquoted_apostrophes_are_not_treated_as_quotes(tmp_path: Path) -> None:
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
2 non-polymer ADENOSINE-5'-DIPHOSPHATE
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
ADP ADENOSINE-5'-DIPHOSPHATE
#
{ATOM_SITE_HEADER}ATOM C CA ALA A 1 1 A 1
HETATM C C5' ADP B 2 . B 101
HETATM O O3' ADP B 2 . B 101
#
""",
        encoding="utf-8",
        newline="\n",
    )

    instances = inv.parse_ligand_instances_from_cif(cif)

    assert len(instances) == 1
    assert instances[0].ligand_comp_id == "ADP"
    assert instances[0].ligand_name == "ADENOSINE-5'-DIPHOSPHATE"
    assert instances[0].atom_count == 2


def test_find_matching_json_exact_and_fallback(tmp_path: Path) -> None:
    cif = tmp_path / "sample_model_0_alphafill.cif"
    cif.write_text("data_sample\n", encoding="utf-8")
    exact = write_json(tmp_path / "sample_model_0_alphafill.json", {})
    fallback = write_json(tmp_path / "sample_model_0_extra_alphafill.json", {})

    assert inv.find_matching_json(cif, "*_alphafill.json") == exact
    exact.unlink()
    assert inv.find_matching_json(cif, "*_alphafill.json") == fallback


def test_parse_alphafill_json_extracts_transplant_context(tmp_path: Path) -> None:
    cif = tmp_path / "mcr_core" / "sample_model_0_alphafill.cif"
    json_path = write_json(cif.with_suffix(".json"), alphafill_json_payload())

    rows = inv.parse_alphafill_json(json_path, target="MCR", model_index="0", cif_path=cif)

    assert len(rows) == 1
    row = rows[0]
    assert row.ligand_comp_id == "F43"
    assert row.donor_compound_id == "M43"
    assert row.biological_class == "F430"
    assert row.inspection_priority == "high"
    assert row.donor_pdb_id == "9QR1"
    assert row.donor_asym_id == "X"
    assert row.donor_auth_seq_id == "1001"
    assert row.recipient_asym_id == "G"
    assert row.local_rmsd == "0.2"
    assert row.global_rmsd == "0.5"
    assert row.clash_count == "2"
    assert row.sequence_identity == "0.75"
    assert row.alignment_length == "100"
    assert row.coverage == "0.8"
    assert row.json_record_path == "hits[0].transplants[0]"
    assert row.raw_json_path == "/remote/model.cif"


def test_recursive_json_parser_carries_parent_donor_context(tmp_path: Path) -> None:
    json_path = write_json(
        tmp_path / "sample_model_0_alphafill.json",
        {
            "outer": {
                "description": "donor structure 1Z69 chain A",
                "records": [
                    {
                        "ligand_id": "F42",
                        "local_rmsd": 0.3,
                        "pdb_auth_seq_id": "401",
                    }
                ],
            }
        },
    )

    rows = inv.parse_alphafill_json(json_path, target="Mer", model_index="0")

    assert len(rows) == 1
    assert rows[0].donor_pdb_id == "1Z69"
    assert rows[0].ligand_comp_id == "F42"
    assert rows[0].inspection_priority == "high"


def test_cli_json_outputs_close_mcr_flags_and_path_column_order(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    outdir = tmp_path / "out"
    cif = write_cif(input_root)
    write_json(cif.with_suffix(".json"), alphafill_json_payload())

    status = inv.main(["--input-root", str(input_root), "--outdir", str(outdir)])

    assert status == 0
    transplants = read_tsv(outdir / "alphafill_json_transplants.tsv")
    donor_summary = read_tsv(outdir / "alphafill_donor_summary.tsv")
    model_summary = read_tsv(outdir / "alphafill_model_summary.tsv")
    target_summary = read_tsv(outdir / "alphafill_target_summary.tsv")
    assert transplants[0]["ligand_comp_id"] == "F43"
    assert transplants[0]["donor_pdb_id"] == "9QR1"
    assert donor_summary[0]["donor_pdb_id"] == "9QR1"
    assert donor_summary[0]["n_transplants"] == "1"
    assert donor_summary[0]["n_clashing"] == "1"
    assert model_summary[0]["json_found"] == "yes"
    assert model_summary[0]["mcr_f43_donor_pdb_ids"] == "9QR1"
    assert model_summary[0]["has_close_anme2d_mcr_donor"] == "yes"
    assert target_summary[0]["models_with_json"] == "1"
    assert target_summary[0]["models_with_close_anme2d_mcr_donor"] == "1"
    assert target_summary[0]["target_donor_interpretation"] == (
        "close_template_dependent_or_needs_exclusion_control"
    )
    assert read_header(outdir / "alphafill_json_transplants.tsv")[-3:] == [
        "cif_path",
        "json_path",
        "raw_json_path",
    ]
    assert read_header(outdir / "alphafill_model_summary.tsv")[-1] == "cif_path"


def test_no_json_and_warn_missing_json_are_nonfatal(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "input"
    write_cif(input_root)

    warn_outdir = tmp_path / "warn_out"
    assert inv.main(
        [
            "--input-root",
            str(input_root),
            "--outdir",
            str(warn_outdir),
            "--warn-missing-json",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "No matching JSON metadata" in captured.err
    assert read_tsv(warn_outdir / "alphafill_json_transplants.tsv") == []

    no_json_outdir = tmp_path / "no_json_out"
    assert inv.main(
        [
            "--input-root",
            str(input_root),
            "--outdir",
            str(no_json_outdir),
            "--no-json",
        ]
    ) == 0
    assert read_tsv(no_json_outdir / "alphafill_donor_summary.tsv") == []


def test_donor_summary_rmsd_median_and_clash_counts(tmp_path: Path) -> None:
    json_path = write_json(
        tmp_path / "sample_model_0_alphafill.json",
        alphafill_json_payload(
            pdb_id="1E6Y",
            transplants=[
                {
                    "compound_id": "F43",
                    "analogue_id": "F43",
                    "asym_id": "A",
                    "local_rmsd": 0.3,
                    "clash": {"clash_count": 0},
                    "pdb_asym_id": "X",
                    "pdb_auth_asym_id": "A",
                    "pdb_auth_seq_id": "1",
                },
                {
                    "compound_id": "F43",
                    "analogue_id": "F43",
                    "asym_id": "B",
                    "local_rmsd": 0.1,
                    "clash": {"clash_count": 2},
                    "pdb_asym_id": "Y",
                    "pdb_auth_asym_id": "A",
                    "pdb_auth_seq_id": "2",
                },
                {
                    "compound_id": "F43",
                    "analogue_id": "F43",
                    "asym_id": "C",
                    "local_rmsd": 0.2,
                    "clash": {"clash_count": 1},
                    "pdb_asym_id": "Z",
                    "pdb_auth_asym_id": "A",
                    "pdb_auth_seq_id": "3",
                },
            ],
        ),
    )
    rows = inv.parse_alphafill_json(json_path, target="MCR", model_index="0")

    summary = inv.create_donor_summaries(rows)[0]

    assert summary.min_local_rmsd == "0.1"
    assert summary.median_local_rmsd == "0.2"
    assert summary.max_local_rmsd == "0.3"
    assert summary.n_clashing == 2
    assert summary.inspection_priority == "high"
