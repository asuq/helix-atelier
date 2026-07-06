from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import af3_input_prep as prep


def options(**overrides: object) -> SimpleNamespace:
    values = {
        "seeds": None,
        "fasta_root": None,
        "allow_noncanonical_aa": False,
        "preserve_ligand_case": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_fasta(path: Path, sequence: str = "MKT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">seq\n{sequence}\n", encoding="utf-8")
    return path


def write_tsv(path: Path, rows: list[str], header: str | None = None) -> Path:
    header = header or "entity_type\tid\tsource\tstoichiometry\tchain_ids\tdescription"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def minimal_tsv(tmp_path: Path, name: str = "job.tsv", sequence: str = "MKT") -> Path:
    write_fasta(tmp_path / "faa" / "a.faa", sequence)
    return write_tsv(tmp_path / name, ["protein\tA\tfaa/a.faa\t\t\tAlpha"])


def test_read_single_record_fasta_normalises_wrapped_lowercase(tmp_path: Path) -> None:
    fasta = tmp_path / "protein.faa"
    fasta.write_text(">p\nmk\n t\n", encoding="utf-8")

    assert prep.read_single_record_fasta(fasta, allow_noncanonical=False) == "MKT"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "no records"),
        ("MKT\n", "no records"),
        (">a\n", "empty sequence"),
        (">a\nMKT\n>b\nMKT\n", "expected exactly one"),
        (">a\nMK1\n", "invalid protein sequence"),
        (">a\nMKX\n", "invalid protein sequence"),
    ],
)
def test_read_single_record_fasta_rejects_invalid_files(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    fasta = tmp_path / "protein.faa"
    fasta.write_text(content, encoding="utf-8")

    with pytest.raises(prep.Af3InputPrepError, match=message):
        prep.read_single_record_fasta(fasta, allow_noncanonical=False)


def test_read_single_record_fasta_allows_noncanonical_when_requested(tmp_path: Path) -> None:
    fasta = tmp_path / "protein.faa"
    fasta.write_text(">p\nMKXJUO\n", encoding="utf-8")

    assert prep.read_single_record_fasta(fasta, allow_noncanonical=True) == "MKXJUO"


def test_read_tsv_requires_columns(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path / "bad.tsv", ["protein\tA"], header="entity_type\tid")

    with pytest.raises(prep.Af3InputPrepError, match="missing required column"):
        prep.read_tsv(tsv)


def test_build_job_resolves_relative_fasta_from_tsv_parent(tmp_path: Path) -> None:
    tsv = minimal_tsv(tmp_path)

    job = prep.build_job_from_tsv(tsv, "job", "local", options())

    assert job["sequences"][0]["protein"]["sequence"] == "MKT"


def test_build_job_uses_fasta_root_override(tmp_path: Path) -> None:
    fasta_root = tmp_path / "root"
    write_fasta(fasta_root / "faa" / "a.faa", "GAS")
    tsv_dir = tmp_path / "specs"
    tsv_dir.mkdir()
    tsv = write_tsv(tsv_dir / "job.tsv", ["protein\tA\tfaa/a.faa\t\t\t"])

    job = prep.build_job_from_tsv(
        tsv,
        "job",
        "local",
        options(fasta_root=fasta_root),
    )

    assert job["sequences"][0]["protein"]["sequence"] == "GAS"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("bad\tA\tfaa/a.faa\t\t\t", "unknown entity_type"),
        ("protein\tA\tfaa/a.faa\t0\t\t", "stoichiometry must be a positive integer"),
        ("protein\tA\tfaa/a.faa\t2\tA\t", "chain_ids count"),
        ("ligand_ccd\tL\t\t\t\t", "source must not be empty"),
    ],
)
def test_tsv_validation_errors(tmp_path: Path, row: str, message: str) -> None:
    write_fasta(tmp_path / "faa" / "a.faa")
    tsv = write_tsv(tmp_path / "bad.tsv", [row])

    with pytest.raises(prep.Af3InputPrepError, match=message):
        prep.build_job_from_tsv(tsv, "job", "local", options())


def test_missing_fasta_path_is_clear(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path / "missing.tsv", ["protein\tA\tmissing.faa\t\t\t"])

    with pytest.raises(prep.Af3InputPrepError, match="Protein FASTA file not found"):
        prep.build_job_from_tsv(tsv, "job", "local", options())


def test_expand_ids_and_chain_ids() -> None:
    assert prep.expand_ids("A", 1, None) == "A"
    assert prep.expand_ids("A", 2, None) == ["A1", "A2"]
    assert prep.expand_ids("A", 2, ["A", "D"]) == ["A", "D"]


def test_duplicate_expanded_ids_are_rejected(tmp_path: Path) -> None:
    tsv = write_tsv(
        tmp_path / "dup.tsv",
        [
            "ligand_ccd\tA\tATP\t\t\t",
            "ion\tA\tMG\t\t\t",
        ],
    )

    with pytest.raises(prep.Af3InputPrepError, match="Duplicate expanded entity ID"):
        prep.build_job_from_tsv(tsv, "job", "local", options())


def test_local_json_shape_and_entities(tmp_path: Path) -> None:
    write_fasta(tmp_path / "faa" / "a.faa", "MKT")
    tsv = write_tsv(
        tmp_path / "local.tsv",
        [
            "protein\tA\tfaa/a.faa\t2\t\tAlpha",
            "ligand_ccd\tF\tF43\t2\t\tF430",
            "ligand_smiles\tL\tCCO\t\t\tEthanol",
            "ion\tM\tmg\t\t\tMagnesium",
        ],
    )

    job = prep.build_job_from_tsv(
        tsv,
        "ANME2d_MCR",
        "local",
        options(seeds="1,2"),
    )

    assert isinstance(job, dict)
    assert job["dialect"] == "alphafold3"
    assert job["version"] == 4
    assert job["modelSeeds"] == [1, 2]
    assert job["sequences"][0]["protein"]["id"] == ["A1", "A2"]
    assert job["sequences"][1]["ligand"]["ccdCodes"] == ["F43"]
    assert job["sequences"][2]["ligand"]["smiles"] == "CCO"
    assert job["sequences"][3]["ligand"]["ccdCodes"] == ["MG"]


def test_server_json_shape_and_entities(tmp_path: Path) -> None:
    write_fasta(tmp_path / "faa" / "a.faa", "MKT")
    tsv = write_tsv(
        tmp_path / "server.tsv",
        [
            "protein\tA\tfaa/a.faa\t2\t\t",
            "ligand_ccd\tF\tf43\t2\t\t",
            "ion\tM\tmg\t3\t\t",
        ],
    )

    job = prep.build_job_from_tsv(tsv, "server_job", "server", options())

    assert job["dialect"] == "alphafoldserver"
    assert job["version"] == 1
    assert job["modelSeeds"] == []
    assert job["sequences"][0] == {"proteinChain": {"sequence": "MKT", "count": 2}}
    assert job["sequences"][1] == {"ligand": {"ligand": "CCD_F43", "count": 2}}
    assert job["sequences"][2] == {"ion": {"ion": "MG", "count": 3}}


def test_server_explicit_seeds_are_strings(tmp_path: Path) -> None:
    tsv = minimal_tsv(tmp_path)

    job = prep.build_job_from_tsv(tsv, "server_job", "server", options(seeds="1,2"))

    assert job["modelSeeds"] == ["1", "2"]


def test_server_smiles_fails_clearly(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path / "smiles.tsv", ["ligand_smiles\tL\tCCO\t\t\t"])

    with pytest.raises(prep.Af3InputPrepError, match="SMILES ligand rows are not supported"):
        prep.build_job_from_tsv(tsv, "server_job", "server", options())


def test_cli_single_writes_one_json(tmp_path: Path) -> None:
    tsv = minimal_tsv(tmp_path)
    outdir = tmp_path / "out"

    status = prep.main(
        [
            "single",
            str(tsv),
            "--style",
            "local",
            "--outdir",
            str(outdir),
            "--prefix",
            "Job One",
            "--quiet",
        ]
    )

    assert status == 0
    output = outdir / "Job_One.json"
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["name"] == "Job One"


def test_cli_single_defaults_prefix_to_input_stem(tmp_path: Path) -> None:
    tsv = minimal_tsv(tmp_path, name="stem_job.tsv")
    outdir = tmp_path / "out"

    status = prep.main(
        [
            "single",
            str(tsv),
            "--style",
            "local",
            "--outdir",
            str(outdir),
            "--quiet",
        ]
    )

    assert status == 0
    output = outdir / "stem_job.json"
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["name"] == "stem_job"


def test_cli_batch_server_writes_one_list_json(tmp_path: Path) -> None:
    tsv1 = minimal_tsv(tmp_path / "one", name="one.tsv")
    tsv2 = minimal_tsv(tmp_path / "two", name="two.tsv")
    tsv3 = minimal_tsv(tmp_path / "three", name="three.tsv")
    outdir = tmp_path / "out"

    status = prep.main(
        [
            "batch",
            str(tsv1),
            str(tsv2),
            str(tsv3),
            "--style",
            "server",
            "--outdir",
            str(outdir),
            "--prefix",
            "methane",
            "--quiet",
        ]
    )

    assert status == 0
    output = outdir / "methane_batch.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 3
    assert [job["name"] for job in payload] == [
        "methane_one",
        "methane_two",
        "methane_three",
    ]


def test_cli_batch_local_writes_json_files_and_manifest(tmp_path: Path) -> None:
    tsv1 = minimal_tsv(tmp_path / "one", name="one.tsv")
    tsv2 = minimal_tsv(tmp_path / "two", name="two.tsv")
    outdir = tmp_path / "out"

    status = prep.main(
        [
            "batch",
            str(tsv1),
            str(tsv2),
            "--style",
            "local",
            "--outdir",
            str(outdir),
            "--prefix",
            "methane",
            "--quiet",
        ]
    )

    assert status == 0
    assert (outdir / "methane_one.json").is_file()
    assert (outdir / "methane_two.json").is_file()
    manifest = outdir / "methane_batch_manifest.tsv"
    assert manifest.is_file()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "job_name\tinput_tsv\toutput_json\tstyle\tstatus\tmessage" in manifest_text
    assert "methane_one" in manifest_text
    assert "\tcreated\t" in manifest_text


def test_cli_existing_output_requires_force(tmp_path: Path) -> None:
    tsv = minimal_tsv(tmp_path)
    outdir = tmp_path / "out"
    args = [
        "single",
        str(tsv),
        "--style",
        "local",
        "--outdir",
        str(outdir),
        "--prefix",
        "job",
        "--quiet",
    ]

    assert prep.main(args) == 0
    assert prep.main(args) == 2
    assert prep.main([*args, "--force"]) == 0
