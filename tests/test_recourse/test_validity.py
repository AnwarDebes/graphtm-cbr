"""Tests for graphtm.recourse.validity.

Coverage:
  - ``validate`` rejects malformed input.
  - ``validate`` accepts ethanol, returns Lipinski + sascore.
  - ``validate_smiles`` matches ``validate(mol_from_smiles)``.
  - Lipinski thresholds enforce Ro5 boundaries.
"""
from __future__ import annotations

import pytest

from graphtm.recourse.validity import (
    ValidityReport,
    _get_sascorer,
    validate,
    validate_smiles,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


class TestValidate:
    def test_ethanol_passes(self):
        mol = Chem.MolFromSmiles("CCO")
        rep = validate(mol)
        assert isinstance(rep, ValidityReport)
        assert rep.sanitize_ok is True
        assert rep.lipinski_ok is True
        # Ethanol passes all Lipinski thresholds easily
        assert rep.mol_weight is not None and rep.mol_weight < 60
        assert rep.hbd is not None and rep.hbd <= 1
        assert rep.hba is not None and rep.hba <= 1
        # SAscore should be very low for ethanol if sascorer is available
        if _get_sascorer() is not None:
            assert rep.sa_score is not None
            assert 1.0 <= rep.sa_score <= 5.0  # very synthesizable
            assert rep.sa_ok is True
        assert rep.overall_ok is True

    def test_none_mol_fails(self):
        rep = validate(None)
        assert rep.overall_ok is False
        assert "mol_is_None" in rep.notes

    def test_malformed_smiles_fails(self):
        # SMILES that RDKit cannot parse
        rep = validate_smiles("X@Y not a smiles")
        assert rep.overall_ok is False
        assert rep.sanitize_ok is False
        # Either invalid_smiles or unparseable_smiles in notes
        assert ("unparseable" in rep.notes) or ("invalid_smiles" in rep.notes)

    def test_empty_smiles_fails(self):
        rep = validate_smiles("")
        assert rep.overall_ok is False

    def test_validate_smiles_ethanol_matches_mol(self):
        rep_s = validate_smiles("CCO")
        rep_m = validate(Chem.MolFromSmiles("CCO"))
        assert rep_s.sanitize_ok == rep_m.sanitize_ok
        assert rep_s.lipinski_ok == rep_m.lipinski_ok
        # MW / LogP should be identical (deterministic computation)
        assert rep_s.mol_weight == pytest.approx(rep_m.mol_weight, abs=1e-3)

    def test_lipinski_rejects_heavy_molecule(self):
        """A clearly Ro5-violating molecule should fail Lipinski.

        Cholesterol has MW ~386 and LogP ~7.4 → fails the LogP threshold.
        """
        cholesterol = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
        rep = validate_smiles(cholesterol)
        # Either logp or MW must breach
        assert rep.lipinski_ok is False
        assert rep.overall_ok is False
        assert "lipinski_fail" in rep.notes

    def test_lipinski_passes_small_drug_like(self):
        """Aspirin: MW 180, LogP ~1.2, all Ro5 thresholds met."""
        rep = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert rep.lipinski_ok is True
        assert rep.sanitize_ok is True

    def test_sa_score_present_for_simple_mol(self):
        """If sascorer is installed, ethanol gets a low SAscore."""
        sa = _get_sascorer()
        if sa is None:
            pytest.skip("sascorer not available in this environment")
        rep = validate_smiles("CCO")
        assert rep.sa_score is not None
        assert rep.sa_score < 6.0
        assert rep.sa_ok is True

    def test_report_is_dataclass(self):
        from dataclasses import is_dataclass

        rep = validate_smiles("CCO")
        assert is_dataclass(rep)


class TestValidityNotes:
    def test_notes_field_documents_failure(self):
        """Failure mode should surface in the notes string."""
        rep = validate_smiles("not_a_real_smiles!!!")
        # Notes string is non-empty when failing
        assert len(rep.notes) > 0
