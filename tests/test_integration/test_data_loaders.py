"""Integration tests for M6, `graphtm.data` dataset loaders.

Covers:
  - `scaffold_split` correctness on a 100-SMILES synthetic set.
  - `load_kazius_toxicophores` returns >=8 alerts with parseable SMARTS.
  - `load_tdc_ames` smoke test: succeeds when TDC available, raises a
    descriptive `MissingDependency` otherwise.

These tests deliberately have minimal dependencies on M1 (the encoder).
Where M1 is unavailable, they exercise the loader path with `encode=False`
so that the parallel build is not blocked.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("rdkit", reason="RDKit is required for M6 data loaders.")


# -------------------------------------------------------------------- helpers


def _hundred_smiles() -> list[str]:
    """Build exactly 100 RDKit-valid SMILES covering many scaffold families.

    The mix is deliberately heterogeneous (chains, mono- and poly-cyclics,
    common heteroaromatics) so that scaffold-based bucketing produces a
    realistic non-degenerate split (i.e. no single 100-member group).
    """
    smiles = [
        # 1-15 acyclic / small functional groups
        "CCO", "CCN", "CCC", "CCCl", "CCBr",
        "CCS", "CCF", "CCI", "CCC#N", "CCC=O",
        "CC(=O)O", "CC(=O)N", "CC(=O)Cl", "CC(=O)OC", "CC(=O)NC",
        # 16-30 monocyclic aromatics
        "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cccnc1",
        "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
        "Cc1ccccc1", "Cc1ccncc1", "Cc1ccccc1C", "Cc1ccc(C)cc1", "Cc1ccc(N)cc1",
        # 31-40 fused aromatics
        "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1ccc2sccc2c1",
        "c1ccc2ncccc2c1", "c1ccc2occc2c1",
        "c1cnc2ncncc2n1", "c1cnc2nccnc2n1", "c1cnc2nccc(N)c2n1",
        "c1cncnc1", "c1cnncn1",
        # 41-55 aliphatic monocycles
        "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "C1CCCCCC1",
        "CC1CCCCC1", "CC1CCCC1", "CC1CCC1", "CC1CC1", "CC1=CCCCC1",
        "O=C1CCCCC1", "O=C1CCCC1", "O=C1CCC1", "O=C1CC1", "O=C(C)C",
        # 56-70 aliphatic N-heterocycles
        "N1CC1", "N1CCC1", "N1CCCC1", "N1CCCCC1", "N1CCCCCC1",
        "CN1CCCCC1", "CN1CCCC1", "CN1CCC1", "CC1NCC1", "CC1NCCC1",
        "O1CC1", "O1CCC1", "O1CCCC1", "O1CCCCC1", "O1CCCCCC1",
        # 71-85 sulfur rings + tertiary chains
        "S1CC1", "S1CCC1", "S1CCCC1", "S1CCCCC1", "S1CCCCCC1",
        "CC(C)(C)O", "CC(C)(C)N", "CC(C)(C)C", "CC(C)(C)Cl", "CC(C)(C)F",
        "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC",
        # 86-100 long heteroatom chains + para-substituted benzene
        "NCCCCC", "NCCCCCC", "NCCCCCCC", "NCCCCCCCC", "NCCCCCCCCC",
        "OCCCCC", "OCCCCCC", "OCCCCCCC", "OCCCCCCCC", "OCCCCCCCCC",
        "c1ccc(C=C)cc1", "c1ccc(C=O)cc1", "c1ccc(C#N)cc1", "c1ccc(CO)cc1", "c1ccc(CN)cc1",
    ]
    assert len(smiles) == 100, f"expected 100 SMILES, got {len(smiles)}"
    return smiles


# ----------------------------------------------------- scaffold_split tests


def test_scaffold_split_100_smiles_disjoint_and_exhaustive():
    """Disjoint, exhaustive, covers all 100 indices."""
    from graphtm.data import scaffold_split

    smiles = _hundred_smiles()
    tr, va, te = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=42)

    assert tr.dtype == np.int64
    assert va.dtype == np.int64
    assert te.dtype == np.int64

    # Sizes sum to 100.
    assert len(tr) + len(va) + len(te) == 100

    # Disjoint.
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0

    # Union covers exactly {0..99}.
    union = np.union1d(np.union1d(tr, va), te)
    assert np.array_equal(union, np.arange(100, dtype=np.int64))


def test_scaffold_split_deterministic_per_seed():
    """Same seed → same split; different seed → different split."""
    from graphtm.data import scaffold_split

    smiles = _hundred_smiles()
    a = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=42)
    b = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)

    c = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=7)
    # Atleast one of the three splits must differ for `seed` to be meaningful.
    differs = any(not np.array_equal(x, y) for x, y in zip(a, c))
    assert differs, "scaffold_split: seed has no effect on the partitioning"


def test_scaffold_split_train_is_largest_split():
    """Convention: train gets the largest scaffolds (most data), train > test."""
    from graphtm.data import scaffold_split

    smiles = _hundred_smiles()
    tr, va, te = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=42)
    assert len(tr) >= len(va) and len(tr) >= len(te)


def test_scaffold_split_empty_input():
    """Edge case: empty SMILES list → three empty arrays."""
    from graphtm.data import scaffold_split

    tr, va, te = scaffold_split([], frac=(0.8, 0.1, 0.1), seed=42)
    assert len(tr) == 0 and len(va) == 0 and len(te) == 0


def test_scaffold_split_fractions_must_sum_to_one():
    """Bad frac argument raises ValueError."""
    from graphtm.data import scaffold_split

    with pytest.raises(ValueError):
        scaffold_split(["CCO"], frac=(0.5, 0.2, 0.2), seed=0)


def test_scaffold_split_handles_unparseable_smiles():
    """Unparseable SMILES fall into the 'no-scaffold' bucket (key '')
    rather than crashing, MoleculeNet-aligned behaviour."""
    from graphtm.data import scaffold_split

    smiles = ["CCO", "NOT-A-SMILES", "CCN", "@@@invalid"]
    tr, va, te = scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=0)
    union = sorted(np.concatenate([tr, va, te]).tolist())
    assert union == [0, 1, 2, 3]


# ----------------------------------------------------- kazius loader tests


def test_load_kazius_toxicophores_returns_at_least_eight():
    """Hard-rule from M6 spec: >= 8 toxicophores."""
    from graphtm.data import load_kazius_toxicophores

    tox = load_kazius_toxicophores()
    assert len(tox) >= 8


def test_load_kazius_toxicophores_smarts_all_rdkit_parseable():
    """Every SMARTS round-trips through `Chem.MolFromSmarts`, no None."""
    from rdkit import Chem

    from graphtm.data import load_kazius_toxicophores

    tox = load_kazius_toxicophores()
    for t in tox:
        patt = Chem.MolFromSmarts(t.smarts)
        assert patt is not None, (
            f"Toxicophore {t.name!r} (kazius_id={t.kazius_id}) has invalid "
            f"SMARTS {t.smarts!r}"
        )
        # Sanity: the SMARTS must encode at least one atom.
        assert patt.GetNumAtoms() >= 1


def test_load_kazius_toxicophores_dataclass_fields():
    """Returned items are `Toxicophore` instances with the M6 contract fields."""
    from graphtm.data import Toxicophore, load_kazius_toxicophores

    tox = load_kazius_toxicophores()
    assert all(isinstance(t, Toxicophore) for t in tox)
    for t in tox:
        assert isinstance(t.name, str) and t.name
        assert isinstance(t.smarts, str) and t.smarts
        assert isinstance(t.kazius_id, int) and t.kazius_id > 0


def test_load_kazius_toxicophores_has_core_set():
    """The eight 'core' (always-representable) toxicophores from the prior
    project must all be present and tier='core' so recourse evaluation
    can filter on `t.tier == "core"`."""
    from graphtm.data import load_kazius_toxicophores

    expected_core_names = {
        "aromatic_nitro",
        "polycyclic_aromatic",
        "halogenated_aromatic",
        "halogen_rich",
        "alpha_beta_unsat_carbonyl",   # Michael acceptor
        "large_hydrophobic_w_heteroatom",
        "aromatic_amine",
        "pure_hydrocarbon",
    }
    tox = load_kazius_toxicophores()
    by_name = {t.name: t for t in tox}
    missing = expected_core_names - set(by_name)
    assert not missing, f"Missing core toxicophores: {missing}"
    for nm in expected_core_names:
        assert by_name[nm].tier == "core", (
            f"Toxicophore {nm!r} should be tier='core'; got {by_name[nm].tier!r}"
        )


def test_kazius_toxicophore_dataclass_is_frozen():
    """`Toxicophore` is frozen so callers cannot accidentally mutate the
    shared alert vocabulary."""
    from dataclasses import FrozenInstanceError

    from graphtm.data import Toxicophore

    t = Toxicophore(name="x", smarts="[C]", kazius_id=1)
    with pytest.raises(FrozenInstanceError):
        t.smarts = "[N]"  # type: ignore[misc]


# ----------------------------------------------------- AMES loader tests


def _tdc_importable() -> bool:
    try:
        import tdc  # noqa: F401
        return True
    except Exception:
        return False


def _ames_cache_exists() -> bool:
    """Pre-existing CSV cache means the loader can run offline-mode without TDC."""
    candidate = Path(os.path.expanduser("~/.graphtm_cache/tdc_ames.csv"))
    return candidate.exists() and candidate.stat().st_size > 0


@pytest.mark.skipif(
    not (_tdc_importable() or _ames_cache_exists()),
    reason=(
        "TDC not installed and no cached AMES CSV; "
        "the smoke test verifies the MissingDependency error path separately."
    ),
)
def test_load_tdc_ames_smoke():
    """When TDC (or a cached CSV) is available, the loader produces >=6500
    graphs and a well-formed metadata dict."""
    from graphtm.data import load_tdc_ames

    # `encode=False` keeps the test independent of M1 (encode_graph).
    graphs, labels, metadata = load_tdc_ames(
        split="scaffold", seed=42, encode=False
    )

    assert isinstance(graphs, list)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert len(graphs) == len(labels)
    # Hansen + Kazius union: canonical TDC source size 7,255; the runtime
    # encoder typically returns 7,278 post-sanitisation. Allow some headroom.
    assert len(graphs) >= 6500, f"Only {len(graphs)} graphs survived"

    assert metadata["n_kept"] == len(graphs)
    assert set(metadata["class_balance"]) == {0, 1}
    assert metadata["class_balance"][0] + metadata["class_balance"][1] == len(graphs)

    si = metadata["split_indices"]
    assert set(si) == {"train", "valid", "test"}
    n_tr, n_va, n_te = (len(si["train"]), len(si["valid"]), len(si["test"]))
    assert n_tr + n_va + n_te == len(graphs)
    # Train is the dominant split.
    assert n_tr >= n_va and n_tr >= n_te


def test_load_tdc_ames_missing_dependency_path(tmp_path, monkeypatch):
    """Drive the cache into an isolated tmpdir with no TDC and no CSV; the
    loader must raise `MissingDependency` with an actionable message."""
    from graphtm.data import MissingDependency, load_tdc_ames

    # Block the `tdc` package even if installed: shadow it with a sentinel
    # that raises on import.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tdc" or name.startswith("tdc."):
            raise ImportError("blocked by test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Force the loader to use an empty cache dir so no CSV is present.
    empty_cache = tmp_path / "no_cache"

    # Also block URL retrieval so the URL-mirror fallback errors out.
    import urllib.request

    def _block_urlretrieve(*a, **kw):
        raise RuntimeError("network blocked by test")

    monkeypatch.setattr(urllib.request, "urlretrieve", _block_urlretrieve)

    with pytest.raises(MissingDependency):
        load_tdc_ames(cache_dir=str(empty_cache), encode=False)


# ----------------------------------------------------- Tox21 loader tests


@pytest.mark.skipif(
    not _tdc_importable(),
    reason="TDC not installed; tox21 smoke deferred to integration phase.",
)
def test_load_tox21_smoke():
    """Tox21 NR-AhR loader runs end-to-end when TDC is available."""
    from graphtm.data import load_tox21

    graphs, labels, metadata = load_tox21(
        task="NR-AhR", split="scaffold", seed=42, encode=False
    )
    assert isinstance(graphs, list)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert len(graphs) == len(labels)
    assert len(graphs) > 0
    assert metadata["task"] == "NR-AhR"


def test_load_tox21_rejects_unknown_task():
    """Invalid task name raises a clear ValueError before any I/O."""
    from graphtm.data import load_tox21

    with pytest.raises(ValueError):
        load_tox21(task="NR-NotARealTask")


def test_tox21_tasks_constant():
    """The exported task list contains the 12 canonical NR/SR endpoints."""
    from graphtm.data import TOX21_TASKS

    assert len(TOX21_TASKS) == 12
    assert "NR-AhR" in TOX21_TASKS


# ----------------------------------------------------- package surface


def test_data_package_exports():
    """All M6 public symbols are reachable from `graphtm.data`."""
    import graphtm.data as gd

    for name in (
        "MissingDependency",
        "Toxicophore",
        "TOX21_TASKS",
        "load_kazius_toxicophores",
        "load_tdc_ames",
        "load_tox21",
        "scaffold_split",
    ):
        assert hasattr(gd, name), f"graphtm.data missing public symbol {name!r}"
