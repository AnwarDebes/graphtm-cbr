"""Kazius 2005 toxicophores, structural-alert vocabulary for Ames recourse.

Why this exists
---------------
ICH M7(R2) §6.1 mandates a *structural-alert + statistical* dual-QSAR system
for impurity mutagenicity. Kazius et al. (J Med Chem 48:312-320, 2005)
distilled the operational ~29 substructural alerts ("toxicophores") that have
become the de-facto alert vocabulary used by Derek, Sarah, ToxAlerts, and
ICH-M7 expert-rule legs.

The HGTM-CBR recourse module needs this vocabulary to:
  (i) score whether a counterfactual edit *removed* a Kazius alert, and
  (ii) restrict the held-out evaluation to molecules whose mutagenicity comes
       from at least one of the canonical 29 alerts (per the paper's
       §7.5-aligned recourse story).

This module loads the alert list from the bundled YAML
(`kazius_toxicophores.yaml`) and validates that every SMARTS is RDKit-parsable
at import time. SMARTS are *not* re-derived here, they are vendored
verbatim from the publication / open-source ToxAlerts mirror, with explicit
attribution in the YAML `notes` field.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from graphtm.data import MissingDependency


# Path to the bundled YAML, kept next to this module so the package is
# distributable without extra data-files plumbing.
_KAZIUS_YAML = Path(__file__).resolve().parent / "kazius_toxicophores.yaml"


@dataclass(frozen=True)
class Toxicophore:
    """A single Kazius 2005 structural alert.

    Attributes
    ----------
    name
        Short snake_case label (e.g. ``"aromatic_nitro"``) suitable for use
        as a recourse-report tag or experiment-config key.
    smarts
        RDKit-parsable SMARTS pattern. Pre-validated by `_load_yaml`.
    kazius_id
        1-indexed alert number in Kazius J Med Chem 48:312 (2005), Table 1.
    tier
        ``"core"`` for the eight unambiguously-representable alerts that the
        prior axiom-coi unified project used for held-out recourse evaluation;
        ``"extended"`` for the remaining alerts.
    notes
        Optional human-readable commentary.
    """

    name: str
    smarts: str
    kazius_id: int
    tier: str = "extended"
    notes: str = ""


def _load_yaml() -> List[Toxicophore]:
    """Parse and validate the bundled `kazius_toxicophores.yaml`."""
    try:
        import yaml
    except ImportError as exc:   # pragma: no cover - env hint
        raise MissingDependency(
            "PyYAML is required to load Kazius toxicophores. "
            "Install via `pip install pyyaml`."
        ) from exc
    try:
        from rdkit import Chem
    except ImportError as exc:   # pragma: no cover - env hint
        raise MissingDependency(
            "rdkit is required to validate Kazius toxicophore SMARTS. "
            "Install via `pip install rdkit` or `conda install -c conda-forge rdkit`."
        ) from exc

    if not _KAZIUS_YAML.exists():
        raise FileNotFoundError(
            f"Kazius alert table not bundled, expected at {_KAZIUS_YAML}."
        )

    with _KAZIUS_YAML.open("r") as f:
        cfg = yaml.safe_load(f)

    if "toxicophores" not in cfg or not isinstance(cfg["toxicophores"], list):
        raise ValueError(
            f"{_KAZIUS_YAML} must contain a top-level `toxicophores:` list."
        )

    out: List[Toxicophore] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for entry in cfg["toxicophores"]:
        try:
            kid = int(entry["kazius_id"])
            name = str(entry["name"])
            smarts = str(entry["smarts"])
        except KeyError as exc:
            raise ValueError(
                f"Malformed toxicophore entry in {_KAZIUS_YAML}: missing {exc}"
            ) from exc
        tier = str(entry.get("tier", "extended"))
        notes = str(entry.get("notes", ""))

        if kid in seen_ids:
            raise ValueError(f"Duplicate kazius_id={kid} in {_KAZIUS_YAML}")
        if name in seen_names:
            raise ValueError(f"Duplicate toxicophore name={name!r} in {_KAZIUS_YAML}")
        if Chem.MolFromSmarts(smarts) is None:
            raise ValueError(
                f"Toxicophore {name!r} (kazius_id={kid}) has unparseable SMARTS "
                f"{smarts!r}; this is a data-file bug, fix the YAML."
            )

        seen_ids.add(kid)
        seen_names.add(name)
        out.append(
            Toxicophore(
                name=name, smarts=smarts, kazius_id=kid, tier=tier, notes=notes
            )
        )

    if len(out) < 8:
        raise ValueError(
            f"Kazius alert table has only {len(out)} entries, need >= 8 "
            f"for the held-out recourse evaluation. Check {_KAZIUS_YAML}."
        )

    return out


def load_kazius_toxicophores() -> List[Toxicophore]:
    """Return the 29 Kazius 2005 toxicophores as a list of `Toxicophore`.

    Every returned alert is guaranteed to have an RDKit-parsable SMARTS
    (validated at load time). The list order matches the YAML file order,
    which in turn matches the 1-indexed table in Kazius J Med Chem 48:312
    (2005). Use the `tier == "core"` filter for the eight always-representable
    alerts used in the prior project's recourse held-out subset.

    Returns
    -------
    List[Toxicophore]
        The full toxicophore vocabulary. Frozen dataclasses (immutable).
    """
    return _load_yaml()


__all__ = ["Toxicophore", "load_kazius_toxicophores"]
