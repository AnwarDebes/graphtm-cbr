"""RDKit-backed chemistry-validity filter for HGTM-CBR recourse.

Implements `research/06_graph_counterfactuals.md` §4.4, a three-stage
chemistry sanity pipeline applied to every committed counterfactual edit:

  1. **RDKit sanitization**, `Chem.SanitizeMol(mol, catchErrors=True)`
     catches valence / kekulization / radical errors. Reject on any
     failure flag.
  2. **Lipinski Rule-of-Five**, Lipinski, Lombardo, Dominy, Feeney 1997
     (DOI: 10.1016/S0169-409X(96)00423-1). The four classic thresholds:
     MW < 500, LogP < 5, HBA < 10, HBD < 5. Reject if any breach.
  3. **Synthetic Accessibility Score (SAscore)**, Ertl & Schuffenhauer
     2009 (DOI: 10.1186/1758-2946-1-8). Threshold ≤ 6 per common practice.
     I import the RDKit-Contrib `sascorer` if available; otherwise the
     ValidityReport still returns sanitize/Lipinski flags and marks the
     SA leg as ``None``.

This module is CPU-only and stateless (modulo the lazy ``sascorer``
import); see the corresponding test for behavioural specs.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, RDConfig
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


# Lazy SAscore import, sascorer is in RDKit's Contrib, not the main API
_SASCORER = None
_SASCORER_IMPORTED = False


def _get_sascorer():
    """Return the imported sascorer module, or ``None`` if unavailable.

    Memoized; the contrib path is appended to ``sys.path`` only once.
    """
    global _SASCORER, _SASCORER_IMPORTED
    if _SASCORER_IMPORTED:
        return _SASCORER
    _SASCORER_IMPORTED = True
    if not _HAVE_RDKIT:
        return None
    try:
        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore[import-not-found]
        _SASCORER = sascorer
    except Exception:
        _SASCORER = None
    return _SASCORER


# Lipinski Ro5 thresholds, strict ``<`` per Lipinski 1997 §2
_LIPINSKI_MW_MAX = 500.0
_LIPINSKI_LOGP_MAX = 5.0
_LIPINSKI_HBA_MAX = 10
_LIPINSKI_HBD_MAX = 5
_SA_MAX = 6.0


@dataclass
class ValidityReport:
    """Pass/fail flags + raw numbers for one molecule.

    overall_ok = sanitize_ok AND lipinski_ok AND sa_ok.

    sa_score is None when sascorer is unavailable; in that case ``sa_ok``
    defaults to ``True`` so the contrib-less environment doesn't block all
    candidates.
    """
    sanitize_ok: bool
    lipinski_ok: bool
    sa_score: Optional[float]
    sa_ok: bool
    overall_ok: bool
    notes: str = ""
    # Detailed numbers, useful for the recourse report
    mol_weight: Optional[float] = None
    logp: Optional[float] = None
    hba: Optional[int] = None
    hbd: Optional[int] = None


def _sanitize(mol: Any) -> tuple[bool, str]:
    """Attempt to sanitize ``mol`` in place; return (ok, message)."""
    try:
        # 0 means "no problems"; otherwise it returns a bitmask of failures
        flag = Chem.SanitizeMol(mol, catchErrors=True)
        if int(flag) != 0:
            return False, f"sanitize_failed (flag={int(flag)})"
        return True, ""
    except Exception as exc:  # pragma: no cover
        return False, f"sanitize_exception: {exc}"


def _check_lipinski(mol: Any) -> tuple[bool, dict, str]:
    """Compute the Ro5 metrics and return (ok, metrics_dict, message).

    Strict `<`-comparisons match Lipinski 1997 §2 ("not more than").
    """
    try:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        hbd = int(Lipinski.NumHDonors(mol))
    except Exception as exc:  # pragma: no cover
        return False, {}, f"lipinski_exception: {exc}"
    breaches = []
    if mw >= _LIPINSKI_MW_MAX:
        breaches.append(f"MW={mw:.1f}")
    if logp >= _LIPINSKI_LOGP_MAX:
        breaches.append(f"LogP={logp:.2f}")
    if hba >= _LIPINSKI_HBA_MAX:
        breaches.append(f"HBA={hba}")
    if hbd >= _LIPINSKI_HBD_MAX:
        breaches.append(f"HBD={hbd}")
    metrics = {"mw": mw, "logp": logp, "hba": hba, "hbd": hbd}
    if breaches:
        return False, metrics, "lipinski_fail: " + ",".join(breaches)
    return True, metrics, ""


def _check_sa(mol: Any) -> tuple[Optional[float], bool, str]:
    """SAscore (Ertl & Schuffenhauer 2009). Returns (score, ok, message).

    Score range 1 (easy) to 10 (hard); I accept ``<= _SA_MAX``.
    If sascorer is unavailable I return (None, True, "sascore_unavailable").
    """
    sa = _get_sascorer()
    if sa is None:
        return None, True, "sascore_unavailable"
    try:
        score = float(sa.calculateScore(mol))
    except Exception as exc:
        return None, False, f"sascore_exception: {exc}"
    ok = score <= _SA_MAX
    msg = "" if ok else f"sa_fail: {score:.2f}>{_SA_MAX}"
    return score, ok, msg


def validate(mol: Any) -> ValidityReport:
    """Sanitize + Lipinski + SAscore pipeline; returns ``ValidityReport``.

    A ``None`` or non-Mol input is treated as a hard fail with an empty
    report. Most "did the edit blow up?" callers can simply check
    ``report.overall_ok``.
    """
    if not _HAVE_RDKIT:
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes="rdkit_unavailable",
        )
    if mol is None:
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes="mol_is_None",
        )
    notes: list[str] = []
    sanitize_ok, msg = _sanitize(mol)
    if msg:
        notes.append(msg)
    if not sanitize_ok:
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes="; ".join(notes) or "sanitize_failed",
        )
    lipinski_ok, metrics, lmsg = _check_lipinski(mol)
    if lmsg:
        notes.append(lmsg)
    sa_score, sa_ok, smsg = _check_sa(mol)
    if smsg:
        notes.append(smsg)
    overall_ok = sanitize_ok and lipinski_ok and sa_ok
    return ValidityReport(
        sanitize_ok=sanitize_ok,
        lipinski_ok=lipinski_ok,
        sa_score=sa_score,
        sa_ok=sa_ok,
        overall_ok=overall_ok,
        notes="; ".join(notes),
        mol_weight=metrics.get("mw"),
        logp=metrics.get("logp"),
        hba=metrics.get("hba"),
        hbd=metrics.get("hbd"),
    )


def validate_smiles(smiles: str) -> ValidityReport:
    """SMILES-string convenience: parses and runs ``validate(mol)``.

    A malformed SMILES becomes a sanitize-fail (mol parse returns None).
    """
    if not _HAVE_RDKIT:
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes="rdkit_unavailable",
        )
    if not smiles or not isinstance(smiles, str):
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes="invalid_smiles",
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ValidityReport(
            sanitize_ok=False, lipinski_ok=False, sa_score=None, sa_ok=False,
            overall_ok=False, notes=f"unparseable_smiles: {smiles!r}",
        )
    return validate(mol)
