"""graphtm.data, TDC AMES, Kazius toxicophores, Tox21 dataset loaders (M6).

Public surface:
  - `MissingDependency`          , sentinel exception for absent optional deps
  - `load_tdc_ames`              , TDC AMES (Hansen 2009) primary loader
  - `load_kazius_toxicophores`   , Kazius 2005 alert vocabulary (29 SMARTS)
  - `load_tox21`                 , Tox21 confirmatory loader, NR-AhR default
  - `scaffold_split`             , Bemis-Murcko scaffold split utility
  - `Toxicophore`                , frozen dataclass for one Kazius alert
  - `TOX21_TASKS`                , canonical 12 Tox21 sub-task names

Design notes
------------
* `MissingDependency` is defined *at the top* of this module before submodule
  imports so that submodules can `from graphtm.data import MissingDependency`
  without a circular-import hazard.
* No data loader depends on M1 internals at import time; M1 is only invoked
  during the optional `encode=True` step inside `load_tdc_ames` /
  `load_tox21`. Sister-agent build order is therefore decoupled.
"""
from __future__ import annotations


class MissingDependency(ImportError):
    """Optional dependency (TDC, RDKit, PyYAML, ...) is not installed.

    Subclass of `ImportError` so callers that already handle `ImportError`
    (e.g. graceful fall-backs in experiments) keep working; the dedicated
    class lets data-layer-aware callers distinguish a true "module missing"
    failure from any other ImportError raised inside `graphtm`.
    """


# Submodule imports, must come *after* `MissingDependency` is defined.
from graphtm.data.ames import load_tdc_ames                         # noqa: E402
from graphtm.data.kazius import Toxicophore, load_kazius_toxicophores  # noqa: E402
from graphtm.data.scaffold_split import scaffold_split                # noqa: E402
from graphtm.data.tox21 import TOX21_TASKS, load_tox21                # noqa: E402


__all__ = [
    "MissingDependency",
    "Toxicophore",
    "TOX21_TASKS",
    "load_kazius_toxicophores",
    "load_tdc_ames",
    "load_tox21",
    "scaffold_split",
]
