"""M1, graph -> BSC hypervector encoding (frozen interface; see ARCHITECTURE.md)."""
from .codebook import (
    ATOM_SYMBOLS,
    BOND_NAMES,
    AtomBondCodebook,
    make_codebook,
)
from .graph_features import GraphTensor, encode_graph
from .hypervectors import (
    bind,
    bundle,
    majority_bundle,
    one_hot,
    permute,
    random_hv,
    similarity,
    sparse_bsc,
    thermometer_encode,
    thermometer_encode_int,
    xor_bind,
)

__all__ = [
    "ATOM_SYMBOLS",
    "BOND_NAMES",
    "AtomBondCodebook",
    "GraphTensor",
    "bind",
    "bundle",
    "encode_graph",
    "majority_bundle",
    "make_codebook",
    "one_hot",
    "permute",
    "random_hv",
    "similarity",
    "sparse_bsc",
    "thermometer_encode",
    "thermometer_encode_int",
    "xor_bind",
]
