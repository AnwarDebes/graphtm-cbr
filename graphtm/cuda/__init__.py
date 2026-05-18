"""graphtm.cuda, Module M2: CUDA-C kernels for Hierarchical Graph TM.

Public surface:
    CudaKernels   , SourceModule loader + forward/feedback dispatchers
    KernelDims    , compile-time #define carrier
    CudaTAState   , device-side bit-plane TA state, host transfers
    TAStateShape  , geometric description of the TA state tensor
    quick_smoke   , minimal forward+feedback round-trip; safe on no-CUDA hosts

Import is side-effect-free (does NOT init CUDA). PyCUDA is imported
lazily inside method calls; absent toolchain raises RuntimeError on
first kernel call.
"""
from ._kernels import CudaKernels, KernelDims, quick_smoke
from .memory import CudaTAState, TAStateShape

__all__ = [
    "CudaKernels",
    "KernelDims",
    "CudaTAState",
    "TAStateShape",
    "quick_smoke",
]
