import logging
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic
logger=logging.getLogger(__name__)
MAX_GRID=512
BLOCK=512
@triton.jit
def _abs_kernel(inp,out,n,BLOCK:tl.constexpr,MAX_GRID:tl.constexpr):
    pid=tl.program_id(0)
    base=pid*BLOCK*MAX_GRID
    for j in range(MAX_GRID):
        offs=base+j*BLOCK+tl.arange(0,BLOCK)
        mask=offs<n
        x=tl.load(inp+offs,mask=mask,other=0)
        tl.store(out+offs,tl.abs(x),mask=mask)
@pointwise_dynamic(promotion_methods=[(0,"COMPLEX_TO_FLOAT")])
@triton.jit
def _fallback(x): return tl.abs(x)
def abs(A):
    logger.debug("GEMS ABS")
    n=A.numel()
    if A.is_contiguous() and n>0:
        grid=(min(MAX_GRID,triton.cdiv(n,BLOCK*MAX_GRID)),)
        return _launch(A,n)
    return _fallback(A)
def _launch(A,n):
    import torch
    out=torch.empty_like(A)
    _abs_kernel[(min(MAX_GRID,triton.cdiv(n,BLOCK*MAX_GRID)),)](A,out,n,BLOCK=BLOCK,MAX_GRID=MAX_GRID)
    return out
def abs_(A):
    n=A.numel()
    if A.is_contiguous() and n>0:
        _abs_kernel[(min(MAX_GRID,triton.cdiv(n,BLOCK*MAX_GRID)),)](A,A,n,BLOCK=BLOCK,MAX_GRID=MAX_GRID)
    else: _fallback(A,out0=A)
    return A
