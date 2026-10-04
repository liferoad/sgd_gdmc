"""8-bit Adam baseline (block-wise quantized optimizer state).

This is the memory-efficient Adam reference the GDMC memory story needs.
Both moment buffers are stored block-wise as int8 plus one float32 scale per
block; the update itself is done in float32 after dequantization.

Storage per parameter is ~10 bytes (4 weight + 4 gradient + 2 x 1 byte per
moment, plus tiny per-block scales), versus 16 bytes for FP32 Adam. That is
the honest comparator for a packed 4-bit GDMC (0.5 + 4 + 4 = 8.5 bytes),
not FP32 Adam.

The quantize/dequantize helpers are **chunked** so they never materialise a
full-size float32 temporary; a naive implementation allocates 2-4 full
buffers per step and its measured peak RSS can exceed FP32 Adam's even
though its state is 4x smaller.

Reference: Dettmers et al., "8-bit Optimizers via Block-wise Quantization",
arXiv:2110.02861.
"""

from __future__ import annotations
from typing import Iterable, Optional
import math

import torch

MAX_CHUNK_ELEMS = 1 << 18  # bound on the float32 scratch per helper call


def _block_size(n: int, block_size: int) -> int:
    return max(1, min(int(block_size), n))


def quantize_blockwise(t: torch.Tensor, block_size: int = 2048,
                       max_elems: int = MAX_CHUNK_ELEMS):
    """Return (int8 codes, float32 per-block scales) for a tensor."""
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n == 0:
        return flat.to(torch.int8), flat.new_zeros(0)
    bs = _block_size(n, block_size)
    nblocks = (n + bs - 1) // bs
    codes = torch.empty(n, dtype=torch.int8, device=flat.device)
    scales = torch.empty(nblocks, dtype=torch.float32, device=flat.device)
    per_chunk = max(1, int(max_elems) // bs)
    for b0 in range(0, nblocks, per_chunk):
        b1 = min(b0 + per_chunk, nblocks)
        s, e = b0 * bs, min(b1 * bs, n)
        nb = b1 - b0
        chunk = flat[s:e].float()
        if chunk.numel() < nb * bs:  # pad the final partial block with zeros
            chunk = torch.cat([chunk, chunk.new_zeros(nb * bs - chunk.numel())])
        blk = chunk.reshape(nb, bs)
        scale = blk.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        q = blk / scale[:, None]
        q.round_().clamp_(-127.0, 127.0)
        codes[s:e] = q.reshape(-1)[:e - s].to(torch.int8)
        scales[b0:b1] = scale
    return codes, scales


def dequantize_blockwise(codes: torch.Tensor, scale: torch.Tensor,
                         shape, block_size: int = 2048,
                         max_elems: int = MAX_CHUNK_ELEMS) -> torch.Tensor:
    """Inverse of quantize_blockwise, trimmed to the given shape."""
    n = 1
    for s in shape:
        n *= int(s)
    out = torch.empty(n, dtype=torch.float32, device=codes.device)
    if n == 0:
        return out.reshape(shape)
    bs = _block_size(n, block_size)
    nblocks = (n + bs - 1) // bs
    per_chunk = max(1, int(max_elems) // bs)
    for b0 in range(0, nblocks, per_chunk):
        b1 = min(b0 + per_chunk, nblocks)
        s, e = b0 * bs, min(b1 * bs, n)
        nb = b1 - b0
        seg = codes[s:e].float()
        if seg.numel() < nb * bs:
            seg = torch.cat([seg, seg.new_zeros(nb * bs - seg.numel())])
        blk = seg.reshape(nb, bs)
        vals = (blk * scale[b0:b1, None]).reshape(-1)
        out[s:e] = vals[:e - s]
    return out.reshape(shape)


class Adam8Bit(torch.optim.Optimizer):
    """Adam whose two moment buffers are stored in 8 bits (block-wise)."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 block_size: int = 2048) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, block_size=block_size)
        super().__init__(params, defaults)
        self.requires_closure = False

    def _init_state(self, state, p, block_size):
        n = p.numel()
        bsize = _block_size(n, block_size)
        nblocks = (n + bsize - 1) // bsize
        state["step"] = 0
        state["bsize"] = bsize
        state["m_c"] = torch.zeros(n, dtype=torch.int8, device=p.data.device)
        state["v_c"] = torch.zeros(n, dtype=torch.uint8, device=p.data.device)
        state["m_s"] = torch.zeros(nblocks, dtype=torch.float32, device=p.data.device)
        state["v_lo"] = torch.zeros(nblocks, dtype=torch.float32, device=p.data.device)
        state["v_hi"] = torch.zeros(nblocks, dtype=torch.float32, device=p.data.device)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Block-streamed update: no full-size float32 moment is materialised."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            block_size = group["block_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    self._init_state(state, p, block_size)
                state["step"] += 1
                t = state["step"]
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                flat_p = p.data.view(-1)
                flat_g = p.grad.reshape(-1)
                bsize = state["bsize"]
                n = flat_p.numel()
                for b in range(state["m_s"].numel()):
                    s = b * bsize
                    e = min(s + bsize, n)
                    gb = flat_g[s:e]
                    # First moment: linear int8 (zeroing a tiny m is harmless).
                    m = state["m_c"][s:e].float().mul_(state["m_s"][b])
                    # Second moment: log-domain uint8. Linear quantization of v
                    # zeroes most of a heavy-tailed block (v spans orders of
                    # magnitude), which collapses the denominator and makes the
                    # update explode.
                    lo = state["v_lo"][b]
                    hi = state["v_hi"][b]
                    if hi > lo:
                        v = torch.pow(2.0, lo + state["v_c"][s:e].float()
                                      * ((hi - lo) / 255.0))
                    else:
                        v = torch.zeros_like(gb)
                    m.mul_(beta1).add_(gb, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(gb, gb, value=1.0 - beta2)
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    flat_p[s:e].addcdiv_(m, denom, value=-lr / bc1)
                    # Re-quantize both moments.
                    ms = m.abs().amax().clamp_min(1e-12) / 127.0
                    state["m_c"][s:e] = (m / ms).round_().clamp_(-127, 127).to(torch.int8)
                    state["m_s"][b] = ms
                    lv = v.clamp_min(1e-30).log2()
                    vlo = float(lv.min().item())
                    vhi = float(lv.max().item())
                    if vhi <= vlo:
                        vhi = vlo + 1e-6
                    state["v_lo"][b] = vlo
                    state["v_hi"][b] = vhi
                    codes = ((lv - vlo) * (255.0 / (vhi - vlo))).round_()
                    state["v_c"][s:e] = codes.clamp_(0, 255).to(torch.uint8)
        return loss

    def state_bytes(self) -> int:
        """Bytes currently held in optimizer state (for tests/diagnostics)."""
        total = 0
        for st in self.state.values():
            for key in ("m_c", "v_c"):
                total += st[key].numel() * st[key].element_size()
            for key in ("m_s", "v_lo", "v_hi"):
                total += st[key].numel() * st[key].element_size()
        return total


def make_adam8bit(params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                  betas=(0.9, 0.999), eps: float = 1e-8,
                  block_size: int = 2048) -> Adam8Bit:
    return Adam8Bit(params, lr=lr, betas=betas, eps=eps, block_size=block_size)
