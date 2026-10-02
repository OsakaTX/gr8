"""D2 MXFP8 (E4M3FN codes + E8M0 scale per 32 inputs) in NumPy, bit-identical to vLLM's
_mxfp8_e4m3_quantize_torch (the function D2 runs on the original BF16 weights at boot):
  amax = max|x| per 32 (clamped to float32 tiny); s = clamp(ceil(log2(amax / 448)) + 127, 0, 254)
  code = float8_e4m3fn(x / 2**(s-127))  (round to nearest even)
The exponent is computed exactly from float bits (frexp), not with a float log2.
"""
import numpy as np


def bf16_to_f32(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def e4m3fn_encode(x):
    """float32 array (|x| <= 448, finite) -> uint8 E4M3FN codes, round-to-nearest-even."""
    x = np.asarray(x, np.float32)
    a = np.abs(x).astype(np.float64)
    sign = np.signbit(x).astype(np.uint8) << 7
    _, ex = np.frexp(a)                           # a = m * 2**ex, m in [0.5, 1)
    e = np.maximum(ex.astype(np.int64) - 1, -6)   # unbiased exponent; subnormals use -6
    q = np.rint(a * np.ldexp(1.0, (3 - e)))       # mantissa steps of 2**(e-3); exact in float64, RNE
    carry = q >= 16
    e = np.where(carry, e + 1, e)
    q = np.where(carry, q / 2, q)
    normal = q >= 8
    code = np.where(normal, ((e + 7) << 3) | (q.astype(np.int64) - 8), q.astype(np.int64))
    if (code > 126).any():
        raise ValueError('E4M3FN overflow (|x| > 448 after rounding)')
    return (code.astype(np.uint8) | sign)


def quantize(w_bf16, chunk_rows=1024):
    """w: [N, K] BF16 bits (uint16). Returns codes uint8 [N, K], scales uint8 [N, K/32] (biased e8m0)."""
    n, k = w_bf16.shape
    if k % 32:
        raise ValueError('K must be a multiple of 32')
    codes = np.empty((n, k), np.uint8)
    scales = np.empty((n, k // 32), np.uint8)
    tiny = np.finfo(np.float32).tiny
    for r0 in range(0, n, chunk_rows):
        x = bf16_to_f32(w_bf16[r0:r0 + chunk_rows]).reshape(-1, k // 32, 32)
        amax = np.maximum(np.abs(x).max(-1), tiny)
        ratio = (amax / np.float32(448.0)).astype(np.float32)      # float32 division, as torch does
        m, ex = np.frexp(ratio.astype(np.float64))
        ceil_log2 = np.where(m == 0.5, ex - 1, ex)                  # exact ceil(log2(ratio))
        s = np.clip(ceil_log2 + 127, 0, 254).astype(np.int64)
        scales[r0:r0 + chunk_rows] = s
        xs = (x.astype(np.float64) * np.ldexp(1.0, (127 - s))[..., None]).astype(np.float32)
        codes[r0:r0 + chunk_rows] = e4m3fn_encode(xs).reshape(-1, k)
    return codes, scales


def e4m3fn_decode(codes):
    u = np.asarray(codes, np.uint8)
    e, m = (u >> 3) & 15, u & 7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1 + m / 8.0, e.astype(np.int64) - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(u & 128, -v, v)


def dequantize(codes, scales):
    n, k = codes.shape
    return (e4m3fn_decode(codes).reshape(n, k // 32, 32) * np.ldexp(1.0, scales.astype(np.int64) - 127)[..., None]).reshape(n, k)
