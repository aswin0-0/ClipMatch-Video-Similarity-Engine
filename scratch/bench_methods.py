import numpy as np
import time
from services.ncc import ncc_score, _alternate_pixels

rng = np.random.default_rng(42)
a = rng.random((256, 256), dtype=np.float32)
b = a + rng.random((256, 256), dtype=np.float32) * 0.1

N = 1000

t0 = time.perf_counter()
for _ in range(N): ncc_score(a, b, method="standard")
std_ms = (time.perf_counter() - t0) * 1000 / N

t0 = time.perf_counter()
for _ in range(N): ncc_score(a, b, method="diagonal")
diag_ms = (time.perf_counter() - t0) * 1000 / N

t0 = time.perf_counter()
for _ in range(N): ncc_score(a, b, method="alternate")
alt_ms = (time.perf_counter() - t0) * 1000 / N

print("Pixel counts:")
print(f"  standard  : {256*256:>7,} pixels")
print(f"  alternate : {len(_alternate_pixels(a)):>7,} pixels")
print(f"  diagonal  : {256:>7} pixels")
print()
print(f"Speed (avg over {N} calls):")
print(f"  standard  : {std_ms:.4f} ms/frame")
print(f"  alternate : {alt_ms:.4f} ms/frame  ({std_ms/alt_ms:.0f}x faster)")
print(f"  diagonal  : {diag_ms:.4f} ms/frame  ({std_ms/diag_ms:.0f}x faster)")
print()
s_std = ncc_score(a, b, method="standard")
s_diag = ncc_score(a, b, method="diagonal")
s_alt = ncc_score(a, b, method="alternate")
print("Score on similar frames (high correlation expected):")
print(f"  standard  : {s_std:.4f}")
print(f"  alternate : {s_alt:.4f}")
print(f"  diagonal  : {s_diag:.4f}")
