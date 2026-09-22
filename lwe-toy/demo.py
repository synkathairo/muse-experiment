"""
Demo: why the error term is the entire security of LWE.

Part 1 -- NO error: Gaussian elimination recovers the secret instantly.
Part 2 -- WITH error: BKW (Blum-Kalai-Wasserman) still recovers it at toy
parameters, but watch the cost grow with n.

Run:  python3 demo.py
"""

import random
import time

from lwe import keygen, recover_secret_noiseless
from bkw import bkw, gen_lwe_samples


def part1_noiseless():
    print("=" * 64)
    print("Part 1: LWE *without* error  ->  just linear algebra")
    print("=" * 64)
    n, p = 16, 101
    random.seed(1)
    s = [random.randrange(p) for _ in range(n)]
    # noiseless public key: b_i = <a_i, s> exactly
    pub = []
    for _ in range(n):
        a = [random.randrange(p) for _ in range(n)]
        b = sum(x * y for x, y in zip(a, s)) % p
        pub.append((a, b))
    t0 = time.time()
    s_rec = recover_secret_noiseless(pub, p)
    print(f"n={n}, p={p}: recovered {s_rec == s} in "
          f"{(time.time() - t0) * 1000:.1f} ms")
    print("Without noise, the 'hard problem' is Gaussian elimination.\n")


def part2_bkw():
    print("=" * 64)
    print("Part 2: LWE *with* error  ->  BKW attack at toy parameters")
    print("=" * 64)
    for n in (9, 12):
        random.seed(0)
        p, bsz, sigma, m = 11, 3, 0.7, 12000
        s = [random.randrange(p) for _ in range(n)]
        samples = gen_lwe_samples(s, p, m, sigma)
        t0 = time.time()
        s_rec, _ = bkw(samples, p, bsz, sigma)
        dt = time.time() - t0
        print(f"n={n:2d}: secret recovered: {s_rec == s}   "
              f"({dt:.1f}s, {m} samples, p^{bsz}={p ** bsz} buckets/stage)")
    print("\nBKW's bill: ~p^bsz bucket entries per stage, and the error")
    print("variance doubles every stage. Shrink p^bsz or the noise and it")
    print("dies; grow n and the sample/noise budget explodes. That tradeoff")
    print("is exactly what real LWE parameters are chosen to survive.")


if __name__ == "__main__":
    part1_noiseless()
    part2_bkw()
