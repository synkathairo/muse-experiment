"""
Toy implementation of Regev's LWE public-key cryptosystem
("On Lattices, Learning with Errors, Random Linear Codes, and Cryptography",
STOC 2005 / arXiv:2401.03703).

PEDAGOGICAL CODE ONLY. Toy parameters, no side-channel protection,
non-constant-time Gaussian sampling. Do not use for real encryption.

The scheme:
    Secret key:  s in Z_p^n (uniform)
    Public key:  m samples (a_i, b_i = <a_i, s> + e_i mod p), e_i <- Psi_alpha
    Encrypt bit x: random subset S of [m]:
        c = (sum_{i in S} a_i mod p, sum_{i in S} b_i + x * floor(p/2) mod p)
    Decrypt (a, b): v = b - <a, s> mod p = sum e_i + x * floor(p/2)
        -> 0 if v is closer to 0, 1 if closer to floor(p/2) (mod p)

Correctness needs |sum_{i in S} e_i| < p/4, which holds w.h.p. when the
error width alpha*p is small relative to p and m is modest.
"""

import math
import random


def sample_error(p, sigma):
    """Sample from a discrete Gaussian-ish error distribution on Z_p.

    Draws round(N(0, sigma^2)) and reduces mod p. A stand-in for Regev's
    Psi_alpha (discretized periodized Gaussian); fine for a toy.
    """
    e = int(round(random.gauss(0.0, sigma)))
    return e % p


def keygen(n, p, m, sigma):
    s = [random.randrange(p) for _ in range(n)]
    pub = []
    for _ in range(m):
        a = [random.randrange(p) for _ in range(n)]
        e = sample_error(p, sigma)
        b = (sum(x * y for x, y in zip(a, s)) + e) % p
        pub.append((a, b))
    return s, pub


def encrypt_bit(pub, p, bit):
    m = len(pub)
    n = len(pub[0][0])
    S = [i for i in range(m) if random.randrange(2)]
    if not S:  # avoid the degenerate empty subset
        S = [random.randrange(m)]
    a_sum = [0] * n
    b_sum = 0
    for i in S:
        a, b = pub[i]
        a_sum = [(x + y) % p for x, y in zip(a_sum, a)]
        b_sum = (b_sum + b) % p
    b_sum = (b_sum + bit * (p // 2)) % p
    return a_sum, b_sum


def decrypt_bit(s, p, ctxt):
    a, b = ctxt
    v = (b - sum(x * y for x, y in zip(a, s))) % p
    # distance of v from 0 vs from floor(p/2), on the cycle Z_p
    d0 = min(v, p - v)
    d1 = min(abs(v - p // 2), p - abs(v - p // 2))
    return 0 if d0 < d1 else 1


def encrypt_bytes(pub, p, data: bytes):
    bits = [(byte >> k) & 1 for byte in data for k in range(7, -1, -1)]
    return [encrypt_bit(pub, p, bit) for bit in bits]


def decrypt_bytes(s, p, ctxts):
    bits = [decrypt_bit(s, p, c) for c in ctxts]
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


# ---------------------------------------------------------------------------
# The "why error matters" demo: with e = 0 the scheme is just linear algebra.
# ---------------------------------------------------------------------------

def modinv(a, p):
    return pow(a, -1, p)


def recover_secret_noiseless(pub, p):
    """Gaussian elimination mod p on noiseless samples (a_i, <a_i, s>).

    Returns s. Demonstrates that without the error term, LWE is trivial.
    """
    n = len(pub[0][0])
    rows = [list(a) + [b] for a, b in pub[:n]]
    # forward elimination
    row = 0
    where = [-1] * n
    for col in range(n):
        piv = next((r for r in range(row, n) if rows[r][col] % p != 0), None)
        if piv is None:
            continue
        rows[row], rows[piv] = rows[piv], rows[row]
        where[col] = row
        inv = modinv(rows[row][col], p)
        rows[row] = [(v * inv) % p for v in rows[row]]
        for r in range(n):
            if r != row and rows[r][col] % p != 0:
                factor = rows[r][col]
                rows[r] = [(v - factor * w) % p for v, w in zip(rows[r], rows[row])]
        row += 1
    return [rows[where[c]][n] % p if where[c] != -1 else 0 for c in range(n)]
