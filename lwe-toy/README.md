# lwe-toy

A toy implementation of Regev's LWE public-key cryptosystem plus a working
BKW attack — built to *feel* where the hardness comes from, not just read
about it.

Based on "On Lattices, Learning with Errors, Random Linear Codes, and
Cryptography" (Regev, STOC 2005; arXiv:2401.03703).

## What's here

- `lwe.py` — the cryptosystem: keygen, encrypt/decrypt bits and bytes, and
  `recover_secret_noiseless`, a Gaussian-elimination attack for the
  error-free case.
- `bkw.py` — the Blum–Kalai–Wasserman attack on search-LWE: block-wise
  sample reduction by bucketing, brute-force of the last block by
  hypothesis testing, back-substitution.
- `demo.py` — the two-part story. Run it:

```
python3 demo.py
```

Stdlib only. No dependencies, no `uv` needed (we'd reach for it if this
ever grows a numpy dependency).

## The idea in one page

LWE samples look like $(\mathbf{a}, b = \langle \mathbf{a}, \mathbf{s} \rangle + e \bmod p)$.
Part 1 of the demo deletes $e$ and recovers $\mathbf{s}$ by Gaussian
elimination in under a millisecond: without noise, there is no hard problem.

Part 2 puts the noise back and runs BKW. Split the $n$ coordinates into
blocks of size $b$. In each stage, bucket samples by the current block's
value and subtract pairs inside each bucket — the block cancels to zero
while the error variance merely doubles. After killing all but the last
block, brute-force it (the right guess leaves small residuals; wrong
guesses look uniform), subtract its contribution, recurse.

BKW's bill, which the demo makes concrete:

- **Samples**: each stage needs on the order of $p^b$ samples to fill the
  bucket table ($11^3 = 1331$ buckets/stage at the demo parameters).
- **Noise**: the error variance doubles every stage, so after $a$ stages
  the final noise has standard deviation $2^{a/2}\sigma$ — it must stay
  well below $p$ or the hypothesis test drowns.
- **Net**: $2^{O(n)}$ time for LWE ($2^{O(n/\log n)}$ for binary LPN).
  Real parameters (Kyber, Dilithium) are chosen so this tradeoff — and
  the stronger lattice attacks, BKZ primal/dual — stays infeasible.

## Parameters

Toy only: $n \in \{9, 12\}$, $p = 11$, block size $3$, $\sigma = 0.7$.
Pedagogical code — non-constant-time sampling, no side-channel
protection. Do not encrypt anything real with this.
