"""
BKW attack (Blum-Kalai-Wasserman, 2003) on search-LWE, toy edition.

Given samples (a, b = <a, s> + e mod p), recover s.

The trick: split the n coordinates into blocks of size `bsz`. In each
stage, bucket samples by the value of the current block; subtracting two
samples from the same bucket zeroes that block (a1 - a2 = 0 there) while
only doubling the error variance. After killing all but the last block,
brute-force the last block by hypothesis testing (the right guess leaves
small residuals; wrong guesses look uniform), subtract its contribution,
and recurse on the remaining coordinates.

Complexity is dominated by the bucket tables: ~p^bsz entries per stage,
and the error grows as 2^(stages/2) -- the fundamental BKW tradeoff.
"""

import itertools
import math
import random
import time


def circ_dist(v, p):
    """Signed distance of v from 0 on the cycle Z_p, in [-(p//2), p//2]."""
    v %= p
    return v - p if v > p // 2 else v


def gen_lwe_samples(s, p, m, sigma):
    n = len(s)
    out = []
    for _ in range(m):
        a = [random.randrange(p) for _ in range(n)]
        e = int(round(random.gauss(0.0, sigma))) % p
        b = (sum(x * y for x, y in zip(a, s)) + e) % p
        out.append((a, b))
    return out


def reduce_stage(samples, p, block, bsz):
    """Zero out one block by differencing samples within equal-value buckets."""
    buckets = {}
    for a, b in samples:
        key = tuple(a[block * bsz:(block + 1) * bsz])
        buckets.setdefault(key, []).append((a, b))
    reduced = []
    for bucket in buckets.values():
        # pair up samples; leftovers are discarded
        for i in range(0, len(bucket) - 1, 2):
            (a1, b1), (a2, b2) = bucket[i], bucket[i + 1]
            a_new = [(x - y) % p for x, y in zip(a1, a2)]
            b_new = (b1 - b2) % p
            reduced.append((a_new, b_new))
    return reduced


def solve_last_block(samples, p, bsz, dim, cap=1500):
    """Brute-force the final block: the right guess minimizes residuals."""
    block = slice(dim - bsz, dim)
    use = samples[:cap]
    best, best_score = None, None
    for guess in itertools.product(range(p), repeat=bsz):
        score = 0
        for a, b in use:
            r = b - sum(x * y for x, y in zip(a[block], guess))
            d = circ_dist(r, p)
            score += d * d
            if best_score is not None and score >= best_score:
                break  # early exit, can't beat current best
        if best_score is None or score < best_score:
            best, best_score = guess, score
    return list(best)


def bkw(samples, p, bsz, sigma, verbose=False):
    """Recover s from LWE samples. Returns (s_recovered, stats)."""
    n = len(samples[0][0])
    s_rec = [0] * n
    t0 = time.time()
    stages_info = []

    # work on a copy; peel off blocks from the end
    cur = [(list(a), b) for a, b in samples]
    dim = n
    while dim > 0:
        take = min(bsz, dim)
        nblocks = (dim + take - 1) // take
        # reduce all blocks except the last `take` coordinates
        for blk in range(nblocks - 1):
            cur = reduce_stage(cur, p, blk, take)
            if len(cur) < 50:
                raise RuntimeError(
                    f"sample starvation at dim={dim}, block={blk}: "
                    f"only {len(cur)} left; need more initial samples")
        s_block = solve_last_block(cur, p, take, dim)
        s_rec[dim - take:dim] = s_block
        stages_info.append((dim, len(cur)))
        if verbose:
            print(f"  dim {dim:3d}: recovered block {s_block}, "
                  f"{len(cur)} reduced samples used")
        # subtract this block's contribution, recurse on the rest
        new_cur = []
        for a, b in samples:
            contrib = sum(x * y for x, y in zip(a[dim - take:dim], s_block))
            new_cur.append((a[:dim - take], (b - contrib) % p))
        # fresh subsample to keep runtime sane; keep it simple: reuse all
        cur = new_cur
        # regenerate reduced-form? No -- recursion re-runs stages on raw form.
        # Instead: run the whole pipeline per block on the original-style data.
        # We do that by re-entering the loop with `samples`-style data below.
        samples = new_cur
        cur = [(list(a), b) for a, b in new_cur]
        dim -= take

    return s_rec, {"time": time.time() - t0, "stages": stages_info}


def demo(n=9, p=11, bsz=3, sigma=0.7, m=12000, seed=0):
    random.seed(seed)
    s = [random.randrange(p) for _ in range(n)]
    print(f"LWE instance: n={n}, p={p}, block size={bsz}, sigma={sigma}, "
          f"samples={m}")
    print(f"true secret: {s}")
    samples = gen_lwe_samples(s, p, m, sigma)
    t0 = time.time()
    s_rec, stats = bkw(samples, p, bsz, sigma, verbose=True)
    dt = time.time() - t0
    ok = s_rec == s
    print(f"recovered:   {s_rec}")
    print(f"match: {ok}   wall time: {dt:.1f}s")
    return ok, dt


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    demo(n=n)
