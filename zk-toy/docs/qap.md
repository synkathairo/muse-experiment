# QAP: compressing all constraints into one divisibility check

The QAP (Quadratic Arithmetic Program) transformation turns the R1CS —
many constraints — into a single statement about polynomials: **"this
polynomial is divisible by that polynomial."** If you understand why
divisibility implies every constraint holds, you understand the core
trick of SNARKs.

## From matrix columns to polynomials

Recall the R1CS matrices A, B, C from [r1cs.md](r1cs.md): 4 rows
(constraints) × 6 columns (variables). Pick one evaluation point rᵢ per
constraint — in our case the 4th roots of unity {1, ω, ω², ω³} in F₂₅₇
(see [field-choice.md](field-choice.md) for why these exist).

Now take **column** j of matrix A — four numbers, one per constraint —
and interpolate the unique degree-≤3 polynomial Aⱼ with

```
Aⱼ(rᵢ) = A[i][j]     for i = 1..4
```

Do this for every column of A, B, and C. You get 18 small polynomials
(Aⱼ, Bⱼ, Cⱼ for j = 0..5). The code checks they are correct by evaluating
each at each root and comparing against the matrix entry
(`column_polys_match_matrix` test in `src/qap.rs`).

## Combining with the witness

Given a witness vector s, form the linear combinations

```
A(x) = Σⱼ sⱼ·A� follows,  B(x) = Σⱼ sⱼ·Bⱼ(x),  C(x) = Σⱼ sⱼ·Cⱼ(x)
```

By construction, evaluating at rᵢ recovers exactly the i-th constraint's
dot products:

```
A(rᵢ) = (A·s)[i],   B(rᵢ) = (B·s)[i],   C(rᵢ) = (C·s)[i]
```

So the R1CS is satisfied **iff** A(rᵢ)·B(rᵢ) − C(rᵢ) = 0 at all four
points rᵢ.

## The divisibility trick

Define the **target polynomial**

```
t(x) = ∏ᵢ (x − rᵢ)
```

t(x) is zero exactly at the four evaluation points (and nowhere else, up
to multiplicity). Since our points are all the 4th roots of unity,
t(x) = x⁴ − 1 — verified by the `target_is_x4_minus_1` test.

Now the key equivalence:

> A(rᵢ)·B(rᵢ) − C(rᵢ) = 0 for all i
> ⟺ t(x) divides A(x)·B(x) − C(x)

(⇒) If the left side vanishes at every rᵢ, then each (x − rᵢ) is a factor
of it, and since the rᵢ are distinct, their product t(x) divides it.
(⇐) If A·B − C = h·t for some h, then evaluating at any rᵢ gives
h(rᵢ)·t(rᵢ) = h(rᵢ)·0 = 0.

So checking "all 4 constraints hold" is equivalent to checking the
**single** polynomial identity

```
A(x)·B(x) − C(x) = h(x)·t(x)
```

for some quotient polynomial h. The prover computes h by plain polynomial
long division (`divmod`); if the witness is bad, the division has a
remainder and proving fails right there (`divisibility_fails_for_bad_witness`).

Degree check: A, B, C have degree ≤ 3 (interpolated through 4 points), so
A·B has degree ≤ 6, and h = (A·B − C)/t has degree ≤ 6 − 4 = 2. The code
asserts this.

## Why a verifier can check this at one point

Here is where Schwartz–Zippel enters. Suppose a cheating prover commits
to polynomials where A·B − C ≠ h·t as polynomials. Their difference
D(x) = A·B − C − h·t is a *nonzero* polynomial of degree ≤ 6, so it has
at most 6 roots in F₂₅₇. The verifier will check the identity at a single
secret random point s. The prover, who does not know s, can only get away
with it if s happens to be one of those ≤ 6 roots — probability ≤ 6/257
≈ 2.3%.

That is the soundness argument in miniature: checking a polynomial
identity at one random point is almost as good as checking it everywhere,
*provided the field is large*. At 257 elements the error is 2.3% —
fine for a toy, useless for security. Real SNARKs use ~256-bit fields,
where the same argument gives an error around 2⁻²⁵⁰: negligible.

This is also why the setup resamples s if t(s) = 0: at a root of t the
divisibility check `a·b − c = hh·t(s)` becomes `a·b − c = 0`, which a
cheater could satisfy without the QAP identity holding. The point s must
be generic.

## What the QAP buys us

Before: 4 constraints to check. After: 1 divisibility to check — and,
crucially, a divisibility that can be verified at a *single hidden point*
using homomorphic hiding, without ever seeing the polynomials themselves.
That is what [protocol.md](protocol.md) builds on.
