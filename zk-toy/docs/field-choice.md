# Why F₂₅₇, and the mod-1543 hiding trick

Two numbers in this codebase look arbitrary — 257 and 1543. They are not.
Each was chosen to make a specific piece of math line up, and the way
they line up with *each other* is the neatest consistency argument in the
whole project.

## Why a prime field at all

Everything — R1CS witnesses, polynomial interpolation, polynomial
division — needs division. In a prime field F_p, every nonzero element
has a multiplicative inverse (Fermat's little theorem: a⁻¹ = a^(p−2)),
so Lagrange interpolation and `divmod` just work. `src/field.rs`
implements inverse exactly this way.

## Why 257 specifically

Three reasons, in increasing order of importance:

1. **It is prime.** (See above.)
2. **p − 1 is smooth.** 257 − 1 = 256 = 2⁸. The multiplicative group
   F_p^* is cyclic of order p − 1, so it contains primitive n-th roots
   of unity for every n dividing 256. We need the **4th roots of unity**
   as our QAP evaluation domain (one point per R1CS constraint), and 4
   divides 256. The code finds them as ω = g⁶⁴ where g = 3 is a primitive
   root (the smallest; found by checking g¹²⁸ ≠ 1, since 256 = 2⁸ means
   primitivity is a single test). Because the four points are *all* the
   4th roots of unity, the target polynomial collapses beautifully:
   t(x) = ∏(x − rᵢ) = x⁴ − 1.
3. **It is small enough to read.** Every intermediate value fits in your
   head. You can hand-check the witness [1, 3, 9, 27, 30, 35], hand-verify
   each R1CS row, and brute-force anything the verifier does. That is the
   entire point of a toy.

The price: soundness. The Schwartz–Zippel argument ([qap.md](qap.md))
gives a cheating prover up to 6/257 ≈ 2.3% success — not negligible.
Real SNARKs use ~256-bit fields (e.g. BN254's scalar field, a 254-bit
prime), where the same argument yields an error around 2⁻²⁵⁰.

## The hiding group: why mod 1543

The homomorphic hiding E(x) = hˣ needs a group where:

- discrete logs are *assumed* hard (hiding),
- exponentiation is homomorphic (E(a)·E(b) = E(a+b)),
- **exponents live in Z₂₅₇** — the same arithmetic as the circuit field,
  so that blind evaluation of F₂₅₇-polynomials is even well-defined.

The trick: work in F₁₅₄₃^*, the multiplicative group of integers mod 1543.
1543 is prime, so F₁₅₄₃^* is cyclic of order 1542 = 6 × 257. Since 257 is
prime, for any g with g⁶ ≠ 1 mod 1543, the element **h = g⁶** satisfies

```
h²⁵⁷ = g¹⁵⁴² = 1,   h ≠ 1
```

and therefore has multiplicative order *exactly* 257. The subgroup ⟨h⟩
has 257 elements, and hˣ cycles with period 257 — exponents are
effectively in Z₂₅₇.

This is the consistency call the whole protocol rests on: the circuit
computes in F₂₅₇, and the hiding function's exponents *are* Z₂₅₇, so
"E(P(s))" is meaningful for F₂₅₇-polynomials P. If the exponent group had
a different order, blind evaluation would silently compute the wrong
thing (exponents reducing mod the wrong number).

`subgroup_generator()` in `src/snark.rs` finds such an h by trial: the
first g ≥ 2 with g⁶ ≠ 1 mod 1543 gives h = g⁶ of order 257. The test
`generator_has_order_257` pins this down.

## The honest cheat

At 257 elements, discrete logs are trivial — `dlog()` just walks
h⁰, h¹, …, h²⁵⁶. The verifier exploits this to recover A(s), B(s), C(s),
H(s) and check the divisibility directly. At real sizes this is
impossible, and that impossibility is *load-bearing* for security: it is
what makes E hiding. Real SNARKs square this circle with **pairings**
(bilinear maps), which let the verifier check
"a·b − c = hh·t(s)" *in the exponent* — confirming the relation between
hidden values without ever recovering them. A pairing is, loosely, the
cryptographic gadget that makes our verifier's brute-force step
unnecessary. Building one is far beyond toy scope, which is why this
project is proudly pairing-free.

## Summary of the number theory

| choice | reason |
|---|---|
| F₂₅₇ | prime (division works); 256 = 2⁸ gives 4th roots of unity; small enough to read |
| 4th roots of unity | one QAP evaluation point per constraint; makes t(x) = x⁴ − 1 |
| mod 1543, h = g⁶ | 1543 prime, 257 \| 1542 ⇒ subgroup of order exactly 257 ⇒ exponents live in Z₂₅₇, matching the circuit field |
| brute-force dlog | toy-only stand-in for pairings; documented everywhere it appears |
