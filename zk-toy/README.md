# zk-toy — a from-scratch zk-SNARK

An educational implementation of a zk-SNARK for the classic statement
**"I know x such that x³ + x + 5 = 35"** (answer: x = 3, which the proof
never reveals). Standard library only — no crypto crates. Every step of
the pipeline is implemented by hand:

```
finite field  →  polynomials  →  R1CS  →  QAP  →  homomorphic hiding
→  blind evaluation  →  KEA check  →  prove / verify
```

Run it:

```sh
cargo test              # full test suite (field, poly, R1CS, QAP, SNARK)
cargo run --release     # demo: prove knowledge of x = 3, verify, show rejections
```

## Background reading

Each stage of the pipeline has a deeper writeup in `docs/` — each one
readable in a single sitting:

- [R1CS](docs/r1cs.md) — flattening, why addition is free, the witness check
- [QAP](docs/qap.md) — the interpolation trick, the divisibility argument, Schwartz–Zippel
- [The protocol](docs/protocol.md) — hiding, trusted setup, blind evaluation, KEA, expanded caveats
- [Field choice](docs/field-choice.md) — why F₂₅₇, roots of unity, the mod-1543 subgroup trick

## 1. The field: F₂₅₇

We do all circuit arithmetic in the prime field F_p with **p = 257**, because:

- 257 is prime, so every nonzero element has an inverse (needed for
  Lagrange interpolation and polynomial division).
- p − 1 = 256 = 2⁸ is *smooth* (a power of two), so the multiplicative
  group F_p^* contains primitive n-th roots of unity for every n dividing
  256 — in particular the **4th roots of unity** we use as the QAP
  evaluation domain (one point per R1CS constraint).
- It is small enough that every intermediate value is human-readable.
  That is the entire point of a toy.

(`src/field.rs`: add, sub, mul, neg, pow, inverse via Fermat's little theorem.)

## 2. Polynomials

Univariate polynomials over F₂₅₇ with naive O(n²) multiplication, Horner
evaluation, long division (`divmod`), Lagrange interpolation, and the
vanishing polynomial ∏(x − rᵢ). Nothing fancy — just correct.
(`src/poly.rs`.)

## 3. R1CS

The computation x³ + x + 5 is *flattened* into multiplication gates, each
gate becoming one constraint of the form `(A·s) · (B·s) = (C·s)`:

| gate | constraint |
|---|---|
| `sym1 = x·x` | `(x) · (x) = (sym1)` |
| `y = sym1·x` | `(sym1) · (x) = (y)` |
| `sym2 = y + x` | `(y + x) · (1) = (sym2)` |
| `out = sym2 + 5` | `(sym2 + 5) · (1) = (out)` |

With solution vector `s = [one, x, sym1, y, sym2, out]`, the matrices are:

```
A = [[0,1,0,0,0,0],   B = [[0,1,0,0,0,0],   C = [[0,0,1,0,0,0],
     [0,0,1,0,0,0],        [0,1,0,0,0,0],        [0,0,0,1,0,0],
     [0,1,0,1,0,0],        [1,0,0,0,0,0],        [0,0,0,0,1,0],
     [5,0,0,0,1,0]]        [1,0,0,0,0,0]]        [0,0,0,0,0,1]]
```

For x = 3: `s = [1, 3, 9, 27, 30, 35]`, and each row checks out, e.g. row 4:
(30 + 5)·1 = 35. (`src/r1cs.rs`.)

## 4. QAP

Pick one evaluation point rᵢ per constraint — the 4th roots of unity
{1, ω, ω², ω³}. Each **column** j of a constraint matrix is interpolated
into a polynomial (Aⱼ, Bⱼ, Cⱼ) with Aⱼ(rᵢ) = A[i][j]. Then

```
A(x) = Σⱼ sⱼ·Aⱼ(x),   B(x) = Σⱼ sⱼ·Bⱼ(x),   C(x) = Σⱼ sⱼ·Cⱼ(x)
```

and the R1CS is satisfied **iff** A(rᵢ)·B(rᵢ) − C(rᵢ) = 0 at all four
points — i.e. iff the *target polynomial* t(x) = ∏ᵢ(x − rᵢ) divides
A(x)·B(x) − C(x). The quotient is h(x):

```
A(x)·B(x) − C(x) = h(x)·t(x)
```

Since our roots are all the 4th roots of unity, t(x) = x⁴ − 1, and
deg(h) ≤ 2 (A·B has degree ≤ 6). The code checks the division is exact
and verifies the polynomial identity directly. (`src/qap.rs`.)

## 5. Homomorphic hiding

We need the prover to evaluate these polynomials at a *secret* point s
without learning s. The hiding function is

```
E(x) = h^x  (mod 1543)
```

where h generates the unique order-257 subgroup of F₁₅₄₃^*. This works
because 1543 is prime and **257 divides 1543 − 1 = 1542 = 6·257**: for any
g with g⁶ ≠ 1 mod 1543, h = g⁶ has order exactly 257 (257 is prime).
Since h²⁵⁷ = 1, exponents live in Z₂₅₇, which we identify with F₂₅₇ —
so the circuit field and the exponent group line up perfectly.

E is homomorphic: `E(a)·E(b) = E(a+b)` and `E(a)^k = E(k·a)`. It is
"discrete-log based": recovering x from E(x) is assumed hard — except
our group is toy-sized, so it is actually brute-forceable, which the
verifier below exploits. (`src/snark.rs`.)

## 6. Trusted setup and blind evaluation

Setup samples secret `s` and `α` ("toxic waste", discarded afterwards)
and publishes the CRS:

```
E(s⁰), E(s¹), E(s²), E(s³)          (for A, B, C — degree ≤ 3)
E(α·s⁰), …, E(α·s³)                 (alpha-shifted, for the KEA check)
E(s⁰), E(s¹), E(s²)                 (for H — degree ≤ 2)
```

The prover, who never learns s, computes `E(P(s)) = ∏ᵢ E(sⁱ)^{pᵢ}` for
their polynomials — *blind evaluation*.

## 7. The KEA (alpha-shift) check

The prover also commits to the alpha-shifted evaluations E(α·A(s)) etc.
The verifier checks `E(α·A(s)) == E(A(s))^α`. Without knowing α, the
prover can only produce a consistent pair by applying the *same*
coefficients to the shifted CRS — the Knowledge-of-Exponent Assumption.
This binds the prover to actually knowing the polynomial they committed to.

## 8. Prove and verify

- **prove(witness)** — checks the witness satisfies the R1CS, builds
  A, B, C, computes h = (A·B − C)/t, and blind-evaluates everything.
  Only the *private* part of A, B, C is committed; variables 0 (`one`)
  and 5 (`out`) are public.
- **verify(proof, public_inputs)** — the verifier adds the public
  contribution itself (`E_full = E_priv · h^{P_pub(s)}`, homomorphically),
  runs the KEA check, recovers a = A(s), b = B(s), c = C(s), hh = H(s)
  by brute-force discrete log, and checks `a·b − c == hh·t(s)` in F₂₅₇.
  Because the verifier mixes in *its own* public inputs, the proof only
  verifies against the claimed `(one=1, out=35)`.

## Honest caveats

This is a teaching toy, not cryptography:

- **Tiny field.** Soundness rests on Schwartz–Zippel: a cheating prover
  who commits to polynomials not satisfying the QAP identity is caught
  unless they get lucky at the secret point s — but with |F| = 257 that
  luck is ~6/257 ≈ 2.3%, not negligible. Real SNARKs use ~256-bit fields.
- **No zero-knowledge.** There is no blinding; worse, the verifier
  brute-forces discrete logs (impossible at real sizes — real SNARKs use
  pairings precisely so the verifier never needs s or discrete logs).
  A verifier willing to do 257 operations can brute-force x itself.
  The demo shows the *mechanics* of proving; it does not hide like the
  real thing.
- **No trusted-setup ceremony.** One party runs setup and knows s and α.
  Real deployments use multi-party computation so no single party knows
  the toxic waste.
- **KEA is an assumption**, and a strong, non-falsifiable one at that.

## Layout

```
zk-toy/
  Cargo.toml          # no dependencies — std only
  src/
    lib.rs            # module tree
    main.rs           # demo binary
    field.rs          # F_257 arithmetic
    poly.rs           # polynomial arithmetic, Lagrange, divmod
    r1cs.rs           # constraint matrices, witness, satisfaction check
    qap.rs            # column interpolation, target poly, quotient h
    snark.rs          # hiding, setup, blind eval, prove, verify
```
