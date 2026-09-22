# The protocol: hiding, setup, blind evaluation, and the KEA check

The QAP ([qap.md](qap.md)) reduced everything to one divisibility:
A(x)·B(x) − C(x) = h(x)·t(x). Now we need a protocol where the prover
convinces the verifier of this **without revealing the witness**, and
where the verifier checks it **at a secret point s without revealing s
to the prover**. This is the SNARK itself.

## 1. Homomorphic hiding

We need a function E with two properties:

- **Hiding**: given E(x), you cannot recover x (discrete-log hardness).
- **Homomorphic**: E(a)·E(b) = E(a+b) and E(a)ᵏ = E(k·a) — you can add
  and scale *inside* the hiding, blindly.

Our toy uses

```
E(x) = hˣ  mod 1543
```

where h generates the unique subgroup of order 257 in F₁₅₄₃^*. Why this
works is explained in [field-choice.md](field-choice.md); the short
version: 1543 is prime, 257 divides 1542, so h = g⁶ has order exactly
257, and exponents live in Z₂₅₇ — identified with our circuit field F₂₅₇.

Homomorphism check: E(a)·E(b) = hᵃ·hᵇ = h^(a+b) = E(a+b). Scaling:
E(a)ᵏ = (hᵃ)ᵏ = h^(ka) = E(k·a). Both are just exponent laws.

The catch, stated plainly: this group is toy-sized, so discrete logs are
brute-forceable (257 tries). The verifier in `src/snark.rs` *actually
does this* — `dlog()` walks h⁰, h¹, … until it hits the target. At real
sizes that is impossible, which is exactly why real SNARKs use pairings:
they let the verifier check the divisibility *without ever recovering the
hidden values*. Our verifier cheats, honestly and on purpose, because a
toy cannot afford pairings.

## 2. The trusted setup and toxic waste

Someone — the "trusted setup" — samples two random secrets:

- **s**: the point at which the divisibility will be checked.
- **α** (alpha): used for the knowledge check below.

and publishes the CRS (common reference string):

```
E(s⁰), E(s¹), E(s²), E(s³)            for A, B, C (degree ≤ 3)
E(α·s⁰), …, E(α·s³)                    alpha-shifted copies
E(s⁰), E(s¹), E(s²)                    for H (degree ≤ 2)
```

then **discards s and α**. This discarded randomness is called *toxic
waste*: anyone who knew s could forge proofs (evaluate the real
polynomials at s directly, or pick s to be a convenient root), and anyone
who knew α could fake the knowledge check. Our setup resamples s if
t(s) = 0, since at a root of t the check becomes vacuous (see
[qap.md](qap.md)).

Real deployments replace this single trusted party with a multi-party
computation ceremony (like the original Zcash ceremony): s and α are
built up as products of many participants' secrets, so no single party
ever knows them, and the ceremony needs only *one* honest participant to
be secure.

## 3. Blind evaluation

The prover knows the witness, hence the polynomials A, B, C and the
quotient h — but never learns s. Using only the CRS, they compute

```
E(P(s)) = ∏ᵢ E(sⁱ)^(pᵢ)
```

for each polynomial P with coefficients pᵢ. Check: ∏ᵢ (h^(sⁱ))^(pᵢ) =
h^(Σ pᵢsⁱ) = h^(P(s)) = E(P(s)). The prover evaluated P at s *without ever
seeing s*. This is the beautiful part.

The proof consists of these blind evaluations: E(A(s)), E(B(s)), E(C(s)),
E(H(s)), plus the alpha-shifted E(α·A(s)), E(α·B(s)), E(α·C(s)).

One subtlety: the prover commits only to the **private** part of A, B, C
(variables x, sym1, y, sym2). Variables 0 (`one`) and 5 (`out`) are
**public** — the verifier adds their contribution itself, homomorphically:
E_full = E_priv · h^(P_pub(s)). Because the verifier mixes in *its own*
public inputs, a proof only verifies against the statement the verifier
claims — you cannot take a proof of "x³+x+5=35" and pass it off as a
proof of "x³+x+5=36".

## 4. The KEA check: proving you know the polynomial

Blind evaluation has a hole: what stops the prover from sending arbitrary
group elements that happen to satisfy the final check, without knowing any
polynomial? The alpha-shift closes it.

The prover also sends E(α·A(s)) etc., computed from the shifted CRS the
same blind way. The verifier checks:

```
E(α·A(s)) == E(A(s))^α
```

The verifier knows α (it ran the setup), the prover does not. To produce
a consistent pair, the prover must have applied the *same* coefficients
to the shifted CRS as to the unshifted one — i.e., they must actually
know the polynomial they committed to. This is the **Knowledge-of-Exponent
Assumption (KEA)**: the only way to produce (E(x), E(α·x)) without knowing
α is to know x as a linear combination of the CRS elements.

KEA is a strong, non-falsifiable assumption — you cannot efficiently test
whether it is false even if it is. Stronger SNARKs (Groth16 and friends)
replace it with assumptions in pairing-friendly groups, but something in
this family is unavoidable: succinctness needs the prover bound to
*knowledge* of the witness, not just existence.

## 5. The final check

The verifier now holds E(A(s)), E(B(s)), E(C(s)), E(H(s)) (with public
inputs mixed in). It brute-forces the discrete logs to recover
a = A(s), b = B(s), c = C(s), hh = H(s) in F₂₅₇, and checks

```
a·b − c == hh·t(s)
```

If the QAP identity holds as polynomials, it holds at s. If it does not
hold as polynomials, the prover's only hope is that s is one of the ≤ 6
roots of the difference polynomial — the 2.3% Schwartz–Zippel gamble from
[qap.md](qap.md).

## What each attack looks like

The test suite in `src/snark.rs` demonstrates every failure mode:

- **Bad witness** → `prove` itself fails: R1CS unsatisfied, QAP division
  has a remainder. You cannot even construct the proof.
- **Wrong public input** (claim out = 36) → the verifier mixes in its own
  public values, the divisibility check fails. Proofs are bound to the
  claimed statement.
- **Tampered alpha-shift** → KEA check fails. You cannot fake knowledge
  of the polynomial.
- **Tampered H commitment** → divisibility check fails.

## Honest caveats, expanded

- **No zero-knowledge.** There is no blinding of the witness polynomials,
  and worse, the verifier brute-forces discrete logs — at toy size it can
  recover x itself. The demo shows the *mechanics* of proving (commitment,
  blind evaluation, verification); it does not hide like the real thing.
  Real SNARKs add random blinding factors to the witness polynomials so
  the proof leaks nothing.
- **Tiny field, 2.3% soundness error.** Real SNARKs use ~256-bit fields
  (e.g. the BN254 scalar field), making the Schwartz–Zippel error
  negligible.
- **No setup ceremony.** One party knows s and α. Real deployments use
  MPC ceremonies; some newer systems (STARKs, and SNARKs with universal
  setups like Plonk's) reduce or restructure this trust.
- **KEA is non-falsifiable.** Real systems use pairing-based knowledge
  assumptions, which are stronger-studied but in the same spirit.
