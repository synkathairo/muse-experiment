# R1CS: the assembly language of zero-knowledge proofs

R1CS stands for **Rank-1 Constraint System**. It is the standard way to
express "this computation was done correctly" as a pile of simple equations
that a proof system can work with. Compilers like circom turn programs into
R1CS; proof systems turn R1CS into QAPs and then into SNARKs. If you
understand this file, you understand the front half of every zk-SNARK.

## The one idea

Take any computation and *flatten* it into individual multiplication gates.
Each gate becomes one constraint of the form

```
(stuff) × (stuff) = (stuff)
```

where each "stuff" is a **linear combination** of your variables. That is
the whole definition. Every constraint has exactly one multiplication —
hence "rank 1".

Addition is free: it never creates a constraint, it just folds into the
linear combinations. Only multiplication costs a constraint. This is why
zk circuit designers count multiplications the way normal programmers
count loop iterations.

## Worked example: x³ + x + 5 = 35

We want to prove knowledge of x with x³ + x + 5 = 35 (answer: x = 3, kept
secret). Flatten the computation into gates, introducing intermediate
variables:

```
sym1 = x * x          # gate 1
y    = sym1 * x       # gate 2
sym2 = y + x          # gate 3 (addition — free, folded into the constraint)
out  = sym2 + 5       # gate 4 (addition — free)
```

Each gate becomes one constraint `(A·s) · (B·s) = (C·s)`, where
`s = [one, x, sym1, y, sym2, out]` is the solution vector:

| gate | constraint |
|---|---|
| `sym1 = x·x` | `(x) · (x) = (sym1)` |
| `y = sym1·x` | `(sym1) · (x) = (y)` |
| `sym2 = y + x` | `(y + x) · (1) = (sym2)` |
| `out = sym2 + 5` | `(sym2 + 5) · (1) = (out)` |

Note gates 3 and 4: the additions `y + x` and `sym2 + 5` live *inside* the
left-hand linear combination, multiplied by the constant `1`. No new
multiplication, no new constraint.

## The matrices

Collect the coefficients of each linear combination into three matrices
A, B, C (one row per constraint, one column per variable). From
`src/r1cs.rs`:

```
s = [one, x, sym1, y, sym2, out]

A = [[0,1,0,0,0,0],   B = [[0,1,0,0,0,0],   C = [[0,0,1,0,0,0],
     [0,0,1,0,0,0],        [0,1,0,0,0,0],        [0,0,0,1,0,0],
     [0,1,0,1,0,0],        [1,0,0,0,0,0],        [0,0,0,0,1,0],
     [5,0,0,0,1,0]]        [1,0,0,0,0,0]]        [0,0,0,0,0,1]]
```

Read row 3 of A as "the left factor of constraint 3 is 0·one + 1·x + 0·sym1
+ 1·y + 0·sym2 + 0·out", i.e. `y + x`. Row 4 of A is `5·one + sym2`, i.e.
`sym2 + 5` — constants enter through the `one` variable (column 0).

## The witness check

A computation is correct **iff** there exists a vector s with

```
(A·s) × (B·s) = (C·s)     for every row
```

For the honest witness x = 3: `s = [1, 3, 9, 27, 30, 35]`, and row by row:

- row 1: 3 × 3 = 9 ✓
- row 2: 9 × 3 = 27 ✓
- row 3: (3 + 27) × 1 = 30 ✓
- row 4: (30 + 5) × 1 = 35 ✓

Lie about anything — set `out = 36`, or claim `x = 4` while keeping the
rest — and at least one row fails. The R1CS checks *internal consistency*;
it does not by itself know which x you used. That is why the protocol
later designates `one` and `out` as **public** variables: the verifier
supplies `(one = 1, out = 35)` itself, so a proof only verifies against
the claimed statement.

## Why this shape?

R1CS is deliberately minimal: one multiplication per constraint, linear
combinations everywhere else. That minimalism is what makes the next step
(the QAP transformation) possible — each column of these matrices becomes
a polynomial, and "all constraints hold" becomes a single statement about
polynomial divisibility. See [qap.md](qap.md).
