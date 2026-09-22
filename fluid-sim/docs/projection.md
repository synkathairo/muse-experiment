# Projection: the incompressibility tax

After advection the velocity field $\mathbf{u}^*$ has nonzero divergence
— numerically, the fluid has compressed. The projection decomposes it
(Helmholtz):

$$\mathbf{u}^* = \mathbf{u} + \nabla p, \qquad \nabla\cdot\mathbf{u} = 0$$

Take the divergence of both sides:

$$\nabla^2 p = \nabla\cdot\mathbf{u}^*$$

Solve this Poisson equation for $p$, subtract $\nabla p$, and what
remains is divergence-free — because
$\nabla\cdot(\mathbf{u}^* - \nabla p) = \nabla\cdot\mathbf{u}^* -
\nabla^2 p = 0$. The pressure here isn't physical pressure; it's the
Lagrange multiplier that enforces the constraint. (`project.rs`.)

## Discretization

On the grid, with spacing $h=1$:

- Divergence (central differences):
  $\mathrm{div} = \tfrac12\big[(u_R-u_L) + (v_T-v_B)\big]$
- Poisson, compact 5-point Laplacian, solved by **Jacobi iteration**:
  $p^{k+1}_C = \tfrac14\big(p_L+p_R+p_T+p_B - \mathrm{div}\big)$
- Gradient subtraction:
  $\mathbf{u} \mathrel{-}= \tfrac12\big(p_R-p_L,\; p_T-p_B\big)$

Jacobi is the simplest iterative solver: replace each cell by the
average of its neighbors (minus the source). It converges slowly —
error decays ~0.99 per iteration for the lowest mode on a 32² grid —
but each iteration is trivially parallel and the demo only needs the
*smooth per-frame residual* gone, not machine precision. 30 iterations
is the default; the slider goes 1–100 so you can feel the cost/quality
tradeoff directly.

## Boundary conditions

- Domain walls: no-slip ($\mathbf{u}=0$) and $\partial p/\partial n = 0$
  (Neumann) — implemented by substituting the center pressure for any
  blocked neighbor, so no pressure gradient leaks through walls.
- Solids: same Neumann treatment; velocity inside the mask is
  overwritten with the body velocity before *and* after the solve.

## The honest footnote: odd-even decoupling

This crate uses a **collocated** grid (u, v, p all at cell centers).
There is a subtle inconsistency: the compact Laplacian in the Poisson
solve is not the exact algebraic inverse of the central-difference
divergence/gradient pair (that would need the stride-2 "wide" stencil).
Consequence: the measured divergence floors around ~0.02 instead of true
zero, and pressure can develop mild checkerboard modes. The classic fix
is a **staggered MAC grid** (velocities on faces, pressure at centers —
what Stam's original paper actually used), where the operators compose
exactly. We kept collocated for readability; the residual is invisible
and the demo's divergence readout shows the true number, floor included.
