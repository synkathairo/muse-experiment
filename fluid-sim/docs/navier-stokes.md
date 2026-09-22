# The Navier-Stokes equations (and what we actually solve)

The incompressible Navier-Stokes equations describe the velocity field
**u**(x, t) of a fluid with constant density:

$$\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u}\cdot\nabla)\mathbf{u} -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}$$

$$\nabla\cdot\mathbf{u} = 0$$

Term by term:

- $-(\mathbf{u}\cdot\nabla)\mathbf{u}$ — *advection*: the fluid carries
  itself along. This is the nonlinear term and the source of all the
  beautiful chaos. Handled in `advect.rs`.
- $-\frac{1}{\rho}\nabla p$ — *pressure*: the force that stops the
  fluid from piling up. We never simulate pressure as a physical
  quantity; we *solve* for whatever pressure field makes the velocity
  divergence-free. Handled in `project.rs`.
- $\nu\nabla^2\mathbf{u}$ — *viscosity*: momentum diffusion.
  $\nu = 0$ is an ideal (Euler) fluid; large $\nu$ is honey. A tiny
  explicit diffusion step in `project.rs`.
- $\mathbf{f}$ — external forces: your mouse, the boat hull,
  vorticity confinement.
- $\nabla\cdot\mathbf{u} = 0$ — *incompressibility*: the velocity
  field has no sources or sinks. This is not evolved — it is
  **enforced**, every frame, by the projection. It is the entire reason
  the sim looks like liquid instead of smoke drifting through a vacuum.

## Operator splitting

Solving everything at once is hard. Stam's insight: split the PDE into
pieces, solve each piece with the best tool for that piece, in sequence:

```
u*  =  add forces (f, viscosity, confinement)
u** =  advect u* through itself        (semi-Lagrangian)
u*** = project u** to divergence-free  (Poisson solve)
```

Each piece is simple and stable; the splitting error is O(dt), invisible
in a toy. This is the same idea as splitting in quantum simulation
(Trotter) or graphics (position-based dynamics): solve the easy
subproblems, alternate fast.

## What "incompressible" buys you

Without the projection step, advected velocity develops divergence and
the dye behaves like a compressible gas — blobs that should swirl
instead pulse and spread. The demo's divergence view and the live
"max divergence" readout exist so you can watch the projection earn its
keep: drag the Jacobi-iterations slider to 4 and see incompressibility
fall apart.
