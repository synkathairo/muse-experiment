# Advection: semi-Lagrangian, stable, smudgy

Advection answers: "the fluid at this cell came from *where*?" Instead of
pushing particles forward (which scatters them and explodes), we trace
*backwards* from each grid cell along the velocity field and interpolate:

$$\mathbf{u}^{n+1}(\mathbf{x}) = \mathbf{u}^n(\mathbf{x} - \Delta t\,\mathbf{u}^n(\mathbf{x}))$$

The departure point $\mathbf{x} - \Delta t\,\mathbf{u}^n(\mathbf{x})$ is
almost never on a grid node, so we bilinearly interpolate the old field.
That is the entire scheme (`advect.rs`, ~40 lines).

## Why it's unconditionally stable

Every new value is a convex combination of old values — interpolation
weights are non-negative and sum to 1. The new field can never exceed the
old field's range, so it cannot blow up, *no matter how large dt is*.
The unit test `hundred_steps_at_large_dt_stay_finite` (dt=4.0, 100 steps)
exists to prove exactly this. Eulerian schemes (finite differences on
the $(\mathbf{u}\cdot\nabla)\mathbf{u}$ term directly) need dt small
enough that fluid crosses less than one cell per step (the CFL
condition); violate it and they explode.

## The price: numerical dissipation

Interpolation is averaging, and averaging destroys detail. Every step
smears the field slightly toward the local mean — fine vortices decay
into mush even with zero physical viscosity. After a few dozen frames a
crisp vortex looks like a blurry blob. This is the dominant visual
artifact of Stable Fluids, and the reason the next section exists.

## Vorticity confinement: the anti-smudge hack

Fedkiw, Stam & Jensen (2001) observed the dissipation kills exactly the
small, pretty vortices. Their fix re-injects swirl where the flow is
already swirling:

1. Compute scalar vorticity $\omega = \nabla\times\mathbf{u}$
   (in 2D: $\omega = \partial v/\partial x - \partial u/\partial y$).
2. Find where $|\omega|$ grows fastest:
   $\mathbf{N} = \nabla|\omega| / |\nabla|\omega||$ (points at vortex cores).
3. Apply $\mathbf{F} = \varepsilon\,(\mathbf{N}\times\omega)$ —
   a force tangential to the vortex, in the direction it's already
   spinning, amplifying it.

It is *positive feedback by design* — controlled energy injection. That
is why the gain must be tiny in grid units (default $\varepsilon=0.05$;
above ~0.3 the sim visibly "boils" as the feedback outruns the
projection). Crank the demo's vorticity slider and watch the max
divergence readout climb: you are watching a numerical hack fight the
incompressibility constraint in real time.

## Not conservative

Semi-Lagrangian advection does not conserve dye mass exactly: in
divergent flow it can create or destroy a few percent per frame (the flow
map's Jacobian is ignored). With the projection keeping divergence near
zero the drift is small; the tests bound it (strict conservation at zero
flow, ≤10% under translation near walls). A fully conservative scheme
would need flux-form advection — more code, less stability.
