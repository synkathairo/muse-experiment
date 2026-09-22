# Moving solids: rasterized masks, no-slip in the body's frame

Obstacles are a boolean mask over the grid (`Fluid::solid`). Each frame:

1. **Rasterize** the body at its current position (`set_circle` /
   `set_hull` in `solid.rs`). Circles test $dx^2+dy^2 \le r^2$;
   polygons use the even-odd rule, sampled at cell centers. The mask is
   rebuilt from scratch every call, so moving bodies never leave stale
   cells behind.
2. **Prescribe velocity**: every solid cell's velocity is overwritten
   with the body's velocity — no-slip in the body's frame. This happens
   *before* advection (so the body stirs the fluid) and *after* the
   projection (so the solve can't soften the body).
3. **Pressure sees walls**: in the Jacobi solve, solid neighbors are
   treated with $\partial p/\partial n = 0$ (center value substituted),
   so pressure doesn't leak through the hull.

## Where the body velocity comes from

The demo tracks the pointer: each frame,
$\mathbf{v}_{body} = (\mathbf{x}_{new} - \mathbf{x}_{old}) / dt$.
The fluid doesn't know about mice — it just sees a mask with a
prescribed velocity.

## Anti-tunneling clamp

If the body jumps more than ~1 cell per frame, fluid leaks through the
gap between the old and new mask positions. `clamp_body_velocity`
caps displacement at `MAX_BODY_CELLS_PER_STEP = 1.5` cells/step,
*inside the solver* (`solid.rs`), not in the UI — the invariant lives
with the numerics. Fast drags get gracefully slowed rather than
glitching through the fluid. (A substepped mask sweep would be the
fancier fix; clamping is the honest toy version.)

## What you get for free

Drag the hull and the wake forms behind it: the body shoves fluid aside
at the bow (divergence source, killed by the projection into a pressure
bump) and drags fluid in its wake (shear → vorticity → the confinement
term keeps the eddies alive). Von Kármán-style shedding emerges without
any special casing — it's just Navier-Stokes with a moving wall.
