# fluid-sim

A from-scratch educational 2D incompressible fluid solver — Jos Stam's
"Stable Fluids" (1999) on a collocated n×n grid, in pure Rust with zero
dependencies. The same crate drives the
[browser demo](../web/site/demos/fluid/index.html) (via
`web/src/fluid-wasm`), the native smoke demo (`cargo run`), and the unit
tests.

## The method, in one paragraph

Each frame: prescribe solid/wall velocities → apply viscosity and
vorticity confinement as forces → advect velocity and dye backwards
through the velocity field (semi-Lagrangian, unconditionally stable) →
fade the dye → **project**: solve ∇²p = ∇·u for pressure with Jacobi
iteration and subtract ∇p to enforce incompressibility → re-prescribe
solids. The projection is the expensive step and the whole reason the
fluid looks like a fluid instead of a compressible gas.

## Layout

- `src/lib.rs` — `Fluid` struct, `step()`, `splat()`, buffer getters, tests
- `src/advect.rs` — semi-Lagrangian advection with bilinear interpolation
- `src/project.rs` — divergence, Jacobi pressure solve, gradient
  subtraction, viscosity diffusion, vorticity confinement
- `src/solid.rs` — rasterized solid masks (circle, boat-hull polygon),
  moving-body velocity with anti-tunneling clamp
- `src/params.rs` — `SimParams`
- `src/main.rs` — headless native demo (`cargo run`)
- `docs/` — four one-sitting explainers (see below)

## Quick start

```rust
use fluid_sim::{Fluid, SimParams, HULL_POLY};

let mut f = Fluid::new(128);
let params = SimParams::default();
// stir:
f.splat(64.0, 64.0, 5.0, 2.0, 1.0, 0.3, 0.1);
// drop in the boat, cruising right:
f.set_hull(&HULL_POLY, (40.0, 64.0), 5.0, (0.8, 0.0));
for _ in 0..60 {
    f.step(params.dt, &params);
}
assert!(f.max_divergence() < 0.5);
```

## Tests

`cargo test` — 6 tests, all must pass before any push (repo rule):

| test | what it proves |
|---|---|
| `projection_kills_divergence` | 3 defect-correction passes cut max divergence 10×+, to < 0.03 |
| `dye_mass_conserved_without_dissipation_or_flow` | zero flow + zero dissipation ⇒ bit-identical dye |
| `dye_mass_conserved_under_uniform_advection` | uniform translation conserves mass within 10% (scheme isn't exactly conservative near divergence sources) |
| `solid_cells_match_body_velocity_after_move` | moving circle/hull cells carry the body velocity through a step |
| `hundred_steps_at_large_dt_stay_finite` | dt=4.0, 100 steps: no NaN/inf (semi-Lagrangian stability) |
| `body_velocity_clamped_against_tunneling` | absurd drag speeds clamp to 1.5 cells/step |

## Known limitations (documented, not hidden)

- **Collocated grid, odd-even decoupling.** The compact 5-point Laplacian
  in the pressure solve doesn't exactly cancel the central-difference
  divergence (that would need the wide stride-2 stencil), so the measured
  divergence floors around ~0.02 instead of true zero. A staggered MAC
  grid would fix it; see `docs/projection.md`.
- **Vorticity confinement is positive feedback by design.** In grid units
  the gain must stay small (default 0.05, useful up to ~0.3); higher values
  visibly "boil". See `docs/advection.md`.
- **Semi-Lagrangian advection dissipates.** Fine vortices smear over
  time — the price of unconditional stability.

## Further reading

Each piece has a deeper explainer, each readable in one sitting:

- [docs/navier-stokes.md](docs/navier-stokes.md) — the equations, what
  incompressibility means, operator splitting
- [docs/advection.md](docs/advection.md) — semi-Lagrangian advection,
  why it's stable, what it dissipates, vorticity confinement
- [docs/projection.md](docs/projection.md) — Helmholtz decomposition,
  the Poisson solve, Jacobi, collocated vs staggered grids
- [docs/moving-solids.md](docs/moving-solids.md) — rasterized masks,
  no-slip in the body frame, anti-tunneling
