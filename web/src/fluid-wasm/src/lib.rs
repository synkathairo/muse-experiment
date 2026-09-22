//! WASM bindings for the fluid-sim demo.
//!
//! Thin stateful boundary over the `fluid-sim` crate: the page owns one
//! `Fluid` behind a static Mutex and drives it per frame:
//!   1. create(n)            -> fresh n×n sim
//!   2. set_params(...)      -> viscosity, vorticity, dye fade, Jacobi iters
//!   3. splat(x,y,dx,dy,r,g,b) -> inject velocity + dye (mouse drags)
//!   4. move_boat(x,y,angle,vx,vy) -> re-rasterize the hull mask (drag/steer the boat)
//!   5. step()               -> advance one frame
//!   6. read dye/velocity/divergence via raw pointers into WASM memory
//! All numerics live in fluid-sim; nothing is computed here.

use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use fluid_sim::{Fluid, SimParams, HULL_POLY};

/// Boat hull scale: the unit polygon is ~3.2 long; at 128² this makes a
/// ~16-cell boat.
const BOAT_SCALE: f32 = 5.0;

struct State {
    fluid: Fluid,
    params: SimParams,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap();
    let st = guard
        .as_mut()
        .expect("fluid-wasm: create(n) must be called first");
    f(st)
}

/// Create a fresh n×n simulation with default parameters.
#[wasm_bindgen]
pub fn create(n: usize) {
    let n = n.clamp(32, 256);
    *STATE.lock().unwrap() = Some(State {
        fluid: Fluid::new(n),
        params: SimParams::default(),
    });
}

/// Advance the simulation one frame (dt = params.dt, default 1).
#[wasm_bindgen]
pub fn step() {
    with_state(|st| {
        let dt = st.params.dt;
        st.fluid.step(dt, &st.params)
    });
}

/// Inject velocity (dx, dy) and dye color (r,g,b ∈ 0..1) at grid coords.
#[wasm_bindgen]
pub fn splat(x: f32, y: f32, dx: f32, dy: f32, r: f32, g: f32, b: f32) {
    with_state(|st| st.fluid.splat(x, y, dx, dy, r, g, b));
}

/// Set simulation parameters. Vorticity is passed in slider units
/// (0..40) and mapped to the solver's epsilon (÷1000 → 0..0.4).
#[wasm_bindgen]
pub fn set_params(viscosity: f32, vorticity_slider: f32, dye_fade: f32, jacobi_iters: u32) {
    with_state(|st| {
        st.params.viscosity = viscosity.clamp(0.0, 0.5);
        st.params.vorticity_confinement = (vorticity_slider / 100.0).clamp(0.0, 0.4);
        st.params.dye_dissipation = dye_fade.clamp(0.0, 0.05);
        st.params.jacobi_iters = jacobi_iters.clamp(1, 100);
        st.params.dt = 1.0;
    });
}

/// Move the boat hull to (x, y) with heading `angle` (radians, 0 = bow
/// pointing +x) and body velocity (vx, vy) in cells/frame.
/// The solver clamps against tunneling internally.
#[wasm_bindgen]
pub fn move_boat(x: f32, y: f32, angle: f32, vx: f32, vy: f32) {
    let (s, c) = angle.sin_cos();
    let rotated: Vec<(f32, f32)> = HULL_POLY
        .iter()
        .map(|&(px, py)| (px * c - py * s, px * s + py * c))
        .collect();
    with_state(|st| st.fluid.set_hull(&rotated, (x, y), BOAT_SCALE, (vx, vy)));
}

/// Remove the boat from the domain.
#[wasm_bindgen]
pub fn clear_boat() {
    with_state(|st| st.fluid.clear_solid());
}

/// Maximum |divergence| — the "incompressibility meter" shown live.
#[wasm_bindgen]
pub fn max_divergence() -> f32 {
    with_state(|st| st.fluid.max_divergence())
}

/// Grid size n (domain is n×n).
#[wasm_bindgen]
pub fn grid_n() -> usize {
    with_state(|st| st.fluid.n())
}

// -- raw buffer views (read by JS each frame; no allocation here) --------

/// Dye channels are planar in the crate: three n*n f32 buffers.
/// JS strides the three pointers into one ImageData.
#[wasm_bindgen]
pub fn dye_r_ptr() -> *const f32 {
    with_state(|st| st.fluid.dye().0.as_ptr())
}
#[wasm_bindgen]
pub fn dye_g_ptr() -> *const f32 {
    with_state(|st| st.fluid.dye().1.as_ptr())
}
#[wasm_bindgen]
pub fn dye_b_ptr() -> *const f32 {
    with_state(|st| st.fluid.dye().2.as_ptr())
}

/// Pointer to the u (x-velocity) buffer: n*n f32. v follows the same
/// layout; JS reads both.
#[wasm_bindgen]
pub fn vel_u_ptr() -> *const f32 {
    with_state(|st| st.fluid.velocity().0.as_ptr())
}
#[wasm_bindgen]
pub fn vel_v_ptr() -> *const f32 {
    with_state(|st| st.fluid.velocity().1.as_ptr())
}

/// Pointer to the divergence buffer: n*n f32 (recomputed each step).
#[wasm_bindgen]
pub fn div_ptr() -> *const f32 {
    with_state(|st| st.fluid.divergence().as_ptr())
}

/// Pointer to the solid mask: n*n bytes (0/1).
#[wasm_bindgen]
pub fn solid_ptr() -> *const u8 {
    with_state(|st| st.fluid.solid_mask().as_ptr() as *const u8)
}
