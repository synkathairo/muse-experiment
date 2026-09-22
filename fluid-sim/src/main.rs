//! Native smoke demo: run the solver headless and print vital signs.
//!
//! `cargo run -p fluid-sim` — no window, just numbers. Useful as a sanity
//! check that the WASM build will behave: same crate, same code path.

use fluid_sim::{Fluid, SimParams, HULL_POLY};

fn main() {
    let mut f = Fluid::new(128);
    // A few dye splats and a cruising boat hull.
    f.splat(40.0, 64.0, 6.0, 2.0, 1.0, 0.2, 0.1);
    f.splat(90.0, 64.0, -6.0, -2.0, 0.1, 0.4, 1.0);
    let params = SimParams::default();
    for step in 0..120 {
        // Boat cruises right, bobbing slightly.
        let t = step as f32;
        let bx = 20.0 + t * 0.5;
        let by = 64.0 + (t * 0.1).sin() * 8.0;
        f.set_hull(&HULL_POLY, (bx, by), 5.0, (0.5, (t * 0.1).cos() * 0.8));
        f.step(params.dt, &params);
        if step % 30 == 0 {
            println!(
                "step {step:3}: max_div={:.4} dye_mass={:.1} finite={}",
                f.max_divergence(),
                f.dye_mass(),
                !f.has_nonfinite()
            );
        }
    }
    println!("done. finite={}", !f.has_nonfinite());
}
