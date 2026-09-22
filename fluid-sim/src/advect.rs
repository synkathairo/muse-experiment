//! Semi-Lagrangian advection (Stam 1999).
//!
//! For each grid cell we trace *backwards* along the velocity field:
//!
//! ```text
//! field^{n+1}(x) = field^n(x - dt * u^n(x))
//! ```
//!
//! The departure point is almost never on a grid node, so we bilinearly
//! interpolate. Because we always sample inside the old field (never
//! extrapolate), the scheme is unconditionally stable — the price is
//! numerical dissipation, which smears out fine vortices over time.

use crate::Fluid;

/// Bilinearly sample `field` at continuous coordinates (x, y), clamping to
/// the domain edges (zero-gradient boundary for the sampled quantity).
pub fn sample(n: usize, field: &[f32], x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (n - 1) as f32);
    let y = y.clamp(0.0, (n - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(n - 1);
    let y1 = (y0 + 1).min(n - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let w = n;
    field[y0 * w + x0] * (1.0 - fx) * (1.0 - fy)
        + field[y0 * w + x1] * fx * (1.0 - fy)
        + field[y1 * w + x0] * (1.0 - fx) * fy
        + field[y1 * w + x1] * fx * fy
}

/// Advect one scalar field through (u, v) into `out`.
fn advect_field(n: usize, dt: f32, u: &[f32], v: &[f32], field: &[f32], out: &mut [f32]) {
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            let sx = x as f32 - dt * u[i];
            let sy = y as f32 - dt * v[i];
            out[i] = sample(n, field, sx, sy);
        }
    }
}

impl Fluid {
    /// Advect velocity and dye through the velocity field.
    ///
    /// Every component advects through the *old* velocity, so u/v are
    /// snapshotted into the scratch buffers first; disjoint field borrows
    /// keep this allocation-free after construction.
    pub(crate) fn advect_all(&mut self, dt: f32) {
        let n = self.n;
        self.buf_a.copy_from_slice(&self.u);
        self.buf_b.copy_from_slice(&self.v);

        advect_field(
            n,
            dt,
            &self.buf_a,
            &self.buf_b,
            &self.buf_a,
            &mut self.scratch,
        );
        std::mem::swap(&mut self.u, &mut self.scratch);

        advect_field(
            n,
            dt,
            &self.buf_a,
            &self.buf_b,
            &self.buf_b,
            &mut self.scratch,
        );
        std::mem::swap(&mut self.v, &mut self.scratch);

        advect_field(
            n,
            dt,
            &self.buf_a,
            &self.buf_b,
            &self.dye_r,
            &mut self.scratch,
        );
        std::mem::swap(&mut self.dye_r, &mut self.scratch);

        advect_field(
            n,
            dt,
            &self.buf_a,
            &self.buf_b,
            &self.dye_g,
            &mut self.scratch,
        );
        std::mem::swap(&mut self.dye_g, &mut self.scratch);

        advect_field(
            n,
            dt,
            &self.buf_a,
            &self.buf_b,
            &self.dye_b,
            &mut self.scratch,
        );
        std::mem::swap(&mut self.dye_b, &mut self.scratch);
    }
}
