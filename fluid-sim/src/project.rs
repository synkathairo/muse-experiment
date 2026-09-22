//! Pressure projection: enforcing incompressibility.
//!
//! After advection the velocity field has nonzero divergence — the fluid
//! appears to compress. The fix is a Helmholtz decomposition: any vector
//! field splits into a divergence-free part plus the gradient of a scalar.
//! Solve the Poisson equation for pressure
//!
//! ```text
//! laplacian(p) = divergence(u*)
//! ```
//!
//! then subtract the pressure gradient:
//!
//! ```text
//! u = u* - grad(p)
//! ```
//!
//! Since `div(grad(p)) = laplacian(p)`, the result is divergence-free.
//! The Poisson solve is the expensive step: we use Jacobi iteration
//! (simple, parallel-friendly, slow to converge — the demo lets you watch
//! what too few iterations does to the sim).
//!
//! Also here: explicit viscosity diffusion and vorticity confinement, both
//! applied as forces before the projection.

use crate::Fluid;

impl Fluid {
    /// Central-difference divergence of (u, v). Solid cells report 0.
    pub(crate) fn compute_divergence(&mut self) {
        let n = self.n;
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                if self.solid[i] {
                    self.divergence[i] = 0.0;
                    continue;
                }
                let xl = if x == 0 { 0 } else { x - 1 };
                let xr = (x + 1).min(n - 1);
                let yd = if y == 0 { 0 } else { y - 1 };
                let yu = (y + 1).min(n - 1);
                self.divergence[i] = 0.5
                    * ((self.u[y * n + xr] - self.u[y * n + xl])
                        + (self.v[yu * n + x] - self.v[yd * n + x]));
            }
        }
    }

    /// Jacobi iteration for `laplacian(p) = divergence`.
    ///
    /// Solid cells and the domain wall use a Neumann condition
    /// (dp/dn = 0): a blocked neighbor is replaced by the center value,
    /// so no pressure gradient leaks through walls.
    pub(crate) fn solve_pressure(&mut self, iters: u32) {
        let n = self.n;
        self.pressure.fill(0.0);
        for _ in 0..iters {
            for y in 0..n {
                for x in 0..n {
                    let i = y * n + x;
                    if self.solid[i] {
                        self.pressure2[i] = 0.0;
                        continue;
                    }
                    // Neighbor pressure, or center value at blocked faces.
                    let p_at = |nx: isize, ny: isize, this: &Fluid| -> f32 {
                        if nx < 0 || ny < 0 || nx >= n as isize || ny >= n as isize {
                            return this.pressure[i];
                        }
                        let j = ny as usize * n + nx as usize;
                        if this.solid[j] {
                            this.pressure[i]
                        } else {
                            this.pressure[j]
                        }
                    };
                    let x = x as isize;
                    let y = y as isize;
                    self.pressure2[i] = 0.25
                        * (p_at(x - 1, y, self)
                            + p_at(x + 1, y, self)
                            + p_at(x, y - 1, self)
                            + p_at(x, y + 1, self)
                            - self.divergence[i]);
                }
            }
            std::mem::swap(&mut self.pressure, &mut self.pressure2);
        }
    }

    /// Subtract the pressure gradient from velocity.
    pub(crate) fn subtract_pressure_gradient(&mut self) {
        let n = self.n;
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                if self.solid[i] {
                    continue;
                }
                let xl = if x == 0 { 0 } else { x - 1 };
                let xr = (x + 1).min(n - 1);
                let yd = if y == 0 { 0 } else { y - 1 };
                let yu = (y + 1).min(n - 1);
                self.u[i] -= 0.5 * (self.pressure[y * n + xr] - self.pressure[y * n + xl]);
                self.v[i] -= 0.5 * (self.pressure[yu * n + x] - self.pressure[yd * n + x]);
            }
        }
    }

    /// Full projection: divergence -> pressure -> gradient subtraction.
    /// Divergence is recomputed at the end so `max_divergence()` reflects
    /// the projected (near-incompressible) field — the demo shows it live.
    pub(crate) fn project(&mut self, iters: u32) {
        self.compute_divergence();
        self.solve_pressure(iters);
        self.subtract_pressure_gradient();
        self.compute_divergence();
    }

    /// Explicit viscosity diffusion: `u += nu*dt*laplacian(u)`.
    /// Clamped to `nu*dt <= 0.2` for explicit stability (2D limit is 0.25).
    /// Allocation-free: each component diffuses into a scratch buffer.
    pub(crate) fn diffuse(&mut self, viscosity: f32, dt: f32) {
        let k = (viscosity * dt).min(0.2);
        if k <= 0.0 {
            return;
        }
        let n = self.n;
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                let xl = if x == 0 { 0 } else { x - 1 };
                let xr = (x + 1).min(n - 1);
                let yd = if y == 0 { 0 } else { y - 1 };
                let yu = (y + 1).min(n - 1);
                let lap_u = self.u[y * n + xl]
                    + self.u[y * n + xr]
                    + self.u[yd * n + x]
                    + self.u[yu * n + x]
                    - 4.0 * self.u[i];
                let lap_v = self.v[y * n + xl]
                    + self.v[y * n + xr]
                    + self.v[yd * n + x]
                    + self.v[yu * n + x]
                    - 4.0 * self.v[i];
                self.buf_a[i] = self.u[i] + k * lap_u;
                self.buf_b[i] = self.v[i] + k * lap_v;
            }
        }
        std::mem::swap(&mut self.u, &mut self.buf_a);
        std::mem::swap(&mut self.v, &mut self.buf_b);
    }

    /// Vorticity confinement (Fedkiw et al.): re-injects the small-scale
    /// swirl that semi-Lagrangian advection dissipates.
    ///
    /// curl = dv/dx - du/dy; N = grad(|curl|) / |grad(|curl|)|;
    /// force = eps * dt * |curl| * (Ny, -Nx).
    pub(crate) fn confine_vorticity(&mut self, eps: f32, dt: f32) {
        if eps <= 0.0 {
            return;
        }
        let n = self.n;
        // curl into scratch
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                let xl = if x == 0 { 0 } else { x - 1 };
                let xr = (x + 1).min(n - 1);
                let yd = if y == 0 { 0 } else { y - 1 };
                let yu = (y + 1).min(n - 1);
                self.scratch[i] = 0.5
                    * ((self.v[y * n + xr] - self.v[y * n + xl])
                        - (self.u[yu * n + x] - self.u[yd * n + x]));
            }
        }

        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let i = y * n + x;
                if self.solid[i] {
                    continue;
                }
                let gx = 0.5 * (self.scratch[i + 1].abs() - self.scratch[i - 1].abs());
                let gy = 0.5 * (self.scratch[i + n].abs() - self.scratch[i - n].abs());
                let len = (gx * gx + gy * gy).sqrt().max(1e-6);
                let (nx, ny) = (gx / len, gy / len);
                let c = self.scratch[i];
                let f = eps * dt * c;
                self.u[i] += f * ny;
                self.v[i] += f * -nx;
            }
        }
    }
}
