//! Simulation parameters for one step of the fluid solver.
//!
//! All quantities are in grid units: velocities are cells per frame,
//! `dt` is in frames (default 1.0 at 60fps). Keeping the units grid-native
//! means the demo page never has to convert between pixels and physics.

/// Tunables for [`crate::Fluid::step`].
#[derive(Debug, Clone, Copy)]
pub struct SimParams {
    /// Explicit viscosity coefficient. The diffusion update is
    /// `u += viscosity * dt * laplacian(u)`; stability requires
    /// `viscosity * dt <= 0.25` in 2D, so the step clamps it to 0.2.
    /// 0 = inviscid (smoke), ~0.1 = watery, higher = honey.
    pub viscosity: f32,
    /// Vorticity confinement strength (Fedkiw et al.). Counteracts the
    /// numerical dissipation of semi-Lagrangian advection by re-injecting
    /// small-scale swirl. It is a positive-feedback force, so the gain is
    /// small in grid units: 0 = off, 0.05 = lively but honest
    /// (max divergence stays ~0.03), 0.3+ = visibly "boily".
    pub vorticity_confinement: f32,
    /// Dye fade per frame, 0 = dye never fades (mass conserved).
    pub dye_dissipation: f32,
    /// Jacobi iterations for the pressure Poisson solve. More iterations =
    /// more incompressible = crisper vortices, at linear cost.
    pub jacobi_iters: u32,
    /// Timestep in frames. Semi-Lagrangian advection is unconditionally
    /// stable, so large dt won't explode — it just gets mushy.
    pub dt: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            viscosity: 0.0,
            vorticity_confinement: 0.05,
            dye_dissipation: 0.002,
            jacobi_iters: 30,
            dt: 1.0,
        }
    }
}
