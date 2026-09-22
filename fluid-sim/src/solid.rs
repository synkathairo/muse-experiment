//! Solid obstacles: rasterized masks with moving-body velocity.
//!
//! A solid is a set of grid cells flagged in `Fluid::solid`. Each step,
//! fluid velocity inside the mask is overwritten with the body's velocity
//! (no-slip in the body's frame), and the pressure solve treats solids as
//! Neumann boundaries (see [`crate::project`]).
//!
//! Moving bodies: the caller re-rasterizes the mask each frame at the new
//! position and passes the body's velocity. To avoid tunneling (the mask
//! jumping over cells between frames), per-frame displacement is clamped
//! to [`MAX_BODY_CELLS_PER_STEP`] cells here in the solver, not in the UI.

use crate::Fluid;

/// Maximum distance a solid may travel in one step, in grid cells.
/// Larger jumps would let fluid leak through the body between frames.
pub const MAX_BODY_CELLS_PER_STEP: f32 = 1.5;

/// A little boat hull, bow pointing +x, in arbitrary units. The demo page
/// scales it to ~14 cells long.
pub const HULL_POLY: [(f32, f32); 7] = [
    (1.6, 0.0), // bow tip
    (1.0, 0.45),
    (-0.6, 0.7),
    (-1.5, 0.5), // stern
    (-1.5, -0.5),
    (-0.6, -0.7),
    (1.0, -0.45),
];

/// Clamp a body velocity so the displacement per step stays within
/// [`MAX_BODY_CELLS_PER_STEP`] cells (anti-tunneling).
pub fn clamp_body_velocity(vx: f32, vy: f32, dt: f32) -> (f32, f32) {
    let max_v = MAX_BODY_CELLS_PER_STEP / dt.max(1e-6);
    let speed = (vx * vx + vy * vy).sqrt();
    if speed > max_v {
        (vx / speed * max_v, vy / speed * max_v)
    } else {
        (vx, vy)
    }
}

/// Even-odd point-in-polygon test for `poly` (list of vertices, unclosed).
fn point_in_poly(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let m = poly.len();
    let mut j = m - 1;
    for i in 0..m {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Fluid {
    /// Clear all solid cells (body velocity reset to zero).
    pub fn clear_solid(&mut self) {
        self.solid.fill(false);
        self.body_vx = 0.0;
        self.body_vy = 0.0;
    }

    /// Rasterize a filled circle as a solid with the given body velocity.
    pub fn set_circle(&mut self, cx: f32, cy: f32, r: f32, vx: f32, vy: f32) {
        let (vx, vy) = clamp_body_velocity(vx, vy, 1.0);
        self.body_vx = vx;
        self.body_vy = vy;
        let n = self.n;
        // Fresh mask each call: the caller re-rasterizes moving bodies
        // every frame, so stale cells from the old position must go.
        self.solid.fill(false);
        let x0 = (cx - r).floor().max(0.0) as usize;
        let x1 = ((cx + r).ceil() as usize).min(n - 1);
        let y0 = (cy - r).floor().max(0.0) as usize;
        let y1 = ((cy + r).ceil() as usize).min(n - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                self.solid[y * n + x] = dx * dx + dy * dy <= r * r;
            }
        }
    }

    /// Rasterize a polygon (given in local coords) translated to `pos`
    /// and scaled by `scale`, as a solid with the given body velocity.
    pub fn set_hull(&mut self, poly: &[(f32, f32)], pos: (f32, f32), scale: f32, vel: (f32, f32)) {
        let (vx, vy) = clamp_body_velocity(vel.0, vel.1, 1.0);
        self.body_vx = vx;
        self.body_vy = vy;
        let n = self.n;
        // Fresh mask each call (see set_circle).
        self.solid.fill(false);
        // World-space polygon for the point-in-poly test.
        let world: Vec<(f32, f32)> = poly
            .iter()
            .map(|(px, py)| (pos.0 + px * scale, pos.1 + py * scale))
            .collect();
        // Conservative bounding box.
        let (mut x0, mut y0) = (n as f32, n as f32);
        let (mut x1, mut y1) = (0.0f32, 0.0f32);
        for (wx, wy) in &world {
            x0 = x0.min(*wx);
            y0 = y0.min(*wy);
            x1 = x1.max(*wx);
            y1 = y1.max(*wy);
        }
        let x0 = (x0.floor().max(0.0) as usize).min(n - 1);
        let x1 = (x1.ceil().max(0.0) as usize).min(n - 1);
        let y0 = (y0.floor().max(0.0) as usize).min(n - 1);
        let y1 = (y1.ceil().max(0.0) as usize).min(n - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                // Sample at cell center for a cleaner edge.
                self.solid[y * n + x] = point_in_poly(x as f32 + 0.5, y as f32 + 0.5, &world);
            }
        }
    }

    /// Overwrite velocity inside solid cells with the body velocity, and
    /// zero the domain walls (no-slip box). Called before advection and
    /// again after projection so solids stay rigid.
    pub(crate) fn apply_solids(&mut self) {
        let n = self.n;
        let (bvx, bvy) = (self.body_vx, self.body_vy);
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                if self.solid[i] {
                    self.u[i] = bvx;
                    self.v[i] = bvy;
                    self.dye_r[i] = 0.0;
                    self.dye_g[i] = 0.0;
                    self.dye_b[i] = 0.0;
                } else if x == 0 || y == 0 || x == n - 1 || y == n - 1 {
                    self.u[i] = 0.0;
                    self.v[i] = 0.0;
                }
            }
        }
    }
}
