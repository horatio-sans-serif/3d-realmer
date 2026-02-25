//! Day/night cycle and sky simulation.
//!
//! Drives the sun direction, sun color, ambient color, fog color, and
//! clear color based on a time-of-day value. The cycle smoothly
//! transitions between dawn, day, dusk, and night phases.

/// Time-of-day phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DayPhase {
    Night,
    Dawn,
    Day,
    Dusk,
}

/// Sky state: drives all global lighting parameters.
pub struct Sky {
    /// Normalized time of day [0, 1). 0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk.
    time_of_day: f32,
    /// Day cycle speed: full cycle in this many seconds.
    cycle_duration: f32,
    /// Pause the cycle (for authoring).
    paused: bool,
}

impl Sky {
    pub fn new(cycle_duration: f32) -> Self {
        Self {
            time_of_day: 0.35, // Start mid-morning
            cycle_duration,
            paused: false,
        }
    }

    /// Advance the cycle by dt seconds.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }
        self.time_of_day += dt / self.cycle_duration;
        self.time_of_day %= 1.0;
    }

    pub fn time_of_day(&self) -> f32 {
        self.time_of_day
    }

    pub fn set_time_of_day(&mut self, t: f32) {
        self.time_of_day = t.rem_euclid(1.0);
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn phase(&self) -> DayPhase {
        let t = self.time_of_day;
        if t < 0.2 || t >= 0.85 {
            DayPhase::Night
        } else if t < 0.3 {
            DayPhase::Dawn
        } else if t < 0.7 {
            DayPhase::Day
        } else {
            DayPhase::Dusk
        }
    }

    /// Sun direction (normalized, pointing toward the sun).
    /// The sun moves in an arc: rises in +X, peaks at +Y, sets in -X.
    pub fn sun_direction(&self) -> [f32; 3] {
        let t = self.time_of_day;
        // Map to sun angle: 0.25=sunrise(0°), 0.5=noon(90°), 0.75=sunset(180°)
        let sun_angle = (t - 0.25) * std::f32::consts::TAU;
        let x = sun_angle.cos();
        let y = sun_angle.sin();
        // Normalize the direction (sun points downward from sky)
        let len = (x * x + y * y).sqrt().max(0.001);
        [x / len, -y.abs() / len, 0.3 / len]
    }

    /// Sun color: warm at dawn/dusk, white at noon, dim at night.
    pub fn sun_color(&self) -> [f32; 3] {
        let t = self.time_of_day;
        let sun_strength = sun_intensity(t);

        if sun_strength < 0.01 {
            return [0.05, 0.05, 0.1]; // Moonlight
        }

        let phase = self.phase();
        match phase {
            DayPhase::Dawn => {
                let f = smoothstep(0.2, 0.3, t);
                lerp3(
                    [1.0, 0.4, 0.2], // Orange sunrise
                    [1.0, 0.95, 0.8], // Bright day
                    f,
                )
            }
            DayPhase::Day => [1.0, 0.95, 0.8],
            DayPhase::Dusk => {
                let f = smoothstep(0.7, 0.85, t);
                lerp3(
                    [1.0, 0.95, 0.8],
                    [1.0, 0.4, 0.2], // Orange sunset
                    f,
                )
            }
            DayPhase::Night => [0.05, 0.05, 0.1],
        }
    }

    /// Ambient color: brighter during day, dark blue at night.
    pub fn ambient_color(&self) -> [f32; 3] {
        let sun = sun_intensity(self.time_of_day);
        lerp3(
            [0.02, 0.03, 0.08], // Night ambient
            [0.15, 0.18, 0.25], // Day ambient
            sun,
        )
    }

    /// Fog color: matches sky color roughly.
    pub fn fog_color(&self) -> [f32; 3] {
        let t = self.time_of_day;
        let sun = sun_intensity(t);
        lerp3(
            [0.02, 0.03, 0.05], // Night fog (dark)
            [0.7, 0.8, 0.95],   // Day fog (sky blue)
            sun,
        )
    }

    /// Clear color (sky background): blue during day, dark at night.
    pub fn clear_color(&self) -> [f64; 4] {
        let t = self.time_of_day;
        let sun = sun_intensity(t);
        let sky = lerp3(
            [0.01, 0.01, 0.03], // Night sky
            [0.4, 0.6, 0.9],    // Day sky
            sun,
        );
        [sky[0] as f64, sky[1] as f64, sky[2] as f64, 1.0]
    }
}

/// Sun intensity curve: 0 at night, 1 at noon.
fn sun_intensity(t: f32) -> f32 {
    // Rises at ~0.22, peaks at 0.5, sets at ~0.78.
    let x = ((t - 0.5) * std::f32::consts::PI * 2.0).cos();
    ((x + 0.3) / 1.3).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_cycle_progression() {
        let mut sky = Sky::new(100.0);
        let initial = sky.time_of_day();

        sky.update(10.0);
        assert!(sky.time_of_day() > initial);

        sky.update(90.0);
        // Should wrap around
        assert!(sky.time_of_day() < 1.0);
    }

    #[test]
    fn pause_stops_cycle() {
        let mut sky = Sky::new(100.0);
        sky.set_paused(true);
        let t = sky.time_of_day();
        sky.update(50.0);
        assert_eq!(sky.time_of_day(), t);
    }

    #[test]
    fn noon_is_brightest() {
        let mut sky = Sky::new(100.0);
        sky.set_time_of_day(0.5); // Noon
        assert_eq!(sky.phase(), DayPhase::Day);

        let sun = sky.sun_color();
        // Sun should be bright white-ish
        assert!(sun[0] > 0.9);
        assert!(sun[1] > 0.9);
    }

    #[test]
    fn midnight_is_darkest() {
        let mut sky = Sky::new(100.0);
        sky.set_time_of_day(0.0); // Midnight
        assert_eq!(sky.phase(), DayPhase::Night);

        let sun = sky.sun_color();
        // Sun should be very dim
        assert!(sun[0] < 0.2);
    }

    #[test]
    fn dawn_and_dusk_phases() {
        let mut sky = Sky::new(100.0);

        sky.set_time_of_day(0.25);
        assert_eq!(sky.phase(), DayPhase::Dawn);

        sky.set_time_of_day(0.75);
        assert_eq!(sky.phase(), DayPhase::Dusk);
    }

    #[test]
    fn clear_color_range() {
        let mut sky = Sky::new(100.0);
        for i in 0..10 {
            sky.set_time_of_day(i as f32 / 10.0);
            let c = sky.clear_color();
            assert!(c[0] >= 0.0 && c[0] <= 1.0);
            assert!(c[1] >= 0.0 && c[1] <= 1.0);
            assert!(c[2] >= 0.0 && c[2] <= 1.0);
        }
    }

    #[test]
    fn set_time_wraps() {
        let mut sky = Sky::new(100.0);
        sky.set_time_of_day(1.5);
        assert!(sky.time_of_day() >= 0.0 && sky.time_of_day() < 1.0);

        sky.set_time_of_day(-0.5);
        assert!(sky.time_of_day() >= 0.0 && sky.time_of_day() < 1.0);
    }
}
