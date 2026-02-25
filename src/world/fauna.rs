//! Ambient fauna: actor-driven creatures (butterflies, birds) that
//! wander the world, providing ambient life.
//!
//! Each fauna entity is a lightweight actor with simple AI:
//! - Butterflies: slow, bobbing flight, random direction changes
//! - Birds: faster, higher altitude, occasional circling
//!
//! Fauna are rendered as colored cubes via the ObjectRenderer
//! (mesh characters deferred to a later phase).

use bitparty::bridge::WorldObjectId;

/// Type of ambient creature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaunaType {
    Butterfly,
    Bird,
}

/// A single fauna entity.
struct FaunaEntity {
    id: WorldObjectId,
    fauna_type: FaunaType,
    position: [f32; 3],
    velocity: [f32; 3],
    /// Time until next direction change.
    wander_timer: f32,
    /// Home position (fauna stay within range of this).
    home: [f32; 3],
    /// Maximum wander distance from home.
    range: f32,
}

/// Manages all ambient fauna entities.
pub struct FaunaManager {
    entities: Vec<FaunaEntity>,
    spawn_seed: u32,
}

impl FaunaManager {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            spawn_seed: 0,
        }
    }

    /// Spawn a group of ambient fauna around a position.
    pub fn spawn_group(&mut self, center: [f32; 3], butterflies: usize, birds: usize) {
        for i in 0..butterflies {
            self.spawn_seed += 1;
            let offset_x = hash_f32(self.spawn_seed * 3 + i as u32) * 20.0 - 10.0;
            let offset_z = hash_f32(self.spawn_seed * 3 + i as u32 + 1) * 20.0 - 10.0;
            let pos = [
                center[0] + offset_x,
                center[1] + 1.0 + hash_f32(self.spawn_seed * 3 + i as u32 + 2) * 3.0,
                center[2] + offset_z,
            ];
            self.entities.push(FaunaEntity {
                id: WorldObjectId::new(),
                fauna_type: FaunaType::Butterfly,
                position: pos,
                velocity: [0.0, 0.0, 0.0],
                wander_timer: hash_f32(self.spawn_seed) * 3.0,
                home: pos,
                range: 8.0,
            });
        }

        for i in 0..birds {
            self.spawn_seed += 1;
            let offset_x = hash_f32(self.spawn_seed * 5 + i as u32) * 40.0 - 20.0;
            let offset_z = hash_f32(self.spawn_seed * 5 + i as u32 + 1) * 40.0 - 20.0;
            let pos = [
                center[0] + offset_x,
                center[1] + 10.0 + hash_f32(self.spawn_seed * 5 + i as u32 + 2) * 10.0,
                center[2] + offset_z,
            ];
            self.entities.push(FaunaEntity {
                id: WorldObjectId::new(),
                fauna_type: FaunaType::Bird,
                position: pos,
                velocity: [0.0, 0.0, 0.0],
                wander_timer: hash_f32(self.spawn_seed) * 2.0,
                home: pos,
                range: 20.0,
            });
        }
    }

    /// Update all fauna positions.
    pub fn update(&mut self, dt: f32, wind_direction: [f32; 2], wind_strength: f32) {
        for entity in &mut self.entities {
            entity.wander_timer -= dt;

            if entity.wander_timer <= 0.0 {
                // Pick new random direction
                let seed = entity.id.as_uuid().as_u128() as u32;
                let time_hash = (entity.position[0] * 100.0) as u32;
                let rx = hash_f32(seed.wrapping_add(time_hash)) * 2.0 - 1.0;
                let rz = hash_f32(seed.wrapping_add(time_hash).wrapping_add(1)) * 2.0 - 1.0;
                let ry = hash_f32(seed.wrapping_add(time_hash).wrapping_add(2)) * 2.0 - 1.0;

                match entity.fauna_type {
                    FaunaType::Butterfly => {
                        let speed = 0.8;
                        entity.velocity = [rx * speed, ry * 0.3, rz * speed];
                        entity.wander_timer = 1.5 + hash_f32(seed.wrapping_add(time_hash).wrapping_add(3)) * 3.0;
                    }
                    FaunaType::Bird => {
                        let speed = 2.5;
                        entity.velocity = [rx * speed, ry * 0.5, rz * speed];
                        entity.wander_timer = 2.0 + hash_f32(seed.wrapping_add(time_hash).wrapping_add(3)) * 4.0;
                    }
                }
            }

            // Apply wind influence
            entity.velocity[0] += wind_direction[0] * wind_strength * 0.1 * dt;
            entity.velocity[2] += wind_direction[1] * wind_strength * 0.1 * dt;

            // Move
            entity.position[0] += entity.velocity[0] * dt;
            entity.position[1] += entity.velocity[1] * dt;
            entity.position[2] += entity.velocity[2] * dt;

            // Bobbing motion for butterflies
            if entity.fauna_type == FaunaType::Butterfly {
                let bob = (entity.position[0] * 2.0 + entity.position[2]).sin() * 0.3;
                entity.position[1] += bob * dt;
            }

            // Keep within range of home
            let dx = entity.position[0] - entity.home[0];
            let dy = entity.position[1] - entity.home[1];
            let dz = entity.position[2] - entity.home[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist > entity.range {
                // Steer back toward home
                let factor = 0.5 * dt;
                entity.velocity[0] -= dx / dist * factor;
                entity.velocity[1] -= dy / dist * factor;
                entity.velocity[2] -= dz / dist * factor;
            }

            // Clamp altitude
            let min_y = match entity.fauna_type {
                FaunaType::Butterfly => entity.home[1] - 2.0,
                FaunaType::Bird => entity.home[1] - 5.0,
            };
            if entity.position[1] < min_y {
                entity.position[1] = min_y;
                entity.velocity[1] = entity.velocity[1].abs();
            }
        }
    }

    /// Get all fauna positions and colors for the ObjectRenderer.
    pub fn instances(&self) -> Vec<FaunaInstance> {
        self.entities
            .iter()
            .map(|e| {
                let color = match e.fauna_type {
                    FaunaType::Butterfly => [0.9, 0.3, 0.8, 1.0], // Pink/purple
                    FaunaType::Bird => [0.1, 0.1, 0.1, 1.0],       // Dark (silhouette)
                };
                let scale = match e.fauna_type {
                    FaunaType::Butterfly => 0.1,
                    FaunaType::Bird => 0.15,
                };
                FaunaInstance {
                    position: e.position,
                    color,
                    scale,
                }
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// Move the home positions of all fauna to follow the player.
    pub fn reanchor(&mut self, center: [f32; 3]) {
        for entity in &mut self.entities {
            let dx = center[0] - entity.home[0];
            let dz = center[2] - entity.home[2];
            let dist = (dx * dx + dz * dz).sqrt();

            // Only reanchor fauna that are far from the new center
            if dist > 50.0 {
                entity.home[0] = center[0] + hash_f32(entity.id.as_uuid().as_u128() as u32) * 20.0 - 10.0;
                entity.home[2] = center[2] + hash_f32(entity.id.as_uuid().as_u128() as u32 + 1) * 20.0 - 10.0;
            }
        }
    }
}

/// Fauna instance data for rendering.
pub struct FaunaInstance {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub scale: f32,
}

/// Simple hash function for deterministic pseudo-random.
fn hash_f32(seed: u32) -> f32 {
    let mut h = seed;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a2d39);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_group() {
        let mut mgr = FaunaManager::new();
        mgr.spawn_group([0.0, 20.0, 0.0], 5, 3);
        assert_eq!(mgr.count(), 8);
    }

    #[test]
    fn update_moves_entities() {
        let mut mgr = FaunaManager::new();
        mgr.spawn_group([0.0, 20.0, 0.0], 2, 0);

        let before: Vec<[f32; 3]> = mgr.entities.iter().map(|e| e.position).collect();
        mgr.update(1.0, [1.0, 0.0], 0.3);
        let after: Vec<[f32; 3]> = mgr.entities.iter().map(|e| e.position).collect();

        // At least some movement should have occurred
        let moved = before.iter().zip(after.iter()).any(|(b, a)| {
            (b[0] - a[0]).abs() > 0.001
                || (b[1] - a[1]).abs() > 0.001
                || (b[2] - a[2]).abs() > 0.001
        });
        assert!(moved, "fauna should have moved");
    }

    #[test]
    fn instances_returns_correct_count() {
        let mut mgr = FaunaManager::new();
        mgr.spawn_group([0.0, 20.0, 0.0], 3, 2);
        assert_eq!(mgr.instances().len(), 5);
    }

    #[test]
    fn hash_range() {
        for i in 0..100 {
            let v = hash_f32(i);
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}
