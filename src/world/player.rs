use glam::Vec3;
use rapier3d::prelude::*;

use super::camera::Camera;
use crate::physics::PhysicsWorld;

/// Player character controller using a Rapier kinematic body.
/// Handles gravity, ground detection, and collision with terrain.
pub struct Player {
    body_handle: RigidBodyHandle,
    #[allow(dead_code)]
    collider_handle: ColliderHandle,
    velocity: Vec3,
    on_ground: bool,
    height: f32,
    radius: f32,
    gravity: f32,
    jump_speed: f32,
    move_speed: f32,
}

impl Player {
    /// Spawn the player at a position.
    pub fn new(position: Vec3, physics: &mut PhysicsWorld) -> Self {
        let height = 1.6;
        let radius = 0.3;

        let (rigid_body_set, collider_set) = physics.body_and_collider_sets_mut();

        let body = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![position.x, position.y, position.z])
            .build();

        let body_handle = rigid_body_set.insert(body);

        let collider = ColliderBuilder::capsule_y(height / 2.0 - radius, radius)
            .build();

        let collider_handle = collider_set.insert_with_parent(
            collider,
            body_handle,
            rigid_body_set,
        );

        Self {
            body_handle,
            collider_handle,
            velocity: Vec3::ZERO,
            on_ground: false,
            height,
            radius,
            gravity: 20.0,
            jump_speed: 8.0,
            move_speed: 6.0,
        }
    }

    /// Update the player position based on camera input and physics.
    pub fn update(&mut self, dt: f32, camera: &Camera, physics: &mut PhysicsWorld) {
        let (rigid_body_set, collider_set, query_pipeline) = physics.physics_context();

        // Get current position
        let body = &rigid_body_set[self.body_handle];
        let pos = body.translation();
        let current = Vec3::new(pos.x, pos.y, pos.z);

        // Ground check: cast a short ray downward from feet
        let feet = point![current.x, current.y - self.height / 2.0, current.z];
        let down = vector![0.0, -1.0, 0.0];
        let ground_check_dist = 0.1;

        let filter = QueryFilter::default().exclude_rigid_body(self.body_handle);
        let ground_hit = query_pipeline.cast_ray(
            rigid_body_set,
            collider_set,
            &Ray::new(feet, down),
            ground_check_dist,
            true,
            filter,
        );

        self.on_ground = ground_hit.is_some();

        // Apply gravity
        if !self.on_ground {
            self.velocity.y -= self.gravity * dt;
        } else if self.velocity.y < 0.0 {
            self.velocity.y = 0.0;
        }

        // Get movement wish direction from camera (horizontal only)
        let front = camera.front();
        let front_flat = Vec3::new(front.x, 0.0, front.z).normalize_or_zero();
        let right = camera.right();
        let right_flat = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

        let mut wish = Vec3::ZERO;
        if camera.is_moving_forward() {
            wish += front_flat;
        }
        if camera.is_moving_backward() {
            wish -= front_flat;
        }
        if camera.is_moving_right() {
            wish += right_flat;
        }
        if camera.is_moving_left() {
            wish -= right_flat;
        }

        if wish.length_squared() > 0.0 {
            wish = wish.normalize();
        }

        // Horizontal velocity from input
        let horizontal = wish * self.move_speed;
        self.velocity.x = horizontal.x;
        self.velocity.z = horizontal.z;

        // Compute next position
        let next_pos = current + self.velocity * dt;

        // Move the kinematic body
        let body = &mut rigid_body_set[self.body_handle];
        body.set_next_kinematic_translation(vector![next_pos.x, next_pos.y, next_pos.z]);
    }

    /// Try to jump (only if on ground).
    pub fn jump(&mut self) {
        if self.on_ground {
            self.velocity.y = self.jump_speed;
            self.on_ground = false;
        }
    }

    /// Get the player's current world position (eye level).
    pub fn eye_position(&self, physics: &PhysicsWorld) -> Vec3 {
        let body = &physics.rigid_body_set()[self.body_handle];
        let pos = body.translation();
        Vec3::new(pos.x, pos.y + self.height / 2.0 - 0.1, pos.z)
    }

    pub fn body_handle(&self) -> RigidBodyHandle {
        self.body_handle
    }

    pub fn on_ground(&self) -> bool {
        self.on_ground
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_creation() {
        let mut physics = PhysicsWorld::new();

        let player = Player::new(
            Vec3::new(0.0, 10.0, 0.0),
            &mut physics,
        );

        assert_eq!(physics.body_count(), 1);
        assert!(!player.on_ground());

        let eye = player.eye_position(&physics);
        assert!((eye.y - 10.7).abs() < 0.2);
    }

    #[test]
    fn player_eye_position_tracks_body() {
        let mut physics = PhysicsWorld::new();

        let player = Player::new(
            Vec3::new(5.0, 20.0, -3.0),
            &mut physics,
        );

        let eye = player.eye_position(&physics);
        assert!((eye.x - 5.0).abs() < 0.01);
        assert!((eye.z - (-3.0)).abs() < 0.01);
    }
}
