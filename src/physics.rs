//! Physics integration via Rapier3D.
//!
//! The PhysicsWorld owns the Rapier pipeline and entity sets.
//! World objects register their rigid bodies and colliders here.
//! Each frame, PhysicsWorld steps the simulation and reports
//! transform updates to actors whose bodies moved.

use std::collections::HashMap;

use bitparty::bridge::WorldObjectId;
use rapier3d::prelude::*;

/// Central physics simulation owner.
pub struct PhysicsWorld {
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,

    /// Maps WorldObjectId -> RigidBodyHandle for cross-runtime addressing.
    body_map: HashMap<WorldObjectId, RigidBodyHandle>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            body_map: HashMap::new(),
        }
    }

    /// Step the physics simulation by `dt` seconds.
    pub fn step(&mut self, _dt: f32) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    /// Register a dynamic rigid body for a world object.
    /// Returns the RigidBodyHandle.
    pub fn register_dynamic(
        &mut self,
        id: WorldObjectId,
        position: [f32; 3],
        collider: ColliderBuilder,
    ) -> RigidBodyHandle {
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![position[0], position[1], position[2]])
            .build();

        let handle = self.rigid_body_set.insert(body);
        self.collider_set
            .insert_with_parent(collider.build(), handle, &mut self.rigid_body_set);
        self.body_map.insert(id, handle);
        handle
    }

    /// Register a static rigid body (terrain, buildings).
    pub fn register_static(
        &mut self,
        id: WorldObjectId,
        position: [f32; 3],
        collider: ColliderBuilder,
    ) -> RigidBodyHandle {
        let body = RigidBodyBuilder::fixed()
            .translation(vector![position[0], position[1], position[2]])
            .build();

        let handle = self.rigid_body_set.insert(body);
        self.collider_set
            .insert_with_parent(collider.build(), handle, &mut self.rigid_body_set);
        self.body_map.insert(id, handle);
        handle
    }

    /// Remove a body from the simulation.
    pub fn remove(&mut self, id: &WorldObjectId) {
        if let Some(handle) = self.body_map.remove(id) {
            self.rigid_body_set.remove(
                handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
        }
    }

    /// Get the current position of a world object's rigid body.
    pub fn position(&self, id: &WorldObjectId) -> Option<[f32; 3]> {
        self.body_map
            .get(id)
            .and_then(|h| self.rigid_body_set.get(*h))
            .map(|body| {
                let t = body.translation();
                [t.x, t.y, t.z]
            })
    }

    /// Cast a ray into the physics world. Returns the WorldObjectId of the
    /// first hit object and the hit distance.
    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
    ) -> Option<(WorldObjectId, f32)> {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![direction[0], direction[1], direction[2]],
        );

        let hit = self.query_pipeline.cast_ray(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_distance,
            true,
            QueryFilter::default(),
        );

        hit.and_then(|(collider_handle, toi)| {
            let collider = self.collider_set.get(collider_handle)?;
            let body_handle = collider.parent()?;
            let world_id = self
                .body_map
                .iter()
                .find(|&(_, h)| *h == body_handle)
                .map(|(id, _)| *id)?;
            Some((world_id, toi))
        })
    }

    pub fn body_count(&self) -> usize {
        self.rigid_body_set.len()
    }

    /// Split borrow for Player: mutable bodies, immutable colliders + query pipeline.
    pub fn physics_context(&mut self) -> (&mut RigidBodySet, &ColliderSet, &QueryPipeline) {
        (&mut self.rigid_body_set, &self.collider_set, &self.query_pipeline)
    }

    /// Mutable access to body and collider sets (for Player initialization).
    pub fn body_and_collider_sets_mut(&mut self) -> (&mut RigidBodySet, &mut ColliderSet) {
        (&mut self.rigid_body_set, &mut self.collider_set)
    }

    /// Read-only access to body set (for reading positions).
    pub fn rigid_body_set(&self) -> &RigidBodySet {
        &self.rigid_body_set
    }

    /// Insert a fixed (static) terrain body with a pre-built collider.
    pub fn insert_terrain_body(&mut self, position: [f32; 3], collider: Collider) -> RigidBodyHandle {
        let body = RigidBodyBuilder::fixed()
            .translation(vector![position[0], position[1], position[2]])
            .build();
        let handle = self.rigid_body_set.insert(body);
        self.collider_set.insert_with_parent(
            collider,
            handle,
            &mut self.rigid_body_set,
        );
        handle
    }

    /// Raycast that returns hit point and surface normal.
    /// Excludes the specified body (e.g., the player).
    pub fn raycast_with_normal(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        exclude_body: RigidBodyHandle,
    ) -> Option<([f32; 3], [f32; 3], f32)> {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![direction[0], direction[1], direction[2]],
        );

        let filter = QueryFilter::default().exclude_rigid_body(exclude_body);
        let hit = self.query_pipeline.cast_ray_and_get_normal(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_distance,
            true,
            filter,
        );

        hit.map(|(_, intersection)| {
            let point = ray.point_at(intersection.time_of_impact);
            let normal = intersection.normal;
            (
                [point.x, point.y, point.z],
                [normal.x, normal.y, normal.z],
                intersection.time_of_impact,
            )
        })
    }

    /// Raycast returning the hit body handle and distance.
    /// Used for identifying which world object was clicked.
    pub fn raycast_hit_body(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        exclude_body: RigidBodyHandle,
    ) -> Option<(RigidBodyHandle, f32)> {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![direction[0], direction[1], direction[2]],
        );

        let filter = QueryFilter::default().exclude_rigid_body(exclude_body);
        let hit = self.query_pipeline.cast_ray(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_distance,
            true,
            filter,
        );

        hit.and_then(|(collider_handle, toi)| {
            let collider = self.collider_set.get(collider_handle)?;
            let body_handle = collider.parent()?;
            Some((body_handle, toi))
        })
    }

    /// Remove a body by its handle (for terrain unloading).
    pub fn remove_body_by_handle(&mut self, handle: RigidBodyHandle) {
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_world_creation() {
        let physics = PhysicsWorld::new();
        assert_eq!(physics.body_count(), 0);
    }

    #[test]
    fn register_and_step() {
        let mut physics = PhysicsWorld::new();
        let id = WorldObjectId::new();

        physics.register_dynamic(id, [0.0, 10.0, 0.0], ColliderBuilder::ball(0.5));
        assert_eq!(physics.body_count(), 1);

        let pos_before = physics.position(&id).unwrap();
        assert_eq!(pos_before[1], 10.0);

        // Step the simulation
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }

        let pos_after = physics.position(&id).unwrap();
        // Ball should have fallen due to gravity
        assert!(pos_after[1] < pos_before[1]);
    }

    #[test]
    fn register_static_and_remove() {
        let mut physics = PhysicsWorld::new();
        let id = WorldObjectId::new();

        physics.register_static(id, [0.0, 0.0, 0.0], ColliderBuilder::cuboid(10.0, 0.5, 10.0));
        assert_eq!(physics.body_count(), 1);

        physics.remove(&id);
        assert_eq!(physics.body_count(), 0);
        assert!(physics.position(&id).is_none());
    }

    #[test]
    fn raycast_hit() {
        let mut physics = PhysicsWorld::new();
        let id = WorldObjectId::new();

        physics.register_static(id, [0.0, 0.0, -5.0], ColliderBuilder::cuboid(1.0, 1.0, 1.0));

        // Update query pipeline
        physics.step(0.0);

        let hit = physics.raycast([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 100.0);
        assert!(hit.is_some());
        let (hit_id, _toi) = hit.unwrap();
        assert_eq!(hit_id, id);
    }
}
