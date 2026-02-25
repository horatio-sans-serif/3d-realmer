//! World object management: tracks placed objects with their physics
//! bodies and visual representations.

use std::collections::HashMap;

use bitparty::actor::ActorId;
use bitparty::bridge::WorldObjectId;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::physics::PhysicsWorld;

/// A placed object in the 3D world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldObject {
    pub id: WorldObjectId,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub object_type: WorldObjectType,
    #[serde(default)]
    pub properties: serde_json::Value,
}

/// Type of world object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldObjectType {
    /// A simple cube block.
    Cube,
    /// A mesh object (GLTF).
    Mesh,
    /// An interactive actor-backed object.
    Interactive,
}

/// Manages all placed world objects and their mappings.
pub struct WorldObjectManager {
    /// All placed objects by WorldObjectId.
    objects: HashMap<WorldObjectId, WorldObject>,
    /// Physics body handles for placed objects.
    body_handles: HashMap<WorldObjectId, RigidBodyHandle>,
    /// Reverse: RigidBodyHandle -> WorldObjectId (for raycast hit identification).
    body_reverse: HashMap<RigidBodyHandle, WorldObjectId>,
    /// WorldObjectId -> ActorId mapping for actor-backed objects.
    actor_map: HashMap<WorldObjectId, ActorId>,
    /// Reverse: ActorId -> WorldObjectId.
    actor_reverse: HashMap<ActorId, WorldObjectId>,
}

impl WorldObjectManager {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            body_handles: HashMap::new(),
            body_reverse: HashMap::new(),
            actor_map: HashMap::new(),
            actor_reverse: HashMap::new(),
        }
    }

    /// Spawn a new object in the world.
    /// Creates a physics body (static cube collider) and registers it.
    pub fn spawn_object(
        &mut self,
        id: WorldObjectId,
        position: [f32; 3],
        object_type: WorldObjectType,
        properties: serde_json::Value,
        physics: &mut PhysicsWorld,
    ) -> &WorldObject {
        // Create a static physics body with a cube collider.
        let half_size = 0.25; // 0.5m cube (matches BLOCK_SIZE)
        let collider = ColliderBuilder::cuboid(half_size, half_size, half_size).build();
        let handle = physics.insert_terrain_body(position, collider);

        let obj = WorldObject {
            id,
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            object_type,
            properties,
        };

        self.objects.insert(id, obj);
        self.body_handles.insert(id, handle);
        self.body_reverse.insert(handle, id);

        self.objects.get(&id).unwrap()
    }

    /// Remove an object from the world.
    pub fn remove_object(
        &mut self,
        id: &WorldObjectId,
        physics: &mut PhysicsWorld,
    ) -> Option<WorldObject> {
        if let Some(handle) = self.body_handles.remove(id) {
            self.body_reverse.remove(&handle);
            physics.remove_body_by_handle(handle);
        }

        // Clean up actor mapping.
        if let Some(actor_id) = self.actor_map.remove(id) {
            self.actor_reverse.remove(&actor_id);
        }

        self.objects.remove(id)
    }

    /// Look up which world object a physics body belongs to (for raycast hit identification).
    pub fn object_for_body(&self, handle: &RigidBodyHandle) -> Option<&WorldObjectId> {
        self.body_reverse.get(handle)
    }

    /// Get an object by its WorldObjectId.
    pub fn get(&self, id: &WorldObjectId) -> Option<&WorldObject> {
        self.objects.get(id)
    }

    /// Get a mutable reference to an object by its WorldObjectId.
    pub fn get_mut(&mut self, id: &WorldObjectId) -> Option<&mut WorldObject> {
        self.objects.get_mut(id)
    }

    /// Get all objects.
    pub fn all_objects(&self) -> Vec<&WorldObject> {
        self.objects.values().collect()
    }

    /// Bind an ActorId to a WorldObjectId.
    pub fn bind_actor(&mut self, world_id: WorldObjectId, actor_id: ActorId) {
        self.actor_map.insert(world_id, actor_id);
        self.actor_reverse.insert(actor_id, world_id);
    }

    /// Get the ActorId for a WorldObjectId.
    pub fn actor_for_object(&self, world_id: &WorldObjectId) -> Option<&ActorId> {
        self.actor_map.get(world_id)
    }

    /// Get the WorldObjectId for an ActorId.
    pub fn object_for_actor(&self, actor_id: &ActorId) -> Option<&WorldObjectId> {
        self.actor_reverse.get(actor_id)
    }

    /// Count of placed objects.
    pub fn count(&self) -> usize {
        self.objects.len()
    }

    /// Find the nearest object to a position within a maximum distance.
    /// Returns (WorldObjectId, &WorldObject) of the closest match, or None.
    pub fn find_nearest(
        &self,
        position: [f32; 3],
        max_distance: f32,
    ) -> Option<(WorldObjectId, &WorldObject)> {
        let max_dist_sq = max_distance * max_distance;
        let mut best: Option<(WorldObjectId, &WorldObject, f32)> = None;

        for obj in self.objects.values() {
            let dx = obj.position[0] - position[0];
            let dy = obj.position[1] - position[1];
            let dz = obj.position[2] - position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= max_dist_sq {
                if best.as_ref().map_or(true, |(_, _, d)| dist_sq < *d) {
                    best = Some((obj.id, obj, dist_sq));
                }
            }
        }

        best.map(|(id, obj, _)| (id, obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_and_get_object() {
        let mut physics = PhysicsWorld::new();
        let mut mgr = WorldObjectManager::new();

        let id = WorldObjectId::new();
        mgr.spawn_object(
            id,
            [5.0, 10.0, -3.0],
            WorldObjectType::Cube,
            serde_json::json!({}),
            &mut physics,
        );

        assert_eq!(mgr.count(), 1);
        let obj = mgr.get(&id).unwrap();
        assert_eq!(obj.position, [5.0, 10.0, -3.0]);
        assert_eq!(obj.object_type, WorldObjectType::Cube);
    }

    #[test]
    fn remove_object() {
        let mut physics = PhysicsWorld::new();
        let mut mgr = WorldObjectManager::new();

        let id = WorldObjectId::new();
        mgr.spawn_object(
            id,
            [0.0, 0.0, 0.0],
            WorldObjectType::Cube,
            serde_json::json!({}),
            &mut physics,
        );

        let removed = mgr.remove_object(&id, &mut physics);
        assert!(removed.is_some());
        assert_eq!(mgr.count(), 0);
        assert!(mgr.get(&id).is_none());
    }

    #[test]
    fn actor_binding() {
        let mut physics = PhysicsWorld::new();
        let mut mgr = WorldObjectManager::new();

        let world_id = WorldObjectId::new();
        mgr.spawn_object(
            world_id,
            [0.0, 0.0, 0.0],
            WorldObjectType::Interactive,
            serde_json::json!({}),
            &mut physics,
        );

        // Create a test ActorId via slotmap's KeyData.
        let actor_id: ActorId = slotmap::KeyData::from_ffi(42).into();
        mgr.bind_actor(world_id, actor_id);

        assert_eq!(mgr.actor_for_object(&world_id), Some(&actor_id));
        assert_eq!(mgr.object_for_actor(&actor_id), Some(&world_id));
    }

    #[test]
    fn all_objects() {
        let mut physics = PhysicsWorld::new();
        let mut mgr = WorldObjectManager::new();

        for i in 0..5 {
            mgr.spawn_object(
                WorldObjectId::new(),
                [i as f32, 0.0, 0.0],
                WorldObjectType::Cube,
                serde_json::json!({}),
                &mut physics,
            );
        }

        assert_eq!(mgr.all_objects().len(), 5);
    }
}
