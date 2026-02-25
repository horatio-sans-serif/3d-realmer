//! NPC actors: actor-driven characters that navigate the voxel world.
//!
//! Each NPC has a state machine (Idle, Wander, WalkTo, Interact)
//! and follows A* paths. NPCs are rendered as colored cubes via the
//! ObjectRenderer (mesh characters deferred).

use bitparty::bridge::WorldObjectId;

use super::pathfinding::{self, NavPos};

/// NPC behavior state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpcState {
    /// Standing still, waiting.
    Idle,
    /// Walking to a random nearby point.
    Wander,
    /// Walking toward a specific target.
    WalkTo,
    /// Engaged in conversation with the player.
    Converse,
}

/// A single NPC in the world.
struct Npc {
    id: WorldObjectId,
    position: [f32; 3],
    /// Current behavior state.
    state: NpcState,
    /// Current path waypoints.
    path: Vec<NavPos>,
    /// Index of current waypoint in path.
    path_index: usize,
    /// Movement speed (blocks per second).
    speed: f32,
    /// Time spent in current state.
    state_timer: f32,
    /// Maximum time to stay in idle before wandering.
    idle_duration: f32,
    /// Home position for wander behavior.
    home: [f32; 3],
    /// Maximum wander distance from home.
    wander_range: f32,
    /// Visual color.
    color: [f32; 4],
    /// NPC name/label.
    label: String,
    /// Whether this NPC can engage in LLM-backed conversation.
    conversational: bool,
    /// Last speech bubble text (displayed near NPC).
    speech_text: Option<String>,
    /// How long the speech bubble has been displayed.
    speech_timer: f32,
}

/// Manages all NPCs.
pub struct NpcManager {
    npcs: Vec<Npc>,
    block_size: f32,
    spawn_seed: u32,
}

impl NpcManager {
    pub fn new(block_size: f32) -> Self {
        Self {
            npcs: Vec::new(),
            block_size,
            spawn_seed: 0,
        }
    }

    /// Spawn an NPC at a position.
    pub fn spawn(
        &mut self,
        position: [f32; 3],
        label: &str,
        color: [f32; 4],
        speed: f32,
    ) -> WorldObjectId {
        let id = WorldObjectId::new();
        self.spawn_seed += 1;

        self.npcs.push(Npc {
            id,
            position,
            state: NpcState::Idle,
            path: Vec::new(),
            path_index: 0,
            speed,
            state_timer: 0.0,
            idle_duration: 2.0 + hash_f32(self.spawn_seed) * 3.0,
            home: position,
            wander_range: 15.0,
            color,
            label: label.to_string(),
            conversational: false,
            speech_text: None,
            speech_timer: 0.0,
        });

        id
    }

    /// Update all NPCs.
    pub fn update(
        &mut self,
        dt: f32,
        is_walkable: &dyn Fn(NavPos) -> bool,
    ) {
        for npc in &mut self.npcs {
            npc.state_timer += dt;

            match npc.state {
                NpcState::Idle => {
                    if npc.state_timer >= npc.idle_duration {
                        // Transition to Wander: pick a random nearby goal.
                        let current_nav = NavPos::from_world(npc.position, 0.5);
                        let seed = npc.id.as_uuid().as_u128() as u32;
                        let time_hash = (npc.state_timer * 100.0) as u32;
                        let dx = (hash_f32(seed.wrapping_add(time_hash)) * 2.0 - 1.0)
                            * npc.wander_range;
                        let dz = (hash_f32(seed.wrapping_add(time_hash).wrapping_add(1)) * 2.0
                            - 1.0)
                            * npc.wander_range;

                        let goal_world = [
                            npc.home[0] + dx,
                            npc.position[1],
                            npc.home[2] + dz,
                        ];
                        let goal_nav = NavPos::from_world(goal_world, 0.5);

                        if let Some(path) = pathfinding::find_path(
                            current_nav,
                            goal_nav,
                            is_walkable,
                            2000,
                        ) {
                            npc.path = path.waypoints;
                            npc.path_index = 0;
                            npc.state = NpcState::Wander;
                            npc.state_timer = 0.0;
                        } else {
                            // Can't find path, reset idle timer
                            npc.state_timer = 0.0;
                            npc.idle_duration =
                                1.0 + hash_f32(seed.wrapping_add(time_hash).wrapping_add(2)) * 2.0;
                        }
                    }
                }
                NpcState::Converse => {
                    // In conversation, NPC stands still and faces the player.
                    // Speech bubble timer counts down.
                    if let Some(_text) = &npc.speech_text {
                        npc.speech_timer += dt;
                        if npc.speech_timer > 10.0 {
                            // Clear old speech after 10 seconds.
                            npc.speech_text = None;
                            npc.speech_timer = 0.0;
                        }
                    }
                    // Stay in Converse until explicitly released.
                }
                NpcState::Wander | NpcState::WalkTo => {
                    if npc.path_index >= npc.path.len() {
                        // Reached end of path, go idle.
                        npc.state = NpcState::Idle;
                        npc.state_timer = 0.0;
                        npc.path.clear();
                        npc.idle_duration = 2.0
                            + hash_f32(
                                (npc.id.as_uuid().as_u128() as u32)
                                    .wrapping_add((npc.state_timer * 100.0) as u32),
                            ) * 3.0;
                        continue;
                    }

                    let target = npc.path[npc.path_index].to_world(0.5);
                    let dx = target[0] - npc.position[0];
                    let dy = target[1] - npc.position[1];
                    let dz = target[2] - npc.position[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist < 0.3 {
                        // Reached waypoint, advance to next.
                        npc.path_index += 1;
                    } else {
                        // Move toward waypoint.
                        let move_dist = npc.speed * dt;
                        if move_dist >= dist {
                            npc.position = target;
                        } else {
                            let factor = move_dist / dist;
                            npc.position[0] += dx * factor;
                            npc.position[1] += dy * factor;
                            npc.position[2] += dz * factor;
                        }
                    }

                    // Check if current waypoint is still walkable
                    if npc.path_index < npc.path.len()
                        && !is_walkable(npc.path[npc.path_index])
                    {
                        // Path invalidated, go idle and re-path next time
                        npc.state = NpcState::Idle;
                        npc.state_timer = 0.0;
                        npc.path.clear();
                    }
                }
            }
        }
    }

    /// Command an NPC to walk to a specific position.
    pub fn walk_to(
        &mut self,
        npc_id: &WorldObjectId,
        target: [f32; 3],
        is_walkable: &dyn Fn(NavPos) -> bool,
    ) -> bool {
        let npc = match self.npcs.iter_mut().find(|n| n.id == *npc_id) {
            Some(n) => n,
            None => return false,
        };

        let start = NavPos::from_world(npc.position, self.block_size);
        let goal = NavPos::from_world(target, self.block_size);

        if let Some(path) = pathfinding::find_path(start, goal, is_walkable, 5000) {
            npc.path = path.waypoints;
            npc.path_index = 0;
            npc.state = NpcState::WalkTo;
            npc.state_timer = 0.0;
            true
        } else {
            false
        }
    }

    /// Get all NPC instances for rendering.
    pub fn instances(&self) -> Vec<NpcInstance> {
        self.npcs
            .iter()
            .map(|npc| NpcInstance {
                position: npc.position,
                color: npc.color,
                state: npc.state,
                speech_text: npc.speech_text.clone(),
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.npcs.len()
    }

    /// Get NPC info by ID.
    pub fn get_info(&self, id: &WorldObjectId) -> Option<NpcInfo> {
        self.npcs.iter().find(|n| n.id == *id).map(|n| NpcInfo {
            id: n.id,
            position: n.position,
            state: n.state,
            label: n.label.clone(),
            conversational: n.conversational,
        })
    }

    /// Spawn a conversational NPC (backed by LLM).
    pub fn spawn_conversational(
        &mut self,
        position: [f32; 3],
        label: &str,
        color: [f32; 4],
        speed: f32,
    ) -> WorldObjectId {
        let id = self.spawn(position, label, color, speed);
        if let Some(npc) = self.npcs.iter_mut().find(|n| n.id == id) {
            npc.conversational = true;
        }
        id
    }

    /// Start a conversation with an NPC (transitions to Converse state).
    pub fn start_conversation(&mut self, npc_id: &WorldObjectId) -> bool {
        if let Some(npc) = self.npcs.iter_mut().find(|n| n.id == *npc_id) {
            if npc.conversational {
                npc.state = NpcState::Converse;
                npc.state_timer = 0.0;
                npc.path.clear();
                return true;
            }
        }
        false
    }

    /// End a conversation with an NPC (returns to Idle state).
    pub fn end_conversation(&mut self, npc_id: &WorldObjectId) {
        if let Some(npc) = self.npcs.iter_mut().find(|n| n.id == *npc_id) {
            npc.state = NpcState::Idle;
            npc.state_timer = 0.0;
            npc.speech_text = None;
            npc.speech_timer = 0.0;
        }
    }

    /// Set the speech bubble text for an NPC.
    pub fn set_speech(&mut self, npc_id: &WorldObjectId, text: &str) {
        if let Some(npc) = self.npcs.iter_mut().find(|n| n.id == *npc_id) {
            npc.speech_text = Some(text.to_string());
            npc.speech_timer = 0.0;
        }
    }

    /// Check if an NPC is conversational.
    pub fn is_conversational(&self, npc_id: &WorldObjectId) -> bool {
        self.npcs
            .iter()
            .find(|n| n.id == *npc_id)
            .is_some_and(|n| n.conversational)
    }

    /// Find the nearest conversational NPC within range of a position.
    pub fn nearest_conversational(
        &self,
        position: [f32; 3],
        max_distance: f32,
    ) -> Option<WorldObjectId> {
        let mut best: Option<(WorldObjectId, f32)> = None;

        for npc in &self.npcs {
            if !npc.conversational {
                continue;
            }
            let dx = npc.position[0] - position[0];
            let dy = npc.position[1] - position[1];
            let dz = npc.position[2] - position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist <= max_distance {
                if best.is_none() || dist < best.unwrap().1 {
                    best = Some((npc.id, dist));
                }
            }
        }

        best.map(|(id, _)| id)
    }
}

/// NPC instance data for rendering.
pub struct NpcInstance {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub state: NpcState,
    pub speech_text: Option<String>,
}

/// Public NPC info (for bridge queries).
pub struct NpcInfo {
    pub id: WorldObjectId,
    pub position: [f32; 3],
    pub state: NpcState,
    pub label: String,
    pub conversational: bool,
}

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

    fn open_world(_pos: NavPos) -> bool {
        true
    }

    #[test]
    fn spawn_npc() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn([5.0, 10.0, 5.0], "Guard", [0.8, 0.2, 0.2, 1.0], 2.0);
        assert_eq!(mgr.count(), 1);
        let info = mgr.get_info(&id).unwrap();
        assert_eq!(info.label, "Guard");
        assert_eq!(info.state, NpcState::Idle);
    }

    #[test]
    fn npc_idle_to_wander() {
        let mut mgr = NpcManager::new(0.5);
        mgr.spawn([5.0, 10.0, 5.0], "Villager", [0.5, 0.3, 0.1, 1.0], 2.0);

        // Advance time past idle duration to trigger wander
        for _ in 0..100 {
            mgr.update(0.1, &open_world);
        }

        let instances = mgr.instances();
        assert_eq!(instances.len(), 1);
        // NPC should have moved from initial position
        let pos = instances[0].position;
        let moved = (pos[0] - 5.0).abs() > 0.1
            || (pos[2] - 5.0).abs() > 0.1;
        assert!(moved, "NPC should have wandered from spawn");
    }

    #[test]
    fn npc_walk_to() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn([0.0, 0.0, 0.0], "Runner", [1.0, 1.0, 1.0, 1.0], 5.0);

        let success = mgr.walk_to(&id, [3.0, 0.0, 0.0], &open_world);
        assert!(success);

        let info = mgr.get_info(&id).unwrap();
        assert_eq!(info.state, NpcState::WalkTo);
    }

    #[test]
    fn npc_instances() {
        let mut mgr = NpcManager::new(0.5);
        mgr.spawn([0.0, 0.0, 0.0], "A", [1.0, 0.0, 0.0, 1.0], 2.0);
        mgr.spawn([5.0, 0.0, 5.0], "B", [0.0, 1.0, 0.0, 1.0], 2.0);

        let instances = mgr.instances();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn conversational_npc() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn_conversational(
            [5.0, 10.0, 5.0],
            "Atlas",
            [0.3, 0.6, 0.9, 1.0],
            1.5,
        );

        assert!(mgr.is_conversational(&id));
        let info = mgr.get_info(&id).unwrap();
        assert!(info.conversational);
        assert_eq!(info.label, "Atlas");
    }

    #[test]
    fn start_end_conversation() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn_conversational(
            [0.0, 0.0, 0.0],
            "Robot",
            [0.5, 0.5, 0.5, 1.0],
            2.0,
        );

        assert!(mgr.start_conversation(&id));
        let info = mgr.get_info(&id).unwrap();
        assert_eq!(info.state, NpcState::Converse);

        mgr.end_conversation(&id);
        let info = mgr.get_info(&id).unwrap();
        assert_eq!(info.state, NpcState::Idle);
    }

    #[test]
    fn non_conversational_npc_rejects_conversation() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn([0.0, 0.0, 0.0], "Guard", [0.8, 0.2, 0.2, 1.0], 2.0);

        assert!(!mgr.is_conversational(&id));
        assert!(!mgr.start_conversation(&id));
    }

    #[test]
    fn nearest_conversational_npc() {
        let mut mgr = NpcManager::new(0.5);
        let _id1 = mgr.spawn([10.0, 0.0, 10.0], "Guard", [0.8, 0.2, 0.2, 1.0], 2.0);
        let id2 = mgr.spawn_conversational(
            [2.0, 0.0, 2.0],
            "Atlas",
            [0.3, 0.6, 0.9, 1.0],
            1.5,
        );
        let _id3 = mgr.spawn_conversational(
            [20.0, 0.0, 20.0],
            "Elder",
            [0.6, 0.3, 0.1, 1.0],
            1.0,
        );

        let nearest = mgr.nearest_conversational([0.0, 0.0, 0.0], 5.0);
        assert_eq!(nearest, Some(id2));

        let none = mgr.nearest_conversational([0.0, 0.0, 0.0], 1.0);
        assert!(none.is_none());
    }

    #[test]
    fn speech_text() {
        let mut mgr = NpcManager::new(0.5);
        let id = mgr.spawn_conversational(
            [0.0, 0.0, 0.0],
            "Robot",
            [0.5, 0.5, 0.5, 1.0],
            2.0,
        );
        mgr.start_conversation(&id);

        mgr.set_speech(&id, "Hello there!");
        let instances = mgr.instances();
        assert_eq!(instances[0].speech_text.as_deref(), Some("Hello there!"));
    }
}
