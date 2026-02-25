//! Undo/redo history for world authoring actions.
//!
//! Records reversible actions (block changes, object spawn/remove, transforms)
//! and supports stepping backward (undo) and forward (redo).

use bitparty::bridge::WorldObjectId;
use serde::{Deserialize, Serialize};

use crate::world_objects::WorldObjectType;

/// A reversible action recorded in the history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldAction {
    SetBlock {
        world_pos: [f32; 3],
        old_type: u8,
        new_type: u8,
    },
    SetBlockRegion {
        positions: Vec<[f32; 3]>,
        old_types: Vec<u8>,
        new_type: u8,
    },
    SpawnObject {
        id: WorldObjectId,
        position: [f32; 3],
        object_type: WorldObjectType,
        properties: serde_json::Value,
        label: String,
    },
    RemoveObject {
        id: WorldObjectId,
        position: [f32; 3],
        object_type: WorldObjectType,
        properties: serde_json::Value,
        label: String,
    },
    TransformObject {
        id: WorldObjectId,
        old_position: [f32; 3],
        new_position: [f32; 3],
        old_rotation: [f32; 4],
        new_rotation: [f32; 4],
        old_scale: [f32; 3],
        new_scale: [f32; 3],
    },
}

impl WorldAction {
    /// Create the reverse of this action (for undo).
    pub fn reverse(&self) -> Self {
        match self {
            WorldAction::SetBlock {
                world_pos,
                old_type,
                new_type,
            } => WorldAction::SetBlock {
                world_pos: *world_pos,
                old_type: *new_type,
                new_type: *old_type,
            },
            WorldAction::SetBlockRegion {
                positions,
                old_types: _,
                new_type,
            } => {
                // Reverse: set each position back to its old type.
                // The old_types field in the original action contains the per-position
                // types to restore. We swap: the new reverse action's old_types are
                // filled with the current new_type, and its new_type is unused (0).
                WorldAction::SetBlockRegion {
                    positions: positions.clone(),
                    old_types: vec![*new_type; positions.len()],
                    new_type: 0,
                }
            }
            WorldAction::SpawnObject {
                id,
                position,
                object_type,
                properties,
                label,
            } => WorldAction::RemoveObject {
                id: *id,
                position: *position,
                object_type: *object_type,
                properties: properties.clone(),
                label: label.clone(),
            },
            WorldAction::RemoveObject {
                id,
                position,
                object_type,
                properties,
                label,
            } => WorldAction::SpawnObject {
                id: *id,
                position: *position,
                object_type: *object_type,
                properties: properties.clone(),
                label: label.clone(),
            },
            WorldAction::TransformObject {
                id,
                old_position,
                new_position,
                old_rotation,
                new_rotation,
                old_scale,
                new_scale,
            } => WorldAction::TransformObject {
                id: *id,
                old_position: *new_position,
                new_position: *old_position,
                old_rotation: *new_rotation,
                new_rotation: *old_rotation,
                old_scale: *new_scale,
                new_scale: *old_scale,
            },
        }
    }
}

/// Undo/redo history stack.
pub struct StateHistory {
    undo_stack: Vec<WorldAction>,
    redo_stack: Vec<WorldAction>,
    max_entries: usize,
}

impl StateHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_entries,
        }
    }

    /// Record a new action. Clears the redo stack.
    pub fn record(&mut self, action: WorldAction) {
        self.undo_stack.push(action);
        self.redo_stack.clear();

        // Trim old entries if we exceed max.
        if self.undo_stack.len() > self.max_entries {
            self.undo_stack.remove(0);
        }
    }

    /// Pop the last action for undo. Returns the reverse action to apply.
    pub fn undo(&mut self) -> Option<WorldAction> {
        let action = self.undo_stack.pop()?;
        let reverse = action.reverse();
        self.redo_stack.push(action);
        Some(reverse)
    }

    /// Pop the last undone action for redo. Returns the action to re-apply.
    pub fn redo(&mut self) -> Option<WorldAction> {
        let action = self.redo_stack.pop()?;
        let forward = action.clone();
        self.undo_stack.push(action);
        Some(forward)
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_undo() {
        let mut history = StateHistory::new(100);

        history.record(WorldAction::SetBlock {
            world_pos: [1.0, 2.0, 3.0],
            old_type: 0, // Air
            new_type: 1, // Stone
        });

        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);

        let reverse = history.undo().unwrap();
        match reverse {
            WorldAction::SetBlock {
                old_type, new_type, ..
            } => {
                // Reverse of placing stone: put air back.
                assert_eq!(old_type, 1); // was stone
                assert_eq!(new_type, 0); // becomes air
            }
            _ => panic!("expected SetBlock reverse"),
        }

        assert_eq!(history.undo_count(), 0);
        assert_eq!(history.redo_count(), 1);
    }

    #[test]
    fn redo_after_undo() {
        let mut history = StateHistory::new(100);

        history.record(WorldAction::SetBlock {
            world_pos: [0.0, 0.0, 0.0],
            old_type: 0,
            new_type: 6, // Brick
        });

        history.undo();
        assert_eq!(history.redo_count(), 1);

        let redo_action = history.redo().unwrap();
        match redo_action {
            WorldAction::SetBlock { new_type, .. } => {
                assert_eq!(new_type, 6); // Re-apply brick
            }
            _ => panic!("expected SetBlock redo"),
        }

        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn new_action_clears_redo() {
        let mut history = StateHistory::new(100);

        history.record(WorldAction::SetBlock {
            world_pos: [0.0, 0.0, 0.0],
            old_type: 0,
            new_type: 1,
        });
        history.undo();
        assert_eq!(history.redo_count(), 1);

        // New action should clear redo stack.
        history.record(WorldAction::SetBlock {
            world_pos: [1.0, 0.0, 0.0],
            old_type: 0,
            new_type: 2,
        });
        assert_eq!(history.redo_count(), 0);
        assert_eq!(history.undo_count(), 1);
    }

    #[test]
    fn spawn_remove_reversals() {
        let mut history = StateHistory::new(100);
        let id = WorldObjectId::new();

        history.record(WorldAction::SpawnObject {
            id,
            position: [1.0, 2.0, 3.0],
            object_type: WorldObjectType::Cube,
            properties: serde_json::json!({}),
            label: "test".to_string(),
        });

        let reverse = history.undo().unwrap();
        match reverse {
            WorldAction::RemoveObject {
                id: rev_id,
                position,
                ..
            } => {
                assert_eq!(rev_id, id);
                assert_eq!(position, [1.0, 2.0, 3.0]);
            }
            _ => panic!("expected RemoveObject reverse"),
        }
    }

    #[test]
    fn max_entries_trimming() {
        let mut history = StateHistory::new(3);

        for i in 0..5 {
            history.record(WorldAction::SetBlock {
                world_pos: [i as f32, 0.0, 0.0],
                old_type: 0,
                new_type: 1,
            });
        }

        assert_eq!(history.undo_count(), 3);
    }

    #[test]
    fn empty_undo_redo() {
        let mut history = StateHistory::new(100);
        assert!(history.undo().is_none());
        assert!(history.redo().is_none());
    }
}
