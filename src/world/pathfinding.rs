//! A* pathfinding on the voxel grid.
//!
//! Operates on the chunk manager's block data: walkable cells are
//! positions where the block below is solid and the block at eye level
//! is air. The nav graph is implicit (neighbors computed on-the-fly).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A grid position for pathfinding (block coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NavPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl NavPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert from world position to nav grid position.
    pub fn from_world(world_pos: [f32; 3], block_size: f32) -> Self {
        Self {
            x: (world_pos[0] / block_size).floor() as i32,
            y: (world_pos[1] / block_size).floor() as i32,
            z: (world_pos[2] / block_size).floor() as i32,
        }
    }

    /// Convert to world position (center of block).
    pub fn to_world(&self, block_size: f32) -> [f32; 3] {
        [
            self.x as f32 * block_size + block_size * 0.5,
            self.y as f32 * block_size + block_size * 0.5,
            self.z as f32 * block_size + block_size * 0.5,
        ]
    }

    fn manhattan_distance(&self, other: &NavPos) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
}

/// A* search node.
#[derive(Debug)]
struct SearchNode {
    pos: NavPos,
    g_cost: f32,
    f_cost: f32,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Callback to check if a position is walkable.
/// Returns true if the entity can stand at this position
/// (solid below, air at feet and head level).
pub type WalkableCheck = Box<dyn Fn(NavPos) -> bool>;

/// Result of a pathfinding query.
pub struct PathResult {
    /// Ordered waypoints from start to goal (inclusive).
    pub waypoints: Vec<NavPos>,
}

impl PathResult {
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    pub fn len(&self) -> usize {
        self.waypoints.len()
    }
}

/// Find a path from start to goal using A*.
/// `is_walkable` checks if the NPC can stand at a given NavPos.
/// `max_iterations` caps the search to prevent frame hitches.
pub fn find_path(
    start: NavPos,
    goal: NavPos,
    is_walkable: &dyn Fn(NavPos) -> bool,
    max_iterations: usize,
) -> Option<PathResult> {
    if start == goal {
        return Some(PathResult {
            waypoints: vec![start],
        });
    }

    if !is_walkable(goal) {
        return None;
    }

    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<NavPos, NavPos> = HashMap::new();
    let mut g_scores: HashMap<NavPos, f32> = HashMap::new();

    g_scores.insert(start, 0.0);
    open.push(SearchNode {
        pos: start,
        g_cost: 0.0,
        f_cost: start.manhattan_distance(&goal) as f32,
    });

    let mut iterations = 0;

    while let Some(current) = open.pop() {
        iterations += 1;
        if iterations > max_iterations {
            return None; // Budget exceeded
        }

        if current.pos == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut pos = goal;
            while let Some(&prev) = came_from.get(&pos) {
                path.push(prev);
                pos = prev;
            }
            path.reverse();
            return Some(PathResult { waypoints: path });
        }

        let current_g = g_scores.get(&current.pos).copied().unwrap_or(f32::MAX);

        // Check all 6 neighbors (plus 4 diagonal neighbors on XZ plane)
        for neighbor in neighbors(current.pos) {
            if !is_walkable(neighbor) {
                continue;
            }

            let move_cost = if neighbor.y != current.pos.y {
                1.5 // Vertical movement is costlier
            } else if neighbor.x != current.pos.x && neighbor.z != current.pos.z {
                1.414 // Diagonal
            } else {
                1.0
            };

            let tentative_g = current_g + move_cost;
            let existing_g = g_scores.get(&neighbor).copied().unwrap_or(f32::MAX);

            if tentative_g < existing_g {
                came_from.insert(neighbor, current.pos);
                g_scores.insert(neighbor, tentative_g);
                let h = neighbor.manhattan_distance(&goal) as f32;
                open.push(SearchNode {
                    pos: neighbor,
                    g_cost: tentative_g,
                    f_cost: tentative_g + h,
                });
            }
        }
    }

    None // No path found
}

/// Generate neighbor positions for A* expansion.
/// Includes 4 cardinal + 4 diagonal on XZ plane + up/down (step climbing).
fn neighbors(pos: NavPos) -> Vec<NavPos> {
    let mut result = Vec::with_capacity(10);

    // Cardinal directions on XZ plane (same Y)
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        result.push(NavPos::new(pos.x + dx, pos.y, pos.z + dz));
    }

    // Diagonal directions on XZ plane
    for (dx, dz) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        result.push(NavPos::new(pos.x + dx, pos.y, pos.z + dz));
    }

    // Step up/down (one block height change)
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        result.push(NavPos::new(pos.x + dx, pos.y + 1, pos.z + dz));
        result.push(NavPos::new(pos.x + dx, pos.y - 1, pos.z + dz));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_world(_pos: NavPos) -> bool {
        true
    }

    #[test]
    fn path_same_position() {
        let start = NavPos::new(0, 0, 0);
        let result = find_path(start, start, &open_world, 1000);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn path_straight_line() {
        let start = NavPos::new(0, 0, 0);
        let goal = NavPos::new(3, 0, 0);
        let result = find_path(start, goal, &open_world, 1000);
        assert!(result.is_some());
        let path = result.unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.waypoints.first().unwrap(), start);
        assert_eq!(*path.waypoints.last().unwrap(), goal);
    }

    #[test]
    fn path_around_wall() {
        // Wall at x=2, z=0..2
        let is_walkable = |pos: NavPos| -> bool {
            !(pos.x == 2 && pos.z >= 0 && pos.z <= 2 && pos.y == 0)
        };

        let start = NavPos::new(0, 0, 1);
        let goal = NavPos::new(4, 0, 1);
        let result = find_path(start, goal, &is_walkable, 5000);
        assert!(result.is_some());
        let path = result.unwrap();
        // Path should go around the wall
        assert!(!path.waypoints.iter().any(|p| p.x == 2 && p.z >= 0 && p.z <= 2 && p.y == 0));
    }

    #[test]
    fn path_unreachable() {
        // Completely blocked
        let is_walkable = |pos: NavPos| -> bool { pos == NavPos::new(0, 0, 0) };

        let start = NavPos::new(0, 0, 0);
        let goal = NavPos::new(5, 0, 5);
        let result = find_path(start, goal, &is_walkable, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn nav_pos_world_conversion() {
        let block_size = 0.5;
        let world = [2.7, 10.3, -1.2];
        let nav = NavPos::from_world(world, block_size);
        let back = nav.to_world(block_size);
        // Should be close to the original (snapped to grid center)
        assert!((back[0] - world[0]).abs() < block_size);
        assert!((back[1] - world[1]).abs() < block_size);
        assert!((back[2] - world[2]).abs() < block_size);
    }

    #[test]
    fn budget_limit() {
        // Very far goal with tiny budget
        let start = NavPos::new(0, 0, 0);
        let goal = NavPos::new(100, 0, 100);
        let result = find_path(start, goal, &open_world, 10);
        assert!(result.is_none());
    }
}
