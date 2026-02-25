use std::collections::HashMap;
use std::path::Path;

use rapier3d::prelude::RigidBodyHandle;

use super::chunk::{Block, BlockType, Chunk, ChunkCoord, BLOCK_SIZE, CHUNK_SIZE};
use super::chunk_render::ChunkRenderer;
use super::chunk_store::ChunkStore;
use super::mesher::mesh_chunk;
use super::terrain::TerrainGenerator;
use super::terrain_collider::collider_from_chunk_mesh;
use crate::physics::PhysicsWorld;

/// Manages chunk lifecycle: generation, meshing, loading/unloading around the player.
pub struct ChunkManager {
    chunks: HashMap<ChunkCoord, Chunk>,
    /// Tracks which chunks have been modified by the player (need saving).
    modified_chunks: std::collections::HashSet<ChunkCoord>,
    terrain: TerrainGenerator,
    store: ChunkStore,
    /// Physics bodies for terrain colliders, keyed by chunk coordinate.
    terrain_bodies: HashMap<ChunkCoord, RigidBodyHandle>,
    load_radius: i32,
    /// Y range of chunks to load (min..=max).
    y_min: i32,
    y_max: i32,
}

impl ChunkManager {
    pub fn new(seed: u32, world_path: &Path) -> Self {
        let store = ChunkStore::open(world_path)
            .expect("failed to open chunk store");
        Self {
            chunks: HashMap::new(),
            modified_chunks: std::collections::HashSet::new(),
            terrain: TerrainGenerator::new(seed),
            store,
            terrain_bodies: HashMap::new(),
            load_radius: 4,
            y_min: -1,
            y_max: 3,
        }
    }

    #[cfg(test)]
    fn new_in_memory(seed: u32) -> Self {
        let store = ChunkStore::open_in_memory()
            .expect("failed to open in-memory chunk store");
        Self {
            chunks: HashMap::new(),
            modified_chunks: std::collections::HashSet::new(),
            terrain: TerrainGenerator::new(seed),
            store,
            terrain_bodies: HashMap::new(),
            load_radius: 4,
            y_min: -1,
            y_max: 3,
        }
    }

    /// Update loaded chunks based on player position.
    /// Returns true if any chunks changed (need re-upload to GPU).
    pub fn update(
        &mut self,
        player_pos: glam::Vec3,
        chunk_renderer: &mut ChunkRenderer,
        device: &wgpu::Device,
        physics: &mut PhysicsWorld,
    ) -> bool {
        let center = ChunkCoord::from_world_pos(player_pos);
        let mut changed = false;

        // Load new chunks within radius
        for dy in self.y_min..=self.y_max {
            for dz in -self.load_radius..=self.load_radius {
                for dx in -self.load_radius..=self.load_radius {
                    let coord = ChunkCoord::new(center.x + dx, center.y + dy, center.z + dz);

                    if !self.chunks.contains_key(&coord) {
                        // Try loading from disk first, fall back to terrain generation
                        let chunk = match self.store.load_chunk(coord) {
                            Ok(Some(c)) => c,
                            _ => self.terrain.generate(coord),
                        };
                        let mesh = mesh_chunk(&chunk);

                        // Upload to GPU for rendering
                        chunk_renderer.upload_chunk(device, coord, &mesh);

                        // Register terrain collider in physics
                        let origin = coord.world_origin();
                        if let Some(collider) = collider_from_chunk_mesh(&mesh, origin) {
                            let handle = physics.insert_terrain_body(
                                [origin.x, origin.y, origin.z],
                                collider,
                            );
                            self.terrain_bodies.insert(coord, handle);
                        }

                        self.chunks.insert(coord, chunk);
                        changed = true;
                    }
                }
            }
        }

        // Unload chunks outside radius (with extra buffer to avoid thrashing)
        let unload_radius = self.load_radius + 2;
        let to_unload: Vec<ChunkCoord> = self
            .chunks
            .keys()
            .filter(|coord| {
                let dx = (coord.x - center.x).abs();
                let dz = (coord.z - center.z).abs();
                let dy_low = coord.y - center.y - self.y_min;
                let dy_high = coord.y - center.y - self.y_max;
                dx > unload_radius || dz > unload_radius || dy_low < -2 || dy_high > 2
            })
            .copied()
            .collect();

        for coord in to_unload {
            // Save modified chunks before unloading
            if self.modified_chunks.remove(&coord) {
                if let Some(chunk) = self.chunks.get(&coord) {
                    let _ = self.store.save_chunk(chunk);
                }
            }

            self.chunks.remove(&coord);
            chunk_renderer.remove_chunk(&coord);
            // Remove terrain collider from physics
            if let Some(handle) = self.terrain_bodies.remove(&coord) {
                physics.remove_body_by_handle(handle);
            }
            changed = true;
        }

        // Re-mesh dirty chunks
        let dirty_coords: Vec<ChunkCoord> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.is_dirty())
            .map(|(coord, _)| *coord)
            .collect();

        for coord in dirty_coords {
            if let Some(chunk) = self.chunks.get_mut(&coord) {
                let mesh = mesh_chunk(chunk);
                chunk_renderer.upload_chunk(device, coord, &mesh);

                // Rebuild terrain collider for this chunk
                if let Some(old_handle) = self.terrain_bodies.remove(&coord) {
                    physics.remove_body_by_handle(old_handle);
                }
                let origin = coord.world_origin();
                if let Some(collider) = collider_from_chunk_mesh(&mesh, origin) {
                    let handle = physics.insert_terrain_body(
                        [origin.x, origin.y, origin.z],
                        collider,
                    );
                    self.terrain_bodies.insert(coord, handle);
                }

                chunk.mark_clean();
                changed = true;
            }
        }

        changed
    }

    /// Set a block at a world position. Returns true if the block was changed.
    pub fn set_block(&mut self, world_pos: glam::Vec3, block_type: BlockType) -> bool {
        let coord = ChunkCoord::from_world_pos(world_pos);
        let chunk = match self.chunks.get_mut(&coord) {
            Some(c) => c,
            None => return false,
        };

        let origin = coord.world_origin();
        let lx = ((world_pos.x - origin.x) / BLOCK_SIZE) as usize;
        let ly = ((world_pos.y - origin.y) / BLOCK_SIZE) as usize;
        let lz = ((world_pos.z - origin.z) / BLOCK_SIZE) as usize;

        if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            return false;
        }

        chunk.set(lx, ly, lz, Block::new(block_type));
        self.modified_chunks.insert(coord);
        true
    }

    /// Get the block type at a world position.
    pub fn get_block(&self, world_pos: glam::Vec3) -> Option<BlockType> {
        let coord = ChunkCoord::from_world_pos(world_pos);
        let chunk = self.chunks.get(&coord)?;

        let origin = coord.world_origin();
        let lx = ((world_pos.x - origin.x) / BLOCK_SIZE) as usize;
        let ly = ((world_pos.y - origin.y) / BLOCK_SIZE) as usize;
        let lz = ((world_pos.z - origin.z) / BLOCK_SIZE) as usize;

        if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            return None;
        }

        Some(chunk.get(lx, ly, lz).block_type)
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn load_radius(&self) -> i32 {
        self.load_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_manager_creation() {
        let cm = ChunkManager::new_in_memory(42);
        assert_eq!(cm.loaded_chunk_count(), 0);
    }

    #[test]
    fn set_and_get_block() {
        let mut cm = ChunkManager::new_in_memory(42);

        // Manually insert a chunk so we can test set/get
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = cm.terrain.generate(coord);
        cm.chunks.insert(coord, chunk);

        // Set a block
        let pos = glam::Vec3::new(1.0, 1.0, 1.0);
        assert!(cm.set_block(pos, BlockType::Brick));
        assert_eq!(cm.get_block(pos), Some(BlockType::Brick));
    }

    #[test]
    fn get_block_outside_loaded_chunks() {
        let cm = ChunkManager::new_in_memory(42);
        let pos = glam::Vec3::new(1000.0, 1000.0, 1000.0);
        assert_eq!(cm.get_block(pos), None);
    }
}
