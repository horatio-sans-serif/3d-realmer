use noise::{NoiseFn, Perlin};

use super::chunk::{Block, BlockType, Chunk, ChunkCoord, BLOCK_SIZE, CHUNK_SIZE};

/// Procedural terrain generator using layered Perlin noise.
pub struct TerrainGenerator {
    elevation_noise: Perlin,
    detail_noise: Perlin,
    seed: u32,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            elevation_noise: Perlin::new(seed),
            detail_noise: Perlin::new(seed.wrapping_add(1)),
            seed,
        }
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Generate terrain for a chunk at the given coordinate.
    pub fn generate(&self, coord: ChunkCoord) -> Chunk {
        let mut chunk = Chunk::new(coord);
        let origin = coord.world_origin();

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = origin.x + lx as f32 * BLOCK_SIZE;
                let world_z = origin.z + lz as f32 * BLOCK_SIZE;

                let height = self.terrain_height(world_x, world_z);

                for ly in 0..CHUNK_SIZE {
                    let world_y = origin.y + ly as f32 * BLOCK_SIZE;

                    let block_type = if world_y > height {
                        BlockType::Air
                    } else if world_y > height - BLOCK_SIZE {
                        // Top layer
                        if height > 2.0 {
                            BlockType::Grass
                        } else {
                            BlockType::Sand
                        }
                    } else if world_y > height - BLOCK_SIZE * 4.0 {
                        BlockType::Dirt
                    } else {
                        BlockType::Stone
                    };

                    chunk.set(lx, ly, lz, Block::new(block_type));
                }
            }
        }

        chunk
    }

    /// Compute terrain height at a world XZ position.
    /// Returns height in world units (meters).
    fn terrain_height(&self, x: f32, z: f32) -> f32 {
        // Large-scale rolling hills
        let scale1 = 0.01;
        let e1 = self.elevation_noise.get([x as f64 * scale1, z as f64 * scale1]) as f32;

        // Medium detail
        let scale2 = 0.04;
        let e2 = self.detail_noise.get([x as f64 * scale2, z as f64 * scale2]) as f32;

        // Combine: base height 8m, hills up to ~16m, detail ±2m
        let base = 8.0;
        let hills = e1 * 8.0;
        let detail = e2 * 2.0;

        base + hills + detail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_generates_non_empty_chunks() {
        let tgen = TerrainGenerator::new(42);

        // Ground-level chunk should have blocks
        let chunk = tgen.generate(ChunkCoord::new(0, 0, 0));
        assert!(!chunk.is_empty());
    }

    #[test]
    fn terrain_sky_chunks_are_empty() {
        let tgen = TerrainGenerator::new(42);

        // Very high chunk should be all air
        let chunk = tgen.generate(ChunkCoord::new(0, 100, 0));
        assert!(chunk.is_empty());
    }

    #[test]
    fn terrain_deep_chunks_are_solid() {
        let tgen = TerrainGenerator::new(42);

        // Deep underground chunk should have no air
        let chunk = tgen.generate(ChunkCoord::new(0, -10, 0));
        for lx in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    assert_ne!(chunk.get(lx, ly, lz).block_type, BlockType::Air);
                }
            }
        }
    }

    #[test]
    fn terrain_deterministic_with_same_seed() {
        let gen1 = TerrainGenerator::new(42);
        let gen2 = TerrainGenerator::new(42);

        let chunk1 = gen1.generate(ChunkCoord::new(5, 0, -3));
        let chunk2 = gen2.generate(ChunkCoord::new(5, 0, -3));

        for lx in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    assert_eq!(
                        chunk1.get(lx, ly, lz).block_type,
                        chunk2.get(lx, ly, lz).block_type,
                    );
                }
            }
        }
    }

    #[test]
    fn terrain_different_seeds_differ() {
        let gen1 = TerrainGenerator::new(42);
        let gen2 = TerrainGenerator::new(99);

        let chunk1 = gen1.generate(ChunkCoord::new(0, 0, 0));
        let chunk2 = gen2.generate(ChunkCoord::new(0, 0, 0));

        let mut differ = false;
        for lx in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    if chunk1.get(lx, ly, lz).block_type != chunk2.get(lx, ly, lz).block_type {
                        differ = true;
                        break;
                    }
                }
            }
        }
        assert!(differ);
    }

    #[test]
    fn terrain_has_grass_on_top() {
        let tgen = TerrainGenerator::new(42);
        let chunk = tgen.generate(ChunkCoord::new(0, 0, 0));

        // Find a column with terrain and check top block
        let mut found_grass = false;
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in (0..CHUNK_SIZE).rev() {
                    let bt = chunk.get(lx, ly, lz).block_type;
                    if bt != BlockType::Air {
                        if bt == BlockType::Grass {
                            found_grass = true;
                        }
                        break;
                    }
                }
            }
        }
        assert!(found_grass);
    }
}
