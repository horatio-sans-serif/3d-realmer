/// Side length of a chunk in blocks.
pub const CHUNK_SIZE: usize = 16;

/// Total number of blocks in a chunk (16^3).
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Block size in meters.
pub const BLOCK_SIZE: f32 = 0.5;

/// Chunk coordinate (signed, allows underground levels with negative Y).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// World-space origin of this chunk (minimum corner).
    pub fn world_origin(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.x as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
            self.y as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
            self.z as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
        )
    }

    /// Convert a world-space position to the chunk coordinate containing it.
    pub fn from_world_pos(pos: glam::Vec3) -> Self {
        let chunk_world_size = CHUNK_SIZE as f32 * BLOCK_SIZE;
        Self {
            x: (pos.x / chunk_world_size).floor() as i32,
            y: (pos.y / chunk_world_size).floor() as i32,
            z: (pos.z / chunk_world_size).floor() as i32,
        }
    }
}

/// Type of a block in the voxel world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Sand = 4,
    Wood = 5,
    Brick = 6,
    Glass = 7,
    Water = 8,
    Leaf = 9,
}

impl BlockType {
    pub fn is_solid(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }

    pub fn is_transparent(self) -> bool {
        matches!(self, BlockType::Air | BlockType::Glass | BlockType::Water)
    }
}

impl Default for BlockType {
    fn default() -> Self {
        BlockType::Air
    }
}

/// A single block in the world.
#[derive(Debug, Clone, Copy, Default)]
pub struct Block {
    pub block_type: BlockType,
}

impl Block {
    pub fn new(block_type: BlockType) -> Self {
        Self { block_type }
    }

    pub fn is_air(&self) -> bool {
        self.block_type == BlockType::Air
    }
}

/// A 16x16x16 chunk of blocks.
pub struct Chunk {
    pub coord: ChunkCoord,
    blocks: Box<[Block; CHUNK_VOLUME]>,
    dirty: bool,
}

impl Chunk {
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            blocks: Box::new([Block::default(); CHUNK_VOLUME]),
            dirty: true,
        }
    }

    /// Get the block at local coordinates (x, y, z) within the chunk.
    /// Each axis ranges from 0 to CHUNK_SIZE-1.
    pub fn get(&self, x: usize, y: usize, z: usize) -> Block {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        self.blocks[Self::index(x, y, z)]
    }

    /// Set the block at local coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, block: Block) {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        self.blocks[Self::index(x, y, z)] = block;
        self.dirty = true;
    }

    /// Whether the chunk mesh needs regeneration.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the chunk as clean (mesh is up to date).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Linear index from 3D coordinates. Y-up, XZY layout for cache-friendly
    /// horizontal iteration (terrain generation iterates X/Z per Y layer).
    fn index(x: usize, y: usize, z: usize) -> usize {
        y * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE + x
    }

    /// Check if the chunk is entirely air.
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(|b| b.is_air())
    }

    /// Fill the entire chunk with a block type.
    pub fn fill(&mut self, block_type: BlockType) {
        let block = Block::new(block_type);
        self.blocks.fill(block);
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_default_is_air() {
        let chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        assert!(chunk.is_empty());
        assert_eq!(chunk.get(0, 0, 0).block_type, BlockType::Air);
    }

    #[test]
    fn chunk_set_and_get() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        chunk.set(5, 10, 3, Block::new(BlockType::Stone));
        assert_eq!(chunk.get(5, 10, 3).block_type, BlockType::Stone);
        assert_eq!(chunk.get(0, 0, 0).block_type, BlockType::Air);
    }

    #[test]
    fn chunk_dirty_tracking() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        assert!(chunk.is_dirty());
        chunk.mark_clean();
        assert!(!chunk.is_dirty());
        chunk.set(0, 0, 0, Block::new(BlockType::Dirt));
        assert!(chunk.is_dirty());
    }

    #[test]
    fn chunk_coord_world_origin() {
        let coord = ChunkCoord::new(1, 0, -1);
        let origin = coord.world_origin();
        assert_eq!(origin.x, 8.0); // 1 * 16 * 0.5
        assert_eq!(origin.y, 0.0);
        assert_eq!(origin.z, -8.0);
    }

    #[test]
    fn chunk_coord_from_world_pos() {
        let coord = ChunkCoord::from_world_pos(glam::Vec3::new(10.0, 0.5, -3.0));
        assert_eq!(coord.x, 1); // 10.0 / 8.0 = 1.25, floor = 1
        assert_eq!(coord.y, 0);
        assert_eq!(coord.z, -1); // -3.0 / 8.0 = -0.375, floor = -1
    }

    #[test]
    fn chunk_fill() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        chunk.fill(BlockType::Stone);
        assert!(!chunk.is_empty());
        assert_eq!(chunk.get(15, 15, 15).block_type, BlockType::Stone);
    }

    #[test]
    fn block_type_properties() {
        assert!(!BlockType::Air.is_solid());
        assert!(BlockType::Stone.is_solid());
        assert!(BlockType::Air.is_transparent());
        assert!(BlockType::Glass.is_transparent());
        assert!(!BlockType::Stone.is_transparent());
        assert!(BlockType::Water.is_transparent());
        assert!(!BlockType::Water.is_solid());
    }
}
