use super::chunk::{BlockType, Chunk, BLOCK_SIZE, CHUNK_SIZE};

/// Vertex for chunk meshes. Position + normal + color.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl ChunkVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: &[wgpu::VertexAttribute] = &[
            // position
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            // normal
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            // color
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ChunkVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

/// Result of meshing a chunk.
pub struct ChunkMesh {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// The six faces of a cube.
#[derive(Debug, Clone, Copy)]
enum Face {
    Top,    // +Y
    Bottom, // -Y
    North,  // +Z
    South,  // -Z
    East,   // +X
    West,   // -X
}

impl Face {
    fn normal(&self) -> [f32; 3] {
        match self {
            Face::Top => [0.0, 1.0, 0.0],
            Face::Bottom => [0.0, -1.0, 0.0],
            Face::North => [0.0, 0.0, 1.0],
            Face::South => [0.0, 0.0, -1.0],
            Face::East => [1.0, 0.0, 0.0],
            Face::West => [-1.0, 0.0, 0.0],
        }
    }
}

/// Color for each block type (simple flat colors for now).
fn block_color(bt: BlockType) -> [f32; 3] {
    match bt {
        BlockType::Air => [0.0; 3],
        BlockType::Stone => [0.5, 0.5, 0.5],
        BlockType::Dirt => [0.45, 0.3, 0.15],
        BlockType::Grass => [0.3, 0.6, 0.2],
        BlockType::Sand => [0.85, 0.8, 0.55],
        BlockType::Wood => [0.55, 0.35, 0.15],
        BlockType::Brick => [0.6, 0.25, 0.2],
        BlockType::Glass => [0.8, 0.9, 1.0],
        BlockType::Water => [0.2, 0.4, 0.8],
        BlockType::Leaf => [0.15, 0.45, 0.1],
    }
}

/// Generate a mesh for a chunk using simple culled face meshing.
/// Only emits faces where a solid block is adjacent to a non-solid block.
pub fn mesh_chunk(chunk: &Chunk) -> ChunkMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let origin = chunk.coord.world_origin();

    for ly in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let block = chunk.get(lx, ly, lz);
                if !block.block_type.is_solid() {
                    continue;
                }

                let color = block_color(block.block_type);
                let bx = origin.x + lx as f32 * BLOCK_SIZE;
                let by = origin.y + ly as f32 * BLOCK_SIZE;
                let bz = origin.z + lz as f32 * BLOCK_SIZE;

                // Check each face
                if should_emit_face(chunk, lx, ly, lz, Face::Top) {
                    emit_face(&mut vertices, &mut indices, bx, by, bz, Face::Top, color);
                }
                if should_emit_face(chunk, lx, ly, lz, Face::Bottom) {
                    emit_face(
                        &mut vertices,
                        &mut indices,
                        bx,
                        by,
                        bz,
                        Face::Bottom,
                        color,
                    );
                }
                if should_emit_face(chunk, lx, ly, lz, Face::North) {
                    emit_face(
                        &mut vertices,
                        &mut indices,
                        bx,
                        by,
                        bz,
                        Face::North,
                        color,
                    );
                }
                if should_emit_face(chunk, lx, ly, lz, Face::South) {
                    emit_face(
                        &mut vertices,
                        &mut indices,
                        bx,
                        by,
                        bz,
                        Face::South,
                        color,
                    );
                }
                if should_emit_face(chunk, lx, ly, lz, Face::East) {
                    emit_face(&mut vertices, &mut indices, bx, by, bz, Face::East, color);
                }
                if should_emit_face(chunk, lx, ly, lz, Face::West) {
                    emit_face(&mut vertices, &mut indices, bx, by, bz, Face::West, color);
                }
            }
        }
    }

    ChunkMesh { vertices, indices }
}

/// Check if a face should be emitted (neighbor is not solid or at chunk boundary).
fn should_emit_face(chunk: &Chunk, x: usize, y: usize, z: usize, face: Face) -> bool {
    let (nx, ny, nz) = match face {
        Face::Top => (x as i32, y as i32 + 1, z as i32),
        Face::Bottom => (x as i32, y as i32 - 1, z as i32),
        Face::North => (x as i32, y as i32, z as i32 + 1),
        Face::South => (x as i32, y as i32, z as i32 - 1),
        Face::East => (x as i32 + 1, y as i32, z as i32),
        Face::West => (x as i32 - 1, y as i32, z as i32),
    };

    // At chunk boundary, always emit face (we'll need cross-chunk culling later)
    if nx < 0
        || ny < 0
        || nz < 0
        || nx >= CHUNK_SIZE as i32
        || ny >= CHUNK_SIZE as i32
        || nz >= CHUNK_SIZE as i32
    {
        return true;
    }

    let neighbor = chunk.get(nx as usize, ny as usize, nz as usize);
    !neighbor.block_type.is_solid()
}

/// Emit a face quad (2 triangles, 4 vertices).
fn emit_face(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    bx: f32,
    by: f32,
    bz: f32,
    face: Face,
    color: [f32; 3],
) {
    let base = vertices.len() as u32;
    let s = BLOCK_SIZE;
    let normal = face.normal();

    let (v0, v1, v2, v3) = match face {
        Face::Top => (
            [bx, by + s, bz],
            [bx + s, by + s, bz],
            [bx + s, by + s, bz + s],
            [bx, by + s, bz + s],
        ),
        Face::Bottom => (
            [bx, by, bz + s],
            [bx + s, by, bz + s],
            [bx + s, by, bz],
            [bx, by, bz],
        ),
        Face::North => (
            [bx + s, by, bz + s],
            [bx, by, bz + s],
            [bx, by + s, bz + s],
            [bx + s, by + s, bz + s],
        ),
        Face::South => (
            [bx, by, bz],
            [bx + s, by, bz],
            [bx + s, by + s, bz],
            [bx, by + s, bz],
        ),
        Face::East => (
            [bx + s, by, bz],
            [bx + s, by, bz + s],
            [bx + s, by + s, bz + s],
            [bx + s, by + s, bz],
        ),
        Face::West => (
            [bx, by, bz + s],
            [bx, by, bz],
            [bx, by + s, bz],
            [bx, by + s, bz + s],
        ),
    };

    vertices.push(ChunkVertex {
        position: v0,
        normal,
        color,
    });
    vertices.push(ChunkVertex {
        position: v1,
        normal,
        color,
    });
    vertices.push(ChunkVertex {
        position: v2,
        normal,
        color,
    });
    vertices.push(ChunkVertex {
        position: v3,
        normal,
        color,
    });

    // Two triangles per quad
    indices.push(base);
    indices.push(base + 1);
    indices.push(base + 2);
    indices.push(base);
    indices.push(base + 2);
    indices.push(base + 3);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::{Block, ChunkCoord};

    #[test]
    fn empty_chunk_produces_no_mesh() {
        let chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        let mesh = mesh_chunk(&chunk);
        assert!(mesh.is_empty());
    }

    #[test]
    fn single_block_produces_6_faces() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        chunk.set(8, 8, 8, Block::new(BlockType::Stone));
        let mesh = mesh_chunk(&chunk);
        // 6 faces * 4 vertices = 24, 6 faces * 6 indices = 36
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn two_adjacent_blocks_share_hidden_face() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        chunk.set(8, 8, 8, Block::new(BlockType::Stone));
        chunk.set(9, 8, 8, Block::new(BlockType::Stone));
        let mesh = mesh_chunk(&chunk);
        // Two blocks with one shared face hidden:
        // Each block has 6 faces, minus 1 shared face each = 10 faces
        // 10 faces * 4 vertices = 40, 10 * 6 indices = 60
        assert_eq!(mesh.vertices.len(), 40);
        assert_eq!(mesh.indices.len(), 60);
    }

    #[test]
    fn full_chunk_only_emits_surface_faces() {
        let mut chunk = Chunk::new(ChunkCoord::new(0, 0, 0));
        chunk.fill(BlockType::Stone);
        let mesh = mesh_chunk(&chunk);
        // Only boundary faces emitted (6 faces of 16x16 = 6*256 = 1536)
        assert_eq!(mesh.indices.len(), 1536 * 6);
    }

    #[test]
    fn vertex_layout_size() {
        assert_eq!(std::mem::size_of::<ChunkVertex>(), 36);
    }

    #[test]
    fn block_colors_are_valid() {
        for bt in [
            BlockType::Stone,
            BlockType::Dirt,
            BlockType::Grass,
            BlockType::Sand,
            BlockType::Wood,
            BlockType::Brick,
            BlockType::Glass,
            BlockType::Water,
            BlockType::Leaf,
        ] {
            let c = block_color(bt);
            assert!(c[0] >= 0.0 && c[0] <= 1.0);
            assert!(c[1] >= 0.0 && c[1] <= 1.0);
            assert!(c[2] >= 0.0 && c[2] <= 1.0);
        }
    }
}
