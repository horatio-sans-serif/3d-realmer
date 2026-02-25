use glam::Vec3;
use rapier3d::prelude::*;

use super::mesher::ChunkMesh;

/// Generate a Rapier trimesh collider from a chunk mesh.
/// Vertices are converted to local space relative to `chunk_origin`,
/// so the collider should be attached to a body positioned at that origin.
/// Returns None if the mesh is empty.
pub fn collider_from_chunk_mesh(mesh: &ChunkMesh, chunk_origin: Vec3) -> Option<Collider> {
    if mesh.is_empty() {
        return None;
    }

    let vertices: Vec<Point<Real>> = mesh
        .vertices
        .iter()
        .map(|v| {
            point![
                v.position[0] - chunk_origin.x,
                v.position[1] - chunk_origin.y,
                v.position[2] - chunk_origin.z
            ]
        })
        .collect();

    let indices: Vec<[u32; 3]> = mesh
        .indices
        .chunks(3)
        .map(|tri| [tri[0], tri[1], tri[2]])
        .collect();

    Some(
        ColliderBuilder::trimesh(vertices, indices)
            .friction(0.8)
            .restitution(0.0)
            .build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::{Block, BlockType, Chunk, ChunkCoord};
    use crate::world::mesher::mesh_chunk;

    #[test]
    fn empty_chunk_no_collider() {
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = Chunk::new(coord);
        let mesh = mesh_chunk(&chunk);
        assert!(collider_from_chunk_mesh(&mesh, coord.world_origin()).is_none());
    }

    #[test]
    fn solid_chunk_produces_collider() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = Chunk::new(coord);
        chunk.fill(BlockType::Stone);
        let mesh = mesh_chunk(&chunk);
        let collider = collider_from_chunk_mesh(&mesh, coord.world_origin());
        assert!(collider.is_some());
    }

    #[test]
    fn single_block_collider() {
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = Chunk::new(coord);
        chunk.set(8, 8, 8, Block::new(BlockType::Stone));
        let mesh = mesh_chunk(&chunk);
        let collider = collider_from_chunk_mesh(&mesh, coord.world_origin());
        assert!(collider.is_some());
    }
}
