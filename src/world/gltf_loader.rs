//! GLTF/GLB model loading.
//!
//! Extracts mesh geometry (positions, normals, indices) from GLTF files
//! for rendering alongside the voxel terrain.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A loaded mesh with vertex and index data.
pub struct LoadedMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

/// Vertex format for loaded meshes.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl MeshVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

/// Cache for loaded GLTF meshes.
pub struct GltfCache {
    meshes: HashMap<PathBuf, LoadedMesh>,
}

impl GltfCache {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    /// Load a GLTF/GLB file. Returns the mesh from cache if already loaded.
    pub fn load(&mut self, path: &Path) -> Result<&LoadedMesh, GltfLoadError> {
        if !self.meshes.contains_key(path) {
            let mesh = load_gltf(path)?;
            self.meshes.insert(path.to_path_buf(), mesh);
        }
        Ok(self.meshes.get(path).unwrap())
    }

    /// Get a previously loaded mesh.
    pub fn get(&self, path: &Path) -> Option<&LoadedMesh> {
        self.meshes.get(path)
    }

    pub fn loaded_count(&self) -> usize {
        self.meshes.len()
    }
}

#[derive(Debug)]
pub enum GltfLoadError {
    Io(std::io::Error),
    Gltf(gltf::Error),
    NoMeshData,
}

impl std::fmt::Display for GltfLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfLoadError::Io(e) => write!(f, "IO error: {}", e),
            GltfLoadError::Gltf(e) => write!(f, "GLTF error: {}", e),
            GltfLoadError::NoMeshData => write!(f, "no mesh data found in GLTF"),
        }
    }
}

impl From<std::io::Error> for GltfLoadError {
    fn from(e: std::io::Error) -> Self {
        GltfLoadError::Io(e)
    }
}

impl From<gltf::Error> for GltfLoadError {
    fn from(e: gltf::Error) -> Self {
        GltfLoadError::Gltf(e)
    }
}

/// Load a GLTF/GLB file and extract the first mesh's geometry.
fn load_gltf(path: &Path) -> Result<LoadedMesh, GltfLoadError> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or(GltfLoadError::NoMeshData)?
                .collect();

            let normals: Vec<[f32; 3]> = if let Some(normals) = reader.read_normals() {
                normals.collect()
            } else {
                // Generate flat normals if not provided.
                vec![[0.0, 1.0, 0.0]; positions.len()]
            };

            let base_vertex = all_vertices.len() as u32;

            for (pos, normal) in positions.iter().zip(normals.iter()) {
                all_vertices.push(MeshVertex {
                    position: *pos,
                    normal: *normal,
                });
            }

            if let Some(indices) = reader.read_indices() {
                for idx in indices.into_u32() {
                    all_indices.push(base_vertex + idx);
                }
            } else {
                // No indices: generate sequential indices.
                let count = positions.len() as u32;
                for i in 0..count {
                    all_indices.push(base_vertex + i);
                }
            }
        }
    }

    if all_vertices.is_empty() {
        return Err(GltfLoadError::NoMeshData);
    }

    Ok(LoadedMesh {
        vertices: all_vertices,
        indices: all_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_vertex_layout() {
        assert_eq!(std::mem::size_of::<MeshVertex>(), 24);
    }

    #[test]
    fn cache_starts_empty() {
        let cache = GltfCache::new();
        assert_eq!(cache.loaded_count(), 0);
    }

    #[test]
    fn load_nonexistent_file() {
        let mut cache = GltfCache::new();
        let result = cache.load(Path::new("nonexistent.glb"));
        assert!(result.is_err());
    }
}
