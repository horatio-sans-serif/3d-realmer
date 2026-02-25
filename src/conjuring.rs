//! AI conjuring pipeline: text prompts -> 3D content.
//!
//! Manages the lifecycle of AI-generated objects:
//! 1. Send prompt to AI API (Meshy -> Tripo -> placeholder fallback)
//! 2. Download generated GLTF/GLB mesh
//! 3. Validate and normalize the mesh
//! 4. Place in world with actor binding
//!
//! Conjuring runs on a background thread to avoid blocking the render loop.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use bitparty::bridge::WorldObjectId;
use serde::{Deserialize, Serialize};

/// Status of a conjuring request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConjureStatus {
    /// Waiting for API response.
    Pending,
    /// Downloading generated model.
    Downloading,
    /// Validating/normalizing the model.
    Validating,
    /// Successfully placed in world.
    Complete,
    /// Failed, using placeholder.
    Failed,
}

/// A conjuring request tracks the lifecycle of a text-to-3D generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConjureRequest {
    pub id: WorldObjectId,
    pub prompt: String,
    pub position: [f32; 3],
    pub status: ConjureStatus,
    pub api_used: Option<String>,
    pub mesh_path: Option<PathBuf>,
    pub error: Option<String>,
}

/// Result delivered back to the main thread.
#[derive(Debug)]
pub struct ConjureResult {
    pub request_id: WorldObjectId,
    pub status: ConjureStatus,
    pub mesh_path: Option<PathBuf>,
    pub api_used: Option<String>,
    pub error: Option<String>,
}

/// Configuration for AI API endpoints.
#[derive(Debug, Clone)]
pub struct ConjureConfig {
    /// Meshy AI API key (primary).
    pub meshy_api_key: Option<String>,
    /// Tripo AI API key (fallback).
    pub tripo_api_key: Option<String>,
    /// Blockade Labs API key (skybox).
    pub blockade_api_key: Option<String>,
    /// Directory to store downloaded meshes.
    pub asset_dir: PathBuf,
}

impl ConjureConfig {
    pub fn from_env(asset_dir: PathBuf) -> Self {
        Self {
            meshy_api_key: std::env::var("MESHY_API_KEY").ok(),
            tripo_api_key: std::env::var("TRIPO_API_KEY").ok(),
            blockade_api_key: std::env::var("BLOCKADE_API_KEY").ok(),
            asset_dir,
        }
    }

    pub fn has_any_api(&self) -> bool {
        self.meshy_api_key.is_some() || self.tripo_api_key.is_some()
    }
}

/// Manages conjuring requests and background processing.
pub struct ConjuringPipeline {
    config: ConjureConfig,
    /// Active requests keyed by WorldObjectId.
    requests: HashMap<WorldObjectId, ConjureRequest>,
    /// Results channel: background thread -> main thread.
    result_rx: crossbeam::channel::Receiver<ConjureResult>,
    result_tx: crossbeam::channel::Sender<ConjureResult>,
}

impl ConjuringPipeline {
    pub fn new(config: ConjureConfig) -> Self {
        let (result_tx, result_rx) = crossbeam::channel::unbounded();

        // Ensure asset directory exists.
        if !config.asset_dir.exists() {
            let _ = std::fs::create_dir_all(&config.asset_dir);
        }

        Self {
            config,
            requests: HashMap::new(),
            result_rx,
            result_tx,
        }
    }

    /// Start a conjuring request. Returns the WorldObjectId for tracking.
    pub fn conjure(&mut self, prompt: &str, position: [f32; 3]) -> WorldObjectId {
        let id = WorldObjectId::new();

        let request = ConjureRequest {
            id,
            prompt: prompt.to_string(),
            position,
            status: ConjureStatus::Pending,
            api_used: None,
            mesh_path: None,
            error: None,
        };

        self.requests.insert(id, request.clone());

        // Spawn background thread for the API call.
        let config = self.config.clone();
        let tx = self.result_tx.clone();
        let prompt = prompt.to_string();

        thread::spawn(move || {
            let result = execute_conjure_pipeline(&config, &id, &prompt);
            let _ = tx.send(result);
        });

        id
    }

    /// Poll for completed conjuring results. Returns all newly completed requests.
    pub fn poll_results(&mut self) -> Vec<ConjureResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            if let Some(req) = self.requests.get_mut(&result.request_id) {
                req.status = result.status.clone();
                req.mesh_path = result.mesh_path.clone();
                req.api_used = result.api_used.clone();
                req.error = result.error.clone();
            }
            results.push(result);
        }
        results
    }

    /// Get a request by ID.
    pub fn get_request(&self, id: &WorldObjectId) -> Option<&ConjureRequest> {
        self.requests.get(id)
    }

    /// Retry a failed conjuring request.
    pub fn retry(&mut self, id: &WorldObjectId) -> bool {
        let req = match self.requests.get(id) {
            Some(r) if r.status == ConjureStatus::Failed => r.clone(),
            _ => return false,
        };

        // Re-submit the same request.
        if let Some(r) = self.requests.get_mut(id) {
            r.status = ConjureStatus::Pending;
            r.error = None;
        }

        let config = self.config.clone();
        let tx = self.result_tx.clone();
        let prompt = req.prompt.clone();
        let request_id = *id;

        thread::spawn(move || {
            let result = execute_conjure_pipeline(&config, &request_id, &prompt);
            let _ = tx.send(result);
        });

        true
    }

    pub fn pending_count(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.status == ConjureStatus::Pending || r.status == ConjureStatus::Downloading)
            .count()
    }
}

/// Execute the full conjuring pipeline (runs on background thread).
fn execute_conjure_pipeline(
    config: &ConjureConfig,
    request_id: &WorldObjectId,
    prompt: &str,
) -> ConjureResult {
    // Try Meshy first, then Tripo, then placeholder.
    if let Some(ref api_key) = config.meshy_api_key {
        match meshy_generate(api_key, prompt, &config.asset_dir, request_id) {
            Ok(path) => {
                return ConjureResult {
                    request_id: *request_id,
                    status: ConjureStatus::Complete,
                    mesh_path: Some(path),
                    api_used: Some("meshy".to_string()),
                    error: None,
                };
            }
            Err(e) => {
                log::warn!("Meshy API failed for '{}': {}", prompt, e);
            }
        }
    }

    if let Some(ref api_key) = config.tripo_api_key {
        match tripo_generate(api_key, prompt, &config.asset_dir, request_id) {
            Ok(path) => {
                return ConjureResult {
                    request_id: *request_id,
                    status: ConjureStatus::Complete,
                    mesh_path: Some(path),
                    api_used: Some("tripo".to_string()),
                    error: None,
                };
            }
            Err(e) => {
                log::warn!("Tripo API failed for '{}': {}", prompt, e);
            }
        }
    }

    // Both APIs failed or no keys configured - use placeholder.
    log::info!(
        "Conjuring '{}' falling back to placeholder (no API keys or APIs failed)",
        prompt
    );

    ConjureResult {
        request_id: *request_id,
        status: ConjureStatus::Failed,
        mesh_path: None,
        api_used: None,
        error: Some("All API providers failed or no API keys configured".to_string()),
    }
}

/// Generate a 3D model via Meshy AI.
/// Returns the path to the downloaded GLTF file.
fn meshy_generate(
    api_key: &str,
    prompt: &str,
    asset_dir: &Path,
    request_id: &WorldObjectId,
) -> Result<PathBuf, String> {
    let client = reqwest::blocking::Client::new();

    // Step 1: Create a text-to-3D task.
    let create_resp = client
        .post("https://api.meshy.ai/v2/text-to-3d")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "mode": "preview",
            "prompt": prompt,
            "art_style": "realistic",
            "negative_prompt": "low quality, distorted",
        }))
        .send()
        .map_err(|e| format!("HTTP error: {}", e))?;

    if !create_resp.status().is_success() {
        return Err(format!(
            "Meshy create failed: {} - {}",
            create_resp.status(),
            create_resp.text().unwrap_or_default()
        ));
    }

    let create_body: serde_json::Value = create_resp
        .json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let task_id = create_body
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or("Missing task ID in Meshy response")?;

    // Step 2: Poll for completion.
    let poll_url = format!("https://api.meshy.ai/v2/text-to-3d/{}", task_id);
    let mut attempts = 0;
    let max_attempts = 60; // 5 minutes at 5s intervals

    loop {
        attempts += 1;
        if attempts > max_attempts {
            return Err("Meshy generation timed out".to_string());
        }

        std::thread::sleep(std::time::Duration::from_secs(5));

        let status_resp = client
            .get(&poll_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .map_err(|e| format!("Poll HTTP error: {}", e))?;

        let status_body: serde_json::Value = status_resp
            .json()
            .map_err(|e| format!("Poll JSON error: {}", e))?;

        let status = status_body
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match status {
            "SUCCEEDED" => {
                // Get the model URL.
                let model_url = status_body
                    .get("model_urls")
                    .and_then(|v| v.get("glb"))
                    .and_then(|v| v.as_str())
                    .ok_or("No GLB URL in Meshy response")?;

                return download_model(model_url, asset_dir, request_id);
            }
            "FAILED" | "EXPIRED" => {
                let error = status_body
                    .get("task_error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                return Err(format!("Meshy generation failed: {}", error));
            }
            _ => {
                // Still processing, continue polling.
                continue;
            }
        }
    }
}

/// Generate a 3D model via Tripo AI.
fn tripo_generate(
    api_key: &str,
    prompt: &str,
    asset_dir: &Path,
    request_id: &WorldObjectId,
) -> Result<PathBuf, String> {
    let client = reqwest::blocking::Client::new();

    // Step 1: Create a text-to-model task.
    let create_resp = client
        .post("https://api.tripo3d.ai/v2/openapi/task")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "type": "text_to_model",
            "prompt": prompt,
        }))
        .send()
        .map_err(|e| format!("HTTP error: {}", e))?;

    if !create_resp.status().is_success() {
        return Err(format!(
            "Tripo create failed: {} - {}",
            create_resp.status(),
            create_resp.text().unwrap_or_default()
        ));
    }

    let create_body: serde_json::Value = create_resp
        .json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let task_id = create_body
        .get("data")
        .and_then(|v| v.get("task_id"))
        .and_then(|v| v.as_str())
        .ok_or("Missing task_id in Tripo response")?;

    // Step 2: Poll for completion.
    let poll_url = format!(
        "https://api.tripo3d.ai/v2/openapi/task/{}",
        task_id
    );
    let mut attempts = 0;
    let max_attempts = 60;

    loop {
        attempts += 1;
        if attempts > max_attempts {
            return Err("Tripo generation timed out".to_string());
        }

        std::thread::sleep(std::time::Duration::from_secs(5));

        let status_resp = client
            .get(&poll_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .map_err(|e| format!("Poll HTTP error: {}", e))?;

        let status_body: serde_json::Value = status_resp
            .json()
            .map_err(|e| format!("Poll JSON error: {}", e))?;

        let status = status_body
            .get("data")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match status {
            "success" => {
                let model_url = status_body
                    .get("data")
                    .and_then(|v| v.get("output"))
                    .and_then(|v| v.get("model"))
                    .and_then(|v| v.as_str())
                    .ok_or("No model URL in Tripo response")?;

                return download_model(model_url, asset_dir, request_id);
            }
            "failed" => {
                return Err("Tripo generation failed".to_string());
            }
            _ => continue,
        }
    }
}

/// Download a model from URL to the asset directory.
fn download_model(
    url: &str,
    asset_dir: &Path,
    request_id: &WorldObjectId,
) -> Result<PathBuf, String> {
    let client = reqwest::blocking::Client::new();

    let resp = client
        .get(url)
        .send()
        .map_err(|e| format!("Download error: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Download failed: {}", resp.status()));
    }

    let bytes = resp
        .bytes()
        .map_err(|e| format!("Read bytes error: {}", e))?;

    let file_name = format!("{}.glb", request_id);
    let file_path = asset_dir.join(&file_name);

    std::fs::write(&file_path, &bytes)
        .map_err(|e| format!("Write file error: {}", e))?;

    log::info!(
        "Downloaded conjured model to {} ({} bytes)",
        file_path.display(),
        bytes.len()
    );

    Ok(file_path)
}

/// Validate a downloaded GLTF/GLB mesh.
/// Returns Ok if valid, Err with description of issues.
pub fn validate_mesh(path: &Path) -> Result<MeshValidation, String> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("GLTF parse error: {}", e))?;

    let mut total_vertices = 0u64;
    let mut total_triangles = 0u64;
    let mut has_normals = true;
    let mut has_uvs = true;
    let mut warnings = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(positions) = primitive.get(&gltf::Semantic::Positions) {
                total_vertices += positions.count() as u64;
            }

            if primitive.get(&gltf::Semantic::Normals).is_none() {
                has_normals = false;
                warnings.push("Missing normals (will be generated)".to_string());
            }

            if primitive.get(&gltf::Semantic::TexCoords(0)).is_none() {
                has_uvs = false;
                warnings.push("Missing UV coordinates".to_string());
            }

            if let Some(indices) = primitive.indices() {
                total_triangles += indices.count() as u64 / 3;
            }
        }
    }

    if total_vertices == 0 {
        return Err("Mesh contains no vertices".to_string());
    }

    // Check for unreasonable sizes
    if total_triangles > 500_000 {
        warnings.push(format!(
            "High triangle count: {} (may impact performance)",
            total_triangles
        ));
    }

    Ok(MeshValidation {
        vertices: total_vertices,
        triangles: total_triangles,
        has_normals,
        has_uvs,
        warnings,
    })
}

/// Results of mesh validation.
#[derive(Debug, Clone)]
pub struct MeshValidation {
    pub vertices: u64,
    pub triangles: u64,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub warnings: Vec<String>,
}

/// Generate a skybox via Blockade Labs.
pub fn generate_skybox(
    api_key: &str,
    prompt: &str,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let client = reqwest::blocking::Client::new();

    // Create skybox generation request.
    let resp = client
        .post("https://backend.blockadelabs.com/api/v1/skybox")
        .header("x-api-key", api_key)
        .json(&serde_json::json!({
            "prompt": prompt,
            "skybox_style_id": 2, // Digital painting style
        }))
        .send()
        .map_err(|e| format!("HTTP error: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Blockade create failed: {}",
            resp.status()
        ));
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let skybox_id = body
        .get("id")
        .and_then(|v| v.as_i64())
        .ok_or("Missing skybox ID")?;

    // Poll for completion.
    let mut attempts = 0;
    let max_attempts = 60;

    loop {
        attempts += 1;
        if attempts > max_attempts {
            return Err("Skybox generation timed out".to_string());
        }

        std::thread::sleep(std::time::Duration::from_secs(3));

        let status_resp = client
            .get(format!(
                "https://backend.blockadelabs.com/api/v1/imagine/requests/{}",
                skybox_id
            ))
            .header("x-api-key", api_key)
            .send()
            .map_err(|e| format!("Poll error: {}", e))?;

        let status_body: serde_json::Value = status_resp
            .json()
            .map_err(|e| format!("Poll JSON error: {}", e))?;

        let status = status_body
            .get("request")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if status == "complete" {
            let image_url = status_body
                .get("request")
                .and_then(|v| v.get("file_url"))
                .and_then(|v| v.as_str())
                .ok_or("No image URL in skybox response")?;

            // Download the skybox image.
            let img_resp = client
                .get(image_url)
                .send()
                .map_err(|e| format!("Download error: {}", e))?;

            let bytes = img_resp
                .bytes()
                .map_err(|e| format!("Read error: {}", e))?;

            let file_path = output_dir.join(format!("skybox_{}.jpg", skybox_id));
            std::fs::write(&file_path, &bytes)
                .map_err(|e| format!("Write error: {}", e))?;

            return Ok(file_path);
        } else if status == "error" {
            return Err("Skybox generation failed".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjure_config_from_env() {
        let config = ConjureConfig::from_env(PathBuf::from("/tmp/test_assets"));
        // API keys depend on environment, but the struct should initialize
        assert_eq!(config.asset_dir, PathBuf::from("/tmp/test_assets"));
    }

    #[test]
    fn conjure_request_lifecycle() {
        let mut pipeline = ConjuringPipeline::new(ConjureConfig {
            meshy_api_key: None,
            tripo_api_key: None,
            blockade_api_key: None,
            asset_dir: PathBuf::from("/tmp/test_conjure"),
        });

        // Without API keys, conjuring should fall back to placeholder.
        let id = pipeline.conjure("a wooden table", [0.0, 10.0, 0.0]);

        // Give the background thread time to complete.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let results = pipeline.poll_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, ConjureStatus::Failed);
        assert!(results[0].error.is_some());

        let req = pipeline.get_request(&id).unwrap();
        assert_eq!(req.prompt, "a wooden table");
    }

    #[test]
    fn conjure_status_serialization() {
        let json = serde_json::to_string(&ConjureStatus::Pending).unwrap();
        assert!(json.contains("Pending"));

        let decoded: ConjureStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ConjureStatus::Pending);
    }

    #[test]
    fn pending_count() {
        let pipeline = ConjuringPipeline::new(ConjureConfig {
            meshy_api_key: None,
            tripo_api_key: None,
            blockade_api_key: None,
            asset_dir: PathBuf::from("/tmp/test_conjure"),
        });
        assert_eq!(pipeline.pending_count(), 0);
    }
}
