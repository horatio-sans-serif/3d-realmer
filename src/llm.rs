//! LLM pipeline: conversation and reasoning via Claude API.
//!
//! Manages the lifecycle of LLM conversations:
//! 1. Accept user/NPC text input
//! 2. Send to Anthropic Claude API with conversation history
//! 3. Parse response including tool calls (conjure objects, etc.)
//! 4. Return structured results to the main thread
//!
//! Runs API calls on background threads to avoid blocking the render loop.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use bitparty::bridge::WorldObjectId;
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};

/// Role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: Role,
    pub content: String,
}

/// A tool call the LLM wants to make.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Result of an LLM conversation turn.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// The NPC or conversation ID this response belongs to.
    pub conversation_id: WorldObjectId,
    /// The text response from the LLM.
    pub text: String,
    /// Tool calls the LLM wants to make (conjure, etc.).
    pub tool_calls: Vec<ToolCall>,
    /// Whether this response indicates an error.
    pub error: Option<String>,
}

/// An LLM request to be processed.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub conversation_id: WorldObjectId,
    pub system_prompt: String,
    pub messages: Vec<ConversationMessage>,
    pub tools: Vec<ToolDefinition>,
}

/// Tool definition for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Configuration for the LLM API.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: u32,
}

impl LlmConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            model: std::env::var("BITPARTY_LLM_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()),
            max_tokens: 1024,
        }
    }

    pub fn is_configured(&self) -> bool {
        self.api_key.is_some()
    }
}

/// The LLM pipeline manages background API calls.
pub struct LlmPipeline {
    config: Arc<LlmConfig>,
    result_rx: Receiver<LlmResponse>,
    result_tx: Sender<LlmResponse>,
}

impl LlmPipeline {
    pub fn new(config: LlmConfig) -> Self {
        let (result_tx, result_rx) = crossbeam::channel::unbounded();
        Self {
            config: Arc::new(config),
            result_rx,
            result_tx,
        }
    }

    /// Submit a conversation request for background processing.
    pub fn submit(&self, request: LlmRequest) {
        let config = self.config.clone();
        let result_tx = self.result_tx.clone();

        thread::Builder::new()
            .name(format!(
                "llm-{}",
                request.conversation_id.as_uuid().as_u128() % 10000
            ))
            .spawn(move || {
                let result = execute_llm_request(&config, &request);
                let _ = result_tx.send(result);
            })
            .ok();
    }

    /// Poll for completed LLM responses (non-blocking).
    pub fn poll_results(&self) -> Vec<LlmResponse> {
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            results.push(result);
        }
        results
    }

    pub fn is_configured(&self) -> bool {
        self.config.is_configured()
    }
}

/// Execute an LLM request against the Anthropic API.
fn execute_llm_request(config: &LlmConfig, request: &LlmRequest) -> LlmResponse {
    let api_key = match &config.api_key {
        Some(key) => key,
        None => {
            return LlmResponse {
                conversation_id: request.conversation_id,
                text: "LLM not configured (set ANTHROPIC_API_KEY)".to_string(),
                tool_calls: Vec::new(),
                error: Some("no_api_key".to_string()),
            };
        }
    };

    // Build the API request body.
    let mut body = serde_json::json!({
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": request.system_prompt,
        "messages": request.messages,
    });

    // Add tools if provided.
    if !request.tools.is_empty() {
        body["tools"] = serde_json::to_value(&request.tools).unwrap_or_default();
    }

    let client = reqwest::blocking::Client::new();

    let response = match client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            return LlmResponse {
                conversation_id: request.conversation_id,
                text: String::new(),
                tool_calls: Vec::new(),
                error: Some(format!("HTTP error: {}", e)),
            };
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body_text = response.text().unwrap_or_default();
        return LlmResponse {
            conversation_id: request.conversation_id,
            text: String::new(),
            tool_calls: Vec::new(),
            error: Some(format!("API error {}: {}", status, body_text)),
        };
    }

    let response_json: serde_json::Value = match response.json() {
        Ok(j) => j,
        Err(e) => {
            return LlmResponse {
                conversation_id: request.conversation_id,
                text: String::new(),
                tool_calls: Vec::new(),
                error: Some(format!("JSON parse error: {}", e)),
            };
        }
    };

    // Parse the response content blocks.
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = response_json.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    tool_calls.push(ToolCall { id, name, input });
                }
                _ => {}
            }
        }
    }

    LlmResponse {
        conversation_id: request.conversation_id,
        text: text_parts.join("\n"),
        tool_calls,
        error: None,
    }
}

/// Standard tools available to NPC conversations.
pub fn npc_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "conjure_object".to_string(),
            description: "Create a 3D object in the world from a text description. Use this when the user asks to see something or when explaining a concept that would benefit from a visual representation.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description of the 3D object to create"
                    },
                    "position_offset": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional [x,y,z] offset from the NPC's position"
                    }
                },
                "required": ["description"]
            }),
        },
        ToolDefinition {
            name: "set_weather".to_string(),
            description: "Change the weather in the world.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "string",
                        "enum": ["clear", "rain", "snow"],
                        "description": "The weather type to set"
                    }
                },
                "required": ["weather"]
            }),
        },
        ToolDefinition {
            name: "set_time".to_string(),
            description: "Change the time of day in the world.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "enum": ["dawn", "morning", "noon", "afternoon", "dusk", "night", "midnight"],
                        "description": "The time of day to set"
                    }
                },
                "required": ["time"]
            }),
        },
    ]
}

/// NPC personality template.
#[derive(Debug, Clone)]
pub struct NpcPersonality {
    pub name: String,
    pub description: String,
    pub traits: Vec<String>,
    pub knowledge_areas: Vec<String>,
}

impl NpcPersonality {
    /// Build a system prompt from this personality.
    pub fn system_prompt(&self) -> String {
        let traits_str = if self.traits.is_empty() {
            String::new()
        } else {
            format!("\n\nPersonality traits: {}", self.traits.join(", "))
        };

        let knowledge_str = if self.knowledge_areas.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nYou have expertise in: {}",
                self.knowledge_areas.join(", ")
            )
        };

        format!(
            "You are {}, {}. You exist as an NPC in a 3D voxel world called BitParty World. \
             You can conjure 3D objects to illustrate concepts using the conjure_object tool. \
             You can also control the weather and time of day.\n\
             Keep responses concise (2-3 sentences). Be helpful and engaging.{}{}\n\n\
             When asked to show or demonstrate something, use the conjure_object tool to create \
             a relevant 3D visualization.",
            self.name, self.description, traits_str, knowledge_str
        )
    }

    pub fn research_robot() -> Self {
        Self {
            name: "Atlas".to_string(),
            description: "a research robot that helps users explore concepts by creating 3D visualizations".to_string(),
            traits: vec![
                "curious".to_string(),
                "helpful".to_string(),
                "visual thinker".to_string(),
            ],
            knowledge_areas: vec![
                "science".to_string(),
                "technology".to_string(),
                "architecture".to_string(),
                "nature".to_string(),
                "art".to_string(),
            ],
        }
    }

    pub fn village_elder() -> Self {
        Self {
            name: "Elder Sage".to_string(),
            description: "a wise village elder who knows the history of the world".to_string(),
            traits: vec![
                "wise".to_string(),
                "patient".to_string(),
                "storyteller".to_string(),
            ],
            knowledge_areas: vec![
                "history".to_string(),
                "philosophy".to_string(),
                "legends".to_string(),
            ],
        }
    }
}

/// Conversation state for an NPC.
#[derive(Debug, Clone)]
pub struct ConversationState {
    pub npc_id: WorldObjectId,
    pub personality: NpcPersonality,
    pub history: Vec<ConversationMessage>,
    pub awaiting_response: bool,
    /// Maximum conversation history to send to the LLM.
    pub max_history: usize,
}

impl ConversationState {
    pub fn new(npc_id: WorldObjectId, personality: NpcPersonality) -> Self {
        Self {
            npc_id,
            personality,
            history: Vec::new(),
            awaiting_response: false,
            max_history: 20,
        }
    }

    /// Add a user message and build an LLM request.
    pub fn user_message(&mut self, text: &str) -> LlmRequest {
        self.history.push(ConversationMessage {
            role: Role::User,
            content: text.to_string(),
        });

        // Trim history to max length.
        if self.history.len() > self.max_history {
            let drain_count = self.history.len() - self.max_history;
            self.history.drain(..drain_count);
        }

        self.awaiting_response = true;

        LlmRequest {
            conversation_id: self.npc_id,
            system_prompt: self.personality.system_prompt(),
            messages: self.history.clone(),
            tools: npc_tools(),
        }
    }

    /// Record the assistant's response.
    pub fn assistant_response(&mut self, text: &str) {
        self.history.push(ConversationMessage {
            role: Role::Assistant,
            content: text.to_string(),
        });
        self.awaiting_response = false;

        // Trim history to max length.
        if self.history.len() > self.max_history {
            let drain_count = self.history.len() - self.max_history;
            self.history.drain(..drain_count);
        }
    }
}

/// Manages all NPC conversations.
pub struct ConversationManager {
    conversations: HashMap<WorldObjectId, ConversationState>,
}

impl ConversationManager {
    pub fn new() -> Self {
        Self {
            conversations: HashMap::new(),
        }
    }

    /// Register an NPC for conversation.
    pub fn register(&mut self, npc_id: WorldObjectId, personality: NpcPersonality) {
        self.conversations.insert(
            npc_id,
            ConversationState::new(npc_id, personality),
        );
    }

    /// Get conversation state for an NPC.
    pub fn get(&self, npc_id: &WorldObjectId) -> Option<&ConversationState> {
        self.conversations.get(npc_id)
    }

    /// Get mutable conversation state for an NPC.
    pub fn get_mut(&mut self, npc_id: &WorldObjectId) -> Option<&mut ConversationState> {
        self.conversations.get_mut(npc_id)
    }

    /// Check if an NPC has a conversation registered.
    pub fn has_conversation(&self, npc_id: &WorldObjectId) -> bool {
        self.conversations.contains_key(npc_id)
    }

    /// Get all registered NPC IDs.
    pub fn npc_ids(&self) -> Vec<WorldObjectId> {
        self.conversations.keys().copied().collect()
    }
}

/// TTS (text-to-speech) output queue entry.
#[derive(Debug, Clone)]
pub struct TtsRequest {
    pub npc_id: WorldObjectId,
    pub text: String,
}

/// Simple TTS synthesis using rodio.
/// Generates a tone sequence as placeholder speech synthesis.
pub struct TtsSynthesizer {
    _stream: Option<rodio::OutputStream>,
    handle: Option<rodio::OutputStreamHandle>,
    enabled: bool,
}

impl TtsSynthesizer {
    pub fn new() -> Self {
        let (stream, handle, enabled) = match rodio::OutputStream::try_default() {
            Ok((s, h)) => (Some(s), Some(h), true),
            Err(e) => {
                log::warn!("TTS audio output unavailable: {}", e);
                (None, None, false)
            }
        };
        Self {
            _stream: stream,
            handle,
            enabled,
        }
    }

    /// Speak text using a simple synthesized voice.
    /// This is a placeholder -- in production, would use a proper TTS engine.
    pub fn speak(&self, text: &str) {
        if !self.enabled {
            return;
        }
        let handle = match &self.handle {
            Some(h) => h,
            None => return,
        };

        // Generate a simple beep sequence proportional to text length.
        // Each "word" gets a short tone.
        let word_count = text.split_whitespace().count().max(1);
        let duration_per_word = 0.08; // seconds
        let total_samples =
            (word_count as f32 * duration_per_word * 44100.0) as usize;

        if let Ok(sink) = rodio::Sink::try_new(handle) {
            let source = SpeechToneSource::new(total_samples, word_count);
            sink.append(source);
            sink.set_volume(0.15);
            sink.detach(); // Play in background
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Simple tone source for placeholder TTS.
struct SpeechToneSource {
    sample_rate: u32,
    sample_index: usize,
    total_samples: usize,
    word_count: usize,
}

impl SpeechToneSource {
    fn new(total_samples: usize, word_count: usize) -> Self {
        Self {
            sample_rate: 44100,
            sample_index: 0,
            total_samples,
            word_count,
        }
    }
}

impl Iterator for SpeechToneSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.sample_index >= self.total_samples {
            return None;
        }

        let t = self.sample_index as f64 / self.sample_rate as f64;
        let samples_per_word = self.total_samples / self.word_count.max(1);
        let word_index = self.sample_index / samples_per_word.max(1);

        // Vary frequency per word to create speech-like cadence.
        let base_freq = 180.0 + (word_index as f64 * 17.0) % 80.0;
        let freq_mod = 1.0 + 0.1 * (t * 5.0).sin();
        let freq = base_freq * freq_mod;

        // Envelope: attack-sustain-release per word.
        let t_in_word = (self.sample_index % samples_per_word.max(1)) as f64
            / samples_per_word.max(1) as f64;
        let envelope = if t_in_word < 0.05 {
            t_in_word / 0.05
        } else if t_in_word > 0.85 {
            (1.0 - t_in_word) / 0.15
        } else {
            1.0
        };

        // Small gap between words.
        let gap = t_in_word > 0.9;
        let sample = if gap {
            0.0
        } else {
            (t * freq * std::f64::consts::TAU).sin() * envelope * 0.3
        };

        self.sample_index += 1;
        Some(sample as f32)
    }
}

impl rodio::Source for SpeechToneSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.total_samples - self.sample_index)
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs_f64(
            self.total_samples as f64 / self.sample_rate as f64,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_config_from_env() {
        let config = LlmConfig::from_env();
        // API key depends on env, but model should have a default.
        assert!(!config.model.is_empty());
        assert!(config.max_tokens > 0);
    }

    #[test]
    fn conversation_state_manages_history() {
        let npc_id = WorldObjectId::new();
        let personality = NpcPersonality::research_robot();
        let mut conv = ConversationState::new(npc_id, personality);

        let request = conv.user_message("What is a fractal?");
        assert_eq!(request.messages.len(), 1);
        assert!(conv.awaiting_response);

        conv.assistant_response("A fractal is a pattern that repeats at every scale.");
        assert!(!conv.awaiting_response);
        assert_eq!(conv.history.len(), 2);
    }

    #[test]
    fn conversation_trims_history() {
        let npc_id = WorldObjectId::new();
        let personality = NpcPersonality::research_robot();
        let mut conv = ConversationState::new(npc_id, personality);
        conv.max_history = 4;

        for i in 0..10 {
            conv.user_message(&format!("Message {}", i));
            conv.assistant_response(&format!("Response {}", i));
        }

        // History should be trimmed to max_history.
        assert!(conv.history.len() <= conv.max_history);
    }

    #[test]
    fn npc_personality_system_prompt() {
        let p = NpcPersonality::research_robot();
        let prompt = p.system_prompt();
        assert!(prompt.contains("Atlas"));
        assert!(prompt.contains("research robot"));
        assert!(prompt.contains("conjure_object"));
    }

    #[test]
    fn npc_tools_defined() {
        let tools = npc_tools();
        assert!(tools.len() >= 3);
        assert!(tools.iter().any(|t| t.name == "conjure_object"));
        assert!(tools.iter().any(|t| t.name == "set_weather"));
        assert!(tools.iter().any(|t| t.name == "set_time"));
    }

    #[test]
    fn conversation_manager_register_and_query() {
        let mut mgr = ConversationManager::new();
        let id = WorldObjectId::new();
        mgr.register(id, NpcPersonality::research_robot());

        assert!(mgr.has_conversation(&id));
        assert!(mgr.get(&id).is_some());
        assert_eq!(mgr.npc_ids().len(), 1);
    }

    #[test]
    fn tool_call_serialization() {
        let tc = ToolCall {
            id: "tc_01".to_string(),
            name: "conjure_object".to_string(),
            input: serde_json::json!({"description": "red cube"}),
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("conjure_object"));
        assert!(json.contains("red cube"));
    }

    #[test]
    fn speech_tone_source_produces_finite_samples() {
        let source = SpeechToneSource::new(4410, 3); // 0.1 seconds
        let samples: Vec<f32> = source.collect();
        assert_eq!(samples.len(), 4410);
        assert!(samples.iter().any(|s| s.abs() > 0.01));
        assert!(samples.iter().all(|s| s.abs() <= 1.0));
    }
}
