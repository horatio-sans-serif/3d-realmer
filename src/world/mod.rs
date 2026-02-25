mod audio;
mod camera;
mod chunk;
mod chunk_manager;
mod chunk_render;
mod chunk_store;
mod fauna;
pub(crate) mod gltf_loader;
mod input;
mod line_render;
mod mesh_render;
mod mesher;
mod npc;
mod object_render;
mod pathfinding;
mod player;
mod renderer;
mod sky;
mod terrain;
mod terrain_collider;
mod weather;

use std::collections::HashMap;
use std::sync::Arc;

use log::info;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use bitparty::actor::{ActorRegistry, Message};
use bitparty::bridge::{BridgeMessage, BridgePlane, WorldObjectId};

use crossbeam::channel::Receiver as CrossbeamReceiver;

use crate::bridge_server::BridgeServer;
use crate::conjuring::{ConjureConfig, ConjureStatus, ConjuringPipeline};
use crate::llm::{
    ConversationManager, LlmConfig, LlmPipeline, NpcPersonality, TtsSynthesizer,
};
use crate::physics::PhysicsWorld;
use crate::state_history::{StateHistory, WorldAction};
use crate::world_object_actor::{
    ActorStateChanged, BridgeRelayActor, ConfigureMessage, InteractMessage, WorldObjectActor,
};
use crate::world_object_store::WorldObjectStore;
use crate::vr::{LocomotionMode, VrPanelContent, VrSystem, VrUiPanel};
use crate::world_objects::{WorldObjectManager, WorldObjectType};

pub use camera::Camera;

const BRIDGE_PORT: u16 = 9090;

/// The application state for the 3D world.
pub struct App {
    state: Option<WorldState>,
}

struct WorldState {
    window: Arc<Window>,
    renderer: renderer::Renderer,
    chunk_renderer: chunk_render::ChunkRenderer,
    object_renderer: object_render::ObjectRenderer,
    line_renderer: line_render::LineRenderer,
    mesh_renderer: mesh_render::MeshRenderer,
    chunk_manager: chunk_manager::ChunkManager,
    camera: Camera,
    player: player::Player,
    input_state: input::InputState,
    registry: ActorRegistry,
    physics: PhysicsWorld,
    bridge: BridgeServer,
    world_objects: WorldObjectManager,
    /// Channel for actor state changes relayed from BridgeRelayActor.
    state_change_rx: CrossbeamReceiver<ActorStateChanged>,
    /// Latest actor visual state per world object (from ActorStateChanged events).
    actor_visual_state: HashMap<WorldObjectId, ActorStateChanged>,
    /// GLTF mesh cache.
    gltf_cache: gltf_loader::GltfCache,
    /// Persistent storage for world objects.
    object_store: WorldObjectStore,
    /// Undo/redo history for authoring actions.
    history: StateHistory,
    /// Path to the world database (for save operations).
    world_db_path: std::path::PathBuf,
    /// Day/night cycle and sky lighting.
    sky: sky::Sky,
    /// Weather particles and wind.
    weather: weather::WeatherSystem,
    /// Ambient fauna (butterflies, birds).
    fauna: fauna::FaunaManager,
    /// NPC actors with pathfinding.
    npcs: npc::NpcManager,
    /// Ambient audio system.
    audio: audio::AmbientAudio,
    /// AI conjuring pipeline.
    conjuring: ConjuringPipeline,
    /// LLM conversation pipeline.
    llm: LlmPipeline,
    /// NPC conversation state manager.
    conversations: ConversationManager,
    /// Text-to-speech synthesizer.
    tts: TtsSynthesizer,
    /// VR system (OpenXR, compile-gated).
    vr: VrSystem,
    /// In-VR UI panels.
    vr_panels: Vec<VrUiPanel>,
    last_frame: std::time::Instant,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("BitParty World")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(event_loop.create_window(window_attrs).expect("create window"));

        let renderer = pollster::block_on(renderer::Renderer::new(window.clone()))
            .expect("create renderer");

        let chunk_renderer = chunk_render::ChunkRenderer::new(
            renderer.device(),
            renderer.surface_format(),
            renderer.global_bind_group_layout(),
        );

        let object_renderer = object_render::ObjectRenderer::new(
            renderer.device(),
            renderer.surface_format(),
            renderer.global_bind_group_layout(),
        );

        let line_renderer = line_render::LineRenderer::new(
            renderer.device(),
            renderer.surface_format(),
            renderer.global_bind_group_layout(),
        );

        let mesh_renderer = mesh_render::MeshRenderer::new(
            renderer.device(),
            renderer.surface_format(),
            renderer.global_bind_group_layout(),
        );

        // Chunk persistence: store in a world database next to the binary
        let world_db = std::env::current_dir()
            .unwrap_or_default()
            .join("world.db");
        let chunk_manager = chunk_manager::ChunkManager::new(42, &world_db);

        let spawn_pos = glam::Vec3::new(0.0, 20.0, 0.0);

        let camera = Camera::new(
            spawn_pos,
            window.inner_size().width as f32 / window.inner_size().height as f32,
        );

        let input_state = input::InputState::new();
        let mut registry = ActorRegistry::new();
        let mut physics = PhysicsWorld::new();

        let player = player::Player::new(spawn_pos, &mut physics);

        // Start the bridge WebSocket server.
        let bridge = BridgeServer::start(BRIDGE_PORT);
        let mut world_objects = WorldObjectManager::new();

        // Spawn the bridge relay actor to forward state changes to the bridge.
        let (state_change_tx, state_change_rx) = crossbeam::channel::unbounded();
        let relay = BridgeRelayActor::new(state_change_tx);
        registry.spawn(Box::new(relay));

        // Open the world object store (same directory as chunk DB).
        let object_store = WorldObjectStore::open(&world_db)
            .expect("failed to open world object store");

        // Load persisted world objects.
        if let Ok(stored_objects) = object_store.load_all() {
            for stored in &stored_objects {
                world_objects.spawn_object(
                    stored.id,
                    stored.position,
                    stored.object_type,
                    stored.properties.clone(),
                    &mut physics,
                );

                // Spawn backing actor.
                let actor = WorldObjectActor::new(stored.id, stored.label.clone());
                let actor_id = registry.spawn(Box::new(actor));
                world_objects.bind_actor(stored.id, actor_id);
            }
            if !stored_objects.is_empty() {
                info!("Loaded {} persisted world objects", stored_objects.len());
            }
        }

        let gltf_cache = gltf_loader::GltfCache::new();
        let history = StateHistory::new(1000);
        let sky = sky::Sky::new(600.0); // 10-minute full day cycle

        // Spawn ambient fauna around the spawn position.
        let mut fauna_mgr = fauna::FaunaManager::new();
        fauna_mgr.spawn_group(spawn_pos.to_array(), 8, 4);

        // Spawn initial NPCs near the player.
        let mut npc_mgr = npc::NpcManager::new(chunk::BLOCK_SIZE);
        npc_mgr.spawn(
            [spawn_pos.x + 3.0, spawn_pos.y, spawn_pos.z + 3.0],
            "Villager",
            [0.7, 0.4, 0.2, 1.0],
            1.5,
        );
        npc_mgr.spawn(
            [spawn_pos.x - 5.0, spawn_pos.y, spawn_pos.z - 2.0],
            "Guard",
            [0.8, 0.2, 0.2, 1.0],
            2.0,
        );

        let ambient_audio = audio::AmbientAudio::new();

        // Initialize the LLM pipeline.
        let llm_config = LlmConfig::from_env();
        if llm_config.is_configured() {
            info!("LLM API configured (model={})", llm_config.model);
        } else {
            info!("LLM: no API key configured (set ANTHROPIC_API_KEY)");
        }
        let llm = LlmPipeline::new(llm_config);
        let mut conversations = ConversationManager::new();
        let tts = TtsSynthesizer::new();

        // Spawn a research robot NPC (conversational, LLM-backed).
        let atlas_id = npc_mgr.spawn_conversational(
            [spawn_pos.x + 5.0, spawn_pos.y, spawn_pos.z],
            "Atlas",
            [0.3, 0.6, 0.9, 1.0],
            1.5,
        );
        conversations.register(atlas_id, NpcPersonality::research_robot());

        // Spawn an elder sage NPC (conversational).
        let elder_id = npc_mgr.spawn_conversational(
            [spawn_pos.x - 3.0, spawn_pos.y, spawn_pos.z + 5.0],
            "Elder Sage",
            [0.6, 0.3, 0.1, 1.0],
            1.0,
        );
        conversations.register(elder_id, NpcPersonality::village_elder());

        // Initialize the AI conjuring pipeline.
        let conjure_asset_dir = world_db.parent().unwrap_or(std::path::Path::new(".")).join("conjured_assets");
        let conjure_config = ConjureConfig::from_env(conjure_asset_dir);
        if conjure_config.has_any_api() {
            info!("AI conjuring APIs configured");
        } else {
            info!("AI conjuring: no API keys configured (set MESHY_API_KEY or TRIPO_API_KEY)");
        }
        let conjuring = ConjuringPipeline::new(conjure_config);

        // Initialize VR system (no-op on macOS without the `vr` feature).
        let vr = VrSystem::new();
        if vr.is_active() {
            info!("VR: OpenXR session active");
        } else {
            info!("VR: inactive (no OpenXR runtime or 'vr' feature disabled)");
        }
        let vr_panels = Vec::new();

        let weather_system = weather::WeatherSystem::new(
            renderer.device(),
            renderer.surface_format(),
            renderer.global_bind_group_layout(),
        );

        info!(
            "World initialized: {}x{}, bridge on port {}",
            window.inner_size().width,
            window.inner_size().height,
            BRIDGE_PORT
        );

        self.state = Some(WorldState {
            window,
            renderer,
            chunk_renderer,
            object_renderer,
            line_renderer,
            mesh_renderer,
            chunk_manager,
            camera,
            player,
            input_state,
            registry,
            physics,
            bridge,
            world_objects,
            state_change_rx,
            actor_visual_state: HashMap::new(),
            gltf_cache,
            object_store,
            history,
            world_db_path: world_db,
            sky,
            weather: weather_system,
            fauna: fauna_mgr,
            npcs: npc_mgr,
            audio: ambient_audio,
            conjuring,
            llm,
            conversations,
            tts,
            vr,
            vr_panels,
            last_frame: std::time::Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.renderer.resize(new_size);
                    state
                        .camera
                        .set_aspect(new_size.width as f32 / new_size.height as f32);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                state.input_state.handle_keyboard(&event, &mut state.camera);
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                state.input_state.handle_mouse_button(button_state, button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state
                    .input_state
                    .handle_cursor_move(position, &mut state.camera);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.1);
                state.last_frame = now;

                // Process bridge commands.
                process_bridge_commands(state);

                // Handle jump input
                if state.input_state.take_jump() {
                    state.player.jump();
                }

                // Handle block interaction (place/remove)
                if let Some(action) = state.input_state.take_block_action() {
                    let cam_pos = state.camera.position();
                    let cam_front = state.camera.front();
                    let max_reach = 5.0;

                    if let Some((hit_point, normal, _toi)) = state.physics.raycast_with_normal(
                        cam_pos.to_array(),
                        cam_front.to_array(),
                        max_reach,
                        state.player.body_handle(),
                    ) {
                        let hp = glam::Vec3::from(hit_point);
                        let n = glam::Vec3::from(normal);
                        let epsilon = 0.01;

                        match action {
                            input::BlockAction::Remove => {
                                let target = hp - n * epsilon;
                                let old_type = state
                                    .chunk_manager
                                    .get_block(target)
                                    .unwrap_or(chunk::BlockType::Air);
                                if state
                                    .chunk_manager
                                    .set_block(target, chunk::BlockType::Air)
                                {
                                    state.history.record(WorldAction::SetBlock {
                                        world_pos: target.to_array(),
                                        old_type: old_type as u8,
                                        new_type: chunk::BlockType::Air as u8,
                                    });
                                }
                            }
                            input::BlockAction::Place => {
                                let target = hp + n * epsilon;
                                let material = state.input_state.selected_material();
                                let old_type = state
                                    .chunk_manager
                                    .get_block(target)
                                    .unwrap_or(chunk::BlockType::Air);
                                if state.chunk_manager.set_block(target, material) {
                                    state.history.record(WorldAction::SetBlock {
                                        world_pos: target.to_array(),
                                        old_type: old_type as u8,
                                        new_type: material as u8,
                                    });
                                }
                            }
                        }
                    }
                }

                // Handle undo/redo
                if state.input_state.take_undo() {
                    handle_undo(state);
                }
                if state.input_state.take_redo() {
                    handle_redo(state);
                }

                // Handle interact (E key) - raycast to find world object
                if state.input_state.take_interact() {
                    handle_interact(state);
                }

                // Step physics (updates query pipeline for ground detection)
                state.physics.step(dt);

                // Update player movement (reads camera direction, applies gravity)
                state.player.update(dt, &state.camera, &mut state.physics);

                // Sync camera to player eye position
                let eye = state.player.eye_position(&state.physics);
                state.camera.set_position(eye);

                // Update VR tracking, input, and camera override.
                update_vr(state, dt);

                state.registry.tick();

                // Forward actor state changes to bridge clients.
                process_actor_state_changes(state);

                // Load/unload chunks around the player
                state.chunk_manager.update(
                    eye,
                    &mut state.chunk_renderer,
                    state.renderer.device(),
                    &mut state.physics,
                );

                // Update world object instances for rendering.
                update_object_instances(state);

                // Update connection lines for rendering.
                update_connection_lines(state);

                // Update mesh instances for rendering.
                update_mesh_instances(state);

                // Advance day/night cycle and push sky-driven uniforms.
                state.sky.update(dt);
                let wind = state.weather.wind();
                state.renderer.set_sky_uniforms(renderer::SkyUniforms {
                    sun_direction: state.sky.sun_direction(),
                    sun_color: state.sky.sun_color(),
                    ambient_color: state.sky.ambient_color(),
                    fog_color: state.sky.fog_color(),
                    fog_start: 64.0,
                    fog_end: 128.0,
                    wind_direction: wind.direction,
                    wind_strength: wind.strength,
                });
                state.renderer.set_clear_color(state.sky.clear_color());

                // Update weather particles.
                let cam_pos = state.camera.position().to_array();
                state.weather.update(dt, cam_pos, state.renderer.queue());

                // Update ambient fauna.
                let wind = state.weather.wind();
                state.fauna.update(dt, wind.direction, wind.strength);
                state.fauna.reanchor(cam_pos);

                // Update NPC pathfinding and movement.
                update_npcs(state, dt);

                // Poll conjuring results and place completed objects.
                process_conjure_results(state);

                // Poll LLM responses and process conversation results.
                process_llm_responses(state);

                // Update ambient audio based on weather and time of day.
                state.audio.update(
                    state.weather.weather(),
                    state.sky.phase(),
                    state.weather.wind().strength,
                );

                match state.renderer.render(
                    &state.camera,
                    &state.chunk_renderer,
                    &state.object_renderer,
                    &state.line_renderer,
                    &state.mesh_renderer,
                    &state.weather,
                )
                {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        state.renderer.resize(state.window.inner_size());
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        event_loop.exit();
                    }
                    Err(e) => {
                        log::warn!("Render error: {e:?}");
                    }
                }

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Drain and process all pending bridge commands from connected clients.
fn process_bridge_commands(state: &mut WorldState) {
    while let Ok(inbound) = state.bridge.command_rx.try_recv() {
        let msg = &inbound.message;
        let client_id = inbound.client_id;

        // Route data plane messages to the target actor.
        if msg.plane == BridgePlane::Data {
            handle_data_plane_message(state, client_id, msg);
            continue;
        }

        // Management plane commands.
        match msg.msg_type.as_str() {
            "spawn-object" => {
                handle_spawn_object(state, client_id, &msg.payload);
            }
            "remove-object" => {
                handle_remove_object(state, client_id, &msg.payload);
            }
            "query-state" => {
                handle_query_state(state, client_id, &msg.payload);
            }
            "query-all-objects" => {
                handle_query_all_objects(state, client_id);
            }
            "send-message" => {
                handle_send_message(state, client_id, &msg.payload);
            }
            "configure-object" => {
                handle_configure_object(state, client_id, &msg.payload);
            }
            "connect-actors" => {
                handle_connect_actors(state, client_id, &msg.payload);
            }
            "save-world" => {
                handle_save_world(state, client_id);
            }
            "import-mesh" => {
                handle_import_mesh(state, client_id, &msg.payload);
            }
            "transform-object" => {
                handle_transform_object(state, client_id, &msg.payload);
            }
            "undo" => {
                handle_undo(state);
            }
            "redo" => {
                handle_redo(state);
            }
            "select-material" => {
                if let Some(mat) = msg.payload.get("material").and_then(|v| v.as_str()) {
                    if let Some(bt) = block_type_from_name(mat) {
                        state.input_state.set_selected_material(bt);
                        log::info!("Bridge: material set to '{}'", mat);
                    }
                }
            }
            "set-time-of-day" => {
                if let Some(t) = msg.payload.get("time").and_then(|v| v.as_f64()) {
                    state.sky.set_time_of_day(t as f32);
                    log::info!("Bridge: time-of-day set to {:.2}", t);
                }
            }
            "pause-cycle" => {
                state.sky.set_paused(true);
                log::info!("Bridge: day/night cycle paused");
            }
            "resume-cycle" => {
                state.sky.set_paused(false);
                log::info!("Bridge: day/night cycle resumed");
            }
            "set-weather" => {
                if let Some(w) = msg.payload.get("weather").and_then(|v| v.as_str()) {
                    let weather_type = match w {
                        "rain" => weather::WeatherType::Rain,
                        "snow" => weather::WeatherType::Snow,
                        _ => weather::WeatherType::Clear,
                    };
                    state.weather.set_weather(weather_type);
                    log::info!("Bridge: weather set to {:?}", weather_type);
                }
            }
            "spawn-npc" => {
                let position = msg
                    .payload
                    .get("position")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        [
                            a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            a.get(1).and_then(|v| v.as_f64()).unwrap_or(10.0) as f32,
                            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        ]
                    })
                    .unwrap_or([0.0, 10.0, 0.0]);
                let label = msg
                    .payload
                    .get("label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("NPC");
                let speed = msg
                    .payload
                    .get("speed")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(2.0) as f32;
                let id = state.npcs.spawn(position, label, [0.6, 0.4, 0.2, 1.0], speed);
                log::info!("Bridge: spawned NPC '{}' at {:?} (id={})", label, position, id);
                let event = BridgeMessage::management(
                    "npc-spawned",
                    serde_json::json!({"id": id.to_string(), "label": label, "position": position}),
                );
                state.bridge.send_to(client_id, &event);
            }
            "npc-walk-to" => {
                if let (Some(id_str), Some(target)) = (
                    msg.payload.get("id").and_then(|v| v.as_str()),
                    msg.payload.get("target").and_then(|v| v.as_array()),
                ) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        let npc_id = WorldObjectId::from_uuid(uuid);
                        let target_pos = [
                            target.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            target.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            target.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        ];
                        let chunk_mgr = &state.chunk_manager;
                        let block_size = chunk::BLOCK_SIZE;
                        let is_walkable = |nav_pos: pathfinding::NavPos| -> bool {
                            let feet = nav_pos.to_world(block_size);
                            let below = [feet[0], feet[1] - block_size, feet[2]];
                            let head = [feet[0], feet[1] + block_size, feet[2]];
                            let b = chunk_mgr.get_block(glam::Vec3::from(below)).unwrap_or(chunk::BlockType::Air);
                            let f = chunk_mgr.get_block(glam::Vec3::from(feet)).unwrap_or(chunk::BlockType::Air);
                            let h = chunk_mgr.get_block(glam::Vec3::from(head)).unwrap_or(chunk::BlockType::Air);
                            b != chunk::BlockType::Air && f == chunk::BlockType::Air && h == chunk::BlockType::Air
                        };
                        state.npcs.walk_to(&npc_id, target_pos, &is_walkable);
                    }
                }
            }
            "set-audio-volume" => {
                if let Some(v) = msg.payload.get("volume").and_then(|v| v.as_f64()) {
                    state.audio.set_volume(v as f32);
                    log::info!("Bridge: audio volume set to {:.2}", v);
                }
            }
            "conjure-object" => {
                handle_conjure_object(state, client_id, &msg.payload);
            }
            "retry-conjure" => {
                if let Some(id_str) = msg.payload.get("id").and_then(|v| v.as_str()) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        let id = WorldObjectId::from_uuid(uuid);
                        if state.conjuring.retry(&id) {
                            log::info!("Bridge: retrying conjure for {}", id);
                        }
                    }
                }
            }
            "conjure-skybox" => {
                handle_conjure_skybox(state, client_id, &msg.payload);
            }
            "toggle-audio" => {
                let enabled = msg
                    .payload
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                state.audio.set_enabled(enabled);
                log::info!("Bridge: audio enabled={}", enabled);
            }
            "talk-to-npc" => {
                handle_talk_to_npc(state, client_id, &msg.payload);
            }
            "end-conversation" => {
                if let Some(id_str) = msg.payload.get("id").and_then(|v| v.as_str()) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        let npc_id = WorldObjectId::from_uuid(uuid);
                        state.npcs.end_conversation(&npc_id);
                        log::info!("Bridge: ended conversation with NPC {}", npc_id);
                    }
                }
            }
            "talk-to-nearest" => {
                handle_talk_to_nearest(state, client_id, &msg.payload);
            }
            "toggle-vr" => {
                handle_toggle_vr(state, client_id);
            }
            "set-locomotion" => {
                if let Some(mode) = msg.payload.get("mode").and_then(|v| v.as_str()) {
                    match mode {
                        "teleport" => state.vr.set_locomotion(LocomotionMode::Teleport),
                        "smooth" => state.vr.set_locomotion(LocomotionMode::Smooth),
                        _ => {}
                    }
                    log::info!("Bridge: VR locomotion set to {}", mode);
                }
            }
            "vr-status" => {
                let status = serde_json::json!({
                    "active": state.vr.is_active(),
                    "locomotion": format!("{:?}", state.vr.locomotion()),
                    "head_position": state.vr.head_position().to_array(),
                    "panels": state.vr_panels.len(),
                });
                let event = BridgeMessage::management("vr-status", status);
                state.bridge.send_to(client_id, &event);
            }
            "vr-show-panel" => {
                handle_vr_show_panel(state, &msg.payload);
            }
            "vr-hide-panels" => {
                state.vr_panels.clear();
                log::info!("Bridge: VR panels cleared");
            }
            "set-wind" => {
                let dir = msg
                    .payload
                    .get("direction")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        [
                            a.first().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        ]
                    })
                    .unwrap_or([1.0, 0.0]);
                let strength = msg
                    .payload
                    .get("strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.3) as f32;
                state.weather.set_wind(dir, strength);
                log::info!(
                    "Bridge: wind set to dir={:?}, strength={}",
                    dir,
                    strength
                );
            }
            other => {
                log::warn!("Bridge: unknown command type: {}", other);
            }
        }
    }
}

fn handle_spawn_object(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let position = match payload.get("position") {
        Some(p) => {
            let arr = p.as_array();
            match arr {
                Some(a) if a.len() >= 3 => [
                    a[0].as_f64().unwrap_or(0.0) as f32,
                    a[1].as_f64().unwrap_or(0.0) as f32,
                    a[2].as_f64().unwrap_or(0.0) as f32,
                ],
                _ => [0.0, 10.0, 0.0],
            }
        }
        None => [0.0, 10.0, 0.0],
    };

    let object_type = match payload.get("object_type").and_then(|v| v.as_str()) {
        Some("cube") => WorldObjectType::Cube,
        Some("mesh") => WorldObjectType::Mesh,
        Some("interactive") => WorldObjectType::Interactive,
        _ => WorldObjectType::Cube,
    };

    let id = WorldObjectId::new();
    let label = payload
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("object")
        .to_string();
    let properties = payload
        .get("properties")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    state.world_objects.spawn_object(
        id,
        position,
        object_type,
        properties.clone(),
        &mut state.physics,
    );

    // Spawn a backing actor for this world object.
    let actor = WorldObjectActor::new(id, label.clone());
    let actor_id = state.registry.spawn(Box::new(actor));
    state.world_objects.bind_actor(id, actor_id);

    log::info!(
        "Bridge: spawned {:?} at {:?} (id={}, actor={:?})",
        object_type,
        position,
        id,
        actor_id,
    );

    // Send confirmation event back to the client.
    let event = BridgeMessage::management(
        "object-spawned",
        serde_json::json!({
            "id": id.to_string(),
            "position": position,
            "object_type": format!("{:?}", object_type).to_lowercase(),
            "label": label,
            "properties": properties,
        }),
    );
    state.bridge.send_to(client_id, &event);

    // Broadcast to all clients.
    state.bridge.broadcast(&event);
}

fn handle_remove_object(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: remove-object missing 'id' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in remove-object: {}", e);
            return;
        }
    };

    let id = WorldObjectId::from_uuid(uuid);

    // Despawn the backing actor first.
    if let Some(&actor_id) = state.world_objects.actor_for_object(&id) {
        state.registry.despawn(actor_id);
    }
    state.actor_visual_state.remove(&id);

    if state
        .world_objects
        .remove_object(&id, &mut state.physics)
        .is_some()
    {
        log::info!("Bridge: removed object {}", id);

        let event = BridgeMessage::management(
            "object-removed",
            serde_json::json!({ "id": id.to_string() }),
        );
        state.bridge.send_to(client_id, &event);
        state.bridge.broadcast(&event);
    } else {
        log::warn!("Bridge: object {} not found for removal", id);
    }
}

fn handle_query_state(state: &WorldState, client_id: usize, payload: &serde_json::Value) {
    let id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: query-state missing 'id' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in query-state: {}", e);
            return;
        }
    };

    let id = WorldObjectId::from_uuid(uuid);

    let response_payload = if let Some(obj) = state.world_objects.get(&id) {
        let mut resp = serde_json::json!({
            "id": obj.id.to_string(),
            "position": obj.position,
            "rotation": obj.rotation,
            "scale": obj.scale,
            "object_type": format!("{:?}", obj.object_type).to_lowercase(),
            "properties": obj.properties,
        });

        // Include actor state if there's a backing actor.
        if let Some(&actor_id) = state.world_objects.actor_for_object(&id) {
            if let Some(actor) = state.registry.get(actor_id) {
                resp["actor"] = serde_json::json!({
                    "type_name": actor.type_name(),
                    "has_actor": true,
                });
            }
        }

        resp
    } else {
        serde_json::json!({ "id": id_str, "error": "not_found" })
    };

    let response = BridgeMessage::management("state-response", response_payload);
    state.bridge.send_to(client_id, &response);
}

fn handle_query_all_objects(state: &WorldState, client_id: usize) {
    let objects: Vec<serde_json::Value> = state
        .world_objects
        .all_objects()
        .iter()
        .map(|obj| {
            let has_actor = state.world_objects.actor_for_object(&obj.id).is_some();
            serde_json::json!({
                "id": obj.id.to_string(),
                "position": obj.position,
                "object_type": format!("{:?}", obj.object_type).to_lowercase(),
                "has_actor": has_actor,
            })
        })
        .collect();

    let response = BridgeMessage::management(
        "all-objects-response",
        serde_json::json!({ "objects": objects }),
    );
    state.bridge.send_to(client_id, &response);
}

/// Handle a data plane message: route to the target actor by WorldObjectId.
fn handle_data_plane_message(
    state: &mut WorldState,
    _client_id: usize,
    bridge_msg: &BridgeMessage,
) {
    let to_str = match &bridge_msg.to {
        Some(s) => s.as_str(),
        None => {
            log::warn!("Bridge: data plane message missing 'to' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(to_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in data plane 'to': {}", e);
            return;
        }
    };

    let world_id = WorldObjectId::from_uuid(uuid);

    let actor_id = match state.world_objects.actor_for_object(&world_id) {
        Some(&id) => id,
        None => {
            log::warn!("Bridge: no actor for world object {}", world_id);
            return;
        }
    };

    // Route based on data plane message type.
    match bridge_msg.msg_type.as_str() {
        "interact" => {
            let action = bridge_msg
                .payload
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("click")
                .to_string();

            let msg = Message::new(
                actor_id,
                InteractMessage {
                    world_object_id: world_id,
                    action,
                    payload: bridge_msg.payload.clone(),
                },
                actor_id,
            );
            let _ = state.registry.send(msg);
        }
        "configure" => {
            let msg = Message::new(
                actor_id,
                ConfigureMessage {
                    world_object_id: world_id,
                    properties: bridge_msg.payload.clone(),
                },
                actor_id,
            );
            let _ = state.registry.send(msg);
        }
        other => {
            log::warn!(
                "Bridge: unknown data plane message type '{}' for {}",
                other,
                world_id
            );
        }
    }
}

/// Handle the management plane "send-message" command.
/// Sends an InteractMessage to a world object's actor.
fn handle_send_message(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: send-message missing 'id' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in send-message: {}", e);
            return;
        }
    };

    let world_id = WorldObjectId::from_uuid(uuid);

    let actor_id = match state.world_objects.actor_for_object(&world_id) {
        Some(&id) => id,
        None => {
            log::warn!("Bridge: no actor for world object {}", world_id);
            let err = BridgeMessage::management(
                "error",
                serde_json::json!({"error": "no_actor", "id": id_str}),
            );
            state.bridge.send_to(client_id, &err);
            return;
        }
    };

    let action = payload
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("interact")
        .to_string();

    let msg_payload = payload
        .get("payload")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let msg = Message::new(
        actor_id,
        InteractMessage {
            world_object_id: world_id,
            action,
            payload: msg_payload,
        },
        actor_id,
    );
    let _ = state.registry.send(msg);

    let ack = BridgeMessage::management(
        "message-sent",
        serde_json::json!({"id": id_str, "status": "delivered"}),
    );
    state.bridge.send_to(client_id, &ack);
}

/// Handle the management plane "configure-object" command.
fn handle_configure_object(
    state: &mut WorldState,
    client_id: usize,
    payload: &serde_json::Value,
) {
    let id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: configure-object missing 'id' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in configure-object: {}", e);
            return;
        }
    };

    let world_id = WorldObjectId::from_uuid(uuid);

    let actor_id = match state.world_objects.actor_for_object(&world_id) {
        Some(&id) => id,
        None => {
            log::warn!("Bridge: no actor for world object {}", world_id);
            return;
        }
    };

    let properties = payload
        .get("properties")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let msg = Message::new(
        actor_id,
        ConfigureMessage {
            world_object_id: world_id,
            properties,
        },
        actor_id,
    );
    let _ = state.registry.send(msg);

    let ack = BridgeMessage::management(
        "object-configured",
        serde_json::json!({"id": id_str}),
    );
    state.bridge.send_to(client_id, &ack);
}

/// Handle the management plane "connect-actors" command.
/// Establishes a visual connection between two world objects.
fn handle_connect_actors(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let from_str = match payload.get("from").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: connect-actors missing 'from' field");
            return;
        }
    };
    let to_str = match payload.get("to").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: connect-actors missing 'to' field");
            return;
        }
    };

    let from_uuid = match uuid::Uuid::parse_str(from_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in connect-actors 'from': {}", e);
            return;
        }
    };
    // Validate that 'to' is a valid UUID (even though we pass it as a string to the actor).
    if uuid::Uuid::parse_str(to_str).is_err() {
        log::warn!("Bridge: invalid UUID in connect-actors 'to'");
        return;
    }

    let from_id = WorldObjectId::from_uuid(from_uuid);

    let actor_id = match state.world_objects.actor_for_object(&from_id) {
        Some(&id) => id,
        None => {
            log::warn!("Bridge: no actor for world object {}", from_id);
            return;
        }
    };

    let connect = payload
        .get("disconnect")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let properties = if connect {
        serde_json::json!({"disconnect_from": to_str})
    } else {
        serde_json::json!({"connect_to": to_str})
    };

    let msg = Message::new(
        actor_id,
        ConfigureMessage {
            world_object_id: from_id,
            properties,
        },
        actor_id,
    );
    let _ = state.registry.send(msg);

    let event = BridgeMessage::management(
        "actors-connected",
        serde_json::json!({
            "from": from_str,
            "to": to_str,
            "connected": !connect,
        }),
    );
    state.bridge.send_to(client_id, &event);
    state.bridge.broadcast(&event);
}

/// Handle 'E' key interact: raycast to find the world object the player is looking at.
fn handle_interact(state: &mut WorldState) {
    let cam_pos = state.camera.position();
    let cam_front = state.camera.front();
    let max_reach = 5.0;

    let hit = state.physics.raycast_hit_body(
        cam_pos.to_array(),
        cam_front.to_array(),
        max_reach,
        state.player.body_handle(),
    );

    let (body_handle, _toi) = match hit {
        Some(h) => h,
        None => return,
    };

    let world_id = match state.world_objects.object_for_body(&body_handle) {
        Some(&id) => id,
        None => return, // Hit terrain, not a world object.
    };

    let actor_id = match state.world_objects.actor_for_object(&world_id) {
        Some(&id) => id,
        None => return,
    };

    log::info!("Player interacting with object {}", world_id);

    let msg = Message::new(
        actor_id,
        InteractMessage {
            world_object_id: world_id,
            action: "interact".to_string(),
            payload: serde_json::json!({"source": "player"}),
        },
        actor_id,
    );
    let _ = state.registry.send(msg);
}

/// Process ActorStateChanged broadcasts and forward them over the bridge.
///
/// The BridgeRelayActor receives ActorStateChanged broadcasts via the actor
/// system and pushes them to a crossbeam channel. We drain that channel here
/// and convert each event to a bridge message for connected clients.
fn process_actor_state_changes(state: &mut WorldState) {
    while let Ok(change) = state.state_change_rx.try_recv() {
        let connections: Vec<String> = change
            .connections
            .iter()
            .map(|c| c.to_string())
            .collect();

        let event = BridgeMessage::management(
            "actor-state-changed",
            serde_json::json!({
                "id": change.world_object_id.to_string(),
                "label": change.label,
                "state": change.state,
                "connections": connections,
            }),
        );
        state.bridge.broadcast(&event);

        // Store latest visual state for rendering.
        state
            .actor_visual_state
            .insert(change.world_object_id, change);
    }
}

/// Collect world object positions and colors, update the instance buffer.
fn update_object_instances(state: &mut WorldState) {
    let mut instances: Vec<object_render::ObjectInstance> = state
        .world_objects
        .all_objects()
        .iter()
        .map(|obj| {
            let color = if let Some(vs) = state.actor_visual_state.get(&obj.id) {
                // Color based on interaction count.
                let count = vs
                    .state
                    .get("interaction_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                if count > 0 {
                    [1.0, 0.6, 0.0, 1.0] // Orange: has been interacted with
                } else {
                    [0.2, 0.6, 1.0, 1.0] // Blue: no interactions yet
                }
            } else {
                [0.5, 0.5, 0.5, 1.0] // Gray: no actor state received yet
            };

            object_render::ObjectInstance {
                offset: obj.position,
                _pad: 0.0,
                color,
            }
        })
        .collect();

    // Add fauna as small colored cubes.
    for fi in state.fauna.instances() {
        instances.push(object_render::ObjectInstance {
            offset: fi.position,
            _pad: fi.scale,
            color: fi.color,
        });
    }

    // Add NPCs as colored cubes.
    for ni in state.npcs.instances() {
        instances.push(object_render::ObjectInstance {
            offset: ni.position,
            _pad: 0.0,
            color: ni.color,
        });
    }

    state
        .object_renderer
        .update_instances(state.renderer.queue(), &instances);
}

/// Convert a block type name to a BlockType.
fn block_type_from_name(name: &str) -> Option<chunk::BlockType> {
    match name.to_lowercase().as_str() {
        "stone" => Some(chunk::BlockType::Stone),
        "dirt" => Some(chunk::BlockType::Dirt),
        "grass" => Some(chunk::BlockType::Grass),
        "sand" => Some(chunk::BlockType::Sand),
        "wood" => Some(chunk::BlockType::Wood),
        "brick" => Some(chunk::BlockType::Brick),
        "glass" => Some(chunk::BlockType::Glass),
        "water" => Some(chunk::BlockType::Water),
        "leaf" => Some(chunk::BlockType::Leaf),
        "air" => Some(chunk::BlockType::Air),
        _ => None,
    }
}

/// Convert a u8 block type discriminant back to a BlockType.
fn block_type_from_u8(v: u8) -> chunk::BlockType {
    match v {
        0 => chunk::BlockType::Air,
        1 => chunk::BlockType::Stone,
        2 => chunk::BlockType::Dirt,
        3 => chunk::BlockType::Grass,
        4 => chunk::BlockType::Sand,
        5 => chunk::BlockType::Wood,
        6 => chunk::BlockType::Brick,
        7 => chunk::BlockType::Glass,
        8 => chunk::BlockType::Water,
        9 => chunk::BlockType::Leaf,
        _ => chunk::BlockType::Air,
    }
}

/// Handle undo: pop the last action and apply its reverse.
fn handle_undo(state: &mut WorldState) {
    if let Some(reverse) = state.history.undo() {
        apply_action(state, &reverse);
        log::info!("Undo applied (remaining: {})", state.history.undo_count());
    }
}

/// Handle redo: re-apply the last undone action.
fn handle_redo(state: &mut WorldState) {
    if let Some(action) = state.history.redo() {
        apply_action(state, &action);
        log::info!("Redo applied (remaining: {})", state.history.redo_count());
    }
}

/// Apply a WorldAction to the world state (used by undo/redo).
fn apply_action(state: &mut WorldState, action: &WorldAction) {
    match action {
        WorldAction::SetBlock {
            world_pos,
            new_type,
            ..
        } => {
            let pos = glam::Vec3::from(*world_pos);
            let bt = block_type_from_u8(*new_type);
            state.chunk_manager.set_block(pos, bt);
        }
        WorldAction::SetBlockRegion {
            positions,
            old_types,
            new_type,
        } => {
            // For region undo, each position gets its individual old_type.
            for (i, pos_arr) in positions.iter().enumerate() {
                let pos = glam::Vec3::from(*pos_arr);
                let bt = if !old_types.is_empty() && i < old_types.len() {
                    block_type_from_u8(old_types[i])
                } else {
                    block_type_from_u8(*new_type)
                };
                state.chunk_manager.set_block(pos, bt);
            }
        }
        WorldAction::SpawnObject {
            id,
            position,
            object_type,
            properties,
            label,
        } => {
            state.world_objects.spawn_object(
                *id,
                *position,
                *object_type,
                properties.clone(),
                &mut state.physics,
            );
            let actor = WorldObjectActor::new(*id, label.clone());
            let actor_id = state.registry.spawn(Box::new(actor));
            state.world_objects.bind_actor(*id, actor_id);
        }
        WorldAction::RemoveObject { id, .. } => {
            if let Some(&actor_id) = state.world_objects.actor_for_object(id) {
                state.registry.despawn(actor_id);
            }
            state.actor_visual_state.remove(id);
            state.world_objects.remove_object(id, &mut state.physics);
        }
        WorldAction::TransformObject {
            id, new_position, ..
        } => {
            if let Some(obj) = state.world_objects.get_mut(id) {
                obj.position = *new_position;
            }
        }
    }
}

/// Save all world objects to the persistent store.
fn handle_save_world(state: &mut WorldState, client_id: usize) {
    let mut saved = 0;
    for obj in state.world_objects.all_objects() {
        let label = state
            .actor_visual_state
            .get(&obj.id)
            .map(|vs| vs.label.as_str())
            .unwrap_or("object");
        if let Err(e) = state.object_store.save_object(obj, label, None) {
            log::warn!("Failed to save object {}: {}", obj.id, e);
        } else {
            saved += 1;
        }
    }

    log::info!("Saved {} world objects", saved);

    let response = BridgeMessage::management(
        "world-saved",
        serde_json::json!({"saved_objects": saved}),
    );
    state.bridge.send_to(client_id, &response);
}

/// Import a GLTF/GLB mesh file and place it as a world object.
fn handle_import_mesh(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let mesh_path = match payload.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            log::warn!("Bridge: import-mesh missing 'path' field");
            return;
        }
    };

    let position = match payload.get("position") {
        Some(p) => {
            let arr = p.as_array();
            match arr {
                Some(a) if a.len() >= 3 => [
                    a[0].as_f64().unwrap_or(0.0) as f32,
                    a[1].as_f64().unwrap_or(0.0) as f32,
                    a[2].as_f64().unwrap_or(0.0) as f32,
                ],
                _ => [0.0, 10.0, 0.0],
            }
        }
        None => [0.0, 10.0, 0.0],
    };

    let label = payload
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("imported mesh")
        .to_string();

    let path = std::path::Path::new(mesh_path);

    // Load GLTF if not already cached.
    match state.gltf_cache.load(path) {
        Ok(loaded_mesh) => {
            // Upload to GPU if not already there.
            if !state.mesh_renderer.has_mesh(path) {
                state
                    .mesh_renderer
                    .upload_mesh(state.renderer.device(), path, loaded_mesh);
            }

            // Create world object.
            let id = WorldObjectId::new();
            let properties = serde_json::json!({"mesh_path": mesh_path});

            state.world_objects.spawn_object(
                id,
                position,
                WorldObjectType::Mesh,
                properties.clone(),
                &mut state.physics,
            );

            // Spawn backing actor.
            let actor = WorldObjectActor::new(id, label.clone());
            let actor_id = state.registry.spawn(Box::new(actor));
            state.world_objects.bind_actor(id, actor_id);

            // Record for undo.
            state.history.record(WorldAction::SpawnObject {
                id,
                position,
                object_type: WorldObjectType::Mesh,
                properties: properties.clone(),
                label: label.clone(),
            });

            log::info!("Imported mesh '{}' at {:?} (id={})", mesh_path, position, id);

            let event = BridgeMessage::management(
                "mesh-imported",
                serde_json::json!({
                    "id": id.to_string(),
                    "path": mesh_path,
                    "position": position,
                    "label": label,
                }),
            );
            state.bridge.send_to(client_id, &event);
            state.bridge.broadcast(&event);
        }
        Err(e) => {
            log::warn!("Failed to load GLTF '{}': {}", mesh_path, e);
            let err = BridgeMessage::management(
                "error",
                serde_json::json!({"error": "mesh_load_failed", "path": mesh_path, "details": e.to_string()}),
            );
            state.bridge.send_to(client_id, &err);
        }
    }
}

/// Handle transform-object command: move/rotate/scale a world object.
fn handle_transform_object(
    state: &mut WorldState,
    client_id: usize,
    payload: &serde_json::Value,
) {
    let id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: transform-object missing 'id' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in transform-object: {}", e);
            return;
        }
    };

    let id = WorldObjectId::from_uuid(uuid);

    let obj = match state.world_objects.get(&id) {
        Some(o) => o,
        None => {
            log::warn!("Bridge: object {} not found for transform", id);
            return;
        }
    };

    let old_position = obj.position;
    let old_rotation = obj.rotation;
    let old_scale = obj.scale;

    let new_position = payload
        .get("position")
        .and_then(|v| v.as_array())
        .map(|a| {
            [
                a.first().and_then(|v| v.as_f64()).unwrap_or(old_position[0] as f64) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(old_position[1] as f64) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(old_position[2] as f64) as f32,
            ]
        })
        .unwrap_or(old_position);

    let new_rotation = payload
        .get("rotation")
        .and_then(|v| v.as_array())
        .map(|a| {
            [
                a.first().and_then(|v| v.as_f64()).unwrap_or(old_rotation[0] as f64) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(old_rotation[1] as f64) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(old_rotation[2] as f64) as f32,
                a.get(3).and_then(|v| v.as_f64()).unwrap_or(old_rotation[3] as f64) as f32,
            ]
        })
        .unwrap_or(old_rotation);

    let new_scale = payload
        .get("scale")
        .and_then(|v| v.as_array())
        .map(|a| {
            [
                a.first().and_then(|v| v.as_f64()).unwrap_or(old_scale[0] as f64) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(old_scale[1] as f64) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(old_scale[2] as f64) as f32,
            ]
        })
        .unwrap_or(old_scale);

    // Apply the transform.
    if let Some(obj) = state.world_objects.get_mut(&id) {
        obj.position = new_position;
        obj.rotation = new_rotation;
        obj.scale = new_scale;
    }

    // Record for undo.
    state.history.record(WorldAction::TransformObject {
        id,
        old_position,
        new_position,
        old_rotation,
        new_rotation,
        old_scale,
        new_scale,
    });

    let event = BridgeMessage::management(
        "object-transformed",
        serde_json::json!({
            "id": id_str,
            "position": new_position,
            "rotation": new_rotation,
            "scale": new_scale,
        }),
    );
    state.bridge.send_to(client_id, &event);
    state.bridge.broadcast(&event);
}

/// Collect connection lines between linked actors, update the line buffer.
fn update_connection_lines(state: &mut WorldState) {
    let mut vertices = Vec::new();
    let line_color = [0.0f32, 1.0, 0.5, 0.8]; // Green glow

    for vs in state.actor_visual_state.values() {
        let from_pos = match state.world_objects.get(&vs.world_object_id) {
            Some(obj) => obj.position,
            None => continue,
        };

        for target_id in &vs.connections {
            let to_pos = match state.world_objects.get(target_id) {
                Some(obj) => obj.position,
                None => continue,
            };

            vertices.push(line_render::LineVertex {
                position: from_pos,
                _pad: 0.0,
                color: line_color,
            });
            vertices.push(line_render::LineVertex {
                position: to_pos,
                _pad: 0.0,
                color: line_color,
            });
        }
    }

    state
        .line_renderer
        .update_lines(state.renderer.queue(), &vertices);
}

/// Collect mesh instances for GLTF objects, update the mesh renderer.
fn update_mesh_instances(state: &mut WorldState) {
    let instances: Vec<mesh_render::MeshInstance> = state
        .world_objects
        .all_objects()
        .iter()
        .filter(|obj| obj.object_type == WorldObjectType::Mesh)
        .filter_map(|obj| {
            let mesh_path = obj
                .properties
                .get("mesh_path")
                .and_then(|v| v.as_str())?;

            Some(mesh_render::MeshInstance {
                world_object_id: obj.id,
                mesh_path: std::path::PathBuf::from(mesh_path),
                position: obj.position,
                rotation: obj.rotation,
                scale: obj.scale,
                color: [0.8, 0.8, 0.8, 1.0], // Light gray default
            })
        })
        .collect();

    state.mesh_renderer.set_instances(instances);
}

/// Handle conjure-object command: start AI text-to-3D generation.
fn handle_conjure_object(
    state: &mut WorldState,
    client_id: usize,
    payload: &serde_json::Value,
) {
    let prompt = match payload.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            log::warn!("Bridge: conjure-object missing 'prompt' field");
            return;
        }
    };

    let position = payload
        .get("position")
        .and_then(|v| v.as_array())
        .map(|a| {
            [
                a.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                a.get(1).and_then(|v| v.as_f64()).unwrap_or(10.0) as f32,
                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            ]
        })
        .unwrap_or([0.0, 10.0, 0.0]);

    let id = state.conjuring.conjure(prompt, position);

    // Immediately place a placeholder cube at the position.
    state.world_objects.spawn_object(
        id,
        position,
        WorldObjectType::Interactive,
        serde_json::json!({"conjure_prompt": prompt, "placeholder": true}),
        &mut state.physics,
    );

    let actor = WorldObjectActor::new(id, format!("conjuring: {}", prompt));
    let actor_id = state.registry.spawn(Box::new(actor));
    state.world_objects.bind_actor(id, actor_id);

    log::info!("Bridge: conjuring '{}' at {:?} (id={})", prompt, position, id);

    let event = BridgeMessage::management(
        "conjuring-started",
        serde_json::json!({
            "id": id.to_string(),
            "prompt": prompt,
            "position": position,
        }),
    );
    state.bridge.send_to(client_id, &event);
    state.bridge.broadcast(&event);
}

/// Handle conjure-skybox command.
fn handle_conjure_skybox(
    state: &mut WorldState,
    client_id: usize,
    payload: &serde_json::Value,
) {
    let prompt = match payload.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            log::warn!("Bridge: conjure-skybox missing 'prompt' field");
            return;
        }
    };

    // TODO: Skybox generation runs in background and applies when complete.
    // For now, log the request.
    log::info!("Bridge: skybox conjure requested: '{}'", prompt);

    let event = BridgeMessage::management(
        "skybox-conjuring-started",
        serde_json::json!({"prompt": prompt}),
    );
    state.bridge.send_to(client_id, &event);
}

/// Process completed conjuring results: replace placeholders with generated meshes.
fn process_conjure_results(state: &mut WorldState) {
    let results = state.conjuring.poll_results();

    for result in results {
        let id = result.request_id;

        match result.status {
            ConjureStatus::Complete => {
                if let Some(mesh_path) = &result.mesh_path {
                    // Load and upload the mesh.
                    match state.gltf_cache.load(mesh_path) {
                        Ok(loaded_mesh) => {
                            if !state.mesh_renderer.has_mesh(mesh_path) {
                                state.mesh_renderer.upload_mesh(
                                    state.renderer.device(),
                                    mesh_path,
                                    loaded_mesh,
                                );
                            }

                            // Update the world object from Interactive (placeholder) to Mesh.
                            if let Some(obj) = state.world_objects.get_mut(&id) {
                                obj.object_type = WorldObjectType::Mesh;
                                obj.properties = serde_json::json!({
                                    "mesh_path": mesh_path.to_string_lossy(),
                                    "conjure_prompt": state.conjuring.get_request(&id)
                                        .map(|r| r.prompt.as_str()).unwrap_or(""),
                                    "api_used": result.api_used.as_deref().unwrap_or(""),
                                });
                            }

                            log::info!(
                                "Conjured object {} placed (api={}, mesh={})",
                                id,
                                result.api_used.as_deref().unwrap_or("unknown"),
                                mesh_path.display()
                            );

                            let event = BridgeMessage::management(
                                "conjuring-complete",
                                serde_json::json!({
                                    "id": id.to_string(),
                                    "mesh_path": mesh_path.to_string_lossy(),
                                    "api_used": result.api_used,
                                }),
                            );
                            state.bridge.broadcast(&event);
                        }
                        Err(e) => {
                            log::warn!("Failed to load conjured mesh {}: {}", mesh_path.display(), e);
                        }
                    }
                }
            }
            ConjureStatus::Failed => {
                // Keep the placeholder, notify clients.
                log::info!("Conjuring failed for {} (placeholder remains)", id);

                let event = BridgeMessage::management(
                    "conjuring-failed",
                    serde_json::json!({
                        "id": id.to_string(),
                        "error": result.error,
                    }),
                );
                state.bridge.broadcast(&event);
            }
            _ => {}
        }
    }
}

/// Update NPC pathfinding and movement.
/// Uses the chunk manager for walkability checks.
fn update_npcs(state: &mut WorldState, dt: f32) {
    let block_size = chunk::BLOCK_SIZE;
    let chunk_mgr = &state.chunk_manager;

    let is_walkable = |nav_pos: pathfinding::NavPos| -> bool {
        // Walkable if: block below is solid, block at feet is air, block at head is air.
        let feet_world = nav_pos.to_world(block_size);
        let below_world = [feet_world[0], feet_world[1] - block_size, feet_world[2]];
        let head_world = [feet_world[0], feet_world[1] + block_size, feet_world[2]];

        let below = chunk_mgr
            .get_block(glam::Vec3::from(below_world))
            .unwrap_or(chunk::BlockType::Air);
        let feet = chunk_mgr
            .get_block(glam::Vec3::from(feet_world))
            .unwrap_or(chunk::BlockType::Air);
        let head = chunk_mgr
            .get_block(glam::Vec3::from(head_world))
            .unwrap_or(chunk::BlockType::Air);

        below != chunk::BlockType::Air
            && feet == chunk::BlockType::Air
            && head == chunk::BlockType::Air
    };

    state.npcs.update(dt, &is_walkable);
}

/// Handle talk-to-npc command: send a message to a specific NPC for LLM conversation.
fn handle_talk_to_npc(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let npc_id_str = match payload.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            log::warn!("Bridge: talk-to-npc missing 'id' field");
            return;
        }
    };

    let text = match payload.get("text").and_then(|v| v.as_str()) {
        Some(t) => t.to_string(),
        None => {
            log::warn!("Bridge: talk-to-npc missing 'text' field");
            return;
        }
    };

    let uuid = match uuid::Uuid::parse_str(npc_id_str) {
        Ok(u) => u,
        Err(e) => {
            log::warn!("Bridge: invalid UUID in talk-to-npc: {}", e);
            return;
        }
    };
    let npc_id = WorldObjectId::from_uuid(uuid);

    if !state.npcs.is_conversational(&npc_id) {
        log::warn!("Bridge: NPC {} is not conversational", npc_id);
        let err = BridgeMessage::management(
            "error",
            serde_json::json!({"error": "not_conversational", "id": npc_id_str}),
        );
        state.bridge.send_to(client_id, &err);
        return;
    }

    // Start conversation if not already in one.
    state.npcs.start_conversation(&npc_id);

    // Build LLM request.
    let request = match state.conversations.get_mut(&npc_id) {
        Some(conv) => conv.user_message(&text),
        None => {
            log::warn!("Bridge: no conversation state for NPC {}", npc_id);
            return;
        }
    };

    if !state.llm.is_configured() {
        // No API key -- respond with a placeholder.
        let fallback_text = "I'd love to chat, but I'm not connected to an LLM right now. Set ANTHROPIC_API_KEY to enable conversations.";
        state.npcs.set_speech(&npc_id, fallback_text);
        if let Some(conv) = state.conversations.get_mut(&npc_id) {
            conv.assistant_response(fallback_text);
        }
        let event = BridgeMessage::management(
            "npc-speech",
            serde_json::json!({
                "id": npc_id.to_string(),
                "text": fallback_text,
                "tool_calls": [],
            }),
        );
        state.bridge.broadcast(&event);
        return;
    }

    state.llm.submit(request);
    log::info!("Bridge: '{}' says to NPC {}: {}", "player", npc_id, text);

    // Acknowledge the message.
    let ack = BridgeMessage::management(
        "npc-thinking",
        serde_json::json!({"id": npc_id.to_string()}),
    );
    state.bridge.send_to(client_id, &ack);
}

/// Handle talk-to-nearest: find the closest conversational NPC and send a message.
fn handle_talk_to_nearest(state: &mut WorldState, client_id: usize, payload: &serde_json::Value) {
    let text = match payload.get("text").and_then(|v| v.as_str()) {
        Some(t) => t.to_string(),
        None => {
            log::warn!("Bridge: talk-to-nearest missing 'text' field");
            return;
        }
    };

    let player_pos = state.camera.position().to_array();
    let max_distance = payload
        .get("range")
        .and_then(|v| v.as_f64())
        .unwrap_or(8.0) as f32;

    let npc_id = match state.npcs.nearest_conversational(player_pos, max_distance) {
        Some(id) => id,
        None => {
            let err = BridgeMessage::management(
                "error",
                serde_json::json!({"error": "no_npc_in_range"}),
            );
            state.bridge.send_to(client_id, &err);
            return;
        }
    };

    // Re-dispatch as a talk-to-npc with the found NPC ID.
    let augmented = serde_json::json!({
        "id": npc_id.to_string(),
        "text": text,
    });
    handle_talk_to_npc(state, client_id, &augmented);
}

/// Process completed LLM responses: update NPC speech, handle tool calls.
fn process_llm_responses(state: &mut WorldState) {
    let responses = state.llm.poll_results();

    for response in responses {
        let npc_id = response.conversation_id;

        if let Some(ref error) = response.error {
            log::warn!("LLM error for NPC {}: {}", npc_id, error);
            state
                .npcs
                .set_speech(&npc_id, "Sorry, I had trouble thinking...");
            if let Some(conv) = state.conversations.get_mut(&npc_id) {
                conv.awaiting_response = false;
            }

            let event = BridgeMessage::management(
                "npc-speech",
                serde_json::json!({
                    "id": npc_id.to_string(),
                    "text": "Sorry, I had trouble thinking...",
                    "error": error,
                }),
            );
            state.bridge.broadcast(&event);
            continue;
        }

        // Record the assistant's response in conversation history.
        if let Some(conv) = state.conversations.get_mut(&npc_id) {
            conv.assistant_response(&response.text);
        }

        // Update NPC speech bubble.
        if !response.text.is_empty() {
            state.npcs.set_speech(&npc_id, &response.text);

            // Play TTS.
            state.tts.speak(&response.text);
        }

        // Process tool calls.
        let tool_calls_json: Vec<serde_json::Value> = response
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "name": tc.name,
                    "input": tc.input,
                })
            })
            .collect();

        for tc in &response.tool_calls {
            match tc.name.as_str() {
                "conjure_object" => {
                    let description = tc
                        .input
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("mysterious object");

                    // Get NPC position for placement offset.
                    let npc_pos = state
                        .npcs
                        .get_info(&npc_id)
                        .map(|info| info.position)
                        .unwrap_or([0.0, 10.0, 0.0]);

                    let offset = tc
                        .input
                        .get("position_offset")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            [
                                a.first().and_then(|v| v.as_f64()).unwrap_or(2.0) as f32,
                                a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                                a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                            ]
                        })
                        .unwrap_or([2.0, 0.0, 2.0]);

                    let position = [
                        npc_pos[0] + offset[0],
                        npc_pos[1] + offset[1],
                        npc_pos[2] + offset[2],
                    ];

                    // Use the conjuring pipeline.
                    let conjure_payload = serde_json::json!({
                        "prompt": description,
                        "position": position,
                    });
                    handle_conjure_object(state, 0, &conjure_payload);

                    log::info!(
                        "NPC {} conjured '{}' at {:?}",
                        npc_id,
                        description,
                        position
                    );
                }
                "set_weather" => {
                    if let Some(w) = tc.input.get("weather").and_then(|v| v.as_str()) {
                        let weather_type = match w {
                            "rain" => weather::WeatherType::Rain,
                            "snow" => weather::WeatherType::Snow,
                            _ => weather::WeatherType::Clear,
                        };
                        state.weather.set_weather(weather_type);
                        log::info!("NPC {} set weather to {:?}", npc_id, weather_type);
                    }
                }
                "set_time" => {
                    if let Some(t) = tc.input.get("time").and_then(|v| v.as_str()) {
                        let time_val = match t {
                            "dawn" => 0.25,
                            "morning" => 0.35,
                            "noon" => 0.5,
                            "afternoon" => 0.6,
                            "dusk" => 0.75,
                            "night" => 0.85,
                            "midnight" => 0.0,
                            _ => 0.5,
                        };
                        state.sky.set_time_of_day(time_val);
                        log::info!("NPC {} set time to {} ({})", npc_id, t, time_val);
                    }
                }
                _ => {
                    log::warn!("Unknown tool call from LLM: {}", tc.name);
                }
            }
        }

        // Broadcast speech event to bridge clients.
        let event = BridgeMessage::management(
            "npc-speech",
            serde_json::json!({
                "id": npc_id.to_string(),
                "text": response.text,
                "tool_calls": tool_calls_json,
            }),
        );
        state.bridge.broadcast(&event);
    }
}

// ---- VR Integration ----

/// Update VR tracking, process input events, and override camera when VR is active.
fn update_vr(state: &mut WorldState, dt: f32) {
    state.vr.update(dt);

    if !state.vr.is_active() {
        return;
    }

    // Override camera with VR head tracking.
    state.camera.set_position(state.vr.head_position());
    let (yaw, pitch, _) = state.vr.head_rotation().to_euler(glam::EulerRot::YXZ);
    state.camera.set_rotation(yaw, pitch);

    // Process VR controller input into world placement intents.
    let intents = state.vr.process_input();
    for intent in &intents {
        match intent {
            crate::vr::VrPlacementIntent::PlaceBlock { position, normal } => {
                let pos = glam::Vec3::from(*position);
                let norm = glam::Vec3::from(*normal);
                let target = pos + norm * 0.01;
                let material = state.input_state.selected_material();
                state.chunk_manager.set_block(target, material);
            }
            crate::vr::VrPlacementIntent::RemoveBlock { position } => {
                let pos = glam::Vec3::from(*position);
                state.chunk_manager.set_block(pos, chunk::BlockType::Air);
            }
            crate::vr::VrPlacementIntent::Interact { world_position } => {
                let pos = glam::Vec3::from(*world_position);
                if let Some((id, _obj)) = state.world_objects.find_nearest(pos.to_array(), 3.0) {
                    if let Some(&actor_id) = state.world_objects.actor_for_object(&id) {
                        let msg = Message::new(
                            actor_id,
                            InteractMessage {
                                world_object_id: id,
                                action: "interact".to_string(),
                                payload: serde_json::json!({"source": "vr_controller"}),
                            },
                            actor_id,
                        );
                        let _ = state.registry.send(msg);
                    }
                }
            }
            crate::vr::VrPlacementIntent::Select { world_position } => {
                let pos = glam::Vec3::from(*world_position);
                if let Some((id, _obj)) = state.world_objects.find_nearest(pos.to_array(), 3.0) {
                    // Show actor inspector panel in VR.
                    let actor_state = state
                        .actor_visual_state
                        .get(&id)
                        .map(|s| s.state.clone())
                        .unwrap_or(serde_json::json!({}));
                    let panel = VrUiPanel::following(VrPanelContent::ActorInspector {
                        label: id.to_string(),
                        state: actor_state,
                    });
                    state.vr_panels.push(panel);
                }
            }
        }
    }

    // Process VR input events (teleport, etc).
    let events = state.vr.take_events();
    for event in &events {
        if let crate::vr::VrInputEvent::TeleportConfirm { target } = event {
            state.vr.teleport_to(glam::Vec3::from(*target));
        }
    }

    // Update VR UI panels to face the user.
    let head_pos = state.vr.head_position();
    let head_rot = state.vr.head_rotation();
    for panel in &mut state.vr_panels {
        panel.face_user(head_pos, head_rot);
    }
}

/// Handle VR toggle bridge command.
fn handle_toggle_vr(state: &mut WorldState, client_id: usize) {
    let active = state.vr.is_active();
    let status = if active { "active" } else { "inactive" };
    log::info!("Bridge: VR status = {}", status);
    let event = BridgeMessage::management(
        "vr-toggled",
        serde_json::json!({ "active": active }),
    );
    state.bridge.send_to(client_id, &event);
}

/// Handle VR panel show command.
fn handle_vr_show_panel(state: &mut WorldState, payload: &serde_json::Value) {
    let panel_type = payload
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("world-info");

    let content = match panel_type {
        "actor-inspector" => {
            let label = payload
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            VrPanelContent::ActorInspector {
                label,
                state: serde_json::json!({}),
            }
        }
        "building-tools" => VrPanelContent::BuildingTools {
            selected_material: format!("{:?}", state.input_state.selected_material()),
            tool_mode: "place".to_string(),
        },
        _ => VrPanelContent::WorldInfo {
            time_of_day: state.sky.time_of_day(),
            weather: format!("{:?}", state.weather.weather()),
            npc_count: state.npcs.count(),
            object_count: state.world_objects.count(),
        },
    };

    let panel = VrUiPanel::following(content);
    state.vr_panels.push(panel);
    log::info!("Bridge: VR panel '{}' shown", panel_type);
}
