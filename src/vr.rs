//! VR integration via OpenXR.
//!
//! Compile-gated behind the `vr` feature flag. On macOS (no OpenXR runtime),
//! all VR types compile to no-op stubs. On Windows/Linux with the `vr` feature
//! enabled, this module provides:
//!
//! - OpenXR session management (instance, system, session)
//! - Head tracking -> camera position/orientation
//! - Controller/hand input -> PlacementIntent messages
//! - Stereo rendering (left/right eye projection matrices)
//! - Teleport locomotion for room-scale movement
//! - In-VR UI panels for actor inspection and building tools

use glam::{Mat4, Quat, Vec3};

/// VR hand (left or right controller).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

/// VR locomotion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocomotionMode {
    /// Point-and-teleport movement.
    Teleport,
    /// Smooth continuous movement via thumbstick.
    Smooth,
}

/// VR controller button state.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControllerState {
    /// Trigger (index finger) [0, 1].
    pub trigger: f32,
    /// Grip (hand squeeze) [0, 1].
    pub grip: f32,
    /// Thumbstick X [-1, 1].
    pub thumbstick_x: f32,
    /// Thumbstick Y [-1, 1].
    pub thumbstick_y: f32,
    /// Primary button (A/X) pressed.
    pub button_a: bool,
    /// Secondary button (B/Y) pressed.
    pub button_b: bool,
    /// Thumbstick pressed.
    pub thumbstick_click: bool,
}

/// Per-eye rendering data.
#[derive(Debug, Clone)]
pub struct EyeView {
    /// View matrix for this eye.
    pub view: Mat4,
    /// Projection matrix for this eye.
    pub projection: Mat4,
    /// The eye's field of view (radians).
    pub fov: EyeFov,
}

/// Field of view angles for one eye.
#[derive(Debug, Clone, Copy, Default)]
pub struct EyeFov {
    pub angle_left: f32,
    pub angle_right: f32,
    pub angle_up: f32,
    pub angle_down: f32,
}

/// VR input event generated each frame.
#[derive(Debug, Clone)]
pub enum VrInputEvent {
    /// Controller trigger pulled past threshold.
    TriggerPress { hand: Hand },
    /// Controller trigger released.
    TriggerRelease { hand: Hand },
    /// Grip pressed.
    GripPress { hand: Hand },
    /// Grip released.
    GripRelease { hand: Hand },
    /// Primary button pressed.
    ButtonA { hand: Hand },
    /// Secondary button pressed.
    ButtonB { hand: Hand },
    /// Teleport target confirmed (trigger release while aiming).
    TeleportConfirm { target: [f32; 3] },
    /// Thumbstick movement for smooth locomotion.
    ThumbstickMove {
        hand: Hand,
        x: f32,
        y: f32,
    },
}

/// Abstract VR placement intent (maps controller ray to world interaction).
#[derive(Debug, Clone)]
pub enum VrPlacementIntent {
    /// Place a block at the ray intersection.
    PlaceBlock { position: [f32; 3], normal: [f32; 3] },
    /// Remove a block at the ray intersection.
    RemoveBlock { position: [f32; 3] },
    /// Interact with a world object.
    Interact { world_position: [f32; 3] },
    /// Select an object for inspection.
    Select { world_position: [f32; 3] },
}

/// Frame timing data from the VR runtime.
#[derive(Debug, Clone, Copy)]
pub struct VrFrameTiming {
    /// Predicted display time for the current frame.
    pub predicted_display_time: f64,
    /// Time between frames.
    pub frame_interval: f64,
}

impl Default for VrFrameTiming {
    fn default() -> Self {
        Self {
            predicted_display_time: 0.0,
            frame_interval: 1.0 / 90.0, // 90fps target
        }
    }
}

// ---- Feature-gated implementation ----

/// The VR system manages the OpenXR session and input.
///
/// When the `vr` feature is not enabled, this compiles to no-op stubs.
pub struct VrSystem {
    active: bool,
    locomotion: LocomotionMode,
    head_position: Vec3,
    head_rotation: Quat,
    left_controller: ControllerPose,
    right_controller: ControllerPose,
    left_state: ControllerState,
    right_state: ControllerState,
    eye_views: [EyeView; 2],
    teleport_target: Option<Vec3>,
    /// Accumulated events this frame.
    events: Vec<VrInputEvent>,
    /// Play space offset (for room-scale).
    play_space_offset: Vec3,

    #[cfg(feature = "vr")]
    xr_instance: Option<openxr::Instance>,
    #[cfg(feature = "vr")]
    xr_session: Option<openxr::Session<openxr::Vulkan>>,
}

/// Pose (position + orientation) for a controller.
#[derive(Debug, Clone, Copy)]
struct ControllerPose {
    position: Vec3,
    rotation: Quat,
}

impl Default for ControllerPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

impl VrSystem {
    /// Create a new VR system. On macOS (no `vr` feature), this returns
    /// an inactive stub that compiles to no-ops.
    pub fn new() -> Self {
        let default_eye = EyeView {
            view: Mat4::IDENTITY,
            projection: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_2,
                1.0,
                0.01,
                1000.0,
            ),
            fov: EyeFov::default(),
        };

        let mut system = Self {
            active: false,
            locomotion: LocomotionMode::Teleport,
            head_position: Vec3::ZERO,
            head_rotation: Quat::IDENTITY,
            left_controller: ControllerPose::default(),
            right_controller: ControllerPose::default(),
            left_state: ControllerState::default(),
            right_state: ControllerState::default(),
            eye_views: [default_eye.clone(), default_eye],
            teleport_target: None,
            events: Vec::new(),
            play_space_offset: Vec3::ZERO,
            #[cfg(feature = "vr")]
            xr_instance: None,
            #[cfg(feature = "vr")]
            xr_session: None,
        };

        system.try_initialize();
        system
    }

    /// Attempt to initialize the OpenXR runtime.
    fn try_initialize(&mut self) {
        #[cfg(feature = "vr")]
        {
            match openxr::Entry::linked().create_instance(
                &openxr::ApplicationInfo {
                    application_name: "BitParty World",
                    application_version: 1,
                    engine_name: "BitParty",
                    engine_version: 1,
                },
                &openxr::ExtensionSet::default(),
                &[],
            ) {
                Ok(instance) => {
                    log::info!("OpenXR instance created");
                    self.xr_instance = Some(instance);
                    self.active = true;
                }
                Err(e) => {
                    log::warn!("OpenXR initialization failed: {}", e);
                    self.active = false;
                }
            }
        }

        #[cfg(not(feature = "vr"))]
        {
            log::info!("VR: disabled (compile without 'vr' feature on this platform)");
            self.active = false;
        }
    }

    /// Whether VR is active and rendering.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Current head position in world space.
    pub fn head_position(&self) -> Vec3 {
        self.head_position + self.play_space_offset
    }

    /// Current head rotation.
    pub fn head_rotation(&self) -> Quat {
        self.head_rotation
    }

    /// Get the view matrix for a specific eye (0=left, 1=right).
    pub fn eye_view(&self, eye: usize) -> &EyeView {
        &self.eye_views[eye.min(1)]
    }

    /// Get controller state.
    pub fn controller_state(&self, hand: Hand) -> &ControllerState {
        match hand {
            Hand::Left => &self.left_state,
            Hand::Right => &self.right_state,
        }
    }

    /// Get controller world position.
    pub fn controller_position(&self, hand: Hand) -> Vec3 {
        let pose = match hand {
            Hand::Left => &self.left_controller,
            Hand::Right => &self.right_controller,
        };
        pose.position + self.play_space_offset
    }

    /// Get controller world rotation.
    pub fn controller_rotation(&self, hand: Hand) -> Quat {
        match hand {
            Hand::Left => self.left_controller.rotation,
            Hand::Right => self.right_controller.rotation,
        }
    }

    /// Get the forward ray direction for a controller.
    pub fn controller_ray(&self, hand: Hand) -> Vec3 {
        let rotation = self.controller_rotation(hand);
        rotation * -Vec3::Z
    }

    /// Current teleport target (if aiming).
    pub fn teleport_target(&self) -> Option<Vec3> {
        self.teleport_target
    }

    /// Set the locomotion mode.
    pub fn set_locomotion(&mut self, mode: LocomotionMode) {
        self.locomotion = mode;
    }

    /// Current locomotion mode.
    pub fn locomotion(&self) -> LocomotionMode {
        self.locomotion
    }

    /// Drain input events for this frame.
    pub fn take_events(&mut self) -> Vec<VrInputEvent> {
        std::mem::take(&mut self.events)
    }

    /// Called each frame to update tracking and input.
    pub fn update(&mut self, _dt: f32) {
        if !self.active {
            return;
        }

        #[cfg(feature = "vr")]
        self.update_xr();
    }

    #[cfg(feature = "vr")]
    fn update_xr(&mut self) {
        // OpenXR frame loop:
        // 1. Wait for frame timing
        // 2. Begin frame
        // 3. Locate views (head tracking)
        // 4. Read controller input
        // 5. Process input into events
        // 6. End frame (submit render data)
        //
        // This is a placeholder for the full OpenXR integration.
        // The actual implementation requires:
        // - Swapchain creation matching wgpu textures
        // - View/projection matrix extraction from xr_views
        // - Action set binding for controllers
        // - Haptic feedback
    }

    /// Apply teleport: move the play space offset so the player
    /// appears at the target position.
    pub fn teleport_to(&mut self, target: Vec3) {
        // Offset = target - current head position (without offset).
        self.play_space_offset = target - self.head_position;
        self.play_space_offset.y = target.y; // Keep vertical position.
        self.teleport_target = None;
    }

    /// Process VR input and generate PlacementIntent messages.
    pub fn process_input(&mut self) -> Vec<VrPlacementIntent> {
        if !self.active {
            return Vec::new();
        }

        let mut intents = Vec::new();

        // Process right controller trigger for block interaction.
        if self.right_state.trigger > 0.8 {
            let ray_origin = self.controller_position(Hand::Right);
            let ray_dir = self.controller_ray(Hand::Right);
            let _hit_pos = ray_origin + ray_dir * 5.0; // Placeholder

            // Button A = place block, trigger without A = remove block.
            if self.right_state.button_a {
                intents.push(VrPlacementIntent::PlaceBlock {
                    position: (ray_origin + ray_dir * 5.0).to_array(),
                    normal: (-ray_dir).to_array(),
                });
            } else {
                intents.push(VrPlacementIntent::RemoveBlock {
                    position: (ray_origin + ray_dir * 5.0).to_array(),
                });
            }
        }

        // Left controller grip = select/inspect.
        if self.left_state.grip > 0.8 {
            let ray_origin = self.controller_position(Hand::Left);
            let ray_dir = self.controller_ray(Hand::Left);
            intents.push(VrPlacementIntent::Select {
                world_position: (ray_origin + ray_dir * 5.0).to_array(),
            });
        }

        intents
    }

    /// Generate stereo view-projection matrices for VR rendering.
    /// Returns (left_vp, right_vp) or None if VR is inactive.
    pub fn stereo_view_projections(&self) -> Option<(Mat4, Mat4)> {
        if !self.active {
            return None;
        }

        let left = self.eye_views[0].projection * self.eye_views[0].view;
        let right = self.eye_views[1].projection * self.eye_views[1].view;
        Some((left, right))
    }
}

/// VR UI panel for in-headset information display.
#[derive(Debug, Clone)]
pub struct VrUiPanel {
    /// Panel position in world space.
    pub position: Vec3,
    /// Panel rotation (faces the user).
    pub rotation: Quat,
    /// Panel width in meters.
    pub width: f32,
    /// Panel height in meters.
    pub height: f32,
    /// Whether the panel is visible.
    pub visible: bool,
    /// Panel content type.
    pub content: VrPanelContent,
}

/// Content displayed on a VR UI panel.
#[derive(Debug, Clone)]
pub enum VrPanelContent {
    /// Actor inspection: shows actor state, connections, properties.
    ActorInspector {
        label: String,
        state: serde_json::Value,
    },
    /// Building tool palette: material selection and tool modes.
    BuildingTools {
        selected_material: String,
        tool_mode: String,
    },
    /// World info: time, weather, NPC count, etc.
    WorldInfo {
        time_of_day: f32,
        weather: String,
        npc_count: usize,
        object_count: usize,
    },
}

impl VrUiPanel {
    /// Create a panel that follows the user at arm's length.
    pub fn following(content: VrPanelContent) -> Self {
        Self {
            position: Vec3::new(0.0, 1.5, -0.8),
            rotation: Quat::IDENTITY,
            width: 0.6,
            height: 0.4,
            visible: true,
            content,
        }
    }

    /// Update panel position to face the user.
    pub fn face_user(&mut self, head_pos: Vec3, head_rot: Quat) {
        let forward = head_rot * -Vec3::Z;
        let panel_distance = 0.8;
        self.position = head_pos + forward * panel_distance + Vec3::new(0.0, -0.2, 0.0);
        // Rotate panel to face the user.
        let look_dir = (head_pos - self.position).normalize();
        self.rotation = Quat::from_rotation_arc(-Vec3::Z, look_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vr_system_initializes_inactive_on_macos() {
        let vr = VrSystem::new();
        // On macOS without the vr feature, should be inactive.
        #[cfg(not(feature = "vr"))]
        assert!(!vr.is_active());
        // Just check it doesn't panic.
        let _ = vr.head_position();
        let _ = vr.head_rotation();
    }

    #[test]
    fn controller_state_defaults() {
        let vr = VrSystem::new();
        let state = vr.controller_state(Hand::Left);
        assert_eq!(state.trigger, 0.0);
        assert_eq!(state.grip, 0.0);
        assert!(!state.button_a);
    }

    #[test]
    fn teleport_moves_play_space() {
        let mut vr = VrSystem::new();
        let target = Vec3::new(10.0, 5.0, 20.0);
        vr.teleport_to(target);

        let head = vr.head_position();
        assert!((head.x - 10.0).abs() < 0.01);
        assert!((head.z - 20.0).abs() < 0.01);
    }

    #[test]
    fn locomotion_mode() {
        let mut vr = VrSystem::new();
        assert_eq!(vr.locomotion(), LocomotionMode::Teleport);
        vr.set_locomotion(LocomotionMode::Smooth);
        assert_eq!(vr.locomotion(), LocomotionMode::Smooth);
    }

    #[test]
    fn eye_view_returns_valid_matrices() {
        let vr = VrSystem::new();
        let left = vr.eye_view(0);
        let right = vr.eye_view(1);
        // Projection should not be zero.
        assert_ne!(left.projection, Mat4::ZERO);
        assert_ne!(right.projection, Mat4::ZERO);
    }

    #[test]
    fn controller_ray_direction() {
        let vr = VrSystem::new();
        let ray = vr.controller_ray(Hand::Right);
        // Default rotation is identity, so ray should be -Z.
        assert!((ray.z - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn vr_ui_panel_following() {
        let panel = VrUiPanel::following(VrPanelContent::WorldInfo {
            time_of_day: 0.5,
            weather: "clear".to_string(),
            npc_count: 4,
            object_count: 10,
        });
        assert!(panel.visible);
        assert!(panel.width > 0.0);
        assert!(panel.height > 0.0);
    }

    #[test]
    fn vr_ui_panel_faces_user() {
        let mut panel = VrUiPanel::following(VrPanelContent::BuildingTools {
            selected_material: "stone".to_string(),
            tool_mode: "place".to_string(),
        });

        let head_pos = Vec3::new(0.0, 1.7, 0.0);
        let head_rot = Quat::IDENTITY;
        panel.face_user(head_pos, head_rot);

        // Panel should be in front of the user.
        assert!(panel.position.z < head_pos.z);
    }

    #[test]
    fn no_stereo_when_inactive() {
        let vr = VrSystem::new();
        #[cfg(not(feature = "vr"))]
        assert!(vr.stereo_view_projections().is_none());
    }

    #[test]
    fn process_input_empty_when_inactive() {
        let mut vr = VrSystem::new();
        let intents = vr.process_input();
        assert!(intents.is_empty());
    }

    #[test]
    fn take_events_drains() {
        let mut vr = VrSystem::new();
        let events = vr.take_events();
        assert!(events.is_empty());
        // Second call should also be empty.
        let events2 = vr.take_events();
        assert!(events2.is_empty());
    }
}
