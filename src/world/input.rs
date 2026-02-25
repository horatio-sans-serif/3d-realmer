use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

use super::camera::Camera;
use super::chunk::BlockType;

/// Block interaction action triggered by mouse click.
#[derive(Debug, Clone, Copy)]
pub enum BlockAction {
    Remove,
    Place,
}

/// Input state tracking for mouse look and keyboard actions.
pub struct InputState {
    last_cursor_pos: Option<(f64, f64)>,
    mouse_captured: bool,
    jump_requested: bool,
    block_action: Option<BlockAction>,
    interact_requested: bool,
    selected_material: BlockType,
    undo_requested: bool,
    redo_requested: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            last_cursor_pos: None,
            mouse_captured: false,
            jump_requested: false,
            block_action: None,
            interact_requested: false,
            selected_material: BlockType::Stone,
            undo_requested: false,
            redo_requested: false,
        }
    }

    pub fn handle_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if button == MouseButton::Right {
            self.mouse_captured = state == ElementState::Pressed;
            if !self.mouse_captured {
                self.last_cursor_pos = None;
            }
        }

        // Left click = remove block, middle click = place block
        if state == ElementState::Pressed {
            match button {
                MouseButton::Left => {
                    self.block_action = Some(BlockAction::Remove);
                }
                MouseButton::Middle => {
                    self.block_action = Some(BlockAction::Place);
                }
                _ => {}
            }
        }
    }

    pub fn handle_cursor_move(
        &mut self,
        position: winit::dpi::PhysicalPosition<f64>,
        camera: &mut Camera,
    ) {
        if !self.mouse_captured {
            self.last_cursor_pos = Some((position.x, position.y));
            return;
        }

        if let Some((last_x, last_y)) = self.last_cursor_pos {
            let dx = (position.x - last_x) as f32;
            let dy = (position.y - last_y) as f32;
            camera.rotate(dx, dy);
        }
        self.last_cursor_pos = Some((position.x, position.y));
    }

    pub fn handle_keyboard(&mut self, event: &KeyEvent, camera: &mut Camera) {
        let pressed = event.state == ElementState::Pressed;

        if let PhysicalKey::Code(key) = event.physical_key {
            match key {
                KeyCode::KeyW => camera.set_move_forward(pressed),
                KeyCode::KeyS => camera.set_move_backward(pressed),
                KeyCode::KeyA => camera.set_move_left(pressed),
                KeyCode::KeyD => camera.set_move_right(pressed),
                KeyCode::Space => {
                    if pressed {
                        self.jump_requested = true;
                    }
                }
                KeyCode::KeyE => {
                    if pressed {
                        self.interact_requested = true;
                    }
                }
                KeyCode::ShiftLeft => camera.set_move_down(pressed),
                // Material selection: number keys 1-9
                KeyCode::Digit1 => {
                    if pressed {
                        self.selected_material = BlockType::Stone;
                    }
                }
                KeyCode::Digit2 => {
                    if pressed {
                        self.selected_material = BlockType::Dirt;
                    }
                }
                KeyCode::Digit3 => {
                    if pressed {
                        self.selected_material = BlockType::Grass;
                    }
                }
                KeyCode::Digit4 => {
                    if pressed {
                        self.selected_material = BlockType::Sand;
                    }
                }
                KeyCode::Digit5 => {
                    if pressed {
                        self.selected_material = BlockType::Wood;
                    }
                }
                KeyCode::Digit6 => {
                    if pressed {
                        self.selected_material = BlockType::Brick;
                    }
                }
                KeyCode::Digit7 => {
                    if pressed {
                        self.selected_material = BlockType::Glass;
                    }
                }
                KeyCode::Digit8 => {
                    if pressed {
                        self.selected_material = BlockType::Water;
                    }
                }
                KeyCode::Digit9 => {
                    if pressed {
                        self.selected_material = BlockType::Leaf;
                    }
                }
                // Undo/redo: Z with modifiers
                KeyCode::KeyZ => {
                    if pressed {
                        // We'll check for Shift modifier in the event handler
                        self.undo_requested = true;
                    }
                }
                KeyCode::KeyY => {
                    if pressed {
                        self.redo_requested = true;
                    }
                }
                _ => {}
            }
        }
    }

    /// Take and clear the jump request flag.
    pub fn take_jump(&mut self) -> bool {
        let j = self.jump_requested;
        self.jump_requested = false;
        j
    }

    /// Take and clear any pending block action.
    pub fn take_block_action(&mut self) -> Option<BlockAction> {
        self.block_action.take()
    }

    /// Take and clear the interact request flag.
    pub fn take_interact(&mut self) -> bool {
        let i = self.interact_requested;
        self.interact_requested = false;
        i
    }

    /// Get the currently selected block material.
    pub fn selected_material(&self) -> BlockType {
        self.selected_material
    }

    /// Set the selected block material (from bridge commands).
    pub fn set_selected_material(&mut self, material: BlockType) {
        self.selected_material = material;
    }

    /// Take and clear the undo request flag.
    pub fn take_undo(&mut self) -> bool {
        let u = self.undo_requested;
        self.undo_requested = false;
        u
    }

    /// Take and clear the redo request flag.
    pub fn take_redo(&mut self) -> bool {
        let r = self.redo_requested;
        self.redo_requested = false;
        r
    }
}
