use glam::{Mat4, Vec3};

/// First-person camera with WASD controls.
pub struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    aspect: f32,
    fov_y: f32,
    near: f32,
    far: f32,
    speed: f32,
    sensitivity: f32,
    velocity: Vec3,
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
}

impl Camera {
    pub fn new(position: Vec3, aspect: f32) -> Self {
        Self {
            position,
            yaw: -90.0_f32.to_radians(),
            pitch: 0.0,
            aspect,
            fov_y: 70.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,
            speed: 10.0,
            sensitivity: 0.003,
            velocity: Vec3::ZERO,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    pub fn front(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(Vec3::Y).normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.front(), Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.sensitivity;
        self.pitch -= dy * self.sensitivity;
        self.pitch = self.pitch.clamp(
            -89.0_f32.to_radians(),
            89.0_f32.to_radians(),
        );
    }

    pub fn set_move_forward(&mut self, active: bool) {
        self.move_forward = active;
    }

    pub fn set_move_backward(&mut self, active: bool) {
        self.move_backward = active;
    }

    pub fn set_move_left(&mut self, active: bool) {
        self.move_left = active;
    }

    pub fn set_move_right(&mut self, active: bool) {
        self.move_right = active;
    }

    pub fn set_move_up(&mut self, active: bool) {
        self.move_up = active;
    }

    pub fn set_move_down(&mut self, active: bool) {
        self.move_down = active;
    }

    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
    }

    pub fn set_rotation(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn is_moving_forward(&self) -> bool {
        self.move_forward
    }

    pub fn is_moving_backward(&self) -> bool {
        self.move_backward
    }

    pub fn is_moving_left(&self) -> bool {
        self.move_left
    }

    pub fn is_moving_right(&self) -> bool {
        self.move_right
    }

    pub fn is_moving_up(&self) -> bool {
        self.move_up
    }

    pub fn is_moving_down(&self) -> bool {
        self.move_down
    }

    pub fn update(&mut self, dt: f32) {
        let front = self.front();
        let right = self.right();

        let mut wish_dir = Vec3::ZERO;

        if self.move_forward {
            wish_dir += front;
        }
        if self.move_backward {
            wish_dir -= front;
        }
        if self.move_right {
            wish_dir += right;
        }
        if self.move_left {
            wish_dir -= right;
        }
        if self.move_up {
            wish_dir += Vec3::Y;
        }
        if self.move_down {
            wish_dir -= Vec3::Y;
        }

        if wish_dir.length_squared() > 0.0 {
            wish_dir = wish_dir.normalize();
        }

        self.velocity = wish_dir * self.speed;
        self.position += self.velocity * dt;
    }
}
