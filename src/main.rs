//! BitParty World - 3D Spatial Creative Environment
//!
//! A 3D environment where actors inhabit a voxel+mesh world.
//! Connects to the TypeScript actor system via the protocol bridge.

mod bridge_server;
mod conjuring;
mod llm;
mod physics;
mod state_history;
mod vr;
mod world;
mod world_object_actor;
mod world_object_store;
mod world_objects;

use anyhow::Result;
use log::info;

fn main() -> Result<()> {
    env_logger::init();
    info!("BitParty World starting...");

    let event_loop = winit::event_loop::EventLoop::new()?;
    let mut app = world::App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
