//! Actor implementation for world objects.
//!
//! Each placed object in the 3D world is backed by a WorldObjectActor.
//! This actor handles interact, configure, and transform messages,
//! and tracks connections to other actors for visualization.

use std::any::TypeId;

use bitparty::actor::{Actor, ActorContext, ActorId, HandleResult};
use bitparty::actor::Message;
use bitparty::bridge::WorldObjectId;

/// Message sent when a user interacts with a world object (e.g., press E near it).
#[derive(Debug, Clone)]
pub struct InteractMessage {
    pub world_object_id: WorldObjectId,
    pub action: String,
    pub payload: serde_json::Value,
}

/// Message to configure an actor-backed object (change label, properties, connections).
#[derive(Debug, Clone)]
pub struct ConfigureMessage {
    pub world_object_id: WorldObjectId,
    pub properties: serde_json::Value,
}

/// Message to transform (move/rotate/scale) an object.
#[derive(Debug, Clone)]
pub struct TransformMessage {
    pub world_object_id: WorldObjectId,
    pub position: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub scale: Option<[f32; 3]>,
}

/// Notification emitted when actor state changes (for bridge broadcast).
#[derive(Debug, Clone)]
pub struct ActorStateChanged {
    pub world_object_id: WorldObjectId,
    pub label: String,
    pub state: serde_json::Value,
    pub connections: Vec<WorldObjectId>,
}

/// A world object backed by a BitParty actor.
pub struct WorldObjectActor {
    id: ActorId,
    world_object_id: WorldObjectId,
    label: String,
    state: serde_json::Value,
    connections: Vec<WorldObjectId>,
    interaction_count: u32,
    handles: Vec<TypeId>,
}

impl WorldObjectActor {
    pub fn new(world_object_id: WorldObjectId, label: String) -> Self {
        Self {
            id: ActorId::default(),
            world_object_id,
            label,
            state: serde_json::json!({}),
            connections: Vec::new(),
            interaction_count: 0,
            handles: vec![
                TypeId::of::<InteractMessage>(),
                TypeId::of::<ConfigureMessage>(),
                TypeId::of::<TransformMessage>(),
            ],
        }
    }

    pub fn world_object_id(&self) -> WorldObjectId {
        self.world_object_id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn state(&self) -> &serde_json::Value {
        &self.state
    }

    pub fn connections(&self) -> &[WorldObjectId] {
        &self.connections
    }

    pub fn interaction_count(&self) -> u32 {
        self.interaction_count
    }

    pub fn add_connection(&mut self, target: WorldObjectId) {
        if !self.connections.contains(&target) {
            self.connections.push(target);
        }
    }

    pub fn remove_connection(&mut self, target: &WorldObjectId) {
        self.connections.retain(|c| c != target);
    }
}

impl Actor for WorldObjectActor {
    fn id(&self) -> ActorId {
        self.id
    }

    fn set_id(&mut self, id: ActorId) {
        self.id = id;
    }

    fn handles(&self) -> &[TypeId] {
        &self.handles
    }

    fn handle(&mut self, msg: Message, ctx: &mut ActorContext) -> HandleResult {
        if let Some(interact) = msg.downcast_ref::<InteractMessage>() {
            self.interaction_count += 1;
            log::info!(
                "WorldObjectActor {}: interact #{} action={}",
                self.world_object_id,
                self.interaction_count,
                interact.action,
            );

            // Update state with interaction info.
            self.state = serde_json::json!({
                "last_action": interact.action,
                "interaction_count": self.interaction_count,
                "last_payload": interact.payload,
            });

            // Broadcast state change.
            ctx.broadcast(ActorStateChanged {
                world_object_id: self.world_object_id,
                label: self.label.clone(),
                state: self.state.clone(),
                connections: self.connections.clone(),
            });

            HandleResult::Handled
        } else if let Some(configure) = msg.downcast_ref::<ConfigureMessage>() {
            log::info!(
                "WorldObjectActor {}: configure",
                self.world_object_id,
            );

            // Update label if provided.
            if let Some(label) = configure.properties.get("label").and_then(|v| v.as_str()) {
                self.label = label.to_string();
            }

            // Merge properties into state.
            if let Some(obj) = configure.properties.as_object() {
                if let Some(state_obj) = self.state.as_object_mut() {
                    for (k, v) in obj {
                        state_obj.insert(k.clone(), v.clone());
                    }
                }
            }

            // Handle connections.
            if let Some(connect_to) = configure
                .properties
                .get("connect_to")
                .and_then(|v| v.as_str())
            {
                if let Ok(uuid) = uuid::Uuid::parse_str(connect_to) {
                    let target = WorldObjectId::from_uuid(uuid);
                    self.add_connection(target);
                }
            }

            if let Some(disconnect_from) = configure
                .properties
                .get("disconnect_from")
                .and_then(|v| v.as_str())
            {
                if let Ok(uuid) = uuid::Uuid::parse_str(disconnect_from) {
                    let target = WorldObjectId::from_uuid(uuid);
                    self.remove_connection(&target);
                }
            }

            ctx.broadcast(ActorStateChanged {
                world_object_id: self.world_object_id,
                label: self.label.clone(),
                state: self.state.clone(),
                connections: self.connections.clone(),
            });

            HandleResult::Handled
        } else if msg.is::<TransformMessage>() {
            // Transform is applied externally by WorldObjectManager.
            HandleResult::Handled
        } else {
            HandleResult::NotHandled
        }
    }

    fn on_spawn(&mut self, ctx: &mut ActorContext) {
        log::info!(
            "WorldObjectActor spawned: {} (world_id={})",
            self.label,
            self.world_object_id,
        );

        ctx.broadcast(ActorStateChanged {
            world_object_id: self.world_object_id,
            label: self.label.clone(),
            state: self.state.clone(),
            connections: self.connections.clone(),
        });
    }

    fn on_despawn(&mut self, _ctx: &mut ActorContext) {
        log::info!(
            "WorldObjectActor despawned: {} (world_id={})",
            self.label,
            self.world_object_id,
        );
    }

    fn type_name(&self) -> &'static str {
        "WorldObjectActor"
    }
}

/// Relay actor that forwards ActorStateChanged broadcasts to a crossbeam channel.
/// The main loop drains this channel and sends events to bridge clients.
pub struct BridgeRelayActor {
    id: ActorId,
    handles: Vec<TypeId>,
    event_tx: crossbeam::channel::Sender<ActorStateChanged>,
}

impl BridgeRelayActor {
    pub fn new(event_tx: crossbeam::channel::Sender<ActorStateChanged>) -> Self {
        Self {
            id: ActorId::default(),
            handles: vec![TypeId::of::<ActorStateChanged>()],
            event_tx,
        }
    }
}

impl Actor for BridgeRelayActor {
    fn id(&self) -> ActorId {
        self.id
    }

    fn set_id(&mut self, id: ActorId) {
        self.id = id;
    }

    fn handles(&self) -> &[TypeId] {
        &self.handles
    }

    fn handle(&mut self, msg: Message, _ctx: &mut ActorContext) -> HandleResult {
        if let Some(state_changed) = msg.downcast_ref::<ActorStateChanged>() {
            let _ = self.event_tx.send(state_changed.clone());
            HandleResult::Handled
        } else {
            HandleResult::NotHandled
        }
    }

    fn type_name(&self) -> &'static str {
        "BridgeRelayActor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitparty::actor::ActorRegistry;

    #[test]
    fn actor_spawns_and_handles_interact() {
        let mut registry = ActorRegistry::new();
        let world_id = WorldObjectId::new();
        let actor = WorldObjectActor::new(world_id, "test cube".to_string());

        let actor_id = registry.spawn(Box::new(actor));

        // Send an interact message.
        let msg = Message::new(
            actor_id,
            InteractMessage {
                world_object_id: world_id,
                action: "click".to_string(),
                payload: serde_json::json!({}),
            },
            actor_id,
        );
        let _ = registry.send(msg);
        registry.tick();

        // Verify actor is alive and correctly typed.
        let actor_ref = registry.get(actor_id).unwrap();
        assert_eq!(actor_ref.type_name(), "WorldObjectActor");

        // Verify the message was delivered (metrics).
        let metrics = registry.metrics().snapshot();
        assert!(metrics.messages_delivered >= 1);
    }

    #[test]
    fn actor_handles_configure() {
        let mut registry = ActorRegistry::new();
        let world_id = WorldObjectId::new();
        let actor = WorldObjectActor::new(world_id, "test".to_string());

        let actor_id = registry.spawn(Box::new(actor));

        let msg = Message::new(
            actor_id,
            ConfigureMessage {
                world_object_id: world_id,
                properties: serde_json::json!({"label": "renamed cube", "color": "red"}),
            },
            actor_id,
        );
        let _ = registry.send(msg);
        registry.tick();

        assert!(registry.get(actor_id).is_some());
    }

    #[test]
    fn actor_connection_management() {
        let world_id = WorldObjectId::new();
        let target_id = WorldObjectId::new();
        let mut actor = WorldObjectActor::new(world_id, "test".to_string());

        assert!(actor.connections().is_empty());

        actor.add_connection(target_id);
        assert_eq!(actor.connections().len(), 1);
        assert_eq!(actor.connections()[0], target_id);

        // Adding same connection again is a no-op.
        actor.add_connection(target_id);
        assert_eq!(actor.connections().len(), 1);

        actor.remove_connection(&target_id);
        assert!(actor.connections().is_empty());
    }

    #[test]
    fn actor_interaction_counter() {
        let world_id = WorldObjectId::new();
        let actor = WorldObjectActor::new(world_id, "counter test".to_string());
        assert_eq!(actor.interaction_count(), 0);
        assert_eq!(actor.label(), "counter test");
        assert_eq!(actor.world_object_id(), world_id);
    }

    #[test]
    fn bridge_relay_forwards_state_changes() {
        let (tx, rx) = crossbeam::channel::unbounded();
        let mut registry = ActorRegistry::new();

        // Spawn the relay actor (listens for ActorStateChanged broadcasts).
        let relay_id = registry.spawn(Box::new(BridgeRelayActor::new(tx)));

        // Verify the relay is registered as a handler for ActorStateChanged.
        let handlers = registry.handlers_for(TypeId::of::<ActorStateChanged>());
        assert!(
            handlers.contains(&relay_id),
            "relay should be registered as ActorStateChanged handler"
        );

        // Spawn a world object actor (broadcasts ActorStateChanged on spawn).
        let world_id = WorldObjectId::new();
        let actor_id = registry.spawn(Box::new(
            WorldObjectActor::new(world_id, "relay test".to_string()),
        ));

        // The on_spawn broadcast was routed during spawn. Tick to process
        // the relay's mailbox (the broadcast was delivered to the relay's
        // inbox during the WOA's spawn, now the relay needs to handle it).
        registry.tick();

        // The relay should have received the state change via the channel.
        // If the first tick didn't process it, try one more (some broadcast
        // routing may defer to the next tick cycle).
        let received = rx.try_recv().or_else(|_| {
            registry.tick();
            rx.try_recv()
        });
        assert!(
            received.is_ok(),
            "relay should have forwarded ActorStateChanged to channel"
        );
        let change = received.unwrap();
        assert_eq!(change.world_object_id, world_id);
        assert_eq!(change.label, "relay test");

        // Send an interact and verify the relay gets the updated state.
        let msg = Message::new(
            actor_id,
            InteractMessage {
                world_object_id: world_id,
                action: "poke".to_string(),
                payload: serde_json::json!({"intent": "test"}),
            },
            actor_id,
        );
        let _ = registry.send(msg);
        registry.tick();

        // InteractMessage is handled by the WOA, which broadcasts
        // ActorStateChanged. That broadcast may be routed to the relay
        // within the same tick or need another tick.
        let received = rx.try_recv().or_else(|_| {
            registry.tick();
            rx.try_recv()
        });
        assert!(
            received.is_ok(),
            "relay should have forwarded updated ActorStateChanged"
        );
        let change = received.unwrap();
        assert_eq!(change.world_object_id, world_id);
        assert_eq!(change.state["last_action"], "poke");
        assert_eq!(change.state["interaction_count"], 1);
    }
}
