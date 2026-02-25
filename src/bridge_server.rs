//! WebSocket bridge server for cross-runtime communication.
//!
//! Runs a WebSocket server on a background thread. Clients send
//! BridgeMessages (newline-delimited JSON). Commands are forwarded
//! to the main thread via a crossbeam channel. Events are sent
//! back to connected clients.

use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam::channel::{self, Receiver, Sender};
use tungstenite::{accept, Message as WsMessage};

use bitparty::bridge::{BridgeMessage, ReadyMessage};

/// Inbound command from a bridge client.
pub struct InboundMessage {
    pub client_id: usize,
    pub message: BridgeMessage,
}

/// Per-client write channel.
struct ClientWriter {
    id: usize,
    tx: Sender<String>,
}

/// A WebSocket bridge server for TypeScript <-> Rust communication.
pub struct BridgeServer {
    /// Receive commands from clients (polled by main thread each frame).
    pub command_rx: Receiver<InboundMessage>,
    /// Connected client writers for broadcasting events.
    clients: Arc<Mutex<Vec<ClientWriter>>>,
}

impl BridgeServer {
    /// Start the bridge server on the given port.
    /// Returns immediately; the server runs on background threads.
    pub fn start(port: u16) -> Self {
        let (command_tx, command_rx) = channel::unbounded::<InboundMessage>();
        let clients: Arc<Mutex<Vec<ClientWriter>>> = Arc::new(Mutex::new(Vec::new()));
        let clients_for_listener = clients.clone();

        thread::Builder::new()
            .name("bridge-listener".into())
            .spawn(move || {
                let listener = match TcpListener::bind(("127.0.0.1", port)) {
                    Ok(l) => l,
                    Err(e) => {
                        log::error!("Bridge server failed to bind port {}: {}", port, e);
                        return;
                    }
                };

                log::info!("Bridge WebSocket server listening on 127.0.0.1:{}", port);

                let mut next_id = 0usize;

                for stream in listener.incoming() {
                    match stream {
                        Ok(stream) => {
                            let client_id = next_id;
                            next_id += 1;

                            let cmd_tx = command_tx.clone();
                            let clients = clients_for_listener.clone();

                            thread::Builder::new()
                                .name(format!("bridge-client-{}", client_id))
                                .spawn(move || {
                                    Self::handle_client(client_id, stream, cmd_tx, clients);
                                })
                                .ok();
                        }
                        Err(e) => {
                            log::warn!("Bridge: failed to accept connection: {}", e);
                        }
                    }
                }
            })
            .expect("spawn bridge listener thread");

        BridgeServer {
            command_rx,
            clients,
        }
    }

    /// Broadcast an event to all connected clients.
    pub fn broadcast(&self, msg: &BridgeMessage) {
        if let Ok(json) = msg.to_ndjson() {
            let clients = self.clients.lock().unwrap();
            for client in clients.iter() {
                let _ = client.tx.send(json.clone());
            }
        }
    }

    /// Send a message to a specific client.
    pub fn send_to(&self, client_id: usize, msg: &BridgeMessage) {
        if let Ok(json) = msg.to_ndjson() {
            let clients = self.clients.lock().unwrap();
            if let Some(client) = clients.iter().find(|c| c.id == client_id) {
                let _ = client.tx.send(json);
            }
        }
    }

    /// Number of connected clients.
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }

    fn handle_client(
        client_id: usize,
        stream: TcpStream,
        command_tx: Sender<InboundMessage>,
        clients: Arc<Mutex<Vec<ClientWriter>>>,
    ) {
        // Set read timeout so the loop can interleave reading and writing.
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .ok();

        let mut ws = match accept(stream) {
            Ok(ws) => ws,
            Err(e) => {
                log::warn!("Bridge: WebSocket handshake failed: {}", e);
                return;
            }
        };

        log::info!("Bridge: client {} connected", client_id);

        // Create per-client write channel.
        let (write_tx, write_rx) = channel::unbounded::<String>();

        // Register the client writer.
        {
            let mut client_list = clients.lock().unwrap();
            client_list.push(ClientWriter {
                id: client_id,
                tx: write_tx,
            });
        }

        // Send ready message to the new client.
        let ready = ReadyMessage::default();
        if let Ok(json) = serde_json::to_string(&ready) {
            let _ = ws.send(WsMessage::Text(json));
        }

        // Main loop: alternate between reading and writing.
        loop {
            // Try to read a message (times out after 50ms if nothing available).
            match ws.read() {
                Ok(WsMessage::Text(text)) => {
                    match BridgeMessage::from_ndjson(&text) {
                        Ok(bridge_msg) => {
                            let _ = command_tx.send(InboundMessage {
                                client_id,
                                message: bridge_msg,
                            });
                        }
                        Err(e) => {
                            log::warn!("Bridge: invalid message from client {}: {}", client_id, e);
                        }
                    }
                }
                Ok(WsMessage::Close(_)) => {
                    log::info!("Bridge: client {} disconnected (close frame)", client_id);
                    break;
                }
                Ok(WsMessage::Ping(data)) => {
                    let _ = ws.send(WsMessage::Pong(data));
                }
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::TimedOut =>
                {
                    // Expected timeout, proceed to write phase.
                }
                Err(e) => {
                    log::info!("Bridge: client {} disconnected: {}", client_id, e);
                    break;
                }
                _ => {} // Binary, Pong — ignore.
            }

            // Drain outbound messages.
            while let Ok(msg) = write_rx.try_recv() {
                if ws.send(WsMessage::Text(msg)).is_err() {
                    log::info!(
                        "Bridge: client {} write failed, disconnecting",
                        client_id
                    );
                    break;
                }
            }
        }

        // Cleanup: remove from client list.
        {
            let mut client_list = clients.lock().unwrap();
            client_list.retain(|c| c.id != client_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tungstenite::connect;

    fn find_open_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    }

    /// Connect a test WebSocket client with a read timeout.
    fn test_connect(port: u16) -> tungstenite::WebSocket<TcpStream> {
        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(2000)))
            .ok();
        let (ws, _) = tungstenite::client::client(
            format!("ws://127.0.0.1:{}", port),
            stream,
        )
        .unwrap();
        ws
    }

    #[test]
    fn server_starts_and_accepts_connection() {
        let port = find_open_port();
        let server = BridgeServer::start(port);
        thread::sleep(Duration::from_millis(100));

        let mut ws = test_connect(port);

        // Should receive the ready message.
        let msg = ws.read().expect("read ready message");
        if let WsMessage::Text(text) = msg {
            let ready: ReadyMessage = serde_json::from_str(&text).expect("parse ready");
            assert_eq!(ready.msg_type, "ready");
            assert_eq!(ready.version, "1.0");
        } else {
            panic!("Expected text message, got {:?}", msg);
        }

        thread::sleep(Duration::from_millis(50));
        assert_eq!(server.client_count(), 1);

        ws.close(None).ok();
    }

    #[test]
    fn send_and_receive_bridge_message() {
        let port = find_open_port();
        let server = BridgeServer::start(port);
        thread::sleep(Duration::from_millis(100));

        let mut ws = test_connect(port);

        // Read and discard ready message.
        let _ = ws.read().unwrap();

        // Send a spawn-object command.
        let msg = BridgeMessage::management(
            "spawn-object",
            serde_json::json!({
                "position": [1.0, 2.0, 3.0],
                "block_type": "stone"
            }),
        );
        let ndjson = msg.to_ndjson().unwrap();
        ws.send(WsMessage::Text(ndjson)).unwrap();

        // The server should forward it to the command channel.
        thread::sleep(Duration::from_millis(200));
        let received = server.command_rx.try_recv();
        assert!(received.is_ok());
        let inbound = received.unwrap();
        assert_eq!(inbound.message.msg_type, "spawn-object");

        ws.close(None).ok();
    }

    #[test]
    fn broadcast_event_to_client() {
        let port = find_open_port();
        let server = BridgeServer::start(port);
        thread::sleep(Duration::from_millis(100));

        let mut ws = test_connect(port);

        // Read and discard ready message.
        let _ = ws.read().unwrap();

        // Wait for client registration.
        thread::sleep(Duration::from_millis(100));

        // Broadcast an event from the main thread.
        let event = BridgeMessage::management(
            "object-spawned",
            serde_json::json!({"id": "test-id"}),
        );
        server.broadcast(&event);

        // Client should receive it (read timeout set in test_connect).
        let received = ws.read();
        assert!(received.is_ok());
        if let Ok(WsMessage::Text(text)) = received {
            let parsed = BridgeMessage::from_ndjson(&text).unwrap();
            assert_eq!(parsed.msg_type, "object-spawned");
        }

        ws.close(None).ok();
    }
}
