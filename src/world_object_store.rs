//! Persistent storage for world objects.
//!
//! Stores world objects in a SQLite table alongside the chunk data.
//! Each object is serialized as JSON for flexibility.

use std::path::Path;

use bitparty::bridge::WorldObjectId;
use rusqlite::{Connection, params};

use crate::world_objects::{WorldObject, WorldObjectType};

/// Persistent storage for world objects in the same database as chunks.
pub struct WorldObjectStore {
    conn: Connection,
}

impl WorldObjectStore {
    /// Open or create the world object store in the given database file.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS world_objects (
                id TEXT PRIMARY KEY,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                position_z REAL NOT NULL,
                rotation_x REAL NOT NULL DEFAULT 0,
                rotation_y REAL NOT NULL DEFAULT 0,
                rotation_z REAL NOT NULL DEFAULT 0,
                rotation_w REAL NOT NULL DEFAULT 1,
                scale_x REAL NOT NULL DEFAULT 1,
                scale_y REAL NOT NULL DEFAULT 1,
                scale_z REAL NOT NULL DEFAULT 1,
                object_type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                label TEXT NOT NULL DEFAULT 'object',
                mesh_path TEXT
            );

            CREATE TABLE IF NOT EXISTS world_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;

        conn.pragma_update(None, "journal_mode", "WAL")?;

        Ok(Self { conn })
    }

    /// Open an in-memory store (for testing).
    pub fn open_in_memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS world_objects (
                id TEXT PRIMARY KEY,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                position_z REAL NOT NULL,
                rotation_x REAL NOT NULL DEFAULT 0,
                rotation_y REAL NOT NULL DEFAULT 0,
                rotation_z REAL NOT NULL DEFAULT 0,
                rotation_w REAL NOT NULL DEFAULT 1,
                scale_x REAL NOT NULL DEFAULT 1,
                scale_y REAL NOT NULL DEFAULT 1,
                scale_z REAL NOT NULL DEFAULT 1,
                object_type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                label TEXT NOT NULL DEFAULT 'object',
                mesh_path TEXT
            );

            CREATE TABLE IF NOT EXISTS world_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;

        Ok(Self { conn })
    }

    /// Save a world object to the database.
    pub fn save_object(&self, obj: &WorldObject, label: &str, mesh_path: Option<&str>) -> rusqlite::Result<()> {
        let type_str = match obj.object_type {
            WorldObjectType::Cube => "cube",
            WorldObjectType::Mesh => "mesh",
            WorldObjectType::Interactive => "interactive",
        };

        let props = serde_json::to_string(&obj.properties).unwrap_or_else(|_| "{}".to_string());

        self.conn.execute(
            "INSERT OR REPLACE INTO world_objects
             (id, position_x, position_y, position_z,
              rotation_x, rotation_y, rotation_z, rotation_w,
              scale_x, scale_y, scale_z,
              object_type, properties, label, mesh_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                obj.id.to_string(),
                obj.position[0] as f64,
                obj.position[1] as f64,
                obj.position[2] as f64,
                obj.rotation[0] as f64,
                obj.rotation[1] as f64,
                obj.rotation[2] as f64,
                obj.rotation[3] as f64,
                obj.scale[0] as f64,
                obj.scale[1] as f64,
                obj.scale[2] as f64,
                type_str,
                props,
                label,
                mesh_path,
            ],
        )?;
        Ok(())
    }

    /// Load all world objects from the database.
    pub fn load_all(&self) -> rusqlite::Result<Vec<StoredWorldObject>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, position_x, position_y, position_z,
                    rotation_x, rotation_y, rotation_z, rotation_w,
                    scale_x, scale_y, scale_z,
                    object_type, properties, label, mesh_path
             FROM world_objects",
        )?;

        let objects = stmt.query_map([], |row| {
            let id_str: String = row.get(0)?;
            let uuid = uuid::Uuid::parse_str(&id_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })?;

            let type_str: String = row.get(11)?;
            let object_type = match type_str.as_str() {
                "cube" => WorldObjectType::Cube,
                "mesh" => WorldObjectType::Mesh,
                "interactive" => WorldObjectType::Interactive,
                _ => WorldObjectType::Cube,
            };

            let props_str: String = row.get(12)?;
            let properties = serde_json::from_str(&props_str).unwrap_or(serde_json::json!({}));

            Ok(StoredWorldObject {
                id: WorldObjectId::from_uuid(uuid),
                position: [row.get::<_, f64>(1)? as f32, row.get::<_, f64>(2)? as f32, row.get::<_, f64>(3)? as f32],
                rotation: [
                    row.get::<_, f64>(4)? as f32,
                    row.get::<_, f64>(5)? as f32,
                    row.get::<_, f64>(6)? as f32,
                    row.get::<_, f64>(7)? as f32,
                ],
                scale: [row.get::<_, f64>(8)? as f32, row.get::<_, f64>(9)? as f32, row.get::<_, f64>(10)? as f32],
                object_type,
                properties,
                label: row.get(13)?,
                mesh_path: row.get(14)?,
            })
        })?;

        objects.collect()
    }

    /// Remove a world object from the database.
    pub fn remove_object(&self, id: &WorldObjectId) -> rusqlite::Result<()> {
        self.conn.execute(
            "DELETE FROM world_objects WHERE id = ?1",
            params![id.to_string()],
        )?;
        Ok(())
    }

    /// Save a metadata key-value pair.
    pub fn set_meta(&self, key: &str, value: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO world_meta (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Load a metadata value.
    pub fn get_meta(&self, key: &str) -> rusqlite::Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT value FROM world_meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        );

        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Count stored objects.
    pub fn object_count(&self) -> rusqlite::Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM world_objects",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// A world object as loaded from the database.
pub struct StoredWorldObject {
    pub id: WorldObjectId,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub object_type: WorldObjectType,
    pub properties: serde_json::Value,
    pub label: String,
    pub mesh_path: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_object(pos: [f32; 3]) -> WorldObject {
        WorldObject {
            id: WorldObjectId::new(),
            position: pos,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            object_type: WorldObjectType::Cube,
            properties: serde_json::json!({"color": "red"}),
        }
    }

    #[test]
    fn save_and_load_object() {
        let store = WorldObjectStore::open_in_memory().unwrap();
        let obj = make_test_object([5.0, 10.0, -3.0]);
        let id = obj.id;

        store.save_object(&obj, "test cube", None).unwrap();
        assert_eq!(store.object_count().unwrap(), 1);

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, id);
        assert_eq!(loaded[0].position, [5.0, 10.0, -3.0]);
        assert_eq!(loaded[0].label, "test cube");
        assert_eq!(loaded[0].object_type, WorldObjectType::Cube);
    }

    #[test]
    fn remove_object() {
        let store = WorldObjectStore::open_in_memory().unwrap();
        let obj = make_test_object([0.0, 0.0, 0.0]);
        let id = obj.id;

        store.save_object(&obj, "temp", None).unwrap();
        assert_eq!(store.object_count().unwrap(), 1);

        store.remove_object(&id).unwrap();
        assert_eq!(store.object_count().unwrap(), 0);
    }

    #[test]
    fn metadata() {
        let store = WorldObjectStore::open_in_memory().unwrap();

        assert!(store.get_meta("seed").unwrap().is_none());

        store.set_meta("seed", "42").unwrap();
        assert_eq!(store.get_meta("seed").unwrap(), Some("42".to_string()));

        store.set_meta("seed", "99").unwrap();
        assert_eq!(store.get_meta("seed").unwrap(), Some("99".to_string()));
    }

    #[test]
    fn save_mesh_object_with_path() {
        let store = WorldObjectStore::open_in_memory().unwrap();
        let mut obj = make_test_object([0.0, 0.0, 0.0]);
        obj.object_type = WorldObjectType::Mesh;

        store
            .save_object(&obj, "desk", Some("assets/desk.glb"))
            .unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].mesh_path, Some("assets/desk.glb".to_string()));
        assert_eq!(loaded[0].object_type, WorldObjectType::Mesh);
    }

    #[test]
    fn multiple_objects() {
        let store = WorldObjectStore::open_in_memory().unwrap();

        for i in 0..5 {
            let obj = make_test_object([i as f32, 0.0, 0.0]);
            store.save_object(&obj, &format!("obj_{}", i), None).unwrap();
        }

        assert_eq!(store.object_count().unwrap(), 5);
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 5);
    }

    #[test]
    fn overwrite_object() {
        let store = WorldObjectStore::open_in_memory().unwrap();
        let mut obj = make_test_object([0.0, 0.0, 0.0]);
        let id = obj.id;

        store.save_object(&obj, "v1", None).unwrap();

        obj.position = [10.0, 20.0, 30.0];
        store.save_object(&obj, "v2", None).unwrap();

        assert_eq!(store.object_count().unwrap(), 1);
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].id, id);
        assert_eq!(loaded[0].position, [10.0, 20.0, 30.0]);
        assert_eq!(loaded[0].label, "v2");
    }
}
