use std::path::Path;

use rusqlite::{Connection, params};

use super::chunk::{Block, BlockType, Chunk, ChunkCoord, CHUNK_SIZE, CHUNK_VOLUME};

/// Persistent chunk storage backed by SQLite.
pub struct ChunkStore {
    conn: Connection,
}

impl ChunkStore {
    /// Open or create a chunk database at the given path.
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                z INTEGER NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (x, y, z)
            );",
        )?;

        // WAL mode for better concurrent read performance
        conn.pragma_update(None, "journal_mode", "WAL")?;

        Ok(Self { conn })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                z INTEGER NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (x, y, z)
            );",
        )?;

        Ok(Self { conn })
    }

    /// Save a chunk to the database.
    pub fn save_chunk(&self, chunk: &Chunk) -> rusqlite::Result<()> {
        let data = serialize_chunk(chunk);
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks (x, y, z, data) VALUES (?1, ?2, ?3, ?4)",
            params![chunk.coord.x, chunk.coord.y, chunk.coord.z, data],
        )?;
        Ok(())
    }

    /// Load a chunk from the database. Returns None if not stored.
    pub fn load_chunk(&self, coord: ChunkCoord) -> rusqlite::Result<Option<Chunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT data FROM chunks WHERE x = ?1 AND y = ?2 AND z = ?3",
        )?;

        let result = stmt.query_row(params![coord.x, coord.y, coord.z], |row| {
            let data: Vec<u8> = row.get(0)?;
            Ok(data)
        });

        match result {
            Ok(data) => Ok(Some(deserialize_chunk(coord, &data))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Check if a chunk exists in the database.
    pub fn has_chunk(&self, coord: ChunkCoord) -> rusqlite::Result<bool> {
        let mut stmt = self.conn.prepare(
            "SELECT 1 FROM chunks WHERE x = ?1 AND y = ?2 AND z = ?3",
        )?;

        let exists = stmt.exists(params![coord.x, coord.y, coord.z])?;
        Ok(exists)
    }

    /// Delete a chunk from the database.
    pub fn delete_chunk(&self, coord: ChunkCoord) -> rusqlite::Result<()> {
        self.conn.execute(
            "DELETE FROM chunks WHERE x = ?1 AND y = ?2 AND z = ?3",
            params![coord.x, coord.y, coord.z],
        )?;
        Ok(())
    }

    /// Count stored chunks.
    pub fn chunk_count(&self) -> rusqlite::Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// Serialize a chunk's block data to bytes.
/// Each block is stored as a single u8 (BlockType discriminant).
fn serialize_chunk(chunk: &Chunk) -> Vec<u8> {
    let mut data = Vec::with_capacity(CHUNK_VOLUME);
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                data.push(chunk.get(x, y, z).block_type as u8);
            }
        }
    }
    data
}

/// Deserialize bytes back into a chunk.
fn deserialize_chunk(coord: ChunkCoord, data: &[u8]) -> Chunk {
    let mut chunk = Chunk::new(coord);

    if data.len() < CHUNK_VOLUME {
        return chunk;
    }

    let mut i = 0;
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let bt = block_type_from_u8(data[i]);
                chunk.set(x, y, z, Block::new(bt));
                i += 1;
            }
        }
    }
    chunk.mark_clean();
    chunk
}

fn block_type_from_u8(v: u8) -> BlockType {
    match v {
        0 => BlockType::Air,
        1 => BlockType::Stone,
        2 => BlockType::Dirt,
        3 => BlockType::Grass,
        4 => BlockType::Sand,
        5 => BlockType::Wood,
        6 => BlockType::Brick,
        7 => BlockType::Glass,
        8 => BlockType::Water,
        9 => BlockType::Leaf,
        _ => BlockType::Air,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_load_empty_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(0, 0, 0);
        let chunk = Chunk::new(coord);

        store.save_chunk(&chunk).unwrap();
        let loaded = store.load_chunk(coord).unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn store_and_load_modified_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(1, -2, 3);
        let mut chunk = Chunk::new(coord);
        chunk.set(5, 10, 3, Block::new(BlockType::Stone));
        chunk.set(0, 0, 0, Block::new(BlockType::Brick));

        store.save_chunk(&chunk).unwrap();
        let loaded = store.load_chunk(coord).unwrap().unwrap();

        assert_eq!(loaded.get(5, 10, 3).block_type, BlockType::Stone);
        assert_eq!(loaded.get(0, 0, 0).block_type, BlockType::Brick);
        assert_eq!(loaded.get(1, 1, 1).block_type, BlockType::Air);
    }

    #[test]
    fn load_nonexistent_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let result = store.load_chunk(ChunkCoord::new(99, 99, 99)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn has_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(0, 0, 0);

        assert!(!store.has_chunk(coord).unwrap());
        store.save_chunk(&Chunk::new(coord)).unwrap();
        assert!(store.has_chunk(coord).unwrap());
    }

    #[test]
    fn delete_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(0, 0, 0);

        store.save_chunk(&Chunk::new(coord)).unwrap();
        assert!(store.has_chunk(coord).unwrap());

        store.delete_chunk(coord).unwrap();
        assert!(!store.has_chunk(coord).unwrap());
    }

    #[test]
    fn chunk_count() {
        let store = ChunkStore::open_in_memory().unwrap();
        assert_eq!(store.chunk_count().unwrap(), 0);

        store.save_chunk(&Chunk::new(ChunkCoord::new(0, 0, 0))).unwrap();
        store.save_chunk(&Chunk::new(ChunkCoord::new(1, 0, 0))).unwrap();
        assert_eq!(store.chunk_count().unwrap(), 2);
    }

    #[test]
    fn overwrite_chunk() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(0, 0, 0);

        let chunk1 = Chunk::new(coord);
        store.save_chunk(&chunk1).unwrap();

        let mut chunk2 = Chunk::new(coord);
        chunk2.fill(BlockType::Stone);
        store.save_chunk(&chunk2).unwrap();

        let loaded = store.load_chunk(coord).unwrap().unwrap();
        assert_eq!(loaded.get(0, 0, 0).block_type, BlockType::Stone);
        assert_eq!(store.chunk_count().unwrap(), 1);
    }

    #[test]
    fn roundtrip_all_block_types() {
        let store = ChunkStore::open_in_memory().unwrap();
        let coord = ChunkCoord::new(0, 0, 0);
        let mut chunk = Chunk::new(coord);

        let types = [
            BlockType::Air, BlockType::Stone, BlockType::Dirt,
            BlockType::Grass, BlockType::Sand, BlockType::Wood,
            BlockType::Brick, BlockType::Glass, BlockType::Water,
            BlockType::Leaf,
        ];

        for (i, bt) in types.iter().enumerate() {
            chunk.set(i, 0, 0, Block::new(*bt));
        }

        store.save_chunk(&chunk).unwrap();
        let loaded = store.load_chunk(coord).unwrap().unwrap();

        for (i, bt) in types.iter().enumerate() {
            assert_eq!(loaded.get(i, 0, 0).block_type, *bt);
        }
    }
}
