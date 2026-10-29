const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const DB_PATH = path.join(__dirname, 'rag.db');
let db = null;

function getDb() {
  if (!db) {
    db = new DatabaseSync(DB_PATH);
    db.exec('PRAGMA foreign_keys = ON;');
    initSchema();
  }
  return db;
}

function initSchema() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      original_text TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      document_id INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      text TEXT NOT NULL,
      embedding TEXT NOT NULL,
      FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    );
  `);
}

function addDocument(title, originalText) {
  const database = getDb();
  const stmt = database.prepare('INSERT INTO documents (title, original_text) VALUES (?, ?)');
  const res = stmt.run(title, originalText);
  return Number(res.lastInsertRowid);
}

function saveChunks(documentId, chunksWithEmbeddings) {
  const database = getDb();
  database.exec('BEGIN TRANSACTION;');
  try {
    const stmt = database.prepare('INSERT INTO chunks (document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)');
    for (let i = 0; i < chunksWithEmbeddings.length; i++) {
      const c = chunksWithEmbeddings[i];
      stmt.run(documentId, i, c.text, JSON.stringify(c.embedding));
    }
    database.exec('COMMIT;');
  } catch (err) {
    database.exec('ROLLBACK;');
    throw err;
  }
}

function deleteDocument(id) {
  const database = getDb();
  const stmt = database.prepare('DELETE FROM documents WHERE id = ?');
  const res = stmt.run(Number(id));
  return res.changes > 0;
}

function getDocument(id) {
  const database = getDb();
  const stmt = database.prepare('SELECT id, title, original_text, created_at FROM documents WHERE id = ?');
  return stmt.get(Number(id));
}

function listDocuments() {
  const database = getDb();
  const stmt = database.prepare(`
    SELECT d.id, d.title, d.created_at, LENGTH(d.original_text) AS text_length, COUNT(c.id) AS chunk_count
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    GROUP BY d.id
    ORDER BY d.id DESC
  `);
  return stmt.all();
}

function clearChunksForDocument(documentId) {
  const database = getDb();
  const stmt = database.prepare('DELETE FROM chunks WHERE document_id = ?');
  stmt.run(Number(documentId));
}

function getAllChunksWithDocument() {
  const database = getDb();
  const stmt = database.prepare(`
    SELECT c.id, c.document_id, c.chunk_index, c.text, c.embedding, d.title AS document_title
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
  `);
  const rows = stmt.all();
  return rows.map(r => ({
    ...r,
    embedding: JSON.parse(r.embedding)
  }));
}

function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function findTopChunks(queryEmbedding, limit = 4) {
  const allChunks = getAllChunksWithDocument();
  if (allChunks.length === 0) return [];

  const scored = allChunks.map(chunk => ({
    ...chunk,
    similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));

  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, limit);
}

module.exports = {
  getDb,
  addDocument,
  saveChunks,
  deleteDocument,
  getDocument,
  listDocuments,
  clearChunksForDocument,
  getAllChunksWithDocument,
  cosineSimilarity,
  findTopChunks
};

