-- D1 Database schema for Lamination DeviceLink

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  google_id TEXT UNIQUE,
  created_at TEXT DEFAULT (datetime('now')),
  last_login TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  adapters TEXT, -- JSON of fitted adapter weights
  metadata TEXT, -- JSON: stats, params
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS training_data (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  substrate TEXT,
  anchor_count INTEGER,
  total_patches INTEGER,
  median_de REAL,
  p95_de REAL,
  max_de REAL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_models_user ON models(user_id);
CREATE INDEX IF NOT EXISTS idx_training_data_created ON training_data(created_at);
