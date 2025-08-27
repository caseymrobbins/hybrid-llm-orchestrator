# src/database.py
import sqlite3
from datetime import datetime

class UserDatabase:
    """Manages the SQLite database for user profiles and interactions."""
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Creates the database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    preferences_json TEXT DEFAULT '{}',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query_hash TEXT NOT NULL,
                    modules_used TEXT NOT NULL,
                    total_cost_cents INTEGER DEFAULT 0,
                    total_latency_ms INTEGER,
                    cache_hit_rate REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                
                CREATE TABLE IF NOT EXISTS module_outputs (
                    output_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    module_name TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    latency_ms INTEGER,
                    model_used TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_interactions_user 
                ON interactions(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_interactions_created 
                ON interactions(created_at);
            """)

    def log_interaction(self, interaction_data: dict):
        """Logs a full interaction and its module outputs to the database."""
        # This method would contain the logic to insert records into the
        # interactions and module_outputs tables.
        pass

    def update_user_profile(self, user_id: int, profile_data: dict):
        """Updates a user's profile JSON."""
        # Logic to update the users table.
        pass