# src/database.py

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class UserDatabase:
    """Manages the SQLite database for user profiles and interactions."""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = Path(db_path)
        self._initialize_schema()
        logger.info(f"Database initialized at: {self.db_path}")
    
    def _initialize_schema(self):
        """Creates the database tables if they don't exist."""
        try:
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
                        query_text TEXT NOT NULL,
                        query_hash TEXT NOT NULL,
                        modules_used TEXT NOT NULL,
                        total_cost_cents INTEGER DEFAULT 0,
                        total_latency_ms INTEGER,
                        cache_hit_rate REAL,
                        final_response TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS module_outputs (
                        output_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id INTEGER NOT NULL,
                        module_name TEXT NOT NULL,
                        module_output TEXT NOT NULL,
                        output_hash TEXT NOT NULL,
                        latency_ms INTEGER,
                        cost_cents INTEGER DEFAULT 0,
                        model_used TEXT,
                        was_cached BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS routing_decisions (
                        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interaction_id INTEGER NOT NULL,
                        module_name TEXT NOT NULL,
                        original_model TEXT NOT NULL,
                        routed_model TEXT NOT NULL,
                        complexity_score REAL,
                        cost_savings_cents INTEGER DEFAULT 0,
                        routing_reason TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
                    );
                    
                    -- Indexes for performance
                    CREATE INDEX IF NOT EXISTS idx_interactions_user 
                    ON interactions(user_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_interactions_created 
                    ON interactions(created_at);
                    
                    CREATE INDEX IF NOT EXISTS idx_interactions_hash 
                    ON interactions(query_hash);
                    
                    CREATE INDEX IF NOT EXISTS idx_module_outputs_interaction 
                    ON module_outputs(interaction_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_module_outputs_hash 
                    ON module_outputs(output_hash);
                    
                    CREATE INDEX IF NOT EXISTS idx_routing_decisions_interaction 
                    ON routing_decisions(interaction_id);
                """)
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise DatabaseError(f"Schema initialization failed: {e}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()

    def _generate_hash(self, text: str) -> str:
        """Generate SHA256 hash of text for deduplication."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def create_user(self, profile_data: Optional[Dict[str, Any]] = None, 
                   preferences: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new user record.
        
        Args:
            profile_data: User profile information
            preferences: User preferences
            
        Returns:
            User ID of the created user
        """
        profile_json = json.dumps(profile_data or {})
        preferences_json = json.dumps(preferences or {})
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO users (profile_json, preferences_json) 
                       VALUES (?, ?)""",
                    (profile_json, preferences_json)
                )
                user_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created new user with ID: {user_id}")
                return user_id
                
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise DatabaseError(f"User creation failed: {e}")

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by ID."""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
                
                if row:
                    return {
                        'user_id': row['user_id'],
                        'profile': json.loads(row['profile_json']),
                        'preferences': json.loads(row['preferences_json']),
                        'created_at': row['created_at'],
                        'last_active': row['last_active']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise DatabaseError(f"User retrieval failed: {e}")

    def update_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """
        Updates a user's profile JSON.
        
        Args:
            user_id: ID of the user to update
            profile_data: New profile data
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            profile_json = json.dumps(profile_data)
            
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """UPDATE users 
                       SET profile_json = ?, last_active = CURRENT_TIMESTAMP
                       WHERE user_id = ?""",
                    (profile_json, user_id)
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False
                
                conn.commit()
                logger.info(f"Updated profile for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {e}")
            raise DatabaseError(f"Profile update failed: {e}")

    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            preferences_json = json.dumps(preferences)
            
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """UPDATE users 
                       SET preferences_json = ?, last_active = CURRENT_TIMESTAMP
                       WHERE user_id = ?""",
                    (preferences_json, user_id)
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False
                
                conn.commit()
                logger.info(f"Updated preferences for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update user preferences {user_id}: {e}")
            raise DatabaseError(f"Preferences update failed: {e}")

    def log_interaction(self, interaction_data: Dict[str, Any]) -> int:
        """
        Logs a full interaction and its module outputs to the database.
        
        Args:
            interaction_data: Dictionary containing:
                - user_id: ID of the user
                - query_text: Original query text
                - modules_used: List of module names used
                - total_cost_cents: Total cost in cents
                - total_latency_ms: Total latency in milliseconds
                - cache_hit_rate: Cache hit rate (0.0 to 1.0)
                - final_response: Final synthesized response
                - module_outputs: List of module output dictionaries
                - routing_decisions: List of routing decision dictionaries
                
        Returns:
            Interaction ID of the logged interaction
        """
        try:
            user_id = interaction_data.get('user_id', 1)
            query_text = interaction_data.get('query_text', '')
            query_hash = self._generate_hash(query_text)
            modules_used = json.dumps(interaction_data.get('modules_used', []))
            total_cost_cents = interaction_data.get('total_cost_cents', 0)
            total_latency_ms = interaction_data.get('total_latency_ms', 0)
            cache_hit_rate = interaction_data.get('cache_hit_rate', 0.0)
            final_response = interaction_data.get('final_response', '')
            
            with self.get_connection() as conn:
                # Insert main interaction record
                cursor = conn.execute(
                    """INSERT INTO interactions 
                       (user_id, query_text, query_hash, modules_used, 
                        total_cost_cents, total_latency_ms, cache_hit_rate, final_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, query_text, query_hash, modules_used,
                     total_cost_cents, total_latency_ms, cache_hit_rate, final_response)
                )
                
                interaction_id = cursor.lastrowid
                
                # Insert module outputs
                module_outputs = interaction_data.get('module_outputs', [])
                for output_data in module_outputs:
                    self._log_module_output(conn, interaction_id, output_data)
                
                # Insert routing decisions
                routing_decisions = interaction_data.get('routing_decisions', [])
                for decision_data in routing_decisions:
                    self._log_routing_decision(conn, interaction_id, decision_data)
                
                # Update user's last_active timestamp
                conn.execute(
                    "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,)
                )
                
                conn.commit()
                
                logger.info(f"Logged interaction {interaction_id} for user {user_id}")
                return interaction_id
                
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            raise DatabaseError(f"Interaction logging failed: {e}")

    def _log_module_output(self, conn: sqlite3.Connection, interaction_id: int, 
                          output_data: Dict[str, Any]) -> None:
        """Log a single module output."""
        module_name = output_data.get('module_name', 'unknown')
        module_output = output_data.get('output', '')
        output_hash = self._generate_hash(module_output)
        latency_ms = output_data.get('latency_ms', 0)
        cost_cents = output_data.get('cost_cents', 0)
        model_used = output_data.get('model_used', 'unknown')
        was_cached = output_data.get('was_cached', False)
        
        conn.execute(
            """INSERT INTO module_outputs 
               (interaction_id, module_name, module_output, output_hash,
                latency_ms, cost_cents, model_used, was_cached)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (interaction_id, module_name, module_output, output_hash,
             latency_ms, cost_cents, model_used, was_cached)
        )

    def _log_routing_decision(self, conn: sqlite3.Connection, interaction_id: int,
                            decision_data: Dict[str, Any]) -> None:
        """Log a single routing decision."""
        module_name = decision_data.get('module_name', 'unknown')
        original_model = decision_data.get('original_model', 'unknown')
        routed_model = decision_data.get('routed_model', 'unknown')
        complexity_score = decision_data.get('complexity_score', 0.0)
        cost_savings_cents = decision_data.get('cost_savings_cents', 0)
        routing_reason = decision_data.get('routing_reason', '')
        
        conn.execute(
            """INSERT INTO routing_decisions 
               (interaction_id, module_name, original_model, routed_model,
                complexity_score, cost_savings_cents, routing_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (interaction_id, module_name, original_model, routed_model,
             complexity_score, cost_savings_cents, routing_reason)
        )

    def get_user_interactions(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent interactions for a user."""
        try:
            with self.get_connection() as conn:
                rows = conn.execute(
                    """SELECT * FROM interactions 
                       WHERE user_id = ? 
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (user_id, limit)
                ).fetchall()
                
                interactions = []
                for row in rows:
                    interactions.append({
                        'interaction_id': row['interaction_id'],
                        'query_text': row['query_text'],
                        'modules_used': json.loads(row['modules_used']),
                        'total_cost_cents': row['total_cost_cents'],
                        'total_latency_ms': row['total_latency_ms'],
                        'cache_hit_rate': row['cache_hit_rate'],
                        'final_response': row['final_response'],
                        'created_at': row['created_at']
                    })
                
                return interactions
                
        except Exception as e:
            logger.error(f"Failed to get user interactions: {e}")
            raise DatabaseError(f"Interaction retrieval failed: {e}")

    def get_similar_queries(self, query_hash: str, user_id: Optional[int] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar queries based on hash (for caching optimization)."""
        try:
            with self.get_connection() as conn:
                if user_id:
                    rows = conn.execute(
                        """SELECT * FROM interactions 
                           WHERE query_hash = ? AND user_id = ?
                           ORDER BY created_at DESC 
                           LIMIT ?""",
                        (query_hash, user_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM interactions 
                           WHERE query_hash = ?
                           ORDER BY created_at DESC 
                           LIMIT ?""",
                        (query_hash, limit)
                    ).fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get similar queries: {e}")
            return []

    def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for the specified number of days."""
        try:
            with self.get_connection() as conn:
                # Basic statistics
                stats = conn.execute(
                    """SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(total_cost_cents) as avg_cost_cents,
                        AVG(total_latency_ms) as avg_latency_ms,
                        AVG(cache_hit_rate) as avg_cache_hit_rate
                       FROM interactions 
                       WHERE created_at >= datetime('now', '-{} days')""".format(days)
                ).fetchone()
                
                # Cost breakdown by module
                module_costs = conn.execute(
                    """SELECT 
                        mo.module_name,
                        COUNT(*) as usage_count,
                        SUM(mo.cost_cents) as total_cost_cents,
                        AVG(mo.latency_ms) as avg_latency_ms
                       FROM module_outputs mo
                       JOIN interactions i ON mo.interaction_id = i.interaction_id
                       WHERE i.created_at >= datetime('now', '-{} days')
                       GROUP BY mo.module_name
                       ORDER BY total_cost_cents DESC""".format(days)
                ).fetchall()
                
                # Routing effectiveness
                routing_stats = conn.execute(
                    """SELECT 
                        COUNT(*) as total_routing_decisions,
                        SUM(cost_savings_cents) as total_savings_cents,
                        AVG(complexity_score) as avg_complexity_score
                       FROM routing_decisions rd
                       JOIN interactions i ON rd.interaction_id = i.interaction_id
                       WHERE i.created_at >= datetime('now', '-{} days')""".format(days)
                ).fetchone()
                
                return {
                    'period_days': days,
                    'overview': dict(stats) if stats else {},
                    'module_performance': [dict(row) for row in module_costs],
                    'routing_effectiveness': dict(routing_stats) if routing_stats else {},
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to generate analytics summary: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old interaction data beyond the retention period."""
        try:
            with self.get_connection() as conn:
                # Delete old interactions (cascades to related tables due to foreign keys)
                cursor = conn.execute(
                    "DELETE FROM interactions WHERE created_at < datetime('now', '-{} days')".format(days_to_keep)
                )
                interactions_deleted = cursor.rowcount
                
                # Clean up orphaned module outputs (just in case)
                cursor = conn.execute(
                    """DELETE FROM module_outputs 
                       WHERE interaction_id NOT IN (SELECT interaction_id FROM interactions)"""
                )
                outputs_deleted = cursor.rowcount
                
                # Clean up orphaned routing decisions
                cursor = conn.execute(
                    """DELETE FROM routing_decisions 
                       WHERE interaction_id NOT IN (SELECT interaction_id FROM interactions)"""
                )
                decisions_deleted = cursor.rowcount
                
                conn.commit()
                
                result = {
                    'interactions_deleted': interactions_deleted,
                    'outputs_deleted': outputs_deleted,
                    'decisions_deleted': decisions_deleted
                }
                
                logger.info(f"Cleaned up old data: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise DatabaseError(f"Data cleanup failed: {e}")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information."""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Table sizes
                for table in ['users', 'interactions', 'module_outputs', 'routing_decisions']:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    stats[f"{table}_count"] = count
                
                # Database file size
                stats['db_file_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Recent activity
                recent_interactions = conn.execute(
                    "SELECT COUNT(*) FROM interactions WHERE created_at >= datetime('now', '-1 day')"
                ).fetchone()[0]
                stats['interactions_last_24h'] = recent_interactions
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_connection() as conn:
                # Test basic connectivity
                conn.execute("SELECT 1").fetchone()
                
                # Check database integrity
                integrity_result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                
                return {
                    'status': 'healthy' if integrity_result == 'ok' else 'degraded',
                    'integrity': integrity_result,
                    'file_exists': self.db_path.exists(),
                    'file_readable': self.db_path.is_file(),
                    'stats': self.get_database_stats()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }