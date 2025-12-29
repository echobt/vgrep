//! PostgreSQL Storage for Challenge Server Mode
//!
//! Provides persistent storage for challenge server running in subnet owner mode.
//! Uses the same PostgreSQL instance as platform-server but with a separate database.

use anyhow::Result;
use deadpool_postgres::{Config, Pool, Runtime};
use serde::{Deserialize, Serialize};
use tokio_postgres::NoTls;
use tracing::{debug, info};

const SCHEMA: &str = r#"
-- Evaluation results from this challenge
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    agent_hash TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    score REAL NOT NULL,
    tasks_passed INTEGER NOT NULL,
    tasks_total INTEGER NOT NULL,
    tasks_failed INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    execution_time_ms BIGINT,
    task_results JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_agent ON evaluations(agent_hash);
CREATE INDEX IF NOT EXISTS idx_evaluations_submission ON evaluations(submission_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created ON evaluations(created_at DESC);

-- Leaderboard for this challenge
CREATE TABLE IF NOT EXISTS leaderboard (
    agent_hash TEXT PRIMARY KEY,
    miner_hotkey TEXT NOT NULL,
    name TEXT,
    best_score REAL NOT NULL,
    avg_score REAL NOT NULL,
    evaluation_count INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    rank INTEGER,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_rank ON leaderboard(rank);
CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON leaderboard(best_score DESC);

-- Pending evaluations (queued for processing)
CREATE TABLE IF NOT EXISTS pending_evaluations (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    agent_hash TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    source_code TEXT NOT NULL,
    api_key TEXT,
    api_provider TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_evaluations(status);

-- Config cache
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub id: String,
    pub submission_id: String,
    pub agent_hash: String,
    pub miner_hotkey: String,
    pub score: f64,
    pub tasks_passed: i32,
    pub tasks_total: i32,
    pub tasks_failed: i32,
    pub total_cost_usd: f64,
    pub execution_time_ms: Option<i64>,
    pub task_results: Option<serde_json::Value>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub agent_hash: String,
    pub miner_hotkey: String,
    pub name: Option<String>,
    pub best_score: f64,
    pub avg_score: f64,
    pub evaluation_count: i32,
    pub total_cost_usd: f64,
    pub rank: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingEvaluation {
    pub id: String,
    pub submission_id: String,
    pub agent_hash: String,
    pub miner_hotkey: String,
    pub source_code: String,
    pub api_key: Option<String>,
    pub api_provider: Option<String>,
    pub status: String,
}

pub struct PgStorage {
    pool: Pool,
}

impl PgStorage {
    /// Create storage from DATABASE_URL
    pub async fn new(database_url: &str) -> Result<Self> {
        let mut config = Config::new();
        config.url = Some(database_url.to_string());
        let pool = config.create_pool(Some(Runtime::Tokio1), NoTls)?;

        // Test connection
        let client = pool.get().await?;
        info!("Connected to PostgreSQL database");

        // Run migrations
        client.batch_execute(SCHEMA).await?;
        info!("Database schema initialized");

        Ok(Self { pool })
    }

    /// Create storage from DATABASE_URL environment variable
    pub async fn from_env() -> Result<Self> {
        let url =
            std::env::var("DATABASE_URL").map_err(|_| anyhow::anyhow!("DATABASE_URL not set"))?;
        Self::new(&url).await
    }

    // ========================================================================
    // EVALUATIONS
    // ========================================================================

    /// Store an evaluation result
    pub async fn store_evaluation(&self, eval: &EvaluationRecord) -> Result<()> {
        let client = self.pool.get().await?;
        client.execute(
            "INSERT INTO evaluations (id, submission_id, agent_hash, miner_hotkey, score, tasks_passed, tasks_total, tasks_failed, total_cost_usd, execution_time_ms, task_results)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
             ON CONFLICT(id) DO UPDATE SET
                score = EXCLUDED.score,
                tasks_passed = EXCLUDED.tasks_passed,
                tasks_total = EXCLUDED.tasks_total,
                tasks_failed = EXCLUDED.tasks_failed,
                total_cost_usd = EXCLUDED.total_cost_usd,
                execution_time_ms = EXCLUDED.execution_time_ms,
                task_results = EXCLUDED.task_results",
            &[
                &eval.id, &eval.submission_id, &eval.agent_hash, &eval.miner_hotkey,
                &eval.score, &eval.tasks_passed, &eval.tasks_total, &eval.tasks_failed,
                &eval.total_cost_usd, &eval.execution_time_ms, &eval.task_results,
            ],
        ).await?;

        // Update leaderboard
        self.update_leaderboard(
            &eval.agent_hash,
            &eval.miner_hotkey,
            eval.score,
            eval.total_cost_usd,
        )
        .await?;

        debug!(
            "Stored evaluation {} for agent {}",
            eval.id, eval.agent_hash
        );
        Ok(())
    }

    /// Get evaluations for an agent
    pub async fn get_evaluations(&self, agent_hash: &str) -> Result<Vec<EvaluationRecord>> {
        let client = self.pool.get().await?;
        let rows = client.query(
            "SELECT id, submission_id, agent_hash, miner_hotkey, score, tasks_passed, tasks_total, tasks_failed, total_cost_usd, execution_time_ms, task_results, EXTRACT(EPOCH FROM created_at)::BIGINT
             FROM evaluations WHERE agent_hash = $1 ORDER BY created_at DESC",
            &[&agent_hash],
        ).await?;

        Ok(rows
            .iter()
            .map(|r| EvaluationRecord {
                id: r.get(0),
                submission_id: r.get(1),
                agent_hash: r.get(2),
                miner_hotkey: r.get(3),
                score: r.get(4),
                tasks_passed: r.get(5),
                tasks_total: r.get(6),
                tasks_failed: r.get(7),
                total_cost_usd: r.get(8),
                execution_time_ms: r.get(9),
                task_results: r.get(10),
                created_at: r.get(11),
            })
            .collect())
    }

    // ========================================================================
    // LEADERBOARD
    // ========================================================================

    /// Update leaderboard entry
    async fn update_leaderboard(
        &self,
        agent_hash: &str,
        miner_hotkey: &str,
        score: f64,
        cost: f64,
    ) -> Result<()> {
        let client = self.pool.get().await?;

        // Upsert leaderboard entry
        client.execute(
            "INSERT INTO leaderboard (agent_hash, miner_hotkey, best_score, avg_score, evaluation_count, total_cost_usd)
             VALUES ($1, $2, $3, $3, 1, $4)
             ON CONFLICT(agent_hash) DO UPDATE SET
                best_score = GREATEST(leaderboard.best_score, EXCLUDED.best_score),
                avg_score = (leaderboard.avg_score * leaderboard.evaluation_count + EXCLUDED.avg_score) / (leaderboard.evaluation_count + 1),
                evaluation_count = leaderboard.evaluation_count + 1,
                total_cost_usd = leaderboard.total_cost_usd + EXCLUDED.total_cost_usd,
                last_updated = NOW()",
            &[&agent_hash, &miner_hotkey, &score, &cost],
        ).await?;

        // Update ranks
        client.execute(
            "UPDATE leaderboard SET rank = subq.new_rank
             FROM (SELECT agent_hash, ROW_NUMBER() OVER (ORDER BY best_score DESC) as new_rank FROM leaderboard) subq
             WHERE leaderboard.agent_hash = subq.agent_hash",
            &[],
        ).await?;

        Ok(())
    }

    /// Get leaderboard
    pub async fn get_leaderboard(&self, limit: i64) -> Result<Vec<LeaderboardEntry>> {
        let client = self.pool.get().await?;
        let rows = client.query(
            "SELECT agent_hash, miner_hotkey, name, best_score, avg_score, evaluation_count, total_cost_usd, rank
             FROM leaderboard ORDER BY rank ASC NULLS LAST LIMIT $1",
            &[&limit],
        ).await?;

        Ok(rows
            .iter()
            .map(|r| LeaderboardEntry {
                agent_hash: r.get(0),
                miner_hotkey: r.get(1),
                name: r.get(2),
                best_score: r.get(3),
                avg_score: r.get(4),
                evaluation_count: r.get(5),
                total_cost_usd: r.get(6),
                rank: r.get(7),
            })
            .collect())
    }

    /// Get leaderboard entry for an agent
    pub async fn get_leaderboard_entry(
        &self,
        agent_hash: &str,
    ) -> Result<Option<LeaderboardEntry>> {
        let client = self.pool.get().await?;
        let row = client.query_opt(
            "SELECT agent_hash, miner_hotkey, name, best_score, avg_score, evaluation_count, total_cost_usd, rank
             FROM leaderboard WHERE agent_hash = $1",
            &[&agent_hash],
        ).await?;

        Ok(row.map(|r| LeaderboardEntry {
            agent_hash: r.get(0),
            miner_hotkey: r.get(1),
            name: r.get(2),
            best_score: r.get(3),
            avg_score: r.get(4),
            evaluation_count: r.get(5),
            total_cost_usd: r.get(6),
            rank: r.get(7),
        }))
    }

    // ========================================================================
    // PENDING EVALUATIONS
    // ========================================================================

    /// Queue an evaluation
    pub async fn queue_evaluation(&self, eval: &PendingEvaluation) -> Result<()> {
        let client = self.pool.get().await?;
        client.execute(
            "INSERT INTO pending_evaluations (id, submission_id, agent_hash, miner_hotkey, source_code, api_key, api_provider, status)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT(id) DO NOTHING",
            &[
                &eval.id, &eval.submission_id, &eval.agent_hash, &eval.miner_hotkey,
                &eval.source_code, &eval.api_key, &eval.api_provider, &eval.status,
            ],
        ).await?;
        Ok(())
    }

    /// Get pending evaluations
    pub async fn get_pending_evaluations(&self, limit: i64) -> Result<Vec<PendingEvaluation>> {
        let client = self.pool.get().await?;
        let rows = client.query(
            "SELECT id, submission_id, agent_hash, miner_hotkey, source_code, api_key, api_provider, status
             FROM pending_evaluations WHERE status = 'pending' ORDER BY created_at ASC LIMIT $1",
            &[&limit],
        ).await?;

        Ok(rows
            .iter()
            .map(|r| PendingEvaluation {
                id: r.get(0),
                submission_id: r.get(1),
                agent_hash: r.get(2),
                miner_hotkey: r.get(3),
                source_code: r.get(4),
                api_key: r.get(5),
                api_provider: r.get(6),
                status: r.get(7),
            })
            .collect())
    }

    /// Update evaluation status
    pub async fn update_evaluation_status(&self, id: &str, status: &str) -> Result<()> {
        let client = self.pool.get().await?;
        client
            .execute(
                "UPDATE pending_evaluations SET status = $1 WHERE id = $2",
                &[&status, &id],
            )
            .await?;
        Ok(())
    }

    /// Delete completed evaluations
    pub async fn cleanup_completed(&self) -> Result<u64> {
        let client = self.pool.get().await?;
        let count = client
            .execute(
                "DELETE FROM pending_evaluations WHERE status IN ('completed', 'failed')",
                &[],
            )
            .await?;
        Ok(count)
    }

    // ========================================================================
    // CONFIG
    // ========================================================================

    /// Set config value
    pub async fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let client = self.pool.get().await?;
        client
            .execute(
                "INSERT INTO config (key, value, updated_at) VALUES ($1, $2, NOW())
             ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                &[&key, &value],
            )
            .await?;
        Ok(())
    }

    /// Get config value
    pub async fn get_config(&self, key: &str) -> Result<Option<String>> {
        let client = self.pool.get().await?;
        let row = client
            .query_opt("SELECT value FROM config WHERE key = $1", &[&key])
            .await?;
        Ok(row.map(|r| r.get(0)))
    }
}
