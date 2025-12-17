//! Core search and indexing functionality.

mod db;
mod embeddings;
mod indexer;
mod search;

pub use db::{ChunkEntry, Database, DatabaseStats, FileEntry, SearchResult as DbSearchResult};
pub use embeddings::EmbeddingEngine;
pub use indexer::{Indexer, ServerIndexer};
pub use search::{SearchEngine, SearchResult};
