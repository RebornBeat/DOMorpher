//! # Chunking Module
//!
//! The chunking module provides functionality for dividing large HTML documents into
//! manageable chunks for processing by LLMs. It implements various chunking strategies
//! including size-based, semantic, hierarchical, and adaptive importance-based chunking.
//!
//! ## Key Components
//!
//! - `ChunkingEngine`: Main processing engine for dividing content
//! - `ChunkingStrategy`: Enum defining available chunking strategies
//! - `Chunk`: Data structure representing a single chunk of content
//! - `ChunkTree`: Hierarchical representation of chunks and their relationships
//! - `SemanticImportanceChunker`: Advanced chunking based on semantic importance

use std::sync::Arc;

// Public re-exports
pub use self::chunk::{Chunk, ChunkMetadata, ChunkRelation, ChunkTree, ChunkType, EnhancedChunk};
pub use self::engine::{ChunkingConfig, ChunkingEngine, ChunkingOptions};
pub use self::semantic_importance::{
    ImportanceConfig, ImportanceMetrics, SemanticImportanceChunker, SemanticSection,
};
pub use self::strategies::{
    ChunkingResult, ChunkingStrategy, HierarchicalChunker, SemanticChunker, SizeBasedChunker,
};

// Module declarations
pub mod chunk;
pub mod engine;
pub mod semantic_importance;
pub mod strategies;

// Internal types not exported directly
mod utils;
