//! # Chunking Engine
//!
//! The chunking engine is responsible for dividing HTML content into manageable chunks
//! for processing by LLMs. It implements various chunking strategies and handles the
//! creation of chunk trees with proper relationships.

use std::collections::HashMap;
use std::sync::Arc;

use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dom::analyzer::{DomAnalyzer, DomNode};
use crate::dom::preprocessor::DomPreprocessor;
use crate::error::{DOMorpherError, Result};
use crate::utils::tokenizer::TokenCounter;

use super::chunk::{
    Chunk, ChunkContext, ChunkMetadata, ChunkPosition, ChunkTree, ChunkType, EnhancedChunk,
};
use super::semantic_importance::{ImportanceConfig, SemanticImportanceChunker, SemanticSection};
use super::strategies::{
    ChunkingResult, ChunkingStrategy, HierarchicalChunker, SemanticChunker, SizeBasedChunker,
};

/// Errors specific to the chunking process
#[derive(Debug, Error)]
pub enum ChunkingError {
    /// Error when the HTML content is too large to process
    #[error("Content too large to process: {0} tokens (max: {1})")]
    ContentTooLarge(usize, usize),

    /// Error when chunking strategy is not supported
    #[error("Unsupported chunking strategy: {0}")]
    UnsupportedStrategy(String),

    /// Error when chunk optimization fails
    #[error("Chunk optimization failed: {0}")]
    OptimizationFailed(String),

    /// Error when chunk generation fails
    #[error("Failed to generate chunks: {0}")]
    ChunkGenerationFailed(String),
}

/// Configuration for the chunking engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// The chunking strategy to use
    pub strategy: ChunkingStrategy,

    /// Maximum token size for a single chunk
    pub max_chunk_size: usize,

    /// Minimum token size for a chunk
    pub min_chunk_size: usize,

    /// Overlap between chunks in tokens
    pub overlap: usize,

    /// Maximum number of chunks to generate
    pub max_chunks: Option<usize>,

    /// Whether to include chunk context
    pub include_context: bool,

    /// Whether to optimize chunks for token efficiency
    pub optimize_tokens: bool,

    /// Configuration for semantic importance-based chunking
    pub importance_config: Option<ImportanceConfig>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Semantic,
            max_chunk_size: 8000,
            min_chunk_size: 1000,
            overlap: 200,
            max_chunks: None,
            include_context: true,
            optimize_tokens: true,
            importance_config: None,
        }
    }
}

/// Runtime options for chunking
#[derive(Debug, Clone)]
pub struct ChunkingOptions {
    /// Document title if available
    pub document_title: Option<String>,

    /// Source URL if available
    pub source_url: Option<String>,

    /// Custom token counter implementation
    pub token_counter: Option<Arc<dyn TokenCounter>>,

    /// DOM preprocessor instance
    pub preprocessor: Option<Arc<DomPreprocessor>>,

    /// DOM analyzer instance
    pub analyzer: Option<Arc<DomAnalyzer>>,

    /// Additional metadata to include with chunks
    pub additional_metadata: HashMap<String, String>,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            document_title: None,
            source_url: None,
            token_counter: None,
            preprocessor: None,
            analyzer: None,
            additional_metadata: HashMap::new(),
        }
    }
}

/// Main chunking engine implementation
#[derive(Debug)]
pub struct ChunkingEngine {
    /// Configuration for chunking
    config: ChunkingConfig,

    /// Default token counter
    default_token_counter: Arc<dyn TokenCounter>,

    /// Default DOM preprocessor
    default_preprocessor: Arc<DomPreprocessor>,

    /// Default DOM analyzer
    default_analyzer: Arc<DomAnalyzer>,
}

impl ChunkingEngine {
    /// Create a new chunking engine with the given configuration
    pub fn new(config: ChunkingConfig) -> Self {
        // Create default implementations
        let default_token_counter = Arc::new(crate::utils::tokenizer::DefaultTokenCounter::new());
        let default_preprocessor = Arc::new(DomPreprocessor::new(Default::default()));
        let default_analyzer = Arc::new(DomAnalyzer::new(Default::default()));

        Self {
            config,
            default_token_counter,
            default_preprocessor,
            default_analyzer,
        }
    }

    /// Create a new chunking engine with default configuration
    pub fn default() -> Self {
        Self::new(ChunkingConfig::default())
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &ChunkingConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ChunkingConfig) {
        self.config = config;
    }

    /// Set the chunking strategy
    pub fn set_strategy(&mut self, strategy: ChunkingStrategy) {
        self.config.strategy = strategy;
    }

    /// Process HTML content into chunks
    pub fn process_html(
        &self,
        html: &str,
        options: Option<ChunkingOptions>,
    ) -> Result<ChunkingResult> {
        let options = options.unwrap_or_default();

        // Get token counter
        let token_counter = options
            .token_counter
            .unwrap_or_else(|| self.default_token_counter.clone());

        // Get preprocessor
        let preprocessor = options
            .preprocessor
            .unwrap_or_else(|| self.default_preprocessor.clone());

        // Get analyzer
        let analyzer = options
            .analyzer
            .unwrap_or_else(|| self.default_analyzer.clone());

        // Preprocess HTML if needed
        let processed_html = preprocessor.preprocess(html)?;

        // Estimate token count
        let estimated_tokens = token_counter.count_tokens(&processed_html);
        debug!("Estimated token count: {}", estimated_tokens);

        // Check if content is too large
        const ABSOLUTE_MAX_TOKENS: usize = 100_000;
        if estimated_tokens > ABSOLUTE_MAX_TOKENS {
            return Err(DOMorpherError::Chunking(
                ChunkingError::ContentTooLarge(estimated_tokens, ABSOLUTE_MAX_TOKENS).to_string(),
            ));
        }

        // If content is small enough to process as a single chunk, do that
        if estimated_tokens <= self.config.max_chunk_size {
            info!("Content is small enough to process as a single chunk");
            let chunk = self.create_single_chunk(&processed_html, &options, &token_counter)?;
            return Ok(ChunkingResult::Single(chunk));
        }

        // Otherwise, apply the configured chunking strategy
        match self.config.strategy {
            ChunkingStrategy::Size => {
                debug!("Using size-based chunking strategy");
                let chunker = SizeBasedChunker::new(
                    self.config.max_chunk_size,
                    self.config.min_chunk_size,
                    self.config.overlap,
                );
                chunker.chunk(&processed_html, &options, &token_counter)
            }
            ChunkingStrategy::Semantic => {
                debug!("Using semantic chunking strategy");
                let chunker = SemanticChunker::new(
                    self.config.max_chunk_size,
                    self.config.min_chunk_size,
                    self.config.overlap,
                    analyzer.clone(),
                );
                chunker.chunk(&processed_html, &options, &token_counter)
            }
            ChunkingStrategy::Hierarchical => {
                debug!("Using hierarchical chunking strategy");
                let chunker = HierarchicalChunker::new(
                    self.config.max_chunk_size,
                    self.config.min_chunk_size,
                    analyzer.clone(),
                );
                chunker.chunk(&processed_html, &options, &token_counter)
            }
            ChunkingStrategy::Adaptive => {
                debug!("Using adaptive chunking strategy based on semantic importance");
                let importance_config = self
                    .config
                    .importance_config
                    .clone()
                    .unwrap_or_else(|| ImportanceConfig::default());

                let chunker = SemanticImportanceChunker::new(
                    self.config.max_chunk_size,
                    self.config.min_chunk_size,
                    self.config.overlap,
                    analyzer.clone(),
                    importance_config,
                );

                chunker.chunk(&processed_html, &options, &token_counter)
            }
        }
    }

    /// Create enhanced chunks with context from a chunking result
    pub fn create_enhanced_chunks(&self, result: ChunkingResult) -> Vec<EnhancedChunk> {
        match result {
            ChunkingResult::Single(chunk) => {
                // Single chunk doesn't need much context
                let context = ChunkContext::with_document_info(
                    chunk.metadata.custom_fields.get("document_title").cloned(),
                    chunk.metadata.source.clone(),
                );

                vec![EnhancedChunk::new(chunk, context)]
            }
            ChunkingResult::Multiple(chunks) => {
                // Create enhanced chunks from a list
                self.create_enhanced_chunks_from_list(chunks)
            }
            ChunkingResult::Tree(tree) => {
                // Create enhanced chunks from a tree
                tree.create_enhanced_chunks()
            }
        }
    }

    /// Create enhanced chunks with context from a list of chunks
    fn create_enhanced_chunks_from_list(&self, chunks: Vec<Chunk>) -> Vec<EnhancedChunk> {
        let mut enhanced_chunks = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            let mut context = ChunkContext::new();

            // Set document info
            if let Some(title) = chunk.metadata.custom_fields.get("document_title") {
                context.document_title = Some(title.clone());
            }

            if let Some(source) = &chunk.metadata.source {
                context.source_url = Some(source.clone());
            }

            // Add preceding context if available
            if i > 0 {
                let prev_chunk = &chunks[i - 1];
                let summary = summarize_content(&prev_chunk.content);
                context.set_preceding_context(&summary);
            }

            // Add following context if available
            if i < chunks.len() - 1 {
                let next_chunk = &chunks[i + 1];
                let summary = summarize_content(&next_chunk.content);
                context.set_following_context(&summary);
            }

            // Create enhanced chunk
            enhanced_chunks.push(EnhancedChunk::new(chunk.clone(), context));
        }

        enhanced_chunks
    }

    /// Create a single chunk from the entire content
    fn create_single_chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<Chunk> {
        let token_count = token_counter.count_tokens(html);

        let mut metadata = ChunkMetadata {
            token_count,
            position: ChunkPosition {
                index: 0,
                start: 0,
                end: html.len(),
                depth: 0,
                parent_index: None,
            },
            content_type: ChunkType::Document,
            importance: 1.0, // Single chunk is maximally important
            source: options.source_url.clone(),
            element_path: None,
            created_at: chrono::Utc::now(),
            custom_fields: options.additional_metadata.clone(),
        };

        // Add document title if available
        if let Some(title) = &options.document_title {
            metadata
                .custom_fields
                .insert("document_title".to_string(), title.clone());
        }

        Ok(Chunk::with_metadata(html, metadata))
    }

    /// Optimize chunks to reduce token usage
    pub fn optimize_chunks(&self, chunks: Vec<EnhancedChunk>) -> Result<Vec<EnhancedChunk>> {
        if !self.config.optimize_tokens {
            return Ok(chunks);
        }

        let mut optimized = Vec::with_capacity(chunks.len());

        for mut chunk in chunks {
            // Strip unnecessary whitespace
            chunk.chunk.content = chunk.chunk.content.trim().to_string();

            // Optimize context
            if chunk
                .context
                .preceding_context
                .as_ref()
                .map_or(false, |s| s.is_empty())
            {
                chunk.context.preceding_context = None;
            }

            if chunk
                .context
                .following_context
                .as_ref()
                .map_or(false, |s| s.is_empty())
            {
                chunk.context.following_context = None;
            }

            // Remove empty breadcrumbs
            if chunk.context.breadcrumbs.is_empty() {
                chunk.context.breadcrumbs = Vec::new();
            }

            optimized.push(chunk);
        }

        Ok(optimized)
    }
}

/// Create a brief summary of HTML content
fn summarize_content(html: &str) -> String {
    // Extract visible text
    let text = html_text::from_read(html.as_bytes(), html_text::ParseOptions::default());

    // Truncate to a reasonable length
    let text = text.trim();
    if text.len() > 200 {
        format!("{}...", &text[..200])
    } else {
        text.to_string()
    }
}
