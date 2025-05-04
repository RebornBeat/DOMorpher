//! # Chunking Strategies
//!
//! This module provides different strategies for dividing HTML content into chunks.
//! Each strategy has its own approach to segmentation based on different criteria.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use log::{debug, info, trace, warn};
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use thiserror::Error;

use crate::dom::analyzer::{DomAnalyzer, DomNode};
use crate::error::{DOMorpherError, Result};
use crate::utils::tokenizer::TokenCounter;

use super::chunk::{Chunk, ChunkMetadata, ChunkPosition, ChunkRelation, ChunkTree, ChunkType};
use super::engine::ChunkingOptions;

/// Result of the chunking process
#[derive(Debug)]
pub enum ChunkingResult {
    /// A single chunk for the entire content
    Single(Chunk),

    /// Multiple chunks without hierarchy
    Multiple(Vec<Chunk>),

    /// A tree of hierarchical chunks
    Tree(ChunkTree),
}

impl ChunkingResult {
    /// Get the total number of chunks
    pub fn chunk_count(&self) -> usize {
        match self {
            ChunkingResult::Single(_) => 1,
            ChunkingResult::Multiple(chunks) => chunks.len(),
            ChunkingResult::Tree(tree) => tree.len(),
        }
    }

    /// Get all chunks as a flat list
    pub fn to_chunks(&self) -> Vec<Chunk> {
        match self {
            ChunkingResult::Single(chunk) => vec![chunk.clone()],
            ChunkingResult::Multiple(chunks) => chunks.clone(),
            ChunkingResult::Tree(tree) => tree
                .get_ordered_chunks()
                .into_iter()
                .map(|chunk_arc| (**chunk_arc).clone())
                .collect(),
        }
    }

    /// Convert to a ChunkTree representation
    pub fn to_tree(&self) -> ChunkTree {
        match self {
            ChunkingResult::Tree(tree) => tree.clone(),
            ChunkingResult::Single(chunk) => {
                let mut tree = ChunkTree::new();
                tree.add_chunk(chunk.clone(), None);
                tree
            }
            ChunkingResult::Multiple(chunks) => {
                let mut tree = ChunkTree::new();
                for chunk in chunks {
                    tree.add_chunk(chunk.clone(), None);
                }
                tree
            }
        }
    }
}

/// Available chunking strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkingStrategy {
    /// Simple size-based chunking
    Size,

    /// Semantic-based chunking that respects document structure
    Semantic,

    /// Hierarchical chunking preserving parent-child relationships
    Hierarchical,

    /// Adaptive chunking based on content importance
    Adaptive,
}

/// Trait for chunking strategies
pub trait Chunker {
    /// Chunk HTML content using the strategy
    fn chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult>;
}

/// Size-based chunking implementation
#[derive(Debug)]
pub struct SizeBasedChunker {
    /// Maximum token size for a chunk
    max_chunk_size: usize,

    /// Minimum token size for a chunk
    min_chunk_size: usize,

    /// Overlap between chunks
    overlap: usize,
}

impl SizeBasedChunker {
    /// Create a new size-based chunker
    pub fn new(max_chunk_size: usize, min_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size,
            min_chunk_size,
            overlap,
        }
    }
}

impl Chunker for SizeBasedChunker {
    fn chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let html_len = html.len();

        // Split by top-level elements to avoid breaking HTML structure
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&html.as_bytes()[..])
            .map_err(|e| DOMorpherError::DomProcessing(format!("Failed to parse HTML: {}", e)))?;

        let top_elements = get_top_level_elements(&dom.document);

        if top_elements.is_empty() {
            // Fall back to basic chunking if no top elements found
            return self.basic_chunk(html, options, token_counter);
        }

        let mut current_chunk = String::new();
        let mut current_tokens = 0;
        let mut chunk_index = 0;

        for (i, node) in top_elements.iter().enumerate() {
            let node_html = get_node_html(node);
            let node_tokens = token_counter.count_tokens(&node_html);

            // If a single element is already too large, we might need to split it further
            if node_tokens > self.max_chunk_size {
                // If the current chunk is not empty, add it first
                if !current_chunk.is_empty() {
                    let chunk = create_chunk(
                        &current_chunk,
                        chunk_index,
                        start,
                        start + current_chunk.len(),
                        current_tokens,
                        options,
                    );
                    chunks.push(chunk);
                    chunk_index += 1;
                    current_chunk.clear();
                    current_tokens = 0;
                }

                // Split the large element further (simplified approach)
                let split_chunks = split_large_element(
                    &node_html,
                    self.max_chunk_size,
                    self.overlap,
                    token_counter,
                );

                for (j, split_content) in split_chunks.iter().enumerate() {
                    let split_tokens = token_counter.count_tokens(split_content);
                    let pos_start = start + node_html.find(split_content).unwrap_or(0);

                    let chunk = create_chunk(
                        split_content,
                        chunk_index + j,
                        pos_start,
                        pos_start + split_content.len(),
                        split_tokens,
                        options,
                    );

                    chunks.push(chunk);
                }

                chunk_index += split_chunks.len();
                start += node_html.len();
                continue;
            }

            // Check if adding this element would exceed the chunk size
            if current_tokens + node_tokens > self.max_chunk_size
                && current_tokens >= self.min_chunk_size
            {
                // Add the current chunk and start a new one
                let chunk = create_chunk(
                    &current_chunk,
                    chunk_index,
                    start,
                    start + current_chunk.len(),
                    current_tokens,
                    options,
                );
                chunks.push(chunk);

                chunk_index += 1;
                start += current_chunk.len();
                current_chunk = node_html;
                current_tokens = node_tokens;
            } else {
                // Add to the current chunk
                current_chunk.push_str(&node_html);
                current_tokens += node_tokens;
            }

            // If this is the last element, add any remaining content
            if i == top_elements.len() - 1 && !current_chunk.is_empty() {
                let chunk = create_chunk(
                    &current_chunk,
                    chunk_index,
                    start,
                    start + current_chunk.len(),
                    current_tokens,
                    options,
                );
                chunks.push(chunk);
            }
        }

        // If we have multiple chunks, add relationship information
        if chunks.len() > 1 {
            for i in 0..chunks.len() {
                if i > 0 {
                    chunks[i].add_relation(&chunks[i - 1].id, ChunkRelation::Previous);
                }
                if i < chunks.len() - 1 {
                    chunks[i].add_relation(&chunks[i + 1].id, ChunkRelation::Next);
                }
            }
        }

        debug!("Created {} chunks using size-based strategy", chunks.len());
        Ok(ChunkingResult::Multiple(chunks))
    }
}

impl SizeBasedChunker {
    /// Fallback method for basic chunking when DOM parsing fails
    fn basic_chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult> {
        let mut chunks = Vec::new();
        let html_len = html.len();

        // Find a reasonable chunking approach - try to split at block elements
        let block_elements = [
            "</div>",
            "</p>",
            "</section>",
            "</article>",
            "</main>",
            "</header>",
            "</footer>",
            "</nav>",
        ];

        // Find potential split points
        let mut split_points = Vec::new();
        for elem in &block_elements {
            let mut start = 0;
            while let Some(pos) = html[start..].find(elem) {
                let absolute_pos = start + pos + elem.len();
                split_points.push(absolute_pos);
                start = absolute_pos;
            }
        }

        // Sort split points
        split_points.sort();

        // If we have no split points, fall back to character-based chunking
        if split_points.is_empty() {
            warn!("No HTML block elements found for chunking, falling back to basic chunking");
            return self.character_based_chunk(html, options, token_counter);
        }

        // Create chunks based on split points
        let mut current_start = 0;
        let mut chunk_index = 0;

        for (i, &split_point) in split_points.iter().enumerate() {
            let chunk_content = &html[current_start..split_point];
            let chunk_tokens = token_counter.count_tokens(chunk_content);

            if chunk_tokens >= self.min_chunk_size && chunk_tokens <= self.max_chunk_size {
                // This is a good chunk size
                let chunk = create_chunk(
                    chunk_content,
                    chunk_index,
                    current_start,
                    split_point,
                    chunk_tokens,
                    options,
                );

                chunks.push(chunk);
                chunk_index += 1;
                current_start = split_point;
            } else if chunk_tokens > self.max_chunk_size {
                // Too large, need to split further
                let sub_chunks = self.character_based_chunk_segment(
                    chunk_content,
                    current_start,
                    options,
                    token_counter,
                )?;

                for sub_chunk in sub_chunks {
                    chunks.push(sub_chunk);
                    chunk_index += 1;
                }

                current_start = split_point;
            }

            // Skip to the end if too small (will accumulate more content)
        }

        // Handle any remaining content
        if current_start < html_len {
            let chunk_content = &html[current_start..];
            let chunk_tokens = token_counter.count_tokens(chunk_content);

            if chunk_tokens <= self.max_chunk_size {
                let chunk = create_chunk(
                    chunk_content,
                    chunk_index,
                    current_start,
                    html_len,
                    chunk_tokens,
                    options,
                );

                chunks.push(chunk);
            } else {
                // Too large, split further
                let sub_chunks = self.character_based_chunk_segment(
                    chunk_content,
                    current_start,
                    options,
                    token_counter,
                )?;

                for sub_chunk in sub_chunks {
                    chunks.push(sub_chunk);
                }
            }
        }

        // Add relationship information between chunks
        if chunks.len() > 1 {
            for i in 0..chunks.len() {
                if i > 0 {
                    chunks[i].add_relation(&chunks[i - 1].id, ChunkRelation::Previous);
                }
                if i < chunks.len() - 1 {
                    chunks[i].add_relation(&chunks[i + 1].id, ChunkRelation::Next);
                }
            }
        }

        debug!(
            "Created {} chunks using basic chunking strategy",
            chunks.len()
        );
        Ok(ChunkingResult::Multiple(chunks))
    }

    /// Character-based chunking as a last resort
    fn character_based_chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult> {
        self.character_based_chunk_segment(html, 0, options, token_counter)
            .map(|chunks| ChunkingResult::Multiple(chunks))
    }

    /// Character-based chunking for a segment of HTML
    fn character_based_chunk_segment(
        &self,
        html: &str,
        offset: usize,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let html_len = html.len();

        // Determine number of characters per token (rough estimate)
        let sample_size = html_len.min(1000);
        let sample = &html[0..sample_size];
        let sample_tokens = token_counter.count_tokens(sample);
        let chars_per_token = if sample_tokens > 0 {
            sample_size as f32 / sample_tokens as f32
        } else {
            4.0 // Default estimate: 4 chars per token
        };

        // Calculate approximate chunk size in characters
        let chunk_size_chars = (self.max_chunk_size as f32 * chars_per_token) as usize;
        let min_chunk_chars = (self.min_chunk_size as f32 * chars_per_token) as usize;
        let overlap_chars = (self.overlap as f32 * chars_per_token) as usize;

        // Create chunks
        let mut start = 0;
        let mut chunk_index = 0;

        while start < html_len {
            let end = (start + chunk_size_chars).min(html_len);

            // Try to find a clean break point (whitespace, punctuation)
            let adjusted_end = if end < html_len {
                let slice = &html[end.saturating_sub(20)..end.min(html_len)];
                if let Some(pos) = slice.rfind(|c: char| {
                    c.is_whitespace() || c == '.' || c == ';' || c == ',' || c == '>' || c == ')'
                }) {
                    end.saturating_sub(20) + pos + 1
                } else {
                    end
                }
            } else {
                end
            };

            let chunk_content = &html[start..adjusted_end];
            let chunk_tokens = token_counter.count_tokens(chunk_content);

            // Create the chunk
            let chunk = create_chunk(
                chunk_content,
                chunk_index,
                offset + start,
                offset + adjusted_end,
                chunk_tokens,
                options,
            );

            chunks.push(chunk);
            chunk_index += 1;

            // Move to the next chunk with overlap
            start = if adjusted_end >= html_len {
                adjusted_end
            } else {
                adjusted_end.saturating_sub(overlap_chars)
            };
        }

        // Add relationship information
        if chunks.len() > 1 {
            for i in 0..chunks.len() {
                if i > 0 {
                    chunks[i].add_relation(&chunks[i - 1].id, ChunkRelation::Previous);
                }
                if i < chunks.len() - 1 {
                    chunks[i].add_relation(&chunks[i + 1].id, ChunkRelation::Next);
                }
            }
        }

        Ok(chunks)
    }
}

/// Semantic chunking implementation
#[derive(Debug)]
pub struct SemanticChunker {
    /// Maximum token size for a chunk
    max_chunk_size: usize,

    /// Minimum token size for a chunk
    min_chunk_size: usize,

    /// Overlap between chunks
    overlap: usize,

    /// DOM analyzer for semantic analysis
    analyzer: Arc<DomAnalyzer>,
}

impl SemanticChunker {
    /// Create a new semantic chunker
    pub fn new(
        max_chunk_size: usize,
        min_chunk_size: usize,
        overlap: usize,
        analyzer: Arc<DomAnalyzer>,
    ) -> Self {
        Self {
            max_chunk_size,
            min_chunk_size,
            overlap,
            analyzer,
        }
    }
}

impl Chunker for SemanticChunker {
    fn chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult> {
        // Parse the HTML document
        let dom = self.analyzer.parse_html(html)?;

        // Identify semantic sections
        let sections = self.analyzer.identify_semantic_sections(&dom)?;
        debug!("Identified {} semantic sections", sections.len());

        if sections.is_empty() {
            // Fall back to size-based chunking if no semantic sections found
            warn!("No semantic sections found, falling back to size-based chunking");
            let size_chunker =
                SizeBasedChunker::new(self.max_chunk_size, self.min_chunk_size, self.overlap);
            return size_chunker.chunk(html, options, token_counter);
        }

        // Group sections into chunks based on token size
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_sections = Vec::new();
        let mut current_tokens = 0;
        let mut chunk_index = 0;
        let mut start_pos = 0;

        for section in sections {
            let section_html = section.html;
            let section_tokens = token_counter.count_tokens(&section_html);

            // If a single section is too large, split it
            if section_tokens > self.max_chunk_size {
                // Add current chunk if not empty
                if !current_chunk.is_empty() {
                    let chunk = create_semantic_chunk(
                        &current_chunk,
                        chunk_index,
                        start_pos,
                        start_pos + current_chunk.len(),
                        current_tokens,
                        &current_sections,
                        options,
                    );
                    chunks.push(chunk);
                    chunk_index += 1;
                    current_chunk.clear();
                    current_sections.clear();
                    current_tokens = 0;
                }

                // Split the large section
                let section_chunks = split_large_element(
                    &section_html,
                    self.max_chunk_size,
                    self.overlap,
                    token_counter,
                );

                for (i, section_chunk) in section_chunks.iter().enumerate() {
                    let section_chunk_tokens = token_counter.count_tokens(section_chunk);
                    let pos_start = html.find(section_chunk).unwrap_or(0);

                    let chunk = create_chunk(
                        section_chunk,
                        chunk_index + i,
                        pos_start,
                        pos_start + section_chunk.len(),
                        section_chunk_tokens,
                        options,
                    );

                    chunks.push(chunk);
                }

                chunk_index += section_chunks.len();
                continue;
            }

            // Check if adding this section would exceed the chunk size
            if current_tokens + section_tokens > self.max_chunk_size
                && current_tokens >= self.min_chunk_size
            {
                // Add the current chunk and start a new one
                let chunk = create_semantic_chunk(
                    &current_chunk,
                    chunk_index,
                    start_pos,
                    start_pos + current_chunk.len(),
                    current_tokens,
                    &current_sections,
                    options,
                );

                chunks.push(chunk);
                chunk_index += 1;
                start_pos += current_chunk.len();
                current_chunk = section_html;
                current_sections = vec![section.clone()];
                current_tokens = section_tokens;
            } else {
                // Add to the current chunk
                current_chunk.push_str(&section_html);
                current_sections.push(section.clone());
                current_tokens += section_tokens;
            }
        }

        // Add any remaining content
        if !current_chunk.is_empty() {
            let chunk = create_semantic_chunk(
                &current_chunk,
                chunk_index,
                start_pos,
                start_pos + current_chunk.len(),
                current_tokens,
                &current_sections,
                options,
            );

            chunks.push(chunk);
        }

        // Add relationship information
        if chunks.len() > 1 {
            for i in 0..chunks.len() {
                if i > 0 {
                    chunks[i].add_relation(&chunks[i - 1].id, ChunkRelation::Previous);
                }
                if i < chunks.len() - 1 {
                    chunks[i].add_relation(&chunks[i + 1].id, ChunkRelation::Next);
                }
            }
        }

        debug!("Created {} chunks using semantic chunking", chunks.len());
        Ok(ChunkingResult::Multiple(chunks))
    }
}

/// Hierarchical chunking implementation
#[derive(Debug)]
pub struct HierarchicalChunker {
    /// Maximum token size for a chunk
    max_chunk_size: usize,

    /// Minimum token size for a chunk
    min_chunk_size: usize,

    /// DOM analyzer for semantic analysis
    analyzer: Arc<DomAnalyzer>,
}

impl HierarchicalChunker {
    /// Create a new hierarchical chunker
    pub fn new(max_chunk_size: usize, min_chunk_size: usize, analyzer: Arc<DomAnalyzer>) -> Self {
        Self {
            max_chunk_size,
            min_chunk_size,
            analyzer,
        }
    }
}

impl Chunker for HierarchicalChunker {
    fn chunk(
        &self,
        html: &str,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<ChunkingResult> {
        // Parse the HTML document
        let dom = self.analyzer.parse_html(html)?;

        // Create a chunk tree
        let mut tree = ChunkTree::new();

        // Process the document recursively
        let dom_node = DomNode::from_handle(dom.document.clone());
        self.process_node(&dom_node, None, 0, &mut tree, options, token_counter)?;

        debug!("Created hierarchical chunk tree with {} chunks", tree.len());
        Ok(ChunkingResult::Tree(tree))
    }
}

impl HierarchicalChunker {
    /// Process a DOM node recursively to build the chunk tree
    fn process_node(
        &self,
        node: &DomNode,
        parent_id: Option<&str>,
        depth: usize,
        tree: &mut ChunkTree,
        options: &ChunkingOptions,
        token_counter: &Arc<dyn TokenCounter>,
    ) -> Result<()> {
        // Skip non-element nodes
        if !node.is_element() {
            return Ok(());
        }

        // Get node HTML
        let node_html = node.to_html();

        // Skip empty nodes
        if node_html.trim().is_empty() {
            return Ok(());
        }

        let token_count = token_counter.count_tokens(&node_html);

        // Determine node type
        let content_type = self.determine_node_type(node);

        // Determine whether to create a chunk for this node
        let should_create_chunk = self.should_create_chunk(node, token_count);

        if should_create_chunk {
            // Create metadata
            let metadata = ChunkMetadata {
                token_count,
                position: ChunkPosition {
                    index: tree.len(),
                    start: 0, // Position info is less relevant in hierarchical chunking
                    end: node_html.len(),
                    depth,
                    parent_index: None, // Will be set by the tree
                },
                content_type,
                importance: self.calculate_importance(node, content_type),
                source: options.source_url.clone(),
                element_path: Some(node.selector_path()),
                created_at: chrono::Utc::now(),
                custom_fields: options.additional_metadata.clone(),
            };

            // Create the chunk
            let chunk = Chunk::with_metadata(&node_html, metadata);
            let chunk_id = chunk.id.clone();

            // Add to the tree
            let chunk_arc = tree.add_chunk(chunk, parent_id);

            // Process children recursively if this node is not too large
            if token_count <= self.max_chunk_size * 2 {
                // Allow some flexibility
                for child in node.children() {
                    if child.is_element() {
                        self.process_node(
                            &child,
                            Some(&chunk_id),
                            depth + 1,
                            tree,
                            options,
                            token_counter,
                        )?;
                    }
                }
            } else {
                // For very large nodes, split the content instead of processing children
                debug!("Node too large for hierarchical processing, splitting content");

                // Split the large node content
                let children_html = node.children_html();
                let size_chunker = SizeBasedChunker::new(
                    self.max_chunk_size,
                    self.min_chunk_size,
                    200, // Overlap
                );

                let options_clone = ChunkingOptions {
                    document_title: options.document_title.clone(),
                    source_url: options.source_url.clone(),
                    token_counter: Some(token_counter.clone()),
                    preprocessor: None,
                    analyzer: None,
                    additional_metadata: options.additional_metadata.clone(),
                };

                let result = size_chunker.chunk(&children_html, &options_clone, token_counter)?;

                if let ChunkingResult::Multiple(chunks) = result {
                    for sub_chunk in chunks {
                        let sub_chunk_id = sub_chunk.id.clone();
                        tree.add_chunk(sub_chunk, Some(&chunk_id));
                    }
                }
            }
        } else {
            // Process children directly without creating a chunk for this node
            for child in node.children() {
                if child.is_element() {
                    self.process_node(&child, parent_id, depth, tree, options, token_counter)?;
                }
            }
        }

        Ok(())
    }

    /// Determine if a node should be its own chunk
    fn should_create_chunk(&self, node: &DomNode, token_count: usize) -> bool {
        // Always create chunks for semantic block elements
        let tag_name = node.tag_name().to_lowercase();
        let semantic_tags = [
            "main", "article", "section", "nav", "aside", "header", "footer", "div", "form",
            "table", "ul", "ol", "dl",
        ];

        if semantic_tags.contains(&tag_name.as_str()) && token_count >= self.min_chunk_size {
            return true;
        }

        // Create chunks for elements with semantic classes or IDs
        let semantic_patterns = [
            "content",
            "main",
            "article",
            "post",
            "entry",
            "page",
            "section",
            "container",
            "wrapper",
            "product",
            "item",
            "card",
            "result",
        ];

        let class_attr = node.attr("class").unwrap_or_default();
        let id_attr = node.attr("id").unwrap_or_default();

        let has_semantic_class = semantic_patterns
            .iter()
            .any(|&pattern| class_attr.contains(pattern));

        let has_semantic_id = semantic_patterns
            .iter()
            .any(|&pattern| id_attr.contains(pattern));

        if (has_semantic_class || has_semantic_id) && token_count >= self.min_chunk_size {
            return true;
        }

        // For other elements, create chunks if they're large enough
        token_count >= self.max_chunk_size / 2
    }

    /// Determine the content type of a node
    fn determine_node_type(&self, node: &DomNode) -> ChunkType {
        let tag_name = node.tag_name().to_lowercase();

        match tag_name.as_str() {
            "html" => ChunkType::Document,
            "table" => ChunkType::Table,
            "ul" | "ol" | "dl" => ChunkType::List,
            "form" => ChunkType::Form,
            "nav" => ChunkType::Navigation,
            "header" => ChunkType::Header,
            "footer" => ChunkType::Footer,
            "main" | "article" => ChunkType::MainContent,
            "aside" => ChunkType::Sidebar,
            _ => {
                // Check for semantic classes or role attributes
                let class_attr = node.attr("class").unwrap_or_default();
                let role_attr = node.attr("role").unwrap_or_default();

                if class_attr.contains("nav") || role_attr == "navigation" {
                    ChunkType::Navigation
                } else if class_attr.contains("main")
                    || class_attr.contains("content")
                    || role_attr == "main"
                {
                    ChunkType::MainContent
                } else if class_attr.contains("sidebar") || role_attr == "complementary" {
                    ChunkType::Sidebar
                } else if class_attr.contains("header") || role_attr == "banner" {
                    ChunkType::Header
                } else if class_attr.contains("footer") || role_attr == "contentinfo" {
                    ChunkType::Footer
                } else {
                    ChunkType::Element
                }
            }
        }
    }

    /// Calculate importance score for a node
    fn calculate_importance(&self, node: &DomNode, content_type: ChunkType) -> f32 {
        let base_importance = match content_type {
            ChunkType::Document => 1.0,
            ChunkType::MainContent => 0.9,
            ChunkType::Element => 0.5,
            ChunkType::Table => 0.7,
            ChunkType::List => 0.6,
            ChunkType::Form => 0.7,
            ChunkType::Navigation => 0.4,
            ChunkType::Header => 0.5,
            ChunkType::Footer => 0.3,
            ChunkType::Sidebar => 0.4,
            ChunkType::Text => 0.5,
            ChunkType::Custom(_) => 0.5,
        };

        // Adjust importance based on factors like headings, links, etc.
        let mut importance_adjustment = 0.0;

        // Check for headings
        let contains_headings = node.find_elements("h1, h2, h3, h4, h5, h6").len() > 0;
        if contains_headings {
            importance_adjustment += 0.1;
        }

        // Check for links density
        let links = node.find_elements("a");
        let link_count = links.len() as f32;
        let text_length = node.text_content().len() as f32;
        let link_density = if text_length > 0.0 {
            link_count / (text_length / 100.0) // Links per 100 characters
        } else {
            0.0
        };

        // High link density might indicate navigation, lower importance
        if link_density > 5.0 {
            importance_adjustment -= 0.1;
        }

        // Finalize importance score
        (base_importance + importance_adjustment).max(0.1).min(1.0)
    }
}

/// Helper function to create a chunk from HTML content
fn create_chunk(
    content: &str,
    index: usize,
    start: usize,
    end: usize,
    token_count: usize,
    options: &ChunkingOptions,
) -> Chunk {
    let metadata = ChunkMetadata {
        token_count,
        position: ChunkPosition {
            index,
            start,
            end,
            depth: 0,
            parent_index: None,
        },
        content_type: ChunkType::Element,
        importance: 0.5, // Default importance
        source: options.source_url.clone(),
        element_path: None,
        created_at: chrono::Utc::now(),
        custom_fields: options.additional_metadata.clone(),
    };

    Chunk::with_metadata(content, metadata)
}

/// Helper function to create a chunk from semantic sections
fn create_semantic_chunk(
    content: &str,
    index: usize,
    start: usize,
    end: usize,
    token_count: usize,
    sections: &[crate::dom::analyzer::SemanticSection],
    options: &ChunkingOptions,
) -> Chunk {
    // Determine the most appropriate content type based on sections
    let content_type = if sections.len() == 1 {
        sections[0].section_type.clone()
    } else {
        // Choose the most specific/important type
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for section in sections {
            *type_counts.entry(section.section_type.clone()).or_insert(0) += 1;
        }

        // Prioritize certain types
        for priority_type in &["main", "article", "section", "content"] {
            if type_counts.contains_key(*priority_type) {
                match *priority_type {
                    "main" => {
                        return create_chunk_with_type(
                            content,
                            index,
                            start,
                            end,
                            token_count,
                            ChunkType::MainContent,
                            options,
                        );
                    }
                    "article" => {
                        return create_chunk_with_type(
                            content,
                            index,
                            start,
                            end,
                            token_count,
                            ChunkType::MainContent,
                            options,
                        );
                    }
                    "section" => {
                        return create_chunk_with_type(
                            content,
                            index,
                            start,
                            end,
                            token_count,
                            ChunkType::Element,
                            options,
                        );
                    }
                    "content" => {
                        return create_chunk_with_type(
                            content,
                            index,
                            start,
                            end,
                            token_count,
                            ChunkType::MainContent,
                            options,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Default to generic Element type
        ChunkType::Element
    };

    create_chunk_with_type(
        content,
        index,
        start,
        end,
        token_count,
        content_type,
        options,
    )
}

/// Helper function to create a chunk with a specific type
fn create_chunk_with_type(
    content: &str,
    index: usize,
    start: usize,
    end: usize,
    token_count: usize,
    content_type: ChunkType,
    options: &ChunkingOptions,
) -> Chunk {
    let metadata = ChunkMetadata {
        token_count,
        position: ChunkPosition {
            index,
            start,
            end,
            depth: 0,
            parent_index: None,
        },
        content_type,
        importance: 0.5, // Default importance
        source: options.source_url.clone(),
        element_path: None,
        created_at: chrono::Utc::now(),
        custom_fields: options.additional_metadata.clone(),
    };

    Chunk::with_metadata(content, metadata)
}

/// Get top-level elements from a DOM tree
fn get_top_level_elements(document: &Handle) -> Vec<Handle> {
    match document.data {
        NodeData::Document => {
            // Get the <html> element
            let mut html_nodes = Vec::new();
            for child in document.children.borrow().iter() {
                match child.data {
                    NodeData::Element { ref name, .. } => {
                        if name.local.as_ref() == "html" {
                            // Get the <body> element
                            for body_child in child.children.borrow().iter() {
                                match body_child.data {
                                    NodeData::Element { ref name, .. } => {
                                        if name.local.as_ref() == "body" {
                                            // Return the children of <body>
                                            return body_child
                                                .children
                                                .borrow()
                                                .iter()
                                                .filter(|node| {
                                                    matches!(node.data, NodeData::Element { .. })
                                                })
                                                .cloned()
                                                .collect();
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            // If no <body>, return children of <html>
                            return child
                                .children
                                .borrow()
                                .iter()
                                .filter(|node| matches!(node.data, NodeData::Element { .. }))
                                .cloned()
                                .collect();
                        }
                    }
                    _ => {}
                }
            }

            // If no <html>, return direct children
            document
                .children
                .borrow()
                .iter()
                .filter(|node| matches!(node.data, NodeData::Element { .. }))
                .cloned()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Get HTML string from a DOM node
fn get_node_html(node: &Handle) -> String {
    let mut html = String::new();
    serialize_node(node, &mut html);
    html
}

/// Serialize a DOM node to HTML
fn serialize_node(node: &Handle, output: &mut String) {
    match node.data {
        NodeData::Element {
            ref name,
            ref attrs,
            ..
        } => {
            output.push('<');
            output.push_str(name.local.as_ref());

            for attr in attrs.borrow().iter() {
                output.push(' ');
                output.push_str(attr.name.local.as_ref());
                output.push_str("=\"");
                output.push_str(&attr.value);
                output.push('"');
            }

            output.push('>');

            for child in node.children.borrow().iter() {
                serialize_node(child, output);
            }

            output.push_str("</");
            output.push_str(name.local.as_ref());
            output.push('>');
        }
        NodeData::Text { ref contents } => {
            output.push_str(&contents.borrow());
        }
        NodeData::Comment { ref contents } => {
            output.push_str("<!--");
            output.push_str(&contents);
            output.push_str("-->");
        }
        NodeData::ProcessingInstruction {
            ref target,
            ref contents,
        } => {
            output.push_str("<?");
            output.push_str(&target);
            output.push(' ');
            output.push_str(&contents);
            output.push_str("?>");
        }
        NodeData::Doctype {
            ref name,
            ref public_id,
            ref system_id,
        } => {
            output.push_str("<!DOCTYPE ");
            output.push_str(&name);

            if !public_id.is_empty() {
                output.push_str(" PUBLIC \"");
                output.push_str(&public_id);
                output.push('"');
            }

            if !system_id.is_empty() {
                output.push_str(" \"");
                output.push_str(&system_id);
                output.push('"');
            }

            output.push('>');
        }
        NodeData::Document => {
            for child in node.children.borrow().iter() {
                serialize_node(child, output);
            }
        }
    }
}

/// Split a large HTML element into smaller chunks
fn split_large_element(
    html: &str,
    max_tokens: usize,
    overlap: usize,
    token_counter: &Arc<dyn TokenCounter>,
) -> Vec<String> {
    // First try to split at meaningful boundaries
    let split_tags = [
        "</p>",
        "</div>",
        "</section>",
        "</article>",
        "</li>",
        "</table>",
        "</tr>",
        "</form>",
        "</fieldset>",
    ];

    let mut split_points = Vec::new();
    for tag in &split_tags {
        let mut start = 0;
        while let Some(pos) = html[start..].find(tag) {
            let absolute_pos = start + pos + tag.len();
            split_points.push(absolute_pos);
            start = absolute_pos;
        }
    }

    // Add fallback split points at regular intervals if needed
    if split_points.len() < 2 {
        // Approximate characters per token
        let sample_size = html.len().min(1000);
        let sample = &html[0..sample_size];
        let sample_tokens = token_counter.count_tokens(sample);
        let chars_per_token = if sample_tokens > 0 {
            sample_size as f32 / sample_tokens as f32
        } else {
            4.0 // Default: 4 chars per token
        };

        let approx_max_chars = (max_tokens as f32 * chars_per_token) as usize;
        let approx_overlap_chars = (overlap as f32 * chars_per_token) as usize;

        // Add regular split points
        let mut pos = approx_max_chars;
        while pos < html.len() {
            // Try to find a clean break point within a window
            let window_start = pos.saturating_sub(50);
            let window_end = (pos + 50).min(html.len());
            let window = &html[window_start..window_end];

            if let Some(break_pos) = find_clean_break(window) {
                split_points.push(window_start + break_pos);
            } else {
                split_points.push(pos);
            }

            pos += approx_max_chars - approx_overlap_chars;
        }
    }

    // Sort split points
    split_points.sort();

    // Create chunks
    let mut chunks = Vec::new();
    let mut start = 0;

    for &split_point in &split_points {
        if split_point > start {
            let chunk = html[start..split_point].to_string();
            let token_count = token_counter.count_tokens(&chunk);

            if token_count <= max_tokens {
                chunks.push(chunk);
                start = split_point - overlap.min(split_point);
            }
        }
    }

    // Add final chunk if needed
    if start < html.len() {
        let final_chunk = html[start..].to_string();
        let token_count = token_counter.count_tokens(&final_chunk);

        if token_count <= max_tokens {
            chunks.push(final_chunk);
        } else {
            // If the final piece is still too large, split it naively
            let approx_chars = (max_tokens as f32 * 4.0) as usize; // Assume 4 chars per token
            let mut piece_start = start;

            while piece_start < html.len() {
                let piece_end = (piece_start + approx_chars).min(html.len());
                chunks.push(html[piece_start..piece_end].to_string());
                piece_start = piece_end;
            }
        }
    }

    // Ensure each chunk has valid HTML structure
    chunks.iter_mut().for_each(|chunk| {
        // Balance tags if needed
        *chunk = balance_html_tags(chunk);
    });

    chunks
}

/// Find a clean break point in HTML text
fn find_clean_break(text: &str) -> Option<usize> {
    // Try to find tag boundaries
    if let Some(pos) = text.rfind('>') {
        return Some(pos + 1);
    }

    // Try to find sentence boundaries
    if let Some(pos) = text.rfind(|c| c == '.' || c == '?' || c == '!') {
        return Some(pos + 1);
    }

    // Try to find paragraph boundaries
    if let Some(pos) = text.rfind("\n\n") {
        return Some(pos + 2);
    }

    // Try to find line breaks
    if let Some(pos) = text.rfind('\n') {
        return Some(pos + 1);
    }

    // Try to find word boundaries
    for i in (0..text.len()).rev() {
        if text.is_char_boundary(i) && i > 0 && i < text.len() {
            let c = text.chars().nth(i - 1).unwrap_or(' ');
            let next = text.chars().nth(i).unwrap_or(' ');

            if c.is_whitespace() && !next.is_whitespace() {
                return Some(i);
            }
        }
    }

    None
}

/// Balance HTML tags in a chunk to ensure valid HTML
fn balance_html_tags(html: &str) -> String {
    // This is a simplified implementation
    // A full implementation would parse the HTML and properly balance tags

    // Check for unclosed tags (simplified approach)
    let mut open_tags = Vec::new();
    let mut in_tag = false;
    let mut in_closing_tag = false;
    let mut current_tag = String::new();

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            current_tag.clear();
        } else if c == '/' && in_tag && current_tag.is_empty() {
            in_closing_tag = true;
        } else if c == '>' && in_tag {
            in_tag = false;

            if !current_tag.is_empty() {
                if in_closing_tag {
                    in_closing_tag = false;

                    // Find and remove matching opening tag
                    if let Some(pos) = open_tags.iter().rposition(|tag| tag == &current_tag) {
                        open_tags.remove(pos);
                    }
                } else {
                    // Self-closing tags don't need to be tracked
                    let is_self_closing = [
                        "img", "br", "hr", "input", "link", "meta", "source", "track", "wbr",
                        "area", "base", "col", "embed", "param",
                    ]
                    .contains(&current_tag.as_str());

                    if !is_self_closing {
                        open_tags.push(current_tag.clone());
                    }
                }
            }

            current_tag.clear();
        } else if in_tag && !c.is_whitespace() {
            current_tag.push(c);
        }
    }

    // No unclosed tags, return original
    if open_tags.is_empty() {
        return html.to_string();
    }

    // Add closing tags
    let mut result = html.to_string();
    for tag in open_tags.iter().rev() {
        result.push_str(&format!("</{}>", tag));
    }

    result
}
