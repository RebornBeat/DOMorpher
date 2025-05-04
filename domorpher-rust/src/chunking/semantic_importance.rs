//! # Semantic Importance-Based Chunking
//!
//! This module implements an advanced chunking technique that analyzes HTML content
//! to identify semantic sections and prioritizes them based on importance metrics
//! such as content density, interactive element concentration, and semantic hierarchy.
//!
//! The algorithm divides content into chunks by:
//! 1. Identifying semantic sections in the document
//! 2. Calculating importance metrics for each section
//! 3. Creating optimized chunks that prioritize important content
//! 4. Ensuring chunk sizes stay within token limits while preserving semantic boundaries
//!
//! This approach ensures that the most relevant content is included in chunks even
//! when dealing with very large documents that exceed context windows.

use crate::chunking::chunk::{Chunk, ChunkMetadata, ChunkRelation, EnhancedChunk};
use crate::chunking::strategies::ChunkingResult;
use crate::dom::analyzer::DomAnalyzer;
use crate::error::{DOMorpherError, Result};
use crate::utils::tokenizer::TokenCounter;

use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use log::{debug, trace, warn};
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

/// Configuration for the semantic importance-based chunking algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceConfig {
    /// Minimum chunk size in tokens
    pub min_chunk_size: usize,

    /// Maximum chunk size in tokens
    pub max_chunk_size: usize,

    /// Overlap between chunks in tokens
    pub overlap: usize,

    /// Weight for hierarchy importance (0.0-1.0)
    pub hierarchy_weight: f32,

    /// Weight for content density (0.0-1.0)
    pub content_weight: f32,

    /// Weight for interactive element concentration (0.0-1.0)
    pub interactivity_weight: f32,

    /// Minimum importance threshold for prioritization (0.0-1.0)
    pub importance_threshold: f32,

    /// Whether to include section context summaries
    pub include_context: bool,

    /// Elements that should never be split across chunks
    pub preserve_elements: HashSet<String>,

    /// Optional selectors to prioritize
    pub priority_selectors: Vec<String>,
}

impl Default for ImportanceConfig {
    fn default() -> Self {
        let mut preserve_elements = HashSet::new();
        preserve_elements.insert("article".to_string());
        preserve_elements.insert("section".to_string());
        preserve_elements.insert("table".to_string());
        preserve_elements.insert("form".to_string());
        preserve_elements.insert("figure".to_string());
        preserve_elements.insert("pre".to_string());

        Self {
            min_chunk_size: 1000,
            max_chunk_size: 8000,
            overlap: 200,
            hierarchy_weight: 0.4,
            content_weight: 0.4,
            interactivity_weight: 0.2,
            importance_threshold: 0.5,
            include_context: true,
            preserve_elements,
            priority_selectors: vec![
                "main".to_string(),
                "article".to_string(),
                ".content".to_string(),
                "#content".to_string(),
                ".main-content".to_string(),
            ],
        }
    }
}

/// Metrics used to calculate importance of document sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceMetrics {
    /// Importance based on document hierarchy (0.0-1.0)
    pub hierarchy_score: f32,

    /// Importance based on content density (0.0-1.0)
    pub content_score: f32,

    /// Importance based on interactive element concentration (0.0-1.0)
    pub interactivity_score: f32,

    /// Combined importance score (0.0-1.0)
    pub overall_score: f32,

    /// Token count for this section
    pub token_count: usize,

    /// Whether this section matches priority selectors
    pub is_priority: bool,
}

impl ImportanceMetrics {
    /// Calculate overall importance score from individual metrics
    pub fn calculate_overall(&mut self, config: &ImportanceConfig) {
        self.overall_score = (self.hierarchy_score * config.hierarchy_weight)
            + (self.content_score * config.content_weight)
            + (self.interactivity_score * config.interactivity_weight);

        // Boost score for priority sections
        if self.is_priority {
            self.overall_score = (self.overall_score + 1.0) / 2.0;
        }

        // Clamp to 0.0-1.0 range
        self.overall_score = self.overall_score.clamp(0.0, 1.0);
    }
}

/// A semantic section of the document with its content and importance metrics
#[derive(Debug, Clone)]
pub struct SemanticSection {
    /// Section ID
    pub id: String,

    /// HTML content of the section
    pub content: String,

    /// Importance metrics for this section
    pub metrics: ImportanceMetrics,

    /// Element selector path to this section
    pub selector_path: String,

    /// Parent section ID (if any)
    pub parent_id: Option<String>,

    /// Child section IDs (if any)
    pub child_ids: Vec<String>,

    /// Document position (top-to-bottom order)
    pub position: usize,

    /// Element tag name
    pub tag_name: String,

    /// Element classes
    pub classes: Vec<String>,

    /// Element ID
    pub element_id: Option<String>,

    /// Whether this section is a container of other sections
    pub is_container: bool,

    /// Depth in the DOM tree
    pub depth: usize,
}

impl SemanticSection {
    /// Create a new semantic section
    pub fn new(
        id: String,
        content: String,
        tag_name: String,
        selector_path: String,
        position: usize,
        depth: usize,
    ) -> Self {
        // Initialize with default metrics
        let metrics = ImportanceMetrics {
            hierarchy_score: 0.0,
            content_score: 0.0,
            interactivity_score: 0.0,
            overall_score: 0.0,
            token_count: 0,
            is_priority: false,
        };

        Self {
            id,
            content,
            metrics,
            selector_path,
            parent_id: None,
            child_ids: Vec::new(),
            position,
            tag_name,
            classes: Vec::new(),
            element_id: None,
            is_container: false,
            depth,
        }
    }

    /// Check if this section should be preserved in a single chunk
    pub fn should_preserve(&self, config: &ImportanceConfig) -> bool {
        config.preserve_elements.contains(&self.tag_name)
    }

    /// Check if this section is a priority section
    pub fn check_priority(&mut self, config: &ImportanceConfig) {
        // Check if this section matches any priority selectors
        for selector in &config.priority_selectors {
            // Check element ID
            if let Some(ref id) = self.element_id {
                if selector.starts_with('#') && selector[1..] == *id {
                    self.metrics.is_priority = true;
                    return;
                }
            }

            // Check element class
            if selector.starts_with('.') {
                let class_name = &selector[1..];
                if self.classes.iter().any(|c| c == class_name) {
                    self.metrics.is_priority = true;
                    return;
                }
            }

            // Check element tag
            if !selector.starts_with('#')
                && !selector.starts_with('.')
                && self.tag_name == *selector
            {
                self.metrics.is_priority = true;
                return;
            }

            // Check selector path
            if self.selector_path.contains(selector) {
                self.metrics.is_priority = true;
                return;
            }
        }
    }
}

/// Advanced chunker that uses semantic importance to optimize chunk creation
pub struct SemanticImportanceChunker {
    /// Configuration for the chunker
    config: ImportanceConfig,

    /// DOM analyzer for extracting structure information
    analyzer: Arc<DomAnalyzer>,

    /// Token counter for measuring content length
    token_counter: TokenCounter,
}

impl SemanticImportanceChunker {
    /// Create a new semantic importance chunker with the given configuration
    pub fn new(config: ImportanceConfig, analyzer: Arc<DomAnalyzer>) -> Self {
        Self {
            config,
            analyzer,
            token_counter: TokenCounter::new(),
        }
    }

    /// Create a new semantic importance chunker with default configuration
    pub fn default(analyzer: Arc<DomAnalyzer>) -> Self {
        Self::new(ImportanceConfig::default(), analyzer)
    }

    /// Process HTML content into chunks based on semantic importance
    pub fn process(&self, html: &str) -> Result<ChunkingResult> {
        debug!("Starting semantic importance-based chunking");

        // Parse and analyze the HTML
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut html.as_bytes())
            .map_err(|e| DOMorpherError::DOMParsingError(format!("Failed to parse HTML: {}", e)))?;

        // Extract semantic sections
        let sections = self.extract_semantic_sections(&dom.document, html)?;

        // Calculate importance metrics for each section
        let sections_with_metrics = self.calculate_importance_metrics(sections)?;

        // Create chunks based on importance
        let chunks = self.create_importance_based_chunks(sections_with_metrics)?;

        // Return result
        Ok(ChunkingResult {
            chunks,
            strategy: crate::chunking::strategies::ChunkingStrategy::Adaptive,
        })
    }

    /// Extract semantic sections from the DOM
    fn extract_semantic_sections(&self, root: &Handle, html: &str) -> Result<Vec<SemanticSection>> {
        debug!("Extracting semantic sections from DOM");

        // Track section IDs to ensure uniqueness
        let mut section_id_counter = 0;
        let generate_id = move || {
            section_id_counter += 1;
            format!("section_{}", section_id_counter)
        };

        // Create sections for semantic elements
        let mut sections: Vec<SemanticSection> = Vec::new();
        let mut position_counter = 0;

        // Process the DOM tree
        let section_tags = vec![
            "body", "article", "section", "div", "main", "aside", "nav", "header", "footer",
            "form", "table",
        ];

        let mut process_node = |node: &Handle, depth: usize, path: &str| -> Result<()> {
            let node_data = &node.data;

            match *node_data {
                NodeData::Element {
                    ref name,
                    ref attrs,
                    ..
                } => {
                    let tag_name = name.local.to_string();

                    // Only process elements that can be semantic sections
                    if section_tags.contains(&tag_name.as_str()) {
                        // Get element attributes
                        let attrs = attrs.borrow();

                        // Create selector path
                        let mut selector_path = path.to_string();
                        if !selector_path.is_empty() {
                            selector_path.push_str(" > ");
                        }
                        selector_path.push_str(&tag_name);

                        // Get element ID and classes
                        let mut element_id = None;
                        let mut classes = Vec::new();

                        for attr in attrs.iter() {
                            if attr.name.local.to_string() == "id" {
                                let id = attr.value.to_string();
                                element_id = Some(id.clone());
                                selector_path.push_str(&format!("#{}", id));
                            } else if attr.name.local.to_string() == "class" {
                                let class_str = attr.value.to_string();
                                classes = class_str
                                    .split_whitespace()
                                    .map(|s| s.to_string())
                                    .collect();

                                for class in &classes {
                                    selector_path.push_str(&format!(".{}", class));
                                }
                            }
                        }

                        // Extract HTML content for this section
                        let section_html = self.extract_html_from_node(node, html)?;

                        // Create section
                        let id = generate_id();
                        position_counter += 1;

                        let mut section = SemanticSection::new(
                            id,
                            section_html,
                            tag_name,
                            selector_path,
                            position_counter,
                            depth,
                        );

                        section.element_id = element_id;
                        section.classes = classes;

                        // Check if this is a priority section
                        section.check_priority(&self.config);

                        // Add to sections list
                        sections.push(section);
                    }
                }
                _ => {}
            }

            Ok(())
        };

        // Traverse the DOM tree
        self.traverse_dom(root, 0, "", &mut process_node)?;

        // Establish parent-child relationships
        self.establish_section_relationships(&mut sections);

        debug!("Extracted {} semantic sections", sections.len());
        Ok(sections)
    }

    /// Traverse DOM tree and process each node
    fn traverse_dom<F>(
        &self,
        node: &Handle,
        depth: usize,
        path: &str,
        process_fn: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&Handle, usize, &str) -> Result<()>,
    {
        // Process current node
        process_fn(node, depth, path)?;

        // Process child nodes
        for child in node.children.borrow().iter() {
            // Get child node data
            let child_data = &child.data;

            // Build path for child node
            let mut child_path = path.to_string();

            if let NodeData::Element { ref name, .. } = *child_data {
                let tag_name = name.local.to_string();

                if !child_path.is_empty() {
                    child_path.push_str(" > ");
                }
                child_path.push_str(&tag_name);
            }

            // Recursively process child node
            self.traverse_dom(child, depth + 1, &child_path, process_fn)?;
        }

        Ok(())
    }

    /// Extract HTML content from a DOM node
    fn extract_html_from_node(&self, node: &Handle, html: &str) -> Result<String> {
        // This is a simplified implementation
        // In a real-world scenario, you'd use a more robust approach to extract HTML

        // For now, we'll use a placeholder approach by serializing the node
        let mut buffer = Vec::new();
        html5ever::serialize::serialize(&mut buffer, node, Default::default()).map_err(|e| {
            DOMorpherError::DOMParsingError(format!("Failed to serialize HTML: {}", e))
        })?;

        let node_html = String::from_utf8(buffer).map_err(|e| {
            DOMorpherError::DOMParsingError(format!("Invalid UTF-8 in HTML: {}", e))
        })?;

        Ok(node_html)
    }

    /// Establish parent-child relationships between sections
    fn establish_section_relationships(&self, sections: &mut Vec<SemanticSection>) {
        // Sort sections by depth (shallow to deep) and position
        sections.sort_by(|a, b| {
            if a.depth == b.depth {
                a.position.cmp(&b.position)
            } else {
                a.depth.cmp(&b.depth)
            }
        });

        // Create a map of section contents to IDs
        let mut content_to_id: HashMap<String, String> = HashMap::new();
        for section in sections.iter() {
            content_to_id.insert(section.content.clone(), section.id.clone());
        }

        // Find parent-child relationships
        for i in 0..sections.len() {
            let section_content = sections[i].content.clone();

            for j in 0..sections.len() {
                if i != j && sections[j].content.contains(&section_content) {
                    // Section j contains section i, so j is a parent of i
                    let parent_id = sections[j].id.clone();
                    let child_id = sections[i].id.clone();

                    // Update parent reference in child
                    sections[i].parent_id = Some(parent_id.clone());

                    // Update child references in parent
                    for section in sections.iter_mut() {
                        if section.id == parent_id {
                            section.child_ids.push(child_id.clone());
                            section.is_container = true;
                            break;
                        }
                    }

                    // No need to check other sections as we found the immediate parent
                    break;
                }
            }
        }

        // Mark sections as containers if they have children
        for section in sections.iter_mut() {
            if !section.child_ids.is_empty() {
                section.is_container = true;
            }
        }
    }

    /// Calculate importance metrics for each section
    fn calculate_importance_metrics(
        &self,
        mut sections: Vec<SemanticSection>,
    ) -> Result<Vec<SemanticSection>> {
        debug!(
            "Calculating importance metrics for {} sections",
            sections.len()
        );

        // Calculate token counts for each section
        for section in sections.iter_mut() {
            section.metrics.token_count = self.token_counter.count_tokens(&section.content);
        }

        // Calculate hierarchy scores based on depth and element type
        self.calculate_hierarchy_scores(&mut sections);

        // Calculate content density scores
        self.calculate_content_scores(&mut sections);

        // Calculate interactivity scores
        self.calculate_interactivity_scores(&mut sections);

        // Calculate overall importance scores
        for section in sections.iter_mut() {
            section.metrics.calculate_overall(&self.config);
        }

        debug!("Importance metrics calculation complete");
        trace!(
            "Section importance scores: {:?}",
            sections
                .iter()
                .map(|s| (s.id.clone(), s.metrics.overall_score))
                .collect::<Vec<_>>()
        );

        Ok(sections)
    }

    /// Calculate hierarchy scores based on depth and element type
    fn calculate_hierarchy_scores(&self, sections: &mut Vec<SemanticSection>) {
        // Get deepest section depth for normalization
        let max_depth = sections.iter().map(|s| s.depth).max().unwrap_or(1);

        // Calculate scores based on depth (shallower elements are more important)
        for section in sections.iter_mut() {
            // Base score inversely proportional to depth
            let depth_score = if max_depth > 1 {
                1.0 - (section.depth as f32 / max_depth as f32)
            } else {
                1.0
            };

            // Adjust score based on tag name
            let tag_bonus = match section.tag_name.as_str() {
                "main" => 0.4,
                "article" => 0.3,
                "section" => 0.2,
                "div" => 0.0,
                "header" => -0.1,
                "footer" => -0.2,
                "aside" => -0.1,
                "nav" => -0.2,
                _ => 0.0,
            };

            // Adjust score based on special classes or IDs
            let mut attribute_bonus = 0.0;

            if let Some(ref id) = section.element_id {
                if id.contains("content") || id.contains("main") || id.contains("article") {
                    attribute_bonus += 0.2;
                }
            }

            for class in &section.classes {
                if class.contains("content") || class.contains("main") || class.contains("article")
                {
                    attribute_bonus += 0.2;
                    break;
                }
            }

            // Combined score
            section.metrics.hierarchy_score =
                (depth_score + tag_bonus + attribute_bonus).clamp(0.0, 1.0);
        }
    }

    /// Calculate content density scores
    fn calculate_content_scores(&self, sections: &mut Vec<SemanticSection>) {
        // Get section with maximum text density
        let mut max_content_length = 0;
        let mut max_token_count = 0;

        for section in sections.iter() {
            let content_length = section.content.len();
            let token_count = section.metrics.token_count;

            if content_length > 0 {
                max_content_length = max_content_length.max(content_length);
                max_token_count = max_token_count.max(token_count);
            }
        }

        // Calculate scores based on content density
        for section in sections.iter_mut() {
            let content_length = section.content.len();
            let token_count = section.metrics.token_count;

            if max_content_length > 0 && max_token_count > 0 {
                // Score based on content length
                let length_score = content_length as f32 / max_content_length as f32;

                // Score based on token count
                let token_score = token_count as f32 / max_token_count as f32;

                // Score based on text to HTML ratio
                let text_content = self.extract_text_content(&section.content);
                let text_ratio = if content_length > 0 {
                    text_content.len() as f32 / content_length as f32
                } else {
                    0.0
                };

                // Combined score
                section.metrics.content_score =
                    ((length_score + token_score) / 2.0 * 0.7 + text_ratio * 0.3).clamp(0.0, 1.0);
            } else {
                section.metrics.content_score = 0.0;
            }
        }
    }

    /// Calculate interactivity scores based on interactive elements
    fn calculate_interactivity_scores(&self, sections: &mut Vec<SemanticSection>) {
        // Count interactive elements in each section
        for section in sections.iter_mut() {
            let interactive_count = count_interactive_elements(&section.content);

            // Score based on presence of interactive elements
            if interactive_count > 0 {
                // More interactivity is valuable, but with diminishing returns
                let interactivity_score =
                    (1.0 - (1.0 / (interactive_count as f32 + 1.0))).clamp(0.0, 1.0);
                section.metrics.interactivity_score = interactivity_score;
            } else {
                section.metrics.interactivity_score = 0.0;
            }
        }
    }

    /// Extract plain text content from HTML
    fn extract_text_content(&self, html: &str) -> String {
        // This is a simplified implementation
        // In a real-world scenario, you'd use a more robust approach

        // Parse HTML and extract text
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut html.as_bytes())
            .unwrap_or_else(|_| RcDom::default());

        let mut text = String::new();
        extract_text_from_node(&dom.document, &mut text);

        text
    }

    /// Create chunks based on section importance
    fn create_importance_based_chunks(
        &self,
        sections: Vec<SemanticSection>,
    ) -> Result<Vec<EnhancedChunk>> {
        debug!(
            "Creating chunks based on importance scores for {} sections",
            sections.len()
        );

        // Sort sections by importance score (highest to lowest)
        let mut sorted_sections = sections;
        sorted_sections.sort_by(|a, b| {
            b.metrics
                .overall_score
                .partial_cmp(&a.metrics.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create a map of section IDs to their indices
        let section_map: HashMap<String, usize> = sorted_sections
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.clone(), i))
            .collect();

        // Keep track of which sections have been allocated to chunks
        let mut allocated_sections: HashSet<String> = HashSet::new();

        // Create chunks
        let mut chunks: Vec<EnhancedChunk> = Vec::new();
        let mut current_chunk_content = String::new();
        let mut current_chunk_sections: Vec<String> = Vec::new();
        let mut current_token_count = 0;

        // Process sorted sections
        for section in &sorted_sections {
            // Skip already allocated sections
            if allocated_sections.contains(&section.id) {
                continue;
            }

            // If this section should be preserved (not split) and it's too large,
            // create a dedicated chunk for it
            if section.should_preserve(&self.config)
                && section.metrics.token_count > self.config.max_chunk_size
            {
                // Create a dedicated chunk for this large section
                let chunk_id = chunks.len() + 1;
                let chunk = self.create_chunk(
                    chunk_id,
                    &section.content,
                    vec![section.id.clone()],
                    section.metrics.token_count,
                )?;

                chunks.push(chunk);
                allocated_sections.insert(section.id.clone());
                continue;
            }

            // Check if adding this section would exceed max chunk size
            if current_token_count + section.metrics.token_count > self.config.max_chunk_size
                && !current_chunk_sections.is_empty()
            {
                // Finalize current chunk
                let chunk_id = chunks.len() + 1;
                let chunk = self.create_chunk(
                    chunk_id,
                    &current_chunk_content,
                    current_chunk_sections.clone(),
                    current_token_count,
                )?;

                chunks.push(chunk);

                // Reset for next chunk
                current_chunk_content = String::new();
                current_chunk_sections = Vec::new();
                current_token_count = 0;
            }

            // Add section to current chunk
            current_chunk_content.push_str(&section.content);
            current_chunk_sections.push(section.id.clone());
            current_token_count += section.metrics.token_count;
            allocated_sections.insert(section.id.clone());

            // Try to add closely related sections (children or siblings)
            self.add_related_sections(
                &section.id,
                &sorted_sections,
                &section_map,
                &mut allocated_sections,
                &mut current_chunk_content,
                &mut current_chunk_sections,
                &mut current_token_count,
            );
        }

        // Add any remaining sections in document order if needed
        if allocated_sections.len() < sorted_sections.len() {
            let mut document_ordered_sections = sorted_sections.clone();
            document_ordered_sections.sort_by(|a, b| a.position.cmp(&b.position));

            for section in &document_ordered_sections {
                if allocated_sections.contains(&section.id) {
                    continue;
                }

                // Check if adding this section would exceed max chunk size
                if current_token_count + section.metrics.token_count > self.config.max_chunk_size
                    && !current_chunk_sections.is_empty()
                {
                    // Finalize current chunk
                    let chunk_id = chunks.len() + 1;
                    let chunk = self.create_chunk(
                        chunk_id,
                        &current_chunk_content,
                        current_chunk_sections.clone(),
                        current_token_count,
                    )?;

                    chunks.push(chunk);

                    // Reset for next chunk
                    current_chunk_content = String::new();
                    current_chunk_sections = Vec::new();
                    current_token_count = 0;
                }

                // Add section to current chunk
                current_chunk_content.push_str(&section.content);
                current_chunk_sections.push(section.id.clone());
                current_token_count += section.metrics.token_count;
                allocated_sections.insert(section.id.clone());
            }
        }

        // Finalize last chunk if there's anything left
        if !current_chunk_sections.is_empty() {
            let chunk_id = chunks.len() + 1;
            let chunk = self.create_chunk(
                chunk_id,
                &current_chunk_content,
                current_chunk_sections,
                current_token_count,
            )?;

            chunks.push(chunk);
        }

        // Add overlaps between chunks if requested
        if self.config.overlap > 0 && chunks.len() > 1 {
            self.add_chunk_overlaps(&mut chunks, &sorted_sections, &section_map)?;
        }

        debug!("Created {} chunks based on importance scores", chunks.len());
        Ok(chunks)
    }

    /// Add related sections to current chunk
    fn add_related_sections(
        &self,
        section_id: &str,
        sections: &[SemanticSection],
        section_map: &HashMap<String, usize>,
        allocated_sections: &mut HashSet<String>,
        current_chunk_content: &mut String,
        current_chunk_sections: &mut Vec<String>,
        current_token_count: &mut usize,
    ) {
        // Find section in the sections list
        let section_idx = match section_map.get(section_id) {
            Some(&idx) => idx,
            None => return,
        };

        let section = &sections[section_idx];

        // Try to add child sections first
        for child_id in &section.child_ids {
            // Skip already allocated sections
            if allocated_sections.contains(child_id) {
                continue;
            }

            // Find child section
            let child_idx = match section_map.get(child_id) {
                Some(&idx) => idx,
                None => continue,
            };

            let child = &sections[child_idx];

            // Check if adding this section would exceed max chunk size
            if *current_token_count + child.metrics.token_count > self.config.max_chunk_size {
                continue;
            }

            // Add child to current chunk
            current_chunk_content.push_str(&child.content);
            current_chunk_sections.push(child.id.clone());
            *current_token_count += child.metrics.token_count;
            allocated_sections.insert(child.id.clone());

            // Recursively try to add child's related sections
            self.add_related_sections(
                child_id,
                sections,
                section_map,
                allocated_sections,
                current_chunk_content,
                current_chunk_sections,
                current_token_count,
            );
        }

        // Try to add sibling sections with parent
        if let Some(ref parent_id) = section.parent_id {
            // Find parent section
            let parent_idx = match section_map.get(parent_id) {
                Some(&idx) => idx,
                None => return,
            };

            let parent = &sections[parent_idx];

            // Process siblings (same parent)
            for sibling_id in &parent.child_ids {
                // Skip already allocated sections and the current section
                if allocated_sections.contains(sibling_id) || sibling_id == section_id {
                    continue;
                }

                // Find sibling section
                let sibling_idx = match section_map.get(sibling_id) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let sibling = &sections[sibling_idx];

                // Check if adding this section would exceed max chunk size
                if *current_token_count + sibling.metrics.token_count > self.config.max_chunk_size {
                    continue;
                }

                // Add sibling to current chunk
                current_chunk_content.push_str(&sibling.content);
                current_chunk_sections.push(sibling.id.clone());
                *current_token_count += sibling.metrics.token_count;
                allocated_sections.insert(sibling.id.clone());
            }
        }
    }

    /// Create a chunk with the given content and metadata
    fn create_chunk(
        &self,
        chunk_id: usize,
        content: &str,
        section_ids: Vec<String>,
        token_count: usize,
    ) -> Result<EnhancedChunk> {
        // Create base chunk
        let chunk = Chunk {
            id: chunk_id,
            content: content.to_string(),
            token_count,
        };

        // Create metadata
        let metadata = ChunkMetadata {
            section_ids,
            chunk_type: crate::chunking::chunk::ChunkType::Semantic,
            position: chunk_id,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
            related_chunk_ids: Vec::new(),
        };

        // Create enhanced chunk
        let enhanced_chunk = EnhancedChunk {
            chunk,
            metadata,
            relations: Vec::new(),
            context: None,
        };

        Ok(enhanced_chunk)
    }

    /// Add overlaps between adjacent chunks
    fn add_chunk_overlaps(
        &self,
        chunks: &mut Vec<EnhancedChunk>,
        sections: &[SemanticSection],
        section_map: &HashMap<String, usize>,
    ) -> Result<()> {
        if chunks.len() <= 1 {
            return Ok(());
        }

        // Process chunks in order
        for i in 0..chunks.len() - 1 {
            let current_chunk = &chunks[i];
            let next_chunk = &chunks[i + 1];

            // Create overlap by adding start of next chunk to end of current
            let mut overlap_content = String::new();
            let mut overlap_tokens = 0;

            // Get sections from the next chunk for overlap
            let section_ids = &next_chunk.metadata.section_ids;

            // Take sections from the beginning of the next chunk up to overlap size
            for section_id in section_ids {
                // Skip if section doesn't exist
                let section_idx = match section_map.get(section_id) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let section = &sections[section_idx];

                // Add section content to overlap if there's room
                if overlap_tokens + section.metrics.token_count <= self.config.overlap {
                    overlap_content.push_str(&section.content);
                    overlap_tokens += section.metrics.token_count;
                } else {
                    // If adding full section exceeds overlap, add partial content
                    let remaining_tokens = self.config.overlap - overlap_tokens;

                    if remaining_tokens > 0 {
                        // This would require more complex logic to split HTML correctly
                        // For now, we'll just stop adding content
                        break;
                    }
                }

                // Stop if we've reached the overlap size
                if overlap_tokens >= self.config.overlap {
                    break;
                }
            }

            // Add overlap content to current chunk
            if !overlap_content.is_empty() {
                let mut updated_chunk = chunks[i].clone();
                updated_chunk
                    .chunk
                    .content
                    .push_str("\n<!-- Overlap with next chunk -->\n");
                updated_chunk.chunk.content.push_str(&overlap_content);
                updated_chunk.chunk.token_count += overlap_tokens;

                // Add relation to next chunk
                updated_chunk.relations.push(ChunkRelation {
                    related_chunk_id: next_chunk.chunk.id,
                    relation_type: "next".to_string(),
                });

                chunks[i] = updated_chunk;

                // Also update the next chunk with a reference to this one
                let mut updated_next_chunk = chunks[i + 1].clone();
                updated_next_chunk.relations.push(ChunkRelation {
                    related_chunk_id: current_chunk.chunk.id,
                    relation_type: "previous".to_string(),
                });

                chunks[i + 1] = updated_next_chunk;
            }
        }

        Ok(())
    }
}

/// Count interactive elements in an HTML string
fn count_interactive_elements(html: &str) -> usize {
    // This is a simplified implementation
    // In a real-world scenario, you'd use a more robust approach

    let mut count = 0;

    // Count buttons
    count += html.matches("<button").count();

    // Count inputs
    count += html.matches("<input").count();

    // Count anchors
    count += html.matches("<a ").count();

    // Count selects
    count += html.matches("<select").count();

    // Count textareas
    count += html.matches("<textarea").count();

    // Count elements with onclick attributes
    count += html.matches("onclick=").count();

    // Count elements with role="button"
    count += html.matches("role=\"button\"").count();

    count
}

/// Extract text content from a DOM node
fn extract_text_from_node(node: &Handle, text: &mut String) {
    let node_data = &node.data;

    match *node_data {
        NodeData::Text { ref contents } => {
            text.push_str(&contents.borrow());
        }
        _ => {
            // Process child nodes
            for child in node.children.borrow().iter() {
                extract_text_from_node(child, text);
            }
        }
    }
}

/// Format for debug output
impl fmt::Display for ImportanceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Score: {:.2} (H: {:.2}, C: {:.2}, I: {:.2}, Tokens: {}{})",
            self.overall_score,
            self.hierarchy_score,
            self.content_score,
            self.interactivity_score,
            self.token_count,
            if self.is_priority { ", Priority" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_importance_metrics_calculation() {
        // Create ImportanceConfig
        let config = ImportanceConfig::default();

        // Create ImportanceMetrics
        let mut metrics = ImportanceMetrics {
            hierarchy_score: 0.8,
            content_score: 0.6,
            interactivity_score: 0.4,
            overall_score: 0.0,
            token_count: 100,
            is_priority: false,
        };

        // Calculate overall score
        metrics.calculate_overall(&config);

        // Expected calculation:
        // (0.8 * 0.4) + (0.6 * 0.4) + (0.4 * 0.2) = 0.32 + 0.24 + 0.08 = 0.64
        assert!((metrics.overall_score - 0.64).abs() < 0.001);

        // Test with priority flag
        metrics.is_priority = true;
        metrics.calculate_overall(&config);

        // Expected calculation with priority:
        // ((0.8 * 0.4) + (0.6 * 0.4) + (0.4 * 0.2) + 1.0) / 2 = (0.64 + 1.0) / 2 = 0.82
        assert!((metrics.overall_score - 0.82).abs() < 0.001);
    }

    // Add more tests as needed
}
