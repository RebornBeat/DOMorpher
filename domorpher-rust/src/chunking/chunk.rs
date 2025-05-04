//! # Chunk Data Structure
//!
//! Defines the core data structures for representing HTML chunks and their relationships.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::dom::analyzer::DomNode;
use crate::error::Result;

/// Type of content contained in a chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChunkType {
    /// Full HTML document
    Document,
    /// HTML element and its children
    Element,
    /// Text content
    Text,
    /// Table structure
    Table,
    /// List structure
    List,
    /// Form structure
    Form,
    /// Navigation section
    Navigation,
    /// Header section
    Header,
    /// Footer section
    Footer,
    /// Main content area
    MainContent,
    /// Sidebar content
    Sidebar,
    /// Custom defined section
    Custom(u32),
}

impl fmt::Display for ChunkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkType::Document => write!(f, "Document"),
            ChunkType::Element => write!(f, "Element"),
            ChunkType::Text => write!(f, "Text"),
            ChunkType::Table => write!(f, "Table"),
            ChunkType::List => write!(f, "List"),
            ChunkType::Form => write!(f, "Form"),
            ChunkType::Navigation => write!(f, "Navigation"),
            ChunkType::Header => write!(f, "Header"),
            ChunkType::Footer => write!(f, "Footer"),
            ChunkType::MainContent => write!(f, "MainContent"),
            ChunkType::Sidebar => write!(f, "Sidebar"),
            ChunkType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Relationship between chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChunkRelation {
    /// Parent-child relationship
    Parent,
    /// Child-parent relationship
    Child,
    /// Sibling relationship
    Sibling,
    /// Previous chunk in sequence
    Previous,
    /// Next chunk in sequence
    Next,
    /// Referenced by this chunk
    References,
    /// References this chunk
    ReferencedBy,
    /// Overlapping content
    Overlaps,
    /// Custom relationship
    Custom(u32),
}

impl fmt::Display for ChunkRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkRelation::Parent => write!(f, "Parent"),
            ChunkRelation::Child => write!(f, "Child"),
            ChunkRelation::Sibling => write!(f, "Sibling"),
            ChunkRelation::Previous => write!(f, "Previous"),
            ChunkRelation::Next => write!(f, "Next"),
            ChunkRelation::References => write!(f, "References"),
            ChunkRelation::ReferencedBy => write!(f, "ReferencedBy"),
            ChunkRelation::Overlaps => write!(f, "Overlaps"),
            ChunkRelation::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Estimated token count
    pub token_count: usize,
    /// Original position in the document
    pub position: ChunkPosition,
    /// Content type
    pub content_type: ChunkType,
    /// Importance score (0.0-1.0)
    pub importance: f32,
    /// Source URL or identifier
    pub source: Option<String>,
    /// Element path (XPath or CSS selector)
    pub element_path: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

impl Default for ChunkMetadata {
    fn default() -> Self {
        Self {
            token_count: 0,
            position: ChunkPosition::default(),
            content_type: ChunkType::Element,
            importance: 0.5,
            source: None,
            element_path: None,
            created_at: Utc::now(),
            custom_fields: HashMap::new(),
        }
    }
}

/// Position information for a chunk
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkPosition {
    /// Index in the original sequence
    pub index: usize,
    /// Start position (character offset)
    pub start: usize,
    /// End position (character offset)
    pub end: usize,
    /// Depth in the document tree
    pub depth: usize,
    /// Parent element index if available
    pub parent_index: Option<usize>,
}

impl Default for ChunkPosition {
    fn default() -> Self {
        Self {
            index: 0,
            start: 0,
            end: 0,
            depth: 0,
            parent_index: None,
        }
    }
}

/// Basic chunk structure representing a portion of HTML content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for the chunk
    pub id: String,
    /// HTML content
    pub content: String,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    /// Related chunks by ID and relationship type
    pub relations: HashMap<String, ChunkRelation>,
}

impl Chunk {
    /// Create a new chunk with the given content
    pub fn new(content: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            metadata: ChunkMetadata::default(),
            relations: HashMap::new(),
        }
    }

    /// Create a new chunk with content and metadata
    pub fn with_metadata(content: &str, metadata: ChunkMetadata) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            metadata,
            relations: HashMap::new(),
        }
    }

    /// Add a relation to another chunk
    pub fn add_relation(&mut self, chunk_id: &str, relation: ChunkRelation) {
        self.relations.insert(chunk_id.to_string(), relation);
    }

    /// Get the estimated token count
    pub fn token_count(&self) -> usize {
        self.metadata.token_count
    }

    /// Get the importance score
    pub fn importance(&self) -> f32 {
        self.metadata.importance
    }

    /// Set the importance score
    pub fn set_importance(&mut self, importance: f32) {
        self.metadata.importance = importance.max(0.0).min(1.0);
    }

    /// Get all related chunks by relation type
    pub fn get_related_by_type(&self, relation_type: ChunkRelation) -> Vec<String> {
        self.relations
            .iter()
            .filter(|(_, rel)| **rel == relation_type)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Check if this chunk has a specific relationship with another chunk
    pub fn has_relation_with(&self, chunk_id: &str, relation: ChunkRelation) -> bool {
        self.relations
            .get(chunk_id)
            .map_or(false, |rel| *rel == relation)
    }

    /// Get the element path if available
    pub fn element_path(&self) -> Option<&str> {
        self.metadata.element_path.as_deref()
    }

    /// Set the element path
    pub fn set_element_path(&mut self, path: &str) {
        self.metadata.element_path = Some(path.to_string());
    }

    /// Convert to an enhanced chunk with context
    pub fn to_enhanced(&self, context: ChunkContext) -> EnhancedChunk {
        EnhancedChunk {
            chunk: self.clone(),
            context,
        }
    }
}

impl PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Chunk {}

impl Hash for Chunk {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Contextual information for a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkContext {
    /// Brief summary of the chunk content
    pub summary: String,
    /// Global document context
    pub document_context: HashMap<String, String>,
    /// Preceding content summary
    pub preceding_context: Option<String>,
    /// Following content summary
    pub following_context: Option<String>,
    /// Ancestor elements summary
    pub ancestors_context: Option<String>,
    /// Document title if available
    pub document_title: Option<String>,
    /// URL source if available
    pub source_url: Option<String>,
    /// Section headings or breadcrumbs
    pub breadcrumbs: Vec<String>,
}

impl ChunkContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context with document title and URL
    pub fn with_document_info(title: Option<String>, url: Option<String>) -> Self {
        Self {
            document_title: title,
            source_url: url,
            ..Default::default()
        }
    }

    /// Add breadcrumb path
    pub fn add_breadcrumb(&mut self, crumb: &str) {
        self.breadcrumbs.push(crumb.to_string());
    }

    /// Set the preceding context
    pub fn set_preceding_context(&mut self, context: &str) {
        self.preceding_context = Some(context.to_string());
    }

    /// Set the following context
    pub fn set_following_context(&mut self, context: &str) {
        self.following_context = Some(context.to_string());
    }

    /// Add document context key-value pair
    pub fn add_document_context(&mut self, key: &str, value: &str) {
        self.document_context
            .insert(key.to_string(), value.to_string());
    }

    /// Get the breadcrumb path as a string
    pub fn breadcrumb_path(&self) -> String {
        self.breadcrumbs.join(" > ")
    }
}

/// Enhanced chunk with contextual information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedChunk {
    /// The base chunk
    pub chunk: Chunk,
    /// Additional context for the chunk
    pub context: ChunkContext,
}

impl EnhancedChunk {
    /// Create a new enhanced chunk from a base chunk and context
    pub fn new(chunk: Chunk, context: ChunkContext) -> Self {
        Self { chunk, context }
    }

    /// Get the HTML content
    pub fn content(&self) -> &str {
        &self.chunk.content
    }

    /// Get the chunk ID
    pub fn id(&self) -> &str {
        &self.chunk.id
    }

    /// Get the metadata
    pub fn metadata(&self) -> &ChunkMetadata {
        &self.chunk.metadata
    }

    /// Get the importance score
    pub fn importance(&self) -> f32 {
        self.chunk.importance()
    }

    /// Get the contextual representation for LLM prompting
    pub fn to_contextual_representation(&self) -> String {
        let mut result = String::new();

        // Add document context if available
        if let Some(title) = &self.context.document_title {
            result.push_str(&format!("Document Title: {}\n\n", title));
        }

        if let Some(url) = &self.context.source_url {
            result.push_str(&format!("Source URL: {}\n\n", url));
        }

        // Add breadcrumbs if available
        if !self.context.breadcrumbs.is_empty() {
            result.push_str(&format!("Location: {}\n\n", self.context.breadcrumb_path()));
        }

        // Add preceding context if available
        if let Some(preceding) = &self.context.preceding_context {
            result.push_str(&format!("Previous Content: {}\n\n", preceding));
        }

        // Add the actual content
        result.push_str("Content:\n");
        result.push_str(&self.chunk.content);
        result.push_str("\n\n");

        // Add following context if available
        if let Some(following) = &self.context.following_context {
            result.push_str(&format!("Following Content: {}\n\n", following));
        }

        result
    }

    /// Get a formatted summary of the chunk
    pub fn summary(&self) -> String {
        let content_preview = if self.chunk.content.len() > 50 {
            format!("{}...", &self.chunk.content[..50])
        } else {
            self.chunk.content.clone()
        };

        format!(
            "Chunk {} - Type: {}, Importance: {:.2}, Tokens: {}, Content: {}",
            self.chunk.id,
            self.chunk.metadata.content_type,
            self.chunk.importance(),
            self.chunk.token_count(),
            content_preview
        )
    }
}

/// Hierarchical tree structure of chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTree {
    /// Root chunks
    pub roots: Vec<Arc<Chunk>>,
    /// All chunks by ID
    pub chunks: HashMap<String, Arc<Chunk>>,
    /// Child relationships
    pub children: HashMap<String, Vec<String>>,
    /// Parent relationships
    pub parents: HashMap<String, String>,
    /// Chunk ordering
    pub order: Vec<String>,
}

impl ChunkTree {
    /// Create a new empty chunk tree
    pub fn new() -> Self {
        Self {
            roots: Vec::new(),
            chunks: HashMap::new(),
            children: HashMap::new(),
            parents: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add a chunk to the tree
    pub fn add_chunk(&mut self, chunk: Chunk, parent_id: Option<&str>) -> Arc<Chunk> {
        let chunk_id = chunk.id.clone();
        let chunk_arc = Arc::new(chunk);

        // Add to all chunks
        self.chunks.insert(chunk_id.clone(), chunk_arc.clone());

        // Add to order
        self.order.push(chunk_id.clone());

        // Set up parent-child relationships
        if let Some(parent) = parent_id {
            self.parents.insert(chunk_id.clone(), parent.to_string());
            self.children
                .entry(parent.to_string())
                .or_insert_with(Vec::new)
                .push(chunk_id.clone());
        } else {
            // This is a root chunk
            self.roots.push(chunk_arc.clone());
        }

        chunk_arc
    }

    /// Get a chunk by ID
    pub fn get_chunk(&self, id: &str) -> Option<&Arc<Chunk>> {
        self.chunks.get(id)
    }

    /// Get all children of a chunk
    pub fn get_children(&self, id: &str) -> Vec<&Arc<Chunk>> {
        if let Some(child_ids) = self.children.get(id) {
            child_ids
                .iter()
                .filter_map(|child_id| self.chunks.get(child_id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the parent of a chunk
    pub fn get_parent(&self, id: &str) -> Option<&Arc<Chunk>> {
        self.parents
            .get(id)
            .and_then(|parent_id| self.chunks.get(parent_id))
    }

    /// Get chunks in order
    pub fn get_ordered_chunks(&self) -> Vec<&Arc<Chunk>> {
        self.order
            .iter()
            .filter_map(|id| self.chunks.get(id))
            .collect()
    }

    /// Traverse the tree in depth-first order
    pub fn traverse_depth_first<F>(&self, f: &mut F)
    where
        F: FnMut(&Arc<Chunk>, usize),
    {
        for root in &self.roots {
            self.traverse_node(root, 0, f);
        }
    }

    /// Helper method for depth-first traversal
    fn traverse_node<F>(&self, node: &Arc<Chunk>, depth: usize, f: &mut F)
    where
        F: FnMut(&Arc<Chunk>, usize),
    {
        f(node, depth);

        if let Some(children) = self.children.get(&node.id) {
            for child_id in children {
                if let Some(child) = self.chunks.get(child_id) {
                    self.traverse_node(child, depth + 1, f);
                }
            }
        }
    }

    /// Get the total number of chunks
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Create enhanced chunks with context
    pub fn create_enhanced_chunks(&self) -> Vec<EnhancedChunk> {
        let mut enhanced_chunks = Vec::new();

        for chunk_id in &self.order {
            if let Some(chunk) = self.chunks.get(chunk_id) {
                let mut context = ChunkContext::new();

                // Add breadcrumbs by traversing up the parent chain
                let mut current_id = chunk_id;
                let mut breadcrumbs = Vec::new();

                while let Some(parent_id) = self.parents.get(current_id) {
                    if let Some(parent) = self.chunks.get(parent_id) {
                        if let Some(heading) = extract_heading(&parent.content) {
                            breadcrumbs.insert(0, heading);
                        }
                    }
                    current_id = parent_id;
                }

                for crumb in breadcrumbs {
                    context.add_breadcrumb(&crumb);
                }

                // Find previous and next chunks in order
                let chunk_index = self.order.iter().position(|id| id == chunk_id);
                if let Some(index) = chunk_index {
                    // Previous chunk
                    if index > 0 {
                        let prev_id = &self.order[index - 1];
                        if let Some(prev_chunk) = self.chunks.get(prev_id) {
                            context.set_preceding_context(&summarize_content(&prev_chunk.content));
                        }
                    }

                    // Next chunk
                    if index < self.order.len() - 1 {
                        let next_id = &self.order[index + 1];
                        if let Some(next_chunk) = self.chunks.get(next_id) {
                            context.set_following_context(&summarize_content(&next_chunk.content));
                        }
                    }
                }

                // Create enhanced chunk
                enhanced_chunks.push(EnhancedChunk::new((**chunk).clone(), context));
            }
        }

        enhanced_chunks
    }
}

/// Extract a heading from HTML content
fn extract_heading(html: &str) -> Option<String> {
    // Simple regex-based extraction of headings
    lazy_static::lazy_static! {
        static ref HEADING_RE: regex::Regex = regex::Regex::new(
            r"<h[1-6][^>]*>(.*?)</h[1-6]>"
        ).unwrap();
    }

    HEADING_RE
        .captures(html)
        .and_then(|caps| caps.get(1))
        .map(|m| {
            // Strip any HTML tags from the heading text
            let text = m.as_str();
            lazy_static::lazy_static! {
                static ref TAG_RE: regex::Regex = regex::Regex::new(
                    r"<[^>]+>"
                ).unwrap();
            }
            TAG_RE.replace_all(text, "").to_string()
        })
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
