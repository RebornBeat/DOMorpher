//! HTML to Semantic Markdown conversion
//!
//! This module provides functionality to convert HTML documents to semantic markdown
//! representation, preserving the document structure, content hierarchy, and semantic
//! meaning while removing unnecessary elements and normalizing the content for better
//! processing by language models.

use ego_tree::NodeRef;
use lazy_static::lazy_static;
use regex::{Regex, RegexBuilder};
use scraper::{ElementRef, Html, Node, Selector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use crate::error::DOMorpherError;
use crate::utils::encoding::decode_html_entities;

// Compile all selectors once
lazy_static! {
    // Main content selectors
    static ref MAIN_CONTENT_SELECTORS: Vec<Selector> = vec![
        Selector::parse("main").unwrap(),
        Selector::parse("article").unwrap(),
        Selector::parse("div[role='main']").unwrap(),
        Selector::parse("[itemprop='mainContentOfPage']").unwrap(),
        Selector::parse(".main-content").unwrap(),
        Selector::parse("#main-content").unwrap(),
        Selector::parse("#content").unwrap(),
        Selector::parse(".content").unwrap(),
        Selector::parse("div.post").unwrap(),
        Selector::parse("div.article").unwrap(),
    ];

    // Skip elements selectors
    static ref SKIP_ELEMENTS_SELECTORS: Vec<Selector> = vec![
        Selector::parse("script").unwrap(),
        Selector::parse("style").unwrap(),
        Selector::parse("noscript").unwrap(),
        Selector::parse("iframe").unwrap(),
        Selector::parse("svg").unwrap(),
        Selector::parse("canvas").unwrap(),
        Selector::parse("template").unwrap(),
        Selector::parse("link").unwrap(),
        Selector::parse("meta").unwrap(),
        Selector::parse("nav").unwrap(),
        Selector::parse("footer").unwrap(),
        Selector::parse("header:not(article header)").unwrap(),
        Selector::parse("aside").unwrap(),
        Selector::parse(".sidebar").unwrap(),
        Selector::parse(".widget").unwrap(),
        Selector::parse(".ad").unwrap(),
        Selector::parse(".advertisement").unwrap(),
        Selector::parse(".banner").unwrap(),
        Selector::parse(".share").unwrap(),
        Selector::parse(".social").unwrap(),
        Selector::parse(".comment").unwrap(),
        Selector::parse(".cookie").unwrap(),
        Selector::parse(".popup").unwrap(),
        Selector::parse(".modal").unwrap(),
        Selector::parse("form#search").unwrap(),
    ];

    // Heading selectors
    static ref H1_SELECTOR: Selector = Selector::parse("h1").unwrap();
    static ref H2_SELECTOR: Selector = Selector::parse("h2").unwrap();
    static ref H3_SELECTOR: Selector = Selector::parse("h3").unwrap();
    static ref H4_SELECTOR: Selector = Selector::parse("h4").unwrap();
    static ref H5_SELECTOR: Selector = Selector::parse("h5").unwrap();
    static ref H6_SELECTOR: Selector = Selector::parse("h6").unwrap();

    // List selectors
    static ref UL_SELECTOR: Selector = Selector::parse("ul").unwrap();
    static ref OL_SELECTOR: Selector = Selector::parse("ol").unwrap();
    static ref LI_SELECTOR: Selector = Selector::parse("li").unwrap();
    static ref DL_SELECTOR: Selector = Selector::parse("dl").unwrap();
    static ref DT_SELECTOR: Selector = Selector::parse("dt").unwrap();
    static ref DD_SELECTOR: Selector = Selector::parse("dd").unwrap();

    // Table selectors
    static ref TABLE_SELECTOR: Selector = Selector::parse("table").unwrap();
    static ref THEAD_SELECTOR: Selector = Selector::parse("thead").unwrap();
    static ref TBODY_SELECTOR: Selector = Selector::parse("tbody").unwrap();
    static ref TR_SELECTOR: Selector = Selector::parse("tr").unwrap();
    static ref TH_SELECTOR: Selector = Selector::parse("th").unwrap();
    static ref TD_SELECTOR: Selector = Selector::parse("td").unwrap();
    static ref CAPTION_SELECTOR: Selector = Selector::parse("caption").unwrap();

    // Inline element selectors
    static ref A_SELECTOR: Selector = Selector::parse("a").unwrap();
    static ref IMG_SELECTOR: Selector = Selector::parse("img").unwrap();
    static ref STRONG_SELECTOR: Selector = Selector::parse("strong, b").unwrap();
    static ref EM_SELECTOR: Selector = Selector::parse("em, i").unwrap();
    static ref CODE_SELECTOR: Selector = Selector::parse("code").unwrap();
    static ref PRE_SELECTOR: Selector = Selector::parse("pre").unwrap();
    static ref BLOCKQUOTE_SELECTOR: Selector = Selector::parse("blockquote").unwrap();
    static ref HR_SELECTOR: Selector = Selector::parse("hr").unwrap();
    static ref BR_SELECTOR: Selector = Selector::parse("br").unwrap();

    // Block level selectors
    static ref P_SELECTOR: Selector = Selector::parse("p").unwrap();
    static ref DIV_SELECTOR: Selector = Selector::parse("div").unwrap();
    static ref SECTION_SELECTOR: Selector = Selector::parse("section").unwrap();
    static ref ARTICLE_SELECTOR: Selector = Selector::parse("article").unwrap();
    static ref FIGURE_SELECTOR: Selector = Selector::parse("figure").unwrap();
    static ref FIGCAPTION_SELECTOR: Selector = Selector::parse("figcaption").unwrap();

    // Form selectors
    static ref FORM_SELECTOR: Selector = Selector::parse("form").unwrap();
    static ref INPUT_SELECTOR: Selector = Selector::parse("input").unwrap();
    static ref TEXTAREA_SELECTOR: Selector = Selector::parse("textarea").unwrap();
    static ref SELECT_SELECTOR: Selector = Selector::parse("select").unwrap();
    static ref OPTION_SELECTOR: Selector = Selector::parse("option").unwrap();
    static ref BUTTON_SELECTOR: Selector = Selector::parse("button").unwrap();
    static ref LABEL_SELECTOR: Selector = Selector::parse("label").unwrap();

    // Special elements
    static ref DETAILS_SELECTOR: Selector = Selector::parse("details").unwrap();
    static ref SUMMARY_SELECTOR: Selector = Selector::parse("summary").unwrap();
    static ref TIME_SELECTOR: Selector = Selector::parse("time").unwrap();
    static ref ABBR_SELECTOR: Selector = Selector::parse("abbr").unwrap();
    static ref CITE_SELECTOR: Selector = Selector::parse("cite").unwrap();
    static ref Q_SELECTOR: Selector = Selector::parse("q").unwrap();
    static ref MARK_SELECTOR: Selector = Selector::parse("mark").unwrap();
    static ref INS_SELECTOR: Selector = Selector::parse("ins").unwrap();
    static ref DEL_SELECTOR: Selector = Selector::parse("del").unwrap();
    static ref SUB_SELECTOR: Selector = Selector::parse("sub").unwrap();
    static ref SUP_SELECTOR: Selector = Selector::parse("sup").unwrap();

    // Metadata selectors
    static ref META_TITLE_SELECTOR: Selector = Selector::parse("title").unwrap();
    static ref META_DESCRIPTION_SELECTOR: Selector = Selector::parse("meta[name='description']").unwrap();
    static ref META_KEYWORDS_SELECTOR: Selector = Selector::parse("meta[name='keywords']").unwrap();
    static ref META_AUTHOR_SELECTOR: Selector = Selector::parse("meta[name='author']").unwrap();
    static ref META_OG_TITLE_SELECTOR: Selector = Selector::parse("meta[property='og:title']").unwrap();
    static ref META_OG_DESCRIPTION_SELECTOR: Selector = Selector::parse("meta[property='og:description']").unwrap();
    static ref META_OG_TYPE_SELECTOR: Selector = Selector::parse("meta[property='og:type']").unwrap();

    // Cleanup patterns
    static ref WHITESPACE_PATTERN: Regex = Regex::new(r"\s+").unwrap();
    static ref MULTI_NEWLINE_PATTERN: Regex = Regex::new(r"\n{3,}").unwrap();
    static ref EMPTY_BRACKETS_PATTERN: Regex = Regex::new(r"\[\s*\]").unwrap();
    static ref CONSECUTIVE_DASHES_PATTERN: Regex = Regex::new(r"-{2,}").unwrap();
    static ref TRAILING_PUNCTUATION_PATTERN: Regex = RegexBuilder::new(r"[,.;:!?]{2,}$")
        .multi_line(true)
        .build()
        .unwrap();
    static ref URL_PATTERN: Regex = Regex::new(
        r"(https?://|www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*(?:/[^\s]*)?"
    ).unwrap();
}

/// Markdown formatting options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkdownFormat {
    /// Standard Markdown format
    Standard,
    /// GitHub Flavored Markdown
    GitHub,
    /// CommonMark standard
    CommonMark,
    /// Simplified Markdown with minimal formatting
    Simplified,
    /// Semantic Markdown that emphasizes structure
    Semantic,
}

impl Default for MarkdownFormat {
    fn default() -> Self {
        MarkdownFormat::Semantic
    }
}

/// Configuration options for HTML to Markdown conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownOptions {
    /// Markdown format to use
    pub format: MarkdownFormat,

    /// Whether to extract main content only
    pub extract_main_content: bool,

    /// Whether to include metadata
    pub include_metadata: bool,

    /// Level of metadata to include
    pub metadata_level: MetadataLevel,

    /// Whether to include tables
    pub include_tables: bool,

    /// Whether to include images
    pub include_images: bool,

    /// Whether to include links
    pub include_links: bool,

    /// Whether to convert links to references
    pub refify_links: bool,

    /// Maximum heading level to use
    pub max_heading_level: u8,

    /// Whether to detect and preserve code blocks
    pub preserve_code_blocks: bool,

    /// Whether to enhance lists with markers
    pub enhance_lists: bool,

    /// Whether to enable table column tracking
    pub enable_table_column_tracking: bool,

    /// Elements to exclude (CSS selectors)
    pub exclude_elements: Vec<String>,

    /// Elements to include (CSS selectors, overrides exclude_elements)
    pub include_elements: Vec<String>,

    /// Maximum line length (0 = no limit)
    pub max_line_length: usize,

    /// Whether to add heading IDs
    pub add_heading_ids: bool,

    /// Whether to emphasize key information
    pub emphasize_key_info: bool,

    /// Whether to use HTML passthrough for complex elements
    pub html_passthrough: bool,

    /// Whether to simplify links (remove URL params)
    pub simplify_links: bool,

    /// Custom element handlers
    pub custom_handlers: HashMap<String, String>,

    /// Whether to add semantic labels to enhance structure
    pub add_semantic_labels: bool,

    /// Whether to normalize whitespace
    pub normalize_whitespace: bool,

    /// Whether to format inline text in JSON mode
    pub json_mode: bool,

    /// Whether to wrap tables
    pub wrap_tables: bool,

    /// Whether to add original element classes as metadata
    pub add_class_metadata: bool,

    /// Whether to add data attribute information
    pub include_data_attributes: bool,

    /// Whether to extract aria attributes
    pub extract_aria: bool,

    /// Custom post-processing function
    pub post_processing: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

impl Default for MarkdownOptions {
    fn default() -> Self {
        Self {
            format: MarkdownFormat::Semantic,
            extract_main_content: true,
            include_metadata: true,
            metadata_level: MetadataLevel::Basic,
            include_tables: true,
            include_images: true,
            include_links: true,
            refify_links: true,
            max_heading_level: 6,
            preserve_code_blocks: true,
            enhance_lists: true,
            enable_table_column_tracking: true,
            exclude_elements: Vec::new(),
            include_elements: Vec::new(),
            max_line_length: 0,
            add_heading_ids: false,
            emphasize_key_info: true,
            html_passthrough: false,
            simplify_links: true,
            custom_handlers: HashMap::new(),
            add_semantic_labels: true,
            normalize_whitespace: true,
            json_mode: false,
            wrap_tables: true,
            add_class_metadata: false,
            include_data_attributes: false,
            extract_aria: true,
            post_processing: None,
        }
    }
}

/// Level of metadata to include in the markdown output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataLevel {
    /// No metadata
    None,
    /// Basic metadata (title, description)
    Basic,
    /// Standard metadata (title, description, keywords, author)
    Standard,
    /// Extended metadata (all available metadata)
    Extended,
    /// Full metadata with structural information
    Full,
}

/// Metadata extracted from HTML document
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title
    pub title: Option<String>,

    /// Document description
    pub description: Option<String>,

    /// Document keywords
    pub keywords: Option<String>,

    /// Document author
    pub author: Option<String>,

    /// Open Graph title
    pub og_title: Option<String>,

    /// Open Graph description
    pub og_description: Option<String>,

    /// Open Graph type
    pub og_type: Option<String>,

    /// Publication date
    pub published_date: Option<String>,

    /// Modified date
    pub modified_date: Option<String>,

    /// Document language
    pub language: Option<String>,

    /// Document canonical URL
    pub canonical_url: Option<String>,

    /// Additional metadata
    pub additional: HashMap<String, String>,
}

/// Main HTML to Markdown converter
#[derive(Clone)]
pub struct HtmlToMarkdown {
    /// Configuration options
    options: MarkdownOptions,

    /// Custom selectors for element exclusion
    exclude_selectors: Vec<Selector>,

    /// Custom selectors for element inclusion
    include_selectors: Vec<Selector>,

    /// Element tracker for link references
    element_tracker: ElementTracker,

    /// Counters for generating unique references
    counters: HashMap<String, usize>,

    /// Current list processing state
    list_state: ListState,
}

/// Element tracker for reference-style links
#[derive(Clone, Default)]
struct ElementTracker {
    /// Link references
    links: HashMap<String, (String, String)>, // URL -> (reference, title)

    /// Image references
    images: HashMap<String, (String, String)>, // URL -> (reference, alt)

    /// Footnotes
    footnotes: HashMap<String, String>, // ID -> content

    /// Heading IDs
    heading_ids: HashMap<String, String>, // Heading text -> ID

    /// Structured data elements
    structured_data: HashMap<String, serde_json::Value>,
}

/// State for list processing
#[derive(Clone, Default)]
struct ListState {
    /// Current list level (nesting depth)
    level: usize,

    /// Whether current list is ordered
    is_ordered: Vec<bool>,

    /// Current item number in ordered list
    current_number: Vec<usize>,

    /// Whether in a definition list
    in_definition_list: bool,

    /// Whether processing a term in a definition list
    in_term: bool,
}

/// Table processing state
#[derive(Clone)]
struct TableState {
    /// Column alignments
    alignments: Vec<ColumnAlignment>,

    /// Column widths
    widths: Vec<usize>,

    /// Current row
    current_row: Vec<String>,

    /// Rows of cells
    rows: Vec<Vec<String>>,

    /// Has header
    has_header: bool,

    /// Column count
    column_count: usize,

    /// Caption
    caption: Option<String>,
}

impl Default for TableState {
    fn default() -> Self {
        Self {
            alignments: Vec::new(),
            widths: Vec::new(),
            current_row: Vec::new(),
            rows: Vec::new(),
            has_header: false,
            column_count: 0,
            caption: None,
        }
    }
}

/// Column alignment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnAlignment {
    /// Left-aligned
    Left,
    /// Center-aligned
    Center,
    /// Right-aligned
    Right,
}

impl Default for ColumnAlignment {
    fn default() -> Self {
        ColumnAlignment::Left
    }
}

impl HtmlToMarkdown {
    /// Create a new HTML to Markdown converter with the given options
    pub fn new(options: MarkdownOptions) -> Self {
        // Compile custom selectors
        let exclude_selectors = options
            .exclude_elements
            .iter()
            .filter_map(|sel| Selector::parse(sel).ok())
            .collect();

        let include_selectors = options
            .include_elements
            .iter()
            .filter_map(|sel| Selector::parse(sel).ok())
            .collect();

        Self {
            options,
            exclude_selectors,
            include_selectors,
            element_tracker: ElementTracker::default(),
            counters: HashMap::new(),
            list_state: ListState::default(),
        }
    }

    /// Create a new HTML to Markdown converter with default options
    pub fn default() -> Self {
        Self::new(MarkdownOptions::default())
    }

    /// Convert HTML to Markdown
    pub fn convert(&self, html: &str) -> Result<String, DOMorpherError> {
        // Parse HTML
        let document = Html::parse_document(html);

        // Extract metadata if needed
        let metadata = if self.options.include_metadata {
            self.extract_metadata(&document)
        } else {
            DocumentMetadata::default()
        };

        // Find main content if enabled
        let root_element = if self.options.extract_main_content {
            self.find_main_content(&document)
                .unwrap_or_else(|| document.root_element())
        } else {
            document.root_element()
        };

        // Process the document
        let mut output = String::new();

        // Add metadata header if needed
        if self.options.include_metadata && self.options.metadata_level != MetadataLevel::None {
            self.add_metadata_header(&mut output, &metadata);
        }

        // Process the elements
        let mut visitor = MarkdownVisitor {
            converter: self,
            output: &mut output,
            metadata: &metadata,
            in_pre_block: false,
            in_list_item: false,
            in_table: false,
            table_state: TableState::default(),
            indentation: 0,
            skip_count: 0,
            current_heading_level: 0,
            inside_link: false,
            inside_heading: false,
            link_text_buffer: String::new(),
            processed_elements: HashSet::new(),
        };

        visitor.process_node(ElementRef::wrap(root_element).unwrap());

        // Add references if needed
        if self.options.refify_links && !self.element_tracker.links.is_empty() {
            output.push_str("\n\n");

            for (url, (reference, title)) in &self.element_tracker.links {
                if title.is_empty() {
                    writeln!(output, "[{}]: {}", reference, url).unwrap();
                } else {
                    writeln!(output, "[{}]: {} \"{}\"", reference, url, title).unwrap();
                }
            }
        }

        // Add image references if needed
        if self.options.refify_links && !self.element_tracker.images.is_empty() {
            if self.element_tracker.links.is_empty() {
                output.push_str("\n\n");
            }

            for (url, (reference, _)) in &self.element_tracker.images {
                writeln!(output, "[{}]: {}", reference, url).unwrap();
            }
        }

        // Add footnotes if needed
        if !self.element_tracker.footnotes.is_empty() {
            output.push_str("\n\n");

            for (id, content) in &self.element_tracker.footnotes {
                writeln!(output, "[^{}]: {}", id, content).unwrap();
            }
        }

        // Clean up the output
        self.clean_output(&output)
    }

    /// Convert HTML string to Markdown string
    pub fn convert_string(&self, html: &str) -> Result<String, DOMorpherError> {
        self.convert(html)
    }

    /// Extract metadata from HTML document
    fn extract_metadata(&self, document: &Html) -> DocumentMetadata {
        let mut metadata = DocumentMetadata::default();

        // Extract basic metadata
        if let Some(title_el) = document.select(&META_TITLE_SELECTOR).next() {
            metadata.title = Some(title_el.inner_html());
        }

        if let Some(desc_el) = document.select(&META_DESCRIPTION_SELECTOR).next() {
            if let Some(content) = desc_el.value().attr("content") {
                metadata.description = Some(content.to_string());
            }
        }

        // Stop here if only basic metadata is needed
        if self.options.metadata_level == MetadataLevel::Basic {
            return metadata;
        }

        // Extract standard metadata
        if let Some(keywords_el) = document.select(&META_KEYWORDS_SELECTOR).next() {
            if let Some(content) = keywords_el.value().attr("content") {
                metadata.keywords = Some(content.to_string());
            }
        }

        if let Some(author_el) = document.select(&META_AUTHOR_SELECTOR).next() {
            if let Some(content) = author_el.value().attr("content") {
                metadata.author = Some(content.to_string());
            }
        }

        // Stop here if only standard metadata is needed
        if self.options.metadata_level == MetadataLevel::Standard {
            return metadata;
        }

        // Extract Open Graph metadata
        if let Some(og_title_el) = document.select(&META_OG_TITLE_SELECTOR).next() {
            if let Some(content) = og_title_el.value().attr("content") {
                metadata.og_title = Some(content.to_string());
            }
        }

        if let Some(og_desc_el) = document.select(&META_OG_DESCRIPTION_SELECTOR).next() {
            if let Some(content) = og_desc_el.value().attr("content") {
                metadata.og_description = Some(content.to_string());
            }
        }

        if let Some(og_type_el) = document.select(&META_OG_TYPE_SELECTOR).next() {
            if let Some(content) = og_type_el.value().attr("content") {
                metadata.og_type = Some(content.to_string());
            }
        }

        // Extract schema.org metadata
        let schema_selector = Selector::parse("script[type='application/ld+json']").unwrap();
        for schema_el in document.select(&schema_selector) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&schema_el.inner_html()) {
                if let Some(schema_type) = value.get("@type") {
                    if let Some(schema_type) = schema_type.as_str() {
                        self.extract_schema_metadata(schema_type, &value, &mut metadata);
                    }
                }
            }
        }

        // Extract dates
        // Try schema.org dates first
        if metadata.published_date.is_none() {
            // Try meta tags
            let published_selector = Selector::parse("meta[property='article:published_time']").unwrap();
            if let Some(el) = document.select(&published_selector).next() {
                if let Some(content) = el.value().attr("content") {
                    metadata.published_date = Some(content.to_string());
                }
            }
        }

        if metadata.modified_date.is_none() {
            let modified_selector = Selector::parse("meta[property='article:modified_time']").unwrap();
            if let Some(el) = document.select(&modified_selector).next() {
                if let Some(content) = el.value().attr("content") {
                    metadata.modified_date = Some(content.to_string());
                }
            }
        }

        // Extract language
        let html_selector = Selector::parse("html").unwrap();
        if let Some(html_el) = document.select(&html_selector).next() {
            if let Some(lang) = html_el.value().attr("lang") {
                metadata.language = Some(lang.to_string());
            }
        }

        // Extract canonical URL
        let canonical_selector = Selector::parse("link[rel='canonical']").unwrap();
        if let Some(el) = document.select(&canonical_selector).next() {
            if let Some(href) = el.value().attr("href") {
                metadata.canonical_url = Some(href.to_string());
            }
        }

        // Additional metadata for Extended or Full level
        if self.options.metadata_level == MetadataLevel::Extended ||
           self.options.metadata_level == MetadataLevel::Full {
            let all_meta_selector = Selector::parse("meta[name], meta[property]").unwrap();
            for meta_el in document.select(&all_meta_selector) {
                let name = meta_el.value().attr("name").or_else(|| meta_el.value().attr("property"));
                let content = meta_el.value().attr("content");

                if let (Some(name), Some(content)) = (name, content) {
                    // Skip already processed metadata
                    if !["description", "keywords", "author"].contains(&name) &&
                       !name.starts_with("og:") &&
                       !name.starts_with("article:") {
                        metadata.additional.insert(name.to_string(), content.to_string());
                    }
                }
            }
        }

        metadata
    }

    /// Extract schema.org metadata
    fn extract_schema_metadata(&self, schema_type: &str, value: &serde_json::Value, metadata: &mut DocumentMetadata) {
        match schema_type {
            "Article" | "NewsArticle" | "BlogPosting" => {
                // Extract article metadata
                if metadata.title.is_none() {
                    if let Some(title) = value.get("headline").and_then(|v| v.as_str()) {
                        metadata.title = Some(title.to_string());
                    }
                }

                if metadata.description.is_none() {
                    if let Some(desc) = value.get("description").and_then(|v| v.as_str()) {
                        metadata.description = Some(desc.to_string());
                    }
                }

                if metadata.author.is_none() {
                    if let Some(author) = value.get("author") {
                        if let Some(name) = author.get("name").and_then(|v| v.as_str()) {
                            metadata.author = Some(name.to_string());
                        }
                    }
                }

                if metadata.published_date.is_none() {
                    if let Some(date) = value.get("datePublished").and_then(|v| v.as_str()) {
                        metadata.published_date = Some(date.to_string());
                    }
                }

                if metadata.modified_date.is_none() {
                    if let Some(date) = value.get("dateModified").and_then(|v| v.as_str()) {
                        metadata.modified_date = Some(date.to_string());
                    }
                }
            },
            "Product" => {
                // Extract product metadata
                if metadata.title.is_none() {
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        metadata.title = Some(name.to_string());
                    }
                }

                if metadata.description.is_none() {
                    if let Some(desc) = value.get("description").and_then(|v| v.as_str()) {
                        metadata.description = Some(desc.to_string());
                    }
                }
            },
            _ => {
                // Extract generic metadata
                if metadata.title.is_none() {
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        metadata.title = Some(name.to_string());
                    }
                }

                if metadata.description.is_none() {
                    if let Some(desc) = value.get("description").and_then(|v| v.as_str()) {
                        metadata.description = Some(desc.to_string());
                    }
                }
            }
        }
    }

    /// Find main content node
    fn find_main_content<'a>(&self, document: &'a Html) -> Option<NodeRef<'a, Node>> {
        // Try each main content selector
        for selector in MAIN_CONTENT_SELECTORS.iter() {
            if let Some(element) = document.select(selector).next() {
                return Some(element.parent_node());
            }
        }

        // If no explicit main content found, try content heuristics

        // Find body
        let body_selector = Selector::parse("body").unwrap();
        let body = document.select(&body_selector).next()?;

        // Find the element with the most content
        let mut best_element = None;
        let mut best_score = 0;

        let paragraphs_selector = Selector::parse("p, article, section").unwrap();

        for element in document.select(&paragraphs_selector) {
            // Skip elements that should be excluded
            if self.should_skip_element(&element) {
                continue;
            }

            // Calculate content score
            let text_length = element.text().collect::<String>().len();
            let paragraph_count = element.select(&P_SELECTOR).count();

            // Score is text length weighted by paragraph count
            let score = text_length * (1 + paragraph_count);

            if score > best_score {
                best_score = score;
                best_element = Some(element.parent_node());
            }
        }

        best_element.or_else(|| Some(body.parent_node()))
    }

    /// Check if an element should be skipped
    fn should_skip_element(&self, element: &ElementRef) -> bool {
        // Check custom include selectors first (they override exclusions)
        for selector in &self.include_selectors {
            if element.matches(selector) {
                return false;
            }
        }

        // Check built-in skip selectors
        for selector in SKIP_ELEMENTS_SELECTORS.iter() {
            if element.matches(selector) {
                return true;
            }
        }

        // Check custom exclude selectors
        for selector in &self.exclude_selectors {
            if element.matches(selector) {
                return true;
            }
        }

        // Check for hidden elements
        if let Some(style) = element.value().attr("style") {
            if style.contains("display: none") || style.contains("visibility: hidden") {
                return true;
            }
        }

        // Check for hidden classes
        if let Some(class) = element.value().attr("class") {
            if class.contains("hidden") || class.contains("hide") || class.contains("invisible") {
                return true;
            }
        }

        false
    }

    /// Add metadata header to output
    fn add_metadata_header(&self, output: &mut String, metadata: &DocumentMetadata) {
        let format_level = self.options.metadata_level;

        if format_level == MetadataLevel::None {
            return;
        }

        // Add YAML-style metadata header
        output.push_str("---\n");

        // Add basic metadata
        if let Some(title) = &metadata.title {
            writeln!(output, "title: \"{}\"", escape_yaml_string(title)).unwrap();
        }

        if let Some(desc) = &metadata.description {
            writeln!(output, "description: \"{}\"", escape_yaml_string(desc)).unwrap();
        }

        // Add standard metadata
        if format_level >= MetadataLevel::Standard {
            if let Some(keywords) = &metadata.keywords {
                writeln!(output, "keywords: \"{}\"", escape_yaml_string(keywords)).unwrap();
            }

            if let Some(author) = &metadata.author {
                writeln!(output, "author: \"{}\"", escape_yaml_string(author)).unwrap();
            }
        }

        // Add extended metadata
        if format_level >= MetadataLevel::Extended {
            if let Some(published) = &metadata.published_date {
                writeln!(output, "date_published: {}", published).unwrap();
            }

            if let Some(modified) = &metadata.modified_date {
                writeln!(output, "date_modified: {}", modified).unwrap();
            }

            if let Some(language) = &metadata.language {
                writeln!(output, "language: {}", language).unwrap();
            }

            if let Some(url) = &metadata.canonical_url {
                writeln!(output, "url: {}", url).unwrap();
            }

            // Add additional metadata
            for (key, value) in &metadata.additional {
                // Skip technical metadata
                if !key.starts_with("fb:") && !key.starts_with("twitter:") {
                    writeln!(output, "{}: \"{}\"", key, escape_yaml_string(value)).unwrap();
                }
            }
        }

        // Add document type hint if available
        if let Some(og_type) = &metadata.og_type {
            writeln!(output, "document_type: {}", og_type).unwrap();
        }

        output.push_str("---\n\n");
    }

    /// Clean up the markdown output
    fn clean_output(&self, output: &str) -> Result<String, DOMorpherError> {
        let mut result = output.to_string();

        // Normalize whitespace if enabled
        if self.options.normalize_whitespace {
            // Replace multiple spaces with a single space
            result = WHITESPACE_PATTERN.replace_all(&result, " ").to_string();

            // Replace multiple newlines with at most two
            result = MULTI_NEWLINE_PATTERN.replace_all(&result, "\n\n").to_string();
        }

        // Clean up markdown artifacts
        result = EMPTY_BRACKETS_PATTERN.replace_all(&result, "").to_string();
        result = CONSECUTIVE_DASHES_PATTERN.replace_all(&result, "-").to_string();
        result = TRAILING_PUNCTUATION_PATTERN.replace_all(&result, "$1").to_string();

        // Apply line length limits if specified
        if self.options.max_line_length > 0 {
            result = self.wrap_text(&result, self.options.max_line_length);
        }

        // Apply custom post-processing if available
        if let Some(post_process) = &self.options.post_processing {
            result = post_process(&result);
        }

        Ok(result)
    }

    /// Wrap text to specified line length
    fn wrap_text(&self, text: &str, max_length: usize) -> String {
        let mut result = String::with_capacity(text.len());
        let mut current_line_length = 0;

        for line in text.lines() {
            if line.len() <= max_length {
                result.push_str(line);
                result.push('\n');
                continue;
            }

            current_line_length = 0;
            let mut words = line.split_whitespace().peekable();

            while let Some(word) = words.next() {
                if current_line_length == 0 {
                    // First word on line
                    result.push_str(word);
                    current_line_length = word.len();
                } else if current_line_length + word.len() + 1 <= max_length {
                    // Word fits on current line
                    result.push(' ');
                    result.push_str(word);
                    current_line_length += word.len() + 1;
                } else {
                    // Word doesn't fit, start new line
                    result.push('\n');
                    result.push_str(word);
                    current_line_length = word.len();
                }

                // Special cases
                if word.ends_with(':') && words.peek().is_some() {
                    // After a colon, prefer to start a new line
                    result.push('\n');
                    current_line_length = 0;
                }
            }

            result.push('\n');
        }

        result
    }

    /// Generate a unique reference for a link
    fn get_link_reference(&self, url: &str, title: &str) -> String {
        if let Some((reference, _)) = self.element_tracker.links.get(url) {
            reference.clone()
        } else {
            let mut reference = String::new();

            // Try to generate a reference from title
            if !title.is_empty() {
                let words: Vec<_> = title
                    .split_whitespace()
                    .take(3)
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            Some(c) => c.to_lowercase().collect::<String>(),
                            None => String::new(),
                        }
                    })
                    .collect();

                if !words.is_empty() {
                    reference = words.join("");
                }
            }

            // If reference is empty, use a numbered reference
            if reference.is_empty() {
                let counter = self.counters.entry("link".to_string()).or_insert(1);
                reference = format!("link{}", counter);
                *counter += 1;
            }

            reference
        }
    }

    /// Generate a unique reference for an image
    fn get_image_reference(&self, url: &str, alt: &str) -> String {
        if let Some((reference, _)) = self.element_tracker.images.get(url) {
            reference.clone()
        } else {
            let mut reference = String::new();

            // Try to generate a reference from alt text
            if !alt.is_empty() {
                let words: Vec<_> = alt
                    .split_whitespace()
                    .take(3)
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            Some(c) => c.to_lowercase().collect::<String>(),
                            None => String::new(),
                        }
                    })
                    .collect();

                if !words.is_empty() {
                    reference = words.join("");
                }
            }

            // If reference is empty, use a numbered reference
            if reference.is_empty() {
                let counter = self.counters.entry("image".to_string()).or_insert(1);
                reference = format!("img{}", counter);
                *counter += 1;
            }

            reference
        }
    }

    /// Get a unique ID for a heading
    fn get_heading_id(&self, text: &str) -> String {
        let text = text.trim();

        if let Some(id) = self.element_tracker.heading_ids.get(text) {
            return id.clone();
        }

        // Generate ID from text
        let id = text
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>();

        // Remove consecutive dashes
        let id = CONSECUTIVE_DASHES_PATTERN.replace_all(&id, "-").to_string();

        // Trim dashes from start and end
        let id = id.trim_matches('-').to_string();

        if id.is_empty() {
            let counter = self.counters.entry("heading".to_string()).or_insert(1);
            let id = format!("heading{}", counter);
            *counter += 1;
            id
        } else {
            id
        }
    }
}

/// Element visitor for generating markdown
struct MarkdownVisitor<'a, 'b> {
    /// Reference to the converter
    converter: &'a HtmlToMarkdown,

    /// Output buffer
    output: &'b mut String,

    /// Document metadata
    metadata: &'a DocumentMetadata,

    /// Whether currently in a pre block
    in_pre_block: bool,

    /// Whether currently in a list item
    in_list_item: bool,

    /// Whether currently in a table
    in_table: bool,

    /// Table processing state
    table_state: TableState,

    /// Current indentation level
    indentation: usize,

    /// Number of elements to skip
    skip_count: usize,

    /// Current heading level
    current_heading_level: u8,

    /// Whether currently inside a link
    inside_link: bool,

    /// Whether currently inside a heading
    inside_heading: bool,

    /// Buffer for link text
    link_text_buffer: String,

    /// Set of already processed elements
    processed_elements: HashSet<*const Node>,
}

impl<'a, 'b> MarkdownVisitor<'a, 'b> {
    /// Process a node and its children
    fn process_node(&mut self, node: ElementRef) {
        let node_ptr = node.value() as *const Node;

        // Skip if already processed
        if self.processed_elements.contains(&node_ptr) {
            return;
        }

        // Mark as processed
        self.processed_elements.insert(node_ptr);

        // Skip element if needed
        if self.converter.should_skip_element(&node) {
            self.skip_count += 1;
            return;
        }

        // Skip children if in a skip count
        if self.skip_count > 0 {
            self.skip_count -= 1;
            return;
        }

        match node.value() {
            Node::Element(element) => {
                let tag_name = element.name.local.as_ref();

                // Process element based on tag name
                match tag_name {
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => self.process_heading(&node, tag_name),
                    "p" => self.process_paragraph(&node),
                    "br" => self.process_line_break(),
                    "hr" => self.process_horizontal_rule(),
                    "ul" => self.process_unordered_list(&node),
                    "ol" => self.process_ordered_list(&node),
                    "li" => self.process_list_item(&node),
                    "dl" => self.process_definition_list(&node),
                    "dt" => self.process_definition_term(&node),
                    "dd" => self.process_definition_description(&node),
                    "a" => self.process_link(&node),
                    "img" => self.process_image(&node),
                    "table" => self.process_table(&node),
                    "thead" => self.process_table_head(&node),
                    "tbody" => self.process_table_body(&node),
                    "tr" => self.process_table_row(&node),
                    "th" => self.process_table_header_cell(&node),
                    "td" => self.process_table_cell(&node),
                    "caption" => self.process_table_caption(&node),
                    "pre" => self.process_pre(&node),
                    "code" => self.process_code(&node),
                    "blockquote" => self.process_blockquote(&node),
                    "strong" | "b" => self.process_strong(&node),
                    "em" | "i" => self.process_emphasis(&node),
                    "s" | "del" => self.process_strikethrough(&node),
                    "sub" => self.process_subscript(&node),
                    "sup" => self.process_superscript(&node),
                    "div" => self.process_div(&node),
                    "span" => self.process_span(&node),
                    "figure" => self.process_figure(&node),
                    "figcaption" => self.process_figcaption(&node),
                    "details" => self.process_details(&node),
                    "summary" => self.process_summary(&node),
                    "mark" => self.process_mark(&node),
                    "time" => self.process_time(&node),
                    "abbr" => self.process_abbreviation(&node),
                    "q" => self.process_quote(&node),
                    "cite" => self.process_citation(&node),
                    _ => self.process_generic_element(&node),
                }
            },
            Node::Text(text) => {
                // Process text differently depending on context
                let text = decode_html_entities(text);

                if self.in_pre_block {
                    // In pre block, preserve whitespace
                    self.output.push_str(&text);
                } else if self.inside_link {
                    // In link, collect text for reference
                    self.link_text_buffer.push_str(&text);
                    self.output.push_str(&text);
                } else {
                    // Normal text
                    if self.converter.options.normalize_whitespace {
                        let text = WHITESPACE_PATTERN.replace_all(&text, " ").to_string();
                        self.output.push_str(&text);
                    } else {
                        self.output.push_str(&text);
                    }
                }
            },
            Node::Comment(_) => {
                // Skip comments
            },
            _ => {
                // Skip other node types
            }
        }
    }

    /// Process a heading element
    fn process_heading(&mut self, node: &ElementRef, tag_name: &str) {
        // Get heading level
        let level = match tag_name {
            "h1" => 1,
            "h2" => 2,
            "h3" => 3,
            "h4" => 4,
            "h5" => 5,
            "h6" => 6,
            _ => 1, // Default to h1
        };

        // Skip if beyond max level
        if level as u8 > self.converter.options.max_heading_level {
            self.process_generic_element(node);
            return;
        }

        // Add newlines before heading
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Set heading mode flags
        self.inside_heading = true;
        self.current_heading_level = level as u8;
        let old_link_text = std::mem::take(&mut self.link_text_buffer);

        // Add semantic marker if enabled
        if self.converter.options.add_semantic_labels {
            if level == 1 {
                self.output.push_str("# [TITLE] ");
            } else {
                self.output.push_str(&format!("{} [SECTION] ", "#".repeat(level)));
            }
        } else {
            // Add heading markers based on format
            match self.converter.options.format {
                MarkdownFormat::Semantic => {
                    self.output.push_str(&format!("{} ", "#".repeat(level)));
                },
                MarkdownFormat::Standard | MarkdownFormat::GitHub | MarkdownFormat::CommonMark => {
                    self.output.push_str(&format!("{} ", "#".repeat(level)));
                },
                MarkdownFormat::Simplified => {
                    if level <= 2 {
                        self.output.push_str(&format!("{} ", "#".repeat(level)));
                    } else {
                        self.output.push_str(&format!("{}. ", "#".repeat(level - 2)));
                    }
                },
            }
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Add ID if enabled
        if self.converter.options.add_heading_ids {
            let heading_text = self.link_text_buffer.trim();
            let id = self.converter.get_heading_id(heading_text);
            self.output.push_str(&format!(" {{#{}}}", id));
        }

        // Add newline after heading
        self.output.push_str("\n\n");

        // Reset heading mode flags
        self.inside_heading = false;
        self.current_heading_level = 0;
        self.link_text_buffer = old_link_text;
    }

    /// Process a paragraph element
    fn process_paragraph(&mut self, node: &ElementRef) {
        // Skip empty paragraphs
        if node.text().collect::<String>().trim().is_empty() {
            return;
        }

        // Don't add extra spacing in lists
        if !self.in_list_item && !self.output.ends_with("\n\n") && !self.output.is_empty() {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Add newline after paragraph (but not in list items)
        if !self.in_list_item {
            self.output.push_str("\n\n");
        } else {
            self.output.push('\n');
        }
    }

    /// Process a line break
    fn process_line_break(&mut self) {
        // In GitHub-flavored markdown, use hard breaks
        match self.converter.options.format {
            MarkdownFormat::GitHub => {
                self.output.push_str("  \n");
            },
            _ => {
                self.output.push('\n');
            }
        }
    }

    /// Process a horizontal rule
    fn process_horizontal_rule(&mut self) {
        // Add spacing before rule
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Add rule
        match self.converter.options.format {
            MarkdownFormat::Semantic => {
                self.output.push_str("[DIVIDER]\n\n---\n\n");
            },
            _ => {
                self.output.push_str("---\n\n");
            }
        }
    }

    /// Process an unordered list
    fn process_unordered_list(&mut self, node: &ElementRef) {
        // Add spacing before list (but not for nested lists)
        if self.converter.list_state.level == 0 && !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Increment list level
        self.converter.list_state.level += 1;

        // Mark as unordered
        self.converter.list_state.is_ordered.push(false);

        // Add list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels &&
           self.converter.list_state.level == 1 {
            self.output.push_str("[LIST]\n");
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // End list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels &&
           self.converter.list_state.level == 1 {
            self.output.push_str("[/LIST]\n");
        }

        // Decrement list level
        self.converter.list_state.level -= 1;
        self.converter.list_state.is_ordered.pop();

        // Add spacing after list (but not for nested lists)
        if self.converter.list_state.level == 0 && !self.output.ends_with("\n\n") {
            self.output.push('\n');
        }
    }

    /// Process an ordered list
    fn process_ordered_list(&mut self, node: &ElementRef) {
        // Add spacing before list (but not for nested lists)
        if self.converter.list_state.level == 0 && !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Increment list level
        self.converter.list_state.level += 1;

        // Mark as ordered
        self.converter.list_state.is_ordered.push(true);

        // Get start attribute
        let start = node.value().attr("start").and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
        self.converter.list_state.current_number.push(start);

        // Add list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels &&
           self.converter.list_state.level == 1 {
            self.output.push_str("[ORDERED_LIST]\n");
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // End list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels &&
           self.converter.list_state.level == 1 {
            self.output.push_str("[/ORDERED_LIST]\n");
        }

        // Decrement list level
        self.converter.list_state.level -= 1;
        self.converter.list_state.is_ordered.pop();
        self.converter.list_state.current_number.pop();

        // Add spacing after list (but not for nested lists)
        if self.converter.list_state.level == 0 && !self.output.ends_with("\n\n") {
            self.output.push('\n');
        }
    }

    /// Process a list item
    fn process_list_item(&mut self, node: &ElementRef) {
        // Get list level and indentation
        let level = self.converter.list_state.level.saturating_sub(1);
        let indent = "  ".repeat(level);

        // Get marker based on list type
        let marker = if self.converter.list_state.is_ordered.last().copied().unwrap_or(false) {
            // For ordered lists, use the current number
            let number = self.converter.list_state.current_number.last_mut().unwrap();
            let marker = format!("{}. ", number);
            *number += 1;
            marker
        } else {
            // For unordered lists, use a bullet
            "- ".to_string()
        };

        // Add indentation and marker
        self.output.push_str(&indent);
        self.output.push_str(&marker);

        // Set list item flag to avoid extra newlines
        self.in_list_item = true;

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                // Handle special case for paragraphs in list items
                if let Node::Element(element) = child_element.value() {
                    if element.name.local.as_ref() == "p" {
                        // Skip paragraph element and directly process its children
                        for p_child in child_element.children() {
                            if let Some(p_child_element) = ElementRef::wrap(p_child) {
                                self.process_node(p_child_element);
                            }
                        }
                        continue;
                    }
                }

                self.process_node(child_element);
            }
        }

        // Add newline if needed
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }

        // Reset list item flag
        self.in_list_item = false;
    }

    /// Process a definition list
    fn process_definition_list(&mut self, node: &ElementRef) {
        // Add spacing before list
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Set definition list state
        self.converter.list_state.in_definition_list = true;

        // Add list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[DEFINITION_LIST]\n");
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // End list marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[/DEFINITION_LIST]\n");
        }

        // Reset definition list state
        self.converter.list_state.in_definition_list = false;

        // Add spacing after list
        if !self.output.ends_with("\n\n") {
            self.output.push('\n');
        }
    }

    /// Process a definition term
    fn process_definition_term(&mut self, node: &ElementRef) {
        // Set term state
        self.converter.list_state.in_term = true;

        // Add term marker
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[TERM] ");
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Add newline
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }

        // Reset term state
        self.converter.list_state.in_term = false;
    }

    /// Process a definition description
    fn process_definition_description(&mut self, node: &ElementRef) {
        // Add description indentation and marker
        self.output.push_str(": ");

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Add newline
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }
    }

    /// Process a link element
    fn process_link(&mut self, node: &ElementRef) {
        // Skip if links should be excluded
        if !self.converter.options.include_links {
            for child in node.children() {
                if let Some(child_element) = ElementRef::wrap(child) {
                    self.process_node(child_element);
                }
            }
            return;
        }

        // Get href and title
        let href = node.value().attr("href").unwrap_or("");
        let title = node.value().attr("title").unwrap_or("");

        // Skip empty links or anchor links
        if href.is_empty() || href == "#" {
            for child in node.children() {
                if let Some(child_element) = ElementRef::wrap(child) {
                    self.process_node(child_element);
                }
            }
            return;
        }

        // Simplify URL if enabled
        let href = if self.converter.options.simplify_links {
            // Remove URL parameters
            if let Some(pos) = href.find('?') {
                &href[0..pos]
            } else {
                href
            }
        } else {
            href
        };

        // Set inside link flag to capture text
        let old_inside_link = self.inside_link;
        let old_link_text = std::mem::take(&mut self.link_text_buffer);

        self.inside_link = true;

        // Process children to collect link text
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        let link_text = if self.link_text_buffer.trim().is_empty() {
            // If link text is empty, use href
            href.to_string()
        } else {
            self.link_text_buffer.clone()
        };

        // Format link based on options
        if self.converter.options.refify_links {
            // Reference-style link
            let reference = self.converter.get_link_reference(href, &link_text);

            // Remove already added link text and replace with reference style
            self.output.truncate(self.output.len() - link_text.len());
            self.output.push_str(&format!("[{}][{}]", link_text, reference));

            // Store link for references section
            self.converter.element_tracker.links
                .entry(href.to_string())
                .or_insert_with(|| (reference, title.to_string()));
        } else {
            // Inline link
            self.output.truncate(self.output.len() - link_text.len());

            if title.is_empty() {
                self.output.push_str(&format!("[{}]({})", link_text, href));
            } else {
                self.output.push_str(&format!("[{}]({} \"{}\")", link_text, href, title));
            }
        }

        // Reset link flags
        self.inside_link = old_inside_link;
        self.link_text_buffer = old_link_text;
    }

    /// Process an image element
    fn process_image(&mut self, node: &ElementRef) {
        // Skip if images should be excluded
        if !self.converter.options.include_images {
            return;
        }

        // Get src, alt, and title
        let src = node.value().attr("src").unwrap_or("");
        let alt = node.value().attr("alt").unwrap_or("");
        let title = node.value().attr("title").unwrap_or("");

        // Skip empty sources
        if src.is_empty() {
            return;
        }

        // Format image based on options
        if self.converter.options.refify_links {
            // Reference-style image
            let reference = self.converter.get_image_reference(src, alt);
            self.output.push_str(&format!("![{}][{}]", alt, reference));

            // Store image for references section
            self.converter.element_tracker.images
                .entry(src.to_string())
                .or_insert_with(|| (reference, alt.to_string()));
        } else {
            // Inline image
            if title.is_empty() {
                self.output.push_str(&format!("![{}]({})", alt, src));
            } else {
                self.output.push_str(&format!("![{}]({} \"{}\")", alt, src, title));
            }
        }

        // Add figure caption if in semantic mode
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            if !alt.is_empty() {
                self.output.push_str(&format!(" [IMAGE: {}]", alt));
            } else {
                self.output.push_str(" [IMAGE]");
            }
        }
    }

    /// Process a table element
    fn process_table(&mut self, node: &ElementRef) {
        // Skip if tables should be excluded
        if !self.converter.options.include_tables {
            self.process_generic_element(node);
            return;
        }

        // Add spacing before table
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Set table state
        self.in_table = true;
        self.table_state = TableState::default();

        // Add table marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[TABLE]\n");
        }

        // Process children to collect table data
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Calculate column widths if enabled
        if self.converter.options.enable_table_column_tracking {
            self.table_state.widths = vec![0; self.table_state.column_count];

            // Calculate maximum width for each column
            for row in &self.table_state.rows {
                for (col_idx, cell) in row.iter().enumerate() {
                    if col_idx < self.table_state.widths.len() {
                        self.table_state.widths[col_idx] = self.table_state.widths[col_idx].max(cell.len());
                    }
                }
            }
        }

        // Render the table
        self.render_table();

        // Reset table state
        self.in_table = false;

        // Add spacing after table
        if !self.output.ends_with("\n\n") {
            self.output.push_str("\n\n");
        }
    }

    /// Process a table head element
    fn process_table_head(&mut self, node: &ElementRef) {
        // Mark table as having a header
        self.table_state.has_header = true;

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }
    }

    /// Process a table body element
    fn process_table_body(&mut self, node: &ElementRef) {
        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }
    }

    /// Process a table row element
    fn process_table_row(&mut self, node: &ElementRef) {
        // Start a new row
        self.table_state.current_row = Vec::new();

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Add row to table
        if !self.table_state.current_row.is_empty() {
            // Update column count if needed
            self.table_state.column_count = self.table_state.column_count.max(self.table_state.current_row.len());

            // Add row to table
            self.table_state.rows.push(self.table_state.current_row.clone());
        }
    }

    /// Process a table header cell element
    fn process_table_header_cell(&mut self, node: &ElementRef) {
        // Capture cell content
        let mut cell_content = String::new();
        let old_output = std::mem::replace(self.output, &mut cell_content);

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Trim whitespace
        let cell_content = cell_content.trim().to_string();

        // Add cell to current row
        self.table_state.current_row.push(cell_content);

        // Get alignment
        if let Some(align) = node.value().attr("align") {
            self.update_column_alignment(&align, self.table_state.current_row.len() - 1);
        } else if let Some(style) = node.value().attr("style") {
            if style.contains("text-align: center") {
                self.update_column_alignment("center", self.table_state.current_row.len() - 1);
            } else if style.contains("text-align: right") {
                self.update_column_alignment("right", self.table_state.current_row.len() - 1);
            }
        }

        // Restore output
        *self.output = old_output;
    }

    /// Process a table cell element
    fn process_table_cell(&mut self, node: &ElementRef) {
        // Capture cell content
        let mut cell_content = String::new();
        let old_output = std::mem::replace(self.output, &mut cell_content);

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Trim whitespace
        let cell_content = cell_content.trim().to_string();

        // Add cell to current row
        self.table_state.current_row.push(cell_content);

        // Get alignment
        if let Some(align) = node.value().attr("align") {
            self.update_column_alignment(&align, self.table_state.current_row.len() - 1);
        } else if let Some(style) = node.value().attr("style") {
            if style.contains("text-align: center") {
                self.update_column_alignment("center", self.table_state.current_row.len() - 1);
            } else if style.contains("text-align: right") {
                self.update_column_alignment("right", self.table_state.current_row.len() - 1);
            }
        }

        // Restore output
        *self.output = old_output;
    }

    /// Process a table caption element
    fn process_table_caption(&mut self, node: &ElementRef) {
        // Capture caption content
        let mut caption_content = String::new();
        let old_output = std::mem::replace(self.output, &mut caption_content);

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Trim whitespace
        let caption_content = caption_content.trim().to_string();

        // Set caption
        self.table_state.caption = Some(caption_content);

        // Restore output
        *self.output = old_output;
    }

    /// Update column alignment
    fn update_column_alignment(&mut self, align: &str, column_index: usize) {
        // Ensure alignments vector is large enough
        while self.table_state.alignments.len() <= column_index {
            self.table_state.alignments.push(ColumnAlignment::Left);
        }

        // Set alignment
        let alignment = match align.to_lowercase().as_str() {
            "center" => ColumnAlignment::Center,
            "right" => ColumnAlignment::Right,
            _ => ColumnAlignment::Left,
        };

        self.table_state.alignments[column_index] = alignment;
    }

    /// Render the table
    fn render_table(&mut self) {
        // Skip empty tables
        if self.table_state.rows.is_empty() {
            return;
        }

        // Add caption if present
        if let Some(caption) = &self.table_state.caption {
            if self.converter.options.format == MarkdownFormat::Semantic &&
               self.converter.options.add_semantic_labels {
                self.output.push_str(&format!("[CAPTION] {}\n\n", caption));
            } else {
                self.output.push_str(&format!("Table: {}\n\n", caption));
            }
        }

        // Calculate total columns
        let columns = self.table_state.column_count;

        // Ensure we have enough alignment info
        while self.table_state.alignments.len() < columns {
            self.table_state.alignments.push(ColumnAlignment::Left);
        }

        // Pad rows to ensure consistent column count
        for row in &mut self.table_state.rows {
            while row.len() < columns {
                row.push(String::new());
            }
        }

        // Render header row
        if !self.table_state.rows.is_empty() {
            self.output.push('|');

            for (col_idx, cell) in self.table_state.rows[0].iter().enumerate() {
                if self.converter.options.enable_table_column_tracking && col_idx < self.table_state.widths.len() {
                    let width = self.table_state.widths[col_idx];
                    let padding = " ".repeat(width.saturating_sub(cell.len()));

                    match self.table_state.alignments[col_idx] {
                        ColumnAlignment::Left => self.output.push_str(&format!(" {}{} |", cell, padding)),
                        ColumnAlignment::Center => {
                            let left_pad = " ".repeat(padding.len() / 2);
                            let right_pad = " ".repeat(padding.len() - left_pad.len());
                            self.output.push_str(&format!(" {}{}{} |", left_pad, cell, right_pad));
                        },
                        ColumnAlignment::Right => self.output.push_str(&format!(" {}{} |", padding, cell)),
                    }
                } else {
                    self.output.push_str(&format!(" {} |", cell));
                }
            }

            self.output.push('\n');

            // Render header separator
            self.output.push('|');

            for (col_idx, _) in self.table_state.rows[0].iter().enumerate() {
                let width = if self.converter.options.enable_table_column_tracking && col_idx < self.table_state.widths.len() {
                    self.table_state.widths[col_idx].max(3)
                } else {
                    3
                };

                match self.table_state.alignments[col_idx] {
                    ColumnAlignment::Left => self.output.push_str(&format!(" :{}-{} |", "-".repeat(width - 2), "")),
                    ColumnAlignment::Center => self.output.push_str(&format!(" :{}-{}: |", "-".repeat(width - 3), "")),
                    ColumnAlignment::Right => self.output.push_str(&format!(" {}-{}: |", "-".repeat(width - 2), "")),
                }
            }

            self.output.push('\n');

            // Render data rows
            let start_idx = if self.table_state.has_header { 1 } else { 0 };

            for row_idx in start_idx..self.table_state.rows.len() {
                self.output.push('|');

                for (col_idx, cell) in self.table_state.rows[row_idx].iter().enumerate() {
                    if self.converter.options.enable_table_column_tracking && col_idx < self.table_state.widths.len() {
                        let width = self.table_state.widths[col_idx];
                        let padding = " ".repeat(width.saturating_sub(cell.len()));

                        match self.table_state.alignments[col_idx] {
                            ColumnAlignment::Left => self.output.push_str(&format!(" {}{} |", cell, padding)),
                            ColumnAlignment::Center => {
                                let left_pad = " ".repeat(padding.len() / 2);
                                let right_pad = " ".repeat(padding.len() - left_pad.len());
                                self.output.push_str(&format!(" {}{}{} |", left_pad, cell, right_pad));
                            },
                            ColumnAlignment::Right => self.output.push_str(&format!(" {}{} |", padding, cell)),
                        }
                    } else {
                        self.output.push_str(&format!(" {} |", cell));
                    }
                }

                self.output.push('\n');
            }
        }
    }

    /// Process a pre element
    fn process_pre(&mut self, node: &ElementRef) {
        // Add spacing before pre
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Set pre mode
        let old_pre_mode = self.in_pre_block;
        self.in_pre_block = true;

        // Check for language
        let mut language = String::new();

        // Look for code element with class
        if let Some(code) = node.select(&CODE_SELECTOR).next() {
            if let Some(class) = code.value().attr("class") {
                // Extract language from class
                for class_name in class.split_whitespace() {
                    if class_name.starts_with("language-") {
                        language = class_name[9..].to_string();
                        break;
                    } else if class_name.starts_with("lang-") {
                        language = class_name[5..].to_string();
                        break;
                    }
                }
            }
        }

        // Add code block markers
        if language.is_empty() {
            self.output.push_str("```\n");
        } else {
            self.output.push_str(&format!("```{}\n", language));
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                // Skip processing the code tag directly
                if let Node::Element(element) = child_element.value() {
                    if element.name.local.as_ref() == "code" {
                        // Process children of code tag directly
                        for code_child in child_element.children() {
                            if let Some(code_child_element) = ElementRef::wrap(code_child) {
                                self.process_node(code_child_element);
                            }
                        }
                        continue;
                    }
                }

                self.process_node(child_element);
            }
        }

        // Ensure code block has newline at the end
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }

        // Close code block
        self.output.push_str("```\n\n");

        // Restore pre mode
        self.in_pre_block = old_pre_mode;
    }

    /// Process a code element
    fn process_code(&mut self, node: &ElementRef) {
        // If inside pre block, process content directly
        if self.in_pre_block {
            for child in node.children() {
                if let Some(child_element) = ElementRef::wrap(child) {
                    self.process_node(child_element);
                }
            }
            return;
        }

        // Inline code
        self.output.push('`');

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        self.output.push('`');
    }

    /// Process a blockquote element
    fn process_blockquote(&mut self, node: &ElementRef) {
        // Add spacing before blockquote
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Add quote marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[QUOTE]\n");
        }

        // Capture blockquote content
        let mut quote_content = String::new();
        let old_output = std::mem::replace(self.output, &mut quote_content);

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Format blockquote
        let quoted_lines = quote_content
            .lines()
            .map(|line| if line.is_empty() { ">" } else { &format!("> {}", line) })
            .collect::<Vec<_>>()
            .join("\n");

        // Restore output and add formatted content
        *self.output = old_output;
        self.output.push_str(&quoted_lines);

        // Add quote end marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("\n[/QUOTE]");
        }

        // Add spacing after blockquote
        self.output.push_str("\n\n");
    }

    /// Process a strong/bold element
    fn process_strong(&mut self, node: &ElementRef) {
        self.output.push_str("**");

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        self.output.push_str("**");
    }

    /// Process an emphasis/italic element
    fn process_emphasis(&mut self, node: &ElementRef) {
        self.output.push('*');

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        self.output.push('*');
    }

    /// Process a strikethrough element
    fn process_strikethrough(&mut self, node: &ElementRef) {
        self.output.push_str("~~");

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        self.output.push_str("~~");
    }

    /// Process a subscript element
    fn process_subscript(&mut self, node: &ElementRef) {
        // Format depends on markdown flavor
        match self.converter.options.format {
            MarkdownFormat::GitHub => self.output.push_str("<sub>"),
            _ => self.output.push('~'),
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Close tag
        match self.converter.options.format {
            MarkdownFormat::GitHub => self.output.push_str("</sub>"),
            _ => self.output.push('~'),
        }
    }

    /// Process a superscript element
    fn process_superscript(&mut self, node: &ElementRef) {
        // Format depends on markdown flavor
        match self.converter.options.format {
            MarkdownFormat::GitHub => self.output.push_str("<sup>"),
            _ => self.output.push('^'),
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }

        // Close tag
        match self.converter.options.format {
            MarkdownFormat::GitHub => self.output.push_str("</sup>"),
            _ => self.output.push('^'),
        }
    }

    /// Process a div element
    fn process_div(&mut self, node: &ElementRef) {
        // If div has special class, handle accordingly
        if let Some(class) = node.value().attr("class") {
            if class.contains("code") || class.contains("highlight") {
                // Treat as code block
                self.process_pre(node);
                return;
            } else if class.contains("quote") || class.contains("blockquote") {
                // Treat as blockquote
                self.process_blockquote(node);
                return;
            }
        }

        // Check for data role
        if let Some(role) = node.value().attr("data-role") {
            if role == "code" || role == "codeblock" {
                // Treat as code block
                self.process_pre(node);
                return;
            } else if role == "quote" || role == "blockquote" {
                // Treat as blockquote
                self.process_blockquote(node);
                return;
            }
        }

        // Regular div processing - just process children without adding extra formatting
        // But add a newline if content is not inline
        let has_block_children = node.children().any(|child| {
            if let Some(child_element) = ElementRef::wrap(child) {
                if let Node::Element(element) = child_element.value() {
                    matches!(
                        element.name.local.as_ref(),
                        "p" | "div" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" |
                        "ul" | "ol" | "table" | "blockquote" | "pre"
                    )
                } else {
                    false
                }
            } else {
                false
            }
        });

        if has_block_children && !self.output.ends_with('\n') {
            self.output.push('\n');
        }

        // Process children
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }
    }

    /// Process a span element
    fn process_span(&mut self, node: &ElementRef) {
        // Check for specific attributes or classes
        if let Some(class) = node.value().attr("class") {
            if class.contains("highlight") || class.contains("code") {
                // Treat as inline code
                self.output.push('`');

                // Process children
                for child in node.children() {
                    if let Some(child_element) = ElementRef::wrap(child) {
                        self.process_node(child_element);
                    }
                }

                self.output.push('`');
                return;
            } else if class.contains("bold") || class.contains("strong") {
                // Treat as bold
                self.process_strong(node);
                return;
            } else if class.contains("italic") || class.contains("emphasis") {
                // Treat as italic
                self.process_emphasis(node);
                return;
            }
        }

        // Regular span processing - just process children without adding extra formatting
        for child in node.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.process_node(child_element);
            }
        }
    }

    /// Process a figure element
    fn process_figure(&mut self, node: &ElementRef) {
        // Add spacing before figure
        if !self.output.ends_with("\n\n") {
            if self.output.ends_with('\n') {
                self.output.push('\n');
            } else {
                self.output.push_str("\n\n");
            }
        }

        // Add figure marker for semantic format
        if self.converter.options.format == MarkdownFormat::Semantic &&
           self.converter.options.add_semantic_labels {
            self.output.push_str("[FIGURE]\n");
        }

        // Process children
        for child in node.children() {
            if let Some(
