//! DOM converter module
//!
//! This module provides functionality for converting DOM structures to different
//! representations suitable for LLM processing.

use html5ever::serialize::SerializableHandle;
use html5ever::serialize::{SerializeOpts, TraversalScope, serialize};
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{RcDom, SerializableHandle as RcDomSerializableHandle};
use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};
use selectors::Element;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// DOM representation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomRepresentation {
    /// Representation type
    pub rep_type: String,

    /// Representation content
    pub content: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// DOM conversion options
#[derive(Debug, Clone)]
pub struct DomConversionOptions {
    /// Whether to generate simplified HTML
    pub generate_simplified_html: bool,

    /// Whether to generate DOM tree text
    pub generate_dom_tree_text: bool,

    /// Whether to generate semantic outline
    pub generate_semantic_outline: bool,

    /// Whether to generate element table
    pub generate_element_table: bool,

    /// Maximum depth for DOM tree
    pub max_depth: Option<usize>,

    /// Whether to include attributes
    pub include_attributes: bool,

    /// Whether to include text content
    pub include_text_content: bool,

    /// Whether to simplify complex structures
    pub simplify_complex_structures: bool,

    /// Whether to focus on content elements
    pub focus_on_content: bool,

    /// Custom representation generators
    pub custom_generators: Vec<CustomRepresentationGenerator>,
}

impl Default for DomConversionOptions {
    fn default() -> Self {
        Self {
            generate_simplified_html: true,
            generate_dom_tree_text: true,
            generate_semantic_outline: true,
            generate_element_table: false,
            max_depth: None,
            include_attributes: true,
            include_text_content: true,
            simplify_complex_structures: true,
            focus_on_content: true,
            custom_generators: Vec::new(),
        }
    }
}

/// Custom DOM representation generator
#[derive(Debug, Clone)]
pub struct CustomRepresentationGenerator {
    /// Generator name
    pub name: String,

    /// Generator function
    #[allow(clippy::type_complexity)]
    pub generator:
        Arc<dyn Fn(&Html) -> Result<DomRepresentation, crate::error::DOMorpherError> + Send + Sync>,
}

/// DOM converter
#[derive(Clone)]
pub struct DomConverter {
    /// Conversion options
    options: DomConversionOptions,
}

impl DomConverter {
    /// Create a new DOM converter with the given options
    pub fn new(options: DomConversionOptions) -> Self {
        Self { options }
    }

    /// Create a new DOM converter with default options
    pub fn default() -> Self {
        Self::new(DomConversionOptions::default())
    }

    /// Convert the DOM to different representations
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DomRepresentation>, DomProcessingError>` - The representations or an error
    pub fn convert(
        &self,
        document: &Html,
    ) -> Result<Vec<DomRepresentation>, crate::error::DOMorpherError> {
        let mut representations = Vec::new();

        // Generate simplified HTML
        if self.options.generate_simplified_html {
            let simplified_html = self.generate_simplified_html(document)?;
            representations.push(simplified_html);
        }

        // Generate DOM tree text
        if self.options.generate_dom_tree_text {
            let dom_tree_text = self.generate_dom_tree_text(document)?;
            representations.push(dom_tree_text);
        }

        // Generate semantic outline
        if self.options.generate_semantic_outline {
            let semantic_outline = self.generate_semantic_outline(document)?;
            representations.push(semantic_outline);
        }

        // Generate element table
        if self.options.generate_element_table {
            let element_table = self.generate_element_table(document)?;
            representations.push(element_table);
        }

        // Apply custom generators
        for generator in &self.options.custom_generators {
            let representation = (generator.generator)(document)?;
            representations.push(representation);
        }

        Ok(representations)
    }

    /// Generate simplified HTML representation
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<DomRepresentation, DomProcessingError>` - The representation or an error
    fn generate_simplified_html(
        &self,
        document: &Html,
    ) -> Result<DomRepresentation, crate::error::DOMorpherError> {
        let html = document.html();
        let simplified_html = self.simplify_html(&html)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "element_count".to_string(),
            self.count_elements(document).to_string(),
        );

        Ok(DomRepresentation {
            rep_type: "simplified_html".to_string(),
            content: simplified_html,
            metadata,
        })
    }

    /// Generate DOM tree text representation
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<DomRepresentation, DomProcessingError>` - The representation or an error
    fn generate_dom_tree_text(
        &self,
        document: &Html,
    ) -> Result<DomRepresentation, crate::error::DOMorpherError> {
        let body_selector = Selector::parse("body").unwrap();

        let mut tree_text = String::new();

        if let Some(body) = document.select(&body_selector).next() {
            self.append_element_to_tree_text(&mut tree_text, body, 0)?;
        } else {
            // Fallback to document root if body not found
            self.append_element_to_tree_text(
                &mut tree_text,
                ElementRef::wrap(document.tree.document()).unwrap(),
                0,
            )?;
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "indented_tree".to_string());
        metadata.insert(
            "element_count".to_string(),
            self.count_elements(document).to_string(),
        );

        Ok(DomRepresentation {
            rep_type: "dom_tree_text".to_string(),
            content: tree_text,
            metadata,
        })
    }

    /// Generate semantic outline representation
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<DomRepresentation, DomProcessingError>` - The representation or an error
    fn generate_semantic_outline(
        &self,
        document: &Html,
    ) -> Result<DomRepresentation, crate::error::DOMorpherError> {
        let mut outline = String::new();

        // Extract title
        let title_selector = Selector::parse("title").unwrap();
        if let Some(title) = document.select(&title_selector).next() {
            outline.push_str(&format!(
                "# {}\n\n",
                title.text().collect::<Vec<_>>().join("")
            ));
        }

        // Extract headings
        let headings_selector = Selector::parse("h1, h2, h3, h4, h5, h6").unwrap();

        for heading in document.select(&headings_selector) {
            let level = match heading.value().name.local.as_ref() {
                "h1" => 1,
                "h2" => 2,
                "h3" => 3,
                "h4" => 4,
                "h5" => 5,
                "h6" => 6,
                _ => 1,
            };

            let heading_text = heading.text().collect::<Vec<_>>().join("");
            let heading_prefix = "#".repeat(level);

            outline.push_str(&format!("{} {}\n\n", heading_prefix, heading_text));

            // Extract content following this heading until the next heading
            let mut next_sibling = heading.next_sibling();

            while let Some(node) = next_sibling {
                if let Some(element) = ElementRef::wrap(node) {
                    if matches!(
                        element.value().name.local.as_ref(),
                        "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                    ) {
                        break;
                    }

                    if matches!(
                        element.value().name.local.as_ref(),
                        "p" | "ul" | "ol" | "blockquote" | "pre"
                    ) {
                        let text = element.text().collect::<Vec<_>>().join("");
                        if !text.trim().is_empty() {
                            outline.push_str(&format!("{}\n\n", text));
                        }
                    }
                }

                next_sibling = node.next_sibling();
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "markdown".to_string());

        Ok(DomRepresentation {
            rep_type: "semantic_outline".to_string(),
            content: outline,
            metadata,
        })
    }

    /// Generate element table representation
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<DomRepresentation, DomProcessingError>` - The representation or an error
    fn generate_element_table(
        &self,
        document: &Html,
    ) -> Result<DomRepresentation, crate::error::DOMorpherError> {
        let mut table = String::new();

        // Add table header
        table.push_str("| Element | Type | Attributes | Text Content |\n");
        table.push_str("|---------|------|------------|-------------|\n");

        // Focus on important elements
        let important_selectors = [
            // Semantic elements
            "main",
            "article",
            "section",
            "nav",
            "header",
            "footer",
            "aside",
            // Headings
            "h1",
            "h2",
            "h3",
            // Interactive elements
            "a",
            "button",
            "form",
            "input[type=submit]",
            "input[type=button]",
            // Media elements
            "img[alt]",
            "video",
            "audio",
            // Lists
            "ul",
            "ol",
            // Tables
            "table",
        ];

        let joined_selector = important_selectors.join(", ");

        if let Ok(selector) = Selector::parse(&joined_selector) {
            for element in document.select(&selector) {
                let tag_name = element.value().name.local.as_ref();

                // Element type
                let element_type = match tag_name {
                    "main" | "article" | "section" => "Content Container",
                    "nav" => "Navigation",
                    "header" => "Header",
                    "footer" => "Footer",
                    "aside" => "Sidebar",
                    "h1" | "h2" | "h3" => "Heading",
                    "a" => "Link",
                    "button" | "input" => "Button",
                    "form" => "Form",
                    "img" => "Image",
                    "video" => "Video",
                    "audio" => "Audio",
                    "ul" | "ol" => "List",
                    "table" => "Table",
                    _ => "Other",
                };

                // Important attributes
                let mut important_attrs = Vec::new();

                if let Some(id) = element.value().attr("id") {
                    important_attrs.push(format!("id=\"{}\"", id));
                }

                if let Some(class) = element.value().attr("class") {
                    important_attrs.push(format!("class=\"{}\"", class));
                }

                if tag_name == "a" {
                    if let Some(href) = element.value().attr("href") {
                        important_attrs.push(format!("href=\"{}\"", href));
                    }
                }

                if tag_name == "img" {
                    if let Some(alt) = element.value().attr("alt") {
                        important_attrs.push(format!("alt=\"{}\"", alt));
                    }

                    if let Some(src) = element.value().attr("src") {
                        important_attrs.push(format!("src=\"{}\"", src));
                    }
                }

                if tag_name == "input" {
                    if let Some(input_type) = element.value().attr("type") {
                        important_attrs.push(format!("type=\"{}\"", input_type));
                    }

                    if let Some(name) = element.value().attr("name") {
                        important_attrs.push(format!("name=\"{}\"", name));
                    }
                }

                // Text content (truncated if too long)
                let text = element.text().collect::<Vec<_>>().join(" ");
                let truncated_text = if text.len() > 50 {
                    format!("{}...", &text[..47])
                } else {
                    text
                };

                // Add row to table
                table.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    tag_name,
                    element_type,
                    important_attrs.join(", "),
                    truncated_text
                ));
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "markdown_table".to_string());

        Ok(DomRepresentation {
            rep_type: "element_table".to_string(),
            content: table,
            metadata,
        })
    }

    /// Simplify HTML by removing unnecessary elements and attributes
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to simplify
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The simplified HTML or an error
    fn simplify_html(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let dom = Html::parse_document(html);

        // Elements to remove completely
        let elements_to_remove = [
            "script",
            "style",
            "noscript",
            "iframe",
            "object",
            "embed",
            "link[rel=stylesheet]",
            "meta",
            "svg",
            "canvas",
            "head",
            "comment",
            "template",
        ];

        // Elements to potentially keep but clean
        let elements_to_clean = [
            "div", "span", "a", "img", "input", "button", "form", "label", "table", "tr", "td",
            "th", "ul", "ol", "li", "p", "h1", "h2", "h3", "h4", "h5", "h6", "main", "article",
            "section", "header", "footer", "nav", "aside",
        ];

        // Attributes to remove from all elements
        let attributes_to_remove = [
            "aria-.*",
            "data-.*",
            "onclick",
            "onload",
            "onsubmit",
            "onchange",
            "oninput",
            "style",
            "class",
            "id",
            "role",
            "tabindex",
            "lang",
            "dir",
            "contenteditable",
            "draggable",
            "spellcheck",
            "translate",
        ];

        // Attributes to keep for specific elements
        let attributes_to_keep = HashMap::from([
            ("a", vec!["href", "title", "target", "rel"]),
            ("img", vec!["src", "alt", "width", "height"]),
            (
                "input",
                vec![
                    "type",
                    "name",
                    "value",
                    "placeholder",
                    "required",
                    "disabled",
                    "checked",
                ],
            ),
            ("button", vec!["type", "name", "value", "disabled"]),
            ("form", vec!["action", "method", "enctype"]),
            ("label", vec!["for"]),
            ("th", vec!["colspan", "rowspan", "scope"]),
            ("td", vec!["colspan", "rowspan"]),
            ("meta", vec!["name", "content", "property"]),
            ("link", vec!["rel", "href", "type"]),
        ]);

        // Create a new document with simplified content
        let mut simplified_html = String::new();

        // Function to clean and serialize an element
        let clean_element =
            |element: ElementRef| -> Result<Option<String>, crate::error::DOMorpherError> {
                let tag_name = element.value().name.local.as_ref();

                // Skip elements that should be removed
                if elements_to_remove.contains(&tag_name) {
                    return Ok(None);
                }

                // Create a new element with cleaned attributes
                let mut clean_html = String::new();
                clean_html.push_str(&format!("<{}", tag_name));

                // Add only allowed attributes
                let allowed_attributes = attributes_to_keep
                    .get(tag_name)
                    .cloned()
                    .unwrap_or_vec(vec![]);
                let attribute_regexes: Vec<Regex> = attributes_to_remove
                    .iter()
                    .map(|pattern| Regex::new(pattern).unwrap())
                    .collect();

                for attr in element.value().attrs.iter() {
                    let attr_name = attr.0.local.as_ref();
                    let attr_value = attr.1.as_ref();

                    let should_remove = attribute_regexes.iter().any(|re| re.is_match(attr_name));

                    if !should_remove || allowed_attributes.contains(&attr_name) {
                        clean_html.push_str(&format!(" {}=\"{}\"", attr_name, attr_value));
                    }
                }

                clean_html.push('>');

                // Recursively process child elements
                for child in element.children() {
                    match child.value() {
                        Node::Element(_) => {
                            if let Some(child_element) = ElementRef::wrap(child) {
                                if let Some(child_html) = clean_element(child_element)? {
                                    clean_html.push_str(&child_html);
                                }
                            }
                        }
                        Node::Text(text) => {
                            // Keep text content if not empty
                            let text_content = text.text.trim();
                            if !text_content.is_empty() {
                                clean_html.push_str(text_content);
                            }
                        }
                        _ => {} // Skip other node types
                    }
                }

                // Close the tag
                clean_html.push_str(&format!("</{}>", tag_name));

                Ok(Some(clean_html))
            };

        // Process the document body
        let body_selector = Selector::parse("body").unwrap();

        if let Some(body) = dom.select(&body_selector).next() {
            if let Some(clean_body) = clean_element(body)? {
                simplified_html.push_str(&clean_body);
            }
        } else {
            // If no body tag found, process the document root
            if let Some(html_element) =
                ElementRef::wrap(dom.tree.document().children.iter().next().unwrap())
            {
                if let Some(clean_html) = clean_element(html_element)? {
                    simplified_html.push_str(&clean_html);
                }
            }
        }

        // If the simplification strategy has removed all content, return a warning message
        if simplified_html.trim().is_empty() {
            return Err(crate::error::DOMorpherError::DomProcessingError(
                "Simplified HTML is empty. Try adjusting simplification options.".to_string(),
            ));
        }

        // Remove excess whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        let simplified_html = whitespace_regex
            .replace_all(&simplified_html, " ")
            .to_string();

        Ok(simplified_html)
    }

    /// Count elements in the HTML document
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `usize` - The number of elements
    fn count_elements(&self, document: &Html) -> usize {
        let all_elements_selector = Selector::parse("*").unwrap();
        document.select(&all_elements_selector).count()
    }

    /// Append element to tree text representation
    ///
    /// # Arguments
    ///
    /// * `tree_text` - The tree text to append to
    /// * `element` - The element to append
    /// * `depth` - The current depth
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Result indicating success or failure
    fn append_element_to_tree_text(
        &self,
        tree_text: &mut String,
        element: ElementRef,
        depth: usize,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Check depth limit
        if let Some(max_depth) = self.options.max_depth {
            if depth > max_depth {
                return Ok(());
            }
        }

        // Get element tag name
        let tag_name = element.value().name.local.as_ref();

        // Skip comment nodes
        if tag_name == "comment" {
            return Ok(());
        }

        // Skip script and style tags if focusing on content
        if self.options.focus_on_content && (tag_name == "script" || tag_name == "style") {
            return Ok(());
        }

        // Create indentation
        let indent = "  ".repeat(depth);

        // Start element line
        tree_text.push_str(&format!("{}<{}", indent, tag_name));

        // Add attributes if enabled
        if self.options.include_attributes {
            for attr in element.value().attrs.iter() {
                let attr_name = attr.0.local.as_ref();
                let attr_value = attr.1.as_ref();

                // Skip data attributes if simplifying
                if self.options.simplify_complex_structures && attr_name.starts_with("data-") {
                    continue;
                }

                tree_text.push_str(&format!(" {}=\"{}\"", attr_name, attr_value));
            }
        }

        tree_text.push_str(">\n");

        // Extract text content
        if self.options.include_text_content {
            let text_content = element.text().collect::<Vec<_>>().join("");

            if !text_content.trim().is_empty() {
                let indent_content = "  ".repeat(depth + 1);
                let truncated_text = if text_content.len() > 100 {
                    format!("{}...", &text_content[..97])
                } else {
                    text_content
                };

                // Only include the first line of text content if it's multiline
                let first_line = truncated_text.lines().next().unwrap_or("");
                tree_text.push_str(&format!("{}\"{}\"\n", indent_content, first_line));
            }
        }

        // Process children elements recursively
        let mut child_count = 0;
        for child in element.children() {
            if let Some(child_element) = ElementRef::wrap(child) {
                self.append_element_to_tree_text(tree_text, child_element, depth + 1)?;
                child_count += 1;

                // Limit children if simplifying complex structures
                if self.options.simplify_complex_structures && child_count >= 10 {
                    let indent_ellipsis = "  ".repeat(depth + 1);
                    tree_text.push_str(&format!(
                        "{}... ({} more children)\n",
                        indent_ellipsis,
                        element.children().count() - 10
                    ));
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get all important elements from the document
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<Vec<ElementRef>, DomProcessingError>` - Important elements or an error
    fn get_important_elements(
        &self,
        document: &Html,
    ) -> Result<Vec<ElementRef>, crate::error::DOMorpherError> {
        let mut important_elements = Vec::new();

        // Selectors for important elements
        let selectors = [
            // Content containers
            "main",
            "article",
            "section",
            // Headings
            "h1",
            "h2",
            "h3",
            "h4",
            // Navigation
            "nav",
            "menu",
            // Interactive elements
            "a",
            "button",
            "input",
            "form",
            "select",
            // Lists
            "ul",
            "ol",
            // Tables
            "table",
            // Media
            "img",
            "video",
            "audio",
            // Semantic elements
            "header",
            "footer",
            "aside",
            "figure",
            "figcaption",
            "details",
            "summary",
            // Text content
            "p",
            "blockquote",
            "pre",
            "code",
        ];

        // Set to track processed elements and avoid duplicates
        let mut processed_elements = HashSet::new();

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    // Create a unique identifier for this element
                    let element_id = format!("{:?}", element.value());

                    // Skip if already processed
                    if processed_elements.contains(&element_id) {
                        continue;
                    }

                    important_elements.push(element);
                    processed_elements.insert(element_id);
                }
            }
        }

        Ok(important_elements)
    }

    /// Extract structural metadata from the document
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<HashMap<String, String>, DomProcessingError>` - Metadata or an error
    pub fn extract_metadata(
        &self,
        document: &Html,
    ) -> Result<HashMap<String, String>, crate::error::DOMorpherError> {
        let mut metadata = HashMap::new();

        // Extract title
        let title_selector = Selector::parse("title").unwrap();
        if let Some(title) = document.select(&title_selector).next() {
            metadata.insert(
                "title".to_string(),
                title.text().collect::<Vec<_>>().join(""),
            );
        }

        // Extract meta description
        let meta_desc_selector = Selector::parse("meta[name=description]").unwrap();
        if let Some(meta_desc) = document.select(&meta_desc_selector).next() {
            if let Some(content) = meta_desc.value().attr("content") {
                metadata.insert("description".to_string(), content.to_string());
            }
        }

        // Extract meta keywords
        let meta_keywords_selector = Selector::parse("meta[name=keywords]").unwrap();
        if let Some(meta_keywords) = document.select(&meta_keywords_selector).next() {
            if let Some(content) = meta_keywords.value().attr("content") {
                metadata.insert("keywords".to_string(), content.to_string());
            }
        }

        // Extract Open Graph metadata
        let og_selectors = [
            ("og:title", "og_title"),
            ("og:description", "og_description"),
            ("og:type", "og_type"),
            ("og:url", "og_url"),
            ("og:image", "og_image"),
            ("og:site_name", "og_site_name"),
        ];

        for (og_property, key) in og_selectors {
            let og_selector =
                Selector::parse(&format!("meta[property='{}']", og_property)).unwrap();
            if let Some(og_meta) = document.select(&og_selector).next() {
                if let Some(content) = og_meta.value().attr("content") {
                    metadata.insert(key.to_string(), content.to_string());
                }
            }
        }

        // Extract element counts
        let element_counts = self.count_element_types(document)?;
        for (element_type, count) in element_counts {
            metadata.insert(format!("count_{}", element_type), count.to_string());
        }

        // Extract URL if available
        let canonical_selector = Selector::parse("link[rel=canonical]").unwrap();
        if let Some(canonical) = document.select(&canonical_selector).next() {
            if let Some(href) = canonical.value().attr("href") {
                metadata.insert("url".to_string(), href.to_string());
            }
        }

        Ok(metadata)
    }

    /// Count the number of elements by type
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<HashMap<String, usize>, DomProcessingError>` - Element counts or an error
    fn count_element_types(
        &self,
        document: &Html,
    ) -> Result<HashMap<String, usize>, crate::error::DOMorpherError> {
        let mut counts = HashMap::new();

        // Element types to count
        let element_types = [
            "div", "p", "a", "img", "ul", "ol", "li", "table", "tr", "td", "form", "input",
            "button", "select", "textarea", "h1", "h2", "h3", "article", "section", "main",
            "header", "footer", "nav", "aside",
        ];

        for element_type in element_types {
            if let Ok(selector) = Selector::parse(element_type) {
                let count = document.select(&selector).count();
                counts.insert(element_type.to_string(), count);
            }
        }

        // Count interactive elements
        let interactive_selector = Selector::parse("a, button, input, select, textarea").unwrap();
        counts.insert(
            "interactive".to_string(),
            document.select(&interactive_selector).count(),
        );

        // Count media elements
        let media_selector = Selector::parse("img, video, audio, canvas, svg").unwrap();
        counts.insert(
            "media".to_string(),
            document.select(&media_selector).count(),
        );

        // Count semantic elements
        let semantic_selector =
            Selector::parse("article, section, nav, aside, header, footer, main").unwrap();
        counts.insert(
            "semantic".to_string(),
            document.select(&semantic_selector).count(),
        );

        Ok(counts)
    }

    /// Create a web page summary for LLM context
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - Summary text or an error
    pub fn create_page_summary(
        &self,
        document: &Html,
    ) -> Result<String, crate::error::DOMorpherError> {
        let mut summary = String::new();

        // Get metadata
        let metadata = self.extract_metadata(document)?;

        // Add title
        if let Some(title) = metadata.get("title") {
            summary.push_str(&format!("Title: {}\n\n", title));
        }

        // Add description
        if let Some(description) = metadata.get("description") {
            summary.push_str(&format!("Description: {}\n\n", description));
        }

        // Add page structure summary
        summary.push_str("Page Structure:\n");

        // Check for header
        let header_selector = Selector::parse("header").unwrap();
        if document.select(&header_selector).next().is_some() {
            summary.push_str("- Has header section\n");
        }

        // Check for navigation
        let nav_selector = Selector::parse("nav").unwrap();
        if document.select(&nav_selector).next().is_some() {
            summary.push_str("- Has navigation menu\n");
        }

        // Check for main content
        let main_selector = Selector::parse("main, article, #content, .content").unwrap();
        if document.select(&main_selector).next().is_some() {
            summary.push_str("- Has main content area\n");
        }

        // Check for sidebar
        let sidebar_selector = Selector::parse("aside, .sidebar, #sidebar").unwrap();
        if document.select(&sidebar_selector).next().is_some() {
            summary.push_str("- Has sidebar\n");
        }

        // Check for footer
        let footer_selector = Selector::parse("footer").unwrap();
        if document.select(&footer_selector).next().is_some() {
            summary.push_str("- Has footer section\n");
        }

        // Report on interactive elements
        let interactive_selector = Selector::parse("a, button, input, select, textarea").unwrap();
        let interactive_count = document.select(&interactive_selector).count();
        if interactive_count > 0 {
            summary.push_str(&format!(
                "- Contains {} interactive elements\n",
                interactive_count
            ));
        }

        // Report on forms
        let form_selector = Selector::parse("form").unwrap();
        let form_count = document.select(&form_selector).count();
        if form_count > 0 {
            summary.push_str(&format!("- Contains {} forms\n", form_count));
        }

        // Report on images
        let img_selector = Selector::parse("img").unwrap();
        let img_count = document.select(&img_selector).count();
        if img_count > 0 {
            summary.push_str(&format!("- Contains {} images\n", img_count));
        }

        // Report on tables
        let table_selector = Selector::parse("table").unwrap();
        let table_count = document.select(&table_selector).count();
        if table_count > 0 {
            summary.push_str(&format!("- Contains {} tables\n", table_count));
        }

        // Report on headings structure
        summary.push_str("\nHeading Structure:\n");
        for level in 1..=6 {
            let heading_selector = Selector::parse(&format!("h{}", level)).unwrap();
            let heading_count = document.select(&heading_selector).count();

            if heading_count > 0 {
                summary.push_str(&format!("- H{}: {} headings\n", level, heading_count));
            }
        }

        Ok(summary)
    }

    /// Create a DOM representation optimized for LLM token efficiency
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - Token-efficient representation or an error
    pub fn create_token_efficient_representation(
        &self,
        document: &Html,
    ) -> Result<String, crate::error::DOMorpherError> {
        let mut representation = String::new();

        // Extract important elements
        let important_elements = self.get_important_elements(document)?;

        // Add metadata summary
        let metadata = self.extract_metadata(document)?;

        if let Some(title) = metadata.get("title") {
            representation.push_str(&format!("TITLE: {}\n", title));
        }

        if let Some(description) = metadata.get("description") {
            representation.push_str(&format!("DESC: {}\n", description));
        }

        representation.push_str("\n");

        // Process important elements
        for element in important_elements {
            let tag_name = element.value().name.local.as_ref();

            // Skip non-content elements if focusing on content
            if self.options.focus_on_content
                && matches!(tag_name, "script" | "style" | "meta" | "link" | "head")
            {
                continue;
            }

            // Format based on element type
            match tag_name {
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                    // Format headings
                    let level = tag_name.chars().last().unwrap().to_digit(10).unwrap() as usize;
                    let prefix = "#".repeat(level);
                    let text = element
                        .text()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string();

                    if !text.is_empty() {
                        representation.push_str(&format!("{} {}\n\n", prefix, text));
                    }
                }
                "p" => {
                    // Format paragraphs
                    let text = element
                        .text()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .trim()
                        .to_string();

                    if !text.is_empty() {
                        representation.push_str(&format!("{}\n\n", text));
                    }
                }
                "a" => {
                    // Format links (only if standalone, not inside other containers)
                    if element.parent_element().map_or(false, |parent| {
                        !matches!(
                            parent.value().name.local.as_ref(),
                            "p" | "li" | "td" | "div"
                        )
                    }) {
                        let text = element
                            .text()
                            .collect::<Vec<_>>()
                            .join(" ")
                            .trim()
                            .to_string();
                        let href = element.value().attr("href").unwrap_or("");

                        if !text.is_empty() {
                            representation.push_str(&format!("LINK: {} ({})\n", text, href));
                        }
                    }
                }
                "img" => {
                    // Format images
                    let alt = element.value().attr("alt").unwrap_or("");
                    let src = element.value().attr("src").unwrap_or("");

                    representation.push_str(&format!("IMG: {} ({})\n", alt, src));
                }
                "ul" | "ol" => {
                    // Format lists
                    representation.push_str(&format!(
                        "{}:\n",
                        if tag_name == "ul" {
                            "LIST"
                        } else {
                            "ORDERED LIST"
                        }
                    ));

                    let li_selector = Selector::parse("li").unwrap();
                    for (i, li) in element.select(&li_selector).enumerate() {
                        let text = li.text().collect::<Vec<_>>().join(" ").trim().to_string();

                        if !text.is_empty() {
                            if tag_name == "ul" {
                                representation.push_str(&format!("- {}\n", text));
                            } else {
                                representation.push_str(&format!("{}. {}\n", i + 1, text));
                            }
                        }
                    }

                    representation.push_str("\n");
                }
                "table" => {
                    // Format tables
                    representation.push_str("TABLE:\n");

                    // Process table headers
                    let th_selector = Selector::parse("th").unwrap();
                    let headers: Vec<String> = element
                        .select(&th_selector)
                        .map(|th| th.text().collect::<Vec<_>>().join(" ").trim().to_string())
                        .collect();

                    if !headers.is_empty() {
                        representation.push_str(&format!("HEADERS: {}\n", headers.join(" | ")));
                    }

                    // Process table rows
                    let tr_selector = Selector::parse("tr").unwrap();
                    let td_selector = Selector::parse("td").unwrap();

                    for tr in element.select(&tr_selector) {
                        let cells: Vec<String> = tr
                            .select(&td_selector)
                            .map(|td| td.text().collect::<Vec<_>>().join(" ").trim().to_string())
                            .collect();

                        if !cells.is_empty() && !cells.iter().all(|c| c.is_empty()) {
                            representation.push_str(&format!("ROW: {}\n", cells.join(" | ")));
                        }
                    }

                    representation.push_str("\n");
                }
                "form" => {
                    // Format forms
                    representation.push_str("FORM:\n");

                    // Process form elements
                    let input_selector =
                        Selector::parse("input, textarea, select, button").unwrap();
                    let label_selector = Selector::parse("label").unwrap();

                    // Collect labels
                    let mut labels = HashMap::new();
                    for label in element.select(&label_selector) {
                        if let Some(for_attr) = label.value().attr("for") {
                            let text = label
                                .text()
                                .collect::<Vec<_>>()
                                .join(" ")
                                .trim()
                                .to_string();
                            labels.insert(for_attr.to_string(), text);
                        }
                    }

                    // Process inputs
                    for input in element.select(&input_selector) {
                        let input_type = input.value().name.local.as_ref();
                        let input_name = input.value().attr("name").unwrap_or("");
                        let input_id = input.value().attr("id").unwrap_or("");

                        // Get label
                        let label = if !input_id.is_empty() {
                            labels.get(input_id).cloned().unwrap_or_default()
                        } else {
                            String::new()
                        };

                        // Format based on input type
                        match input_type {
                            "input" => {
                                let input_type_attr = input.value().attr("type").unwrap_or("text");
                                let placeholder = input.value().attr("placeholder").unwrap_or("");

                                representation.push_str(&format!(
                                    "INPUT: type={}, name={}, label={}, placeholder={}\n",
                                    input_type_attr, input_name, label, placeholder
                                ));
                            }
                            "textarea" => {
                                let placeholder = input.value().attr("placeholder").unwrap_or("");

                                representation.push_str(&format!(
                                    "TEXTAREA: name={}, label={}, placeholder={}\n",
                                    input_name, label, placeholder
                                ));
                            }
                            "select" => {
                                representation.push_str(&format!(
                                    "SELECT: name={}, label={}\n",
                                    input_name, label
                                ));

                                // Get options
                                let option_selector = Selector::parse("option").unwrap();
                                let options: Vec<String> = input
                                    .select(&option_selector)
                                    .map(|opt| {
                                        opt.text().collect::<Vec<_>>().join(" ").trim().to_string()
                                    })
                                    .collect();

                                if !options.is_empty() {
                                    representation
                                        .push_str(&format!("  OPTIONS: {}\n", options.join(", ")));
                                }
                            }
                            "button" => {
                                let text = input
                                    .text()
                                    .collect::<Vec<_>>()
                                    .join(" ")
                                    .trim()
                                    .to_string();
                                let button_type = input.value().attr("type").unwrap_or("button");

                                representation.push_str(&format!(
                                    "BUTTON: type={}, text={}\n",
                                    button_type, text
                                ));
                            }
                            _ => {}
                        }
                    }

                    representation.push_str("\n");
                }
                "article" | "section" | "main" => {
                    // For main content containers, we'll process their headings and paragraphs
                    let heading_selector = Selector::parse("h1, h2, h3, h4, h5, h6").unwrap();
                    let paragraph_selector = Selector::parse("p").unwrap();

                    representation.push_str(&format!("{}:\n", tag_name.to_uppercase()));

                    // Process headings
                    for heading in element.select(&heading_selector) {
                        let h_tag = heading.value().name.local.as_ref();
                        let level = h_tag.chars().last().unwrap().to_digit(10).unwrap() as usize;
                        let prefix = "#".repeat(level);
                        let text = heading
                            .text()
                            .collect::<Vec<_>>()
                            .join(" ")
                            .trim()
                            .to_string();

                        if !text.is_empty() {
                            representation.push_str(&format!("{} {}\n", prefix, text));
                        }
                    }

                    // Process paragraphs (max 3)
                    for (i, p) in element.select(&paragraph_selector).enumerate() {
                        if i >= 3 {
                            representation.push_str("...\n");
                            break;
                        }

                        let text = p.text().collect::<Vec<_>>().join(" ").trim().to_string();

                        if !text.is_empty() {
                            if text.len() > 200 {
                                representation.push_str(&format!("{:.197}...\n", text));
                            } else {
                                representation.push_str(&format!("{}\n", text));
                            }
                        }
                    }

                    representation.push_str("\n");
                }
                _ => {}
            }
        }

        Ok(representation)
    }
}
