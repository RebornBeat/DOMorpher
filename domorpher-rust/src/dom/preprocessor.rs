//! DOM preprocessor module
//!
//! This module provides functionality for normalizing and optimizing HTML content
//! for further processing. It handles common HTML issues and standardizes the structure.

use html5ever::parse_document;
use html5ever::rcdom::{Handle, NodeData, RcDom};
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{SerializableHandle};

use scraper::{Html, Selector};
use selectors::Element;
use std::default::Default;
use std::io::Cursor;
use std::str::FromStr;
use std::sync::Arc;

/// Options for HTML preprocessing
#[derive(Debug, Clone)]
pub struct PreprocessingOptions {
    /// Cleaning mode
    pub cleaning_mode: CleaningMode,

    /// Whether to fix unclosed tags
    pub fix_unclosed_tags: bool,

    /// Whether to normalize attributes
    pub normalize_attributes: bool,

    /// Whether to remove scripts
    pub remove_scripts: bool,

    /// Whether to remove styles
    pub remove_styles: bool,

    /// Whether to remove comments
    pub remove_comments: bool,

    /// Whether to remove hidden elements
    pub remove_hidden_elements: bool,

    /// Whether to simplify complex structures
    pub simplify_complex_structures: bool,

    /// Whether to enhance semantic structure
    pub enhance_semantic_structure: bool,

    /// Specific elements to remove (selectors)
    pub elements_to_remove: Vec<String>,

    /// Specific elements to preserve (selectors)
    pub elements_to_preserve: Vec<String>,

    /// Whether to convert non-semantic elements to semantic ones
    pub enhance_semantics: bool,

    /// Content type hints
    pub content_type_hints: Vec<String>,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            cleaning_mode: CleaningMode::Standard,
            fix_unclosed_tags: true,
            normalize_attributes: true,
            remove_scripts: true,
            remove_styles: true,
            remove_comments: true,
            remove_hidden_elements: true,
            simplify_complex_structures: true,
            enhance_semantic_structure: true,
            elements_to_remove: vec![
                "script".to_string(),
                "style".to_string(),
                "iframe".to_string(),
                "noscript".to_string(),
                "svg".to_string(),
                "link[rel='stylesheet']".to_string(),
                "meta".to_string(),
            ],
            elements_to_preserve: vec![],
            enhance_semantics: true,
            content_type_hints: vec![],
        }
    }
}

/// HTML cleaning mode
#[derive(Debug, Clone, PartialEq)]
pub enum CleaningMode {
    /// Minimal cleaning
    Minimal,

    /// Standard cleaning
    Standard,

    /// Aggressive cleaning
    Aggressive,
}

/// DOM preprocessor
#[derive(Clone)]
pub struct DomPreprocessor {
    /// Options for preprocessing
    options: PreprocessingOptions,

    /// Element selectors to remove
    remove_selectors: Vec<Selector>,

    /// Element selectors to preserve
    preserve_selectors: Vec<Selector>,
}

impl DomPreprocessor {
    /// Create a new DOM preprocessor with the given options
    pub fn new(options: PreprocessingOptions) -> Self {
        // Compile selectors
        let remove_selectors = options.elements_to_remove
            .iter()
            .filter_map(|selector| Selector::parse(selector).ok())
            .collect();

        let preserve_selectors = options.elements_to_preserve
            .iter()
            .filter_map(|selector| Selector::parse(selector).ok())
            .collect();

        Self {
            options,
            remove_selectors,
            preserve_selectors,
        }
    }

    /// Create a new DOM preprocessor with default options
    pub fn default() -> Self {
        Self::new(PreprocessingOptions::default())
    }

    /// Preprocess the HTML content
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to preprocess
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The preprocessed HTML or an error
    pub fn preprocess(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Apply preprocessing steps
        let mut processed_html = html.to_string();

        // Fix unclosed tags
        if self.options.fix_unclosed_tags {
            processed_html = self.fix_unclosed_tags(&processed_html)?;
        }

        // Parse the fixed HTML
        let document = Html::parse_document(&processed_html);

        // Remove elements
        if !self.remove_selectors.is_empty() {
            processed_html = self.remove_elements(&document)?;
        }

        // Simplify complex structures
        if self.options.simplify_complex_structures {
            processed_html = self.simplify_complex_structures(&processed_html)?;
        }

        // Enhance semantic structure
        if self.options.enhance_semantic_structure {
            processed_html = self.enhance_semantic_structure(&processed_html)?;
        }

        // Normalize attributes
        if self.options.normalize_attributes {
            processed_html = self.normalize_attributes(&processed_html)?;
        }

        Ok(processed_html)
    }

    /// Fix unclosed tags in HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to fix
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The fixed HTML or an error
    fn fix_unclosed_tags(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Use html5ever to parse and serialize the HTML
        let parser = parse_document(RcDom::default(), Default::default());
        let dom = parser.one(html.into());

        match dom {
            Ok(rcdom) => {
                // Serialize back to HTML
                let mut bytes = Vec::new();
                html5ever::serialize::serialize(&mut bytes, &SerializableHandle(rcdom.document), Default::default())
                    .map_err(|e| crate::error::DOMorpherError::DomParsingError(
                        format!("Failed to serialize HTML: {}", e)
                    ))?;

                let fixed_html = String::from_utf8(bytes)
                    .map_err(|e| crate::error::DOMorpherError::DomParsingError(
                        format!("Failed to convert serialized HTML to UTF-8: {}", e)
                    ))?;

                Ok(fixed_html)
            },
            Err(e) => Err(crate::error::DOMorpherError::DomParsingError(
                format!("Failed to parse HTML: {}", e)
            )),
        }
    }

    /// Remove specified elements from the HTML
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The HTML with elements removed or an error
    fn remove_elements(&self, document: &Html) -> Result<String, crate::error::DOMorpherError> {
        // Create a mutable copy of the document
        let mut document_clone = document.clone();
        let dom = document_clone.tree.get();

        // Track elements to remove
        let mut to_remove = Vec::new();

        // First, identify all elements to remove
        for selector in &self.remove_selectors {
            for element in document.select(selector) {
                // Check if the element should be preserved
                let should_preserve = self.preserve_selectors.iter().any(|preserve_selector| {
                    document.select(preserve_selector).any(|preserved_element| {
                        preserved_element.id() == element.id()
                    })
                });

                if !should_preserve {
                    to_remove.push(element.id());
                }
            }
        }

        // Then remove elements (this is just a placeholder as we can't directly modify scraper::Html)
        // In a real implementation, we would need to create a modified DOM tree

        // For simplicity in this example, we'll serialize and parse again
        // This is not efficient for production but demonstrates the concept
        let html_string = document.html();

        // Apply removals through string manipulation
        // This is a simplified approach
        let mut modified_html = html_string.clone();

        for selector in &self.remove_selectors {
            if let Ok(re) = regex::Regex::from_str(&format!("<{0}[^>]*>.*?</{0}>", selector.to_string())) {
                modified_html = re.replace_all(&modified_html, "").to_string();
            }
        }

        Ok(modified_html)
    }

    /// Simplify complex nested structures in the HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to simplify
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The simplified HTML or an error
    fn simplify_complex_structures(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Target deeply nested divs and spans without meaningful attributes
        let div_selector = Selector::parse("div > div > div > div").unwrap();
        let span_selector = Selector::parse("span > span > span").unwrap();

        // Create a mutable copy of the HTML
        let mut simplified_html = html.to_string();

        // Replace deeply nested divs with simpler structure
        // This is a simplified approach using regex
        let nested_div_pattern = regex::Regex::new(r"<div[^>]*>\s*<div[^>]*>\s*<div[^>]*>\s*<div[^>]*>(.*?)</div>\s*</div>\s*</div>\s*</div>").unwrap();
        simplified_html = nested_div_pattern.replace_all(&simplified_html, r"<div>$1</div>").to_string();

        // Replace deeply nested spans with simpler structure
        let nested_span_pattern = regex::Regex::new(r"<span[^>]*>\s*<span[^>]*>\s*<span[^>]*>(.*?)</span>\s*</span>\s*</span>").unwrap();
        simplified_html = nested_span_pattern.replace_all(&simplified_html, r"<span>$1</span>").to_string();

        Ok(simplified_html)
    }

    /// Enhance the semantic structure of the HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to enhance
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The enhanced HTML or an error
    fn enhance_semantic_structure(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Create a mutable copy of the HTML
        let mut enhanced_html = html.to_string();

        // Convert non-semantic divs to semantic elements based on content and position
        // This is a simplified approach using regex patterns

        // Convert div with navigation links to nav
        let nav_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:navigation|menu|nav)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = nav_pattern.replace_all(&enhanced_html, r"<nav>$1</nav>").to_string();

        // Convert div with heading content to header
        let header_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:header|masthead|site-header)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = header_pattern.replace_all(&enhanced_html, r"<header>$1</header>").to_string();

        // Convert div with footer content to footer
        let footer_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:footer|site-footer)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = footer_pattern.replace_all(&enhanced_html, r"<footer>$1</footer>").to_string();

        // Convert div with main content to main
        let main_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:main|content|main-content)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = main_pattern.replace_all(&enhanced_html, r"<main>$1</main>").to_string();

        // Convert div with article content to article
        let article_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:article|post|entry)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = article_pattern.replace_all(&enhanced_html, r"<article>$1</article>").to_string();

        // Convert div with section content to section
        let section_pattern = regex::Regex::new(r"<div[^>]*class=\"[^\"]*(?:section)[^\"]*\"[^>]*>(.*?)</div>").unwrap();
        enhanced_html = section_pattern.replace_all(&enhanced_html, r"<section>$1</section>").to_string();

        // Add ARIA landmarks where appropriate
        let main_aria_pattern = regex::Regex::new(r"<main([^>]*)>").unwrap();
        enhanced_html = main_aria_pattern.replace_all(&enhanced_html, r#"<main$1 role="main">"#).to_string();

        let nav_aria_pattern = regex::Regex::new(r"<nav([^>]*)>").unwrap();
        enhanced_html = nav_aria_pattern.replace_all(&enhanced_html, r#"<nav$1 role="navigation">"#).to_string();

        Ok(enhanced_html)
    }

    /// Normalize attributes in the HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to normalize
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The normalized HTML or an error
    fn normalize_attributes(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Create a mutable copy of the HTML
        let mut normalized_html = html.to_string();

        // Normalize boolean attributes
        let boolean_attrs = ["checked", "selected", "disabled", "readonly", "required", "multiple", "hidden"];

        for attr in &boolean_attrs {
            // Convert attributes with values to boolean
            let attr_pattern = regex::Regex::new(&format!(r#"{}="[^"]*""#, attr)).unwrap();
            normalized_html = attr_pattern.replace_all(&normalized_html, *attr).to_string();
        }

        // Convert inline event handlers to data attributes for safer handling
        let event_attrs = ["onclick", "onmouseover", "onmouseout", "onchange", "onsubmit", "onload"];

        for event in &event_attrs {
            let event_pattern = regex::Regex::new(&format!(r#"{}="([^"]*)""#, event)).unwrap();
            normalized_html = event_pattern.replace_all(&normalized_html, r#"data-event-$1="$2""#).to_string();
        }

        // Normalize URLs in href and src attributes
        let url_pattern = regex::Regex::new(r#"(href|src)="([^"]+)""#).unwrap();
        normalized_html = url_pattern.replace_all(&normalized_html, |caps: &regex::Captures| {
            let attr = &caps[1];
            let mut url = caps[2].to_string();

            // Resolve relative URLs (simplified example)
            if !url.starts_with("http") && !url.starts_with("https") && !url.starts_with("#") && !url.starts_with("data:") {
                // This is a basic placeholder for URL resolution logic
                if url.starts_with("/") {
                    url = format!("https://example.com{}", url);
                } else {
                    url = format!("https://example.com/{}", url);
                }
            }

            format!(r#"{}="{}""#, attr, url)
        }).to_string();

        Ok(normalized_html)
    }

    /// Convert HTML to a clean text representation
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to convert
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The text content or an error
    pub fn html_to_text(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Preprocess the HTML first
        let preprocessed_html = self.preprocess(html)?;

        // Parse the HTML
        let document = Html::parse_document(&preprocessed_html);

        // Extract the text content
        let text = self.extract_text(&document);

        // Clean up whitespace
        let clean_text = self.clean_whitespace(&text);

        Ok(clean_text)
    }

    /// Extract the text content from a DOM
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `String` - The extracted text content
    fn extract_text(&self, document: &Html) -> String {
        // Access the document's tree and extract text recursively
        // For simplicity, we'll use a more basic approach here

        // Remove script and style elements first
        let script_selector = Selector::parse("script, style").unwrap();

        // This is a simplification as we can't directly modify the DOM with scraper
        // In a real implementation, you would recursively traverse and rebuild the tree

        // For this example, we'll just use a simple text extraction
        let mut text = document.root_element()
            .text()
            .collect::<Vec<_>>()
            .join(" ");

        Ok(text)
    }

    /// Clean whitespace in text
    ///
    /// # Arguments
    ///
    /// * `text` - The text to clean
    ///
    /// # Returns
    ///
    /// * `String` - The cleaned text
    fn clean_whitespace(&self, text: &str) -> String {
        // Replace multiple whitespace characters with a single space
        let whitespace_pattern = regex::Regex::new(r"\s+").unwrap();
        let cleaned = whitespace_pattern.replace_all(text, " ").to_string();

        // Trim leading and trailing whitespace
        cleaned.trim().to_string()
    }
}

// Implementation of additional utility methods
impl DomPreprocessor {
    /// Remove boilerplate content from HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to process
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The processed HTML or an error
    pub fn remove_boilerplate(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Common boilerplate selectors
        let boilerplate_selectors = [
            "header", "footer", "nav", ".navigation", "#header", "#footer", "#navigation",
            ".sidebar", "#sidebar", ".menu", "#menu", ".ad", "#ad", ".advertisement",
            ".banner", "#banner", ".share", ".social", ".comment-section", ".disqus",
            ".related-articles", ".recommended", ".popular-posts", ".newsletter",
            ".subscription", ".cookie-notice", ".popup", ".modal"
        ];

        // Create a mutable copy of the HTML
        let mut cleaned_html = html.to_string();

        // Remove each boilerplate element
        for selector_str in boilerplate_selectors.iter() {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    // In real implementation, we would remove the element from the DOM
                    // Here we use a naive regex-based approach for illustration
                    let element_html = element.html();
                    cleaned_html = cleaned_html.replace(&element_html, "");
                }
            }
        }

        Ok(cleaned_html)
    }

    /// Extract the main content from HTML
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to process
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The main content HTML or an error
    pub fn extract_main_content(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Try to find the main content using common selectors
        let main_content_selectors = [
            "main", "article", "#main", "#content", ".main", ".content",
            "#main-content", ".main-content", "[role=main]", ".post", ".article",
            ".entry", ".post-content", ".article-content", ".entry-content"
        ];

        // First, try to find an element matching main content selectors
        for selector_str in main_content_selectors.iter() {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    return Ok(element.html());
                }
            }
        }

        // If no main content selectors match, try a heuristic approach
        // Find the element with the most text content

        // For simplicity, this example just returns the body content
        let body_selector = Selector::parse("body").unwrap();
        if let Some(body) = document.select(&body_selector).next() {
            return Ok(body.html());
        }

        // Fallback: return the original HTML
        Err(crate::error::DOMorpherError::ExtractionError(
            "Failed to identify main content".to_string()
        ))
    }

    /// Add context clues to HTML for improved LLM understanding
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to enhance
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The enhanced HTML or an error
    pub fn add_context_clues(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Parse the HTML
        let document = Html::parse_document(html);

        // Create a mutable copy of the HTML
        let mut enhanced_html = html.to_string();

        // Add aria-description attributes to ambiguous elements
        let links_pattern = regex::Regex::new(r"<a([^>]*)>([^<]*)(</a>)").unwrap();
        enhanced_html = links_pattern.replace_all(&enhanced_html, |caps: &regex::Captures| {
            let attrs = &caps[1];
            let text = &caps[2];
            let close_tag = &caps[3];

            // Don't modify if it already has aria attributes
            if attrs.contains("aria-") {
                return format!("<a{}>{}{}", attrs, text, close_tag);
            }

            format!("<a{} aria-description=\"Link to: {}\"{}{}", attrs, text, text, close_tag)
        }).to_string();

        // Add context comments for sections
        let section_begin_pattern = regex::Regex::new(r"<(section|article|main|header|footer|nav|aside)([^>]*)>").unwrap();
        enhanced_html = section_begin_pattern.replace_all(&enhanced_html, |caps: &regex::Captures| {
            let tag = &caps[1];
            let attrs = &caps[2];

            format!("<!-- Begin {} --><{}{}>{}", tag, tag, attrs)
        }).to_string();

        let section_end_pattern = regex::Regex::new(r"</(section|article|main|header|footer|nav|aside)>").unwrap();
        enhanced_html = section_end_pattern.replace_all(&enhanced_html, |caps: &regex::Captures| {
            let tag = &caps[1];

            format!("</{}>{}", tag, format!("<!-- End {} -->", tag))
        }).to_string();

        Ok(enhanced_html)
    }
}
