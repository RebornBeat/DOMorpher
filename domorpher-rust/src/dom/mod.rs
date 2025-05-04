//! DOM processing module for DOMorpher
//!
//! This module provides functionality for parsing, preprocessing, analyzing,
//! and converting DOM structures to facilitate intelligent extraction.

use std::sync::Arc;

// Re-export public items from submodules
pub use self::accessibility::{AccessibilityAnalyzer, AccessibilityFeature, AccessibilityReport};
pub use self::analyzer::{
    DomAnalysis, DomAnalyzer, DomNode, ElementInfo, ElementRole, ElementType,
};
pub use self::converter::{DomConversionOptions, DomConverter, DomRepresentation};
pub use self::preprocessor::{CleaningMode, DomPreprocessor, PreprocessingOptions};
pub use self::semantic_markdown::{HtmlToMarkdown, MarkdownFormat, MarkdownOptions};

// Declare submodules
pub mod accessibility;
pub mod analyzer;
pub mod converter;
pub mod preprocessor;
pub mod semantic_markdown;

/// DOM processing configuration
#[derive(Debug, Clone)]
pub struct DomProcessingConfig {
    /// Preprocessing options
    pub preprocessing: PreprocessingOptions,

    /// Analysis options
    pub analysis: AnalysisOptions,

    /// Conversion options
    pub conversion: DomConversionOptions,

    /// Accessibility analysis options
    pub accessibility: AccessibilityOptions,

    /// Semantic markdown options
    pub markdown: MarkdownOptions,
}

impl Default for DomProcessingConfig {
    fn default() -> Self {
        Self {
            preprocessing: PreprocessingOptions::default(),
            analysis: AnalysisOptions::default(),
            conversion: DomConversionOptions::default(),
            accessibility: AccessibilityOptions::default(),
            markdown: MarkdownOptions::default(),
        }
    }
}

/// DOM analysis options
#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    /// Whether to analyze content structure
    pub analyze_content_structure: bool,

    /// Whether to analyze element relationships
    pub analyze_relationships: bool,

    /// Whether to identify main content areas
    pub identify_main_content: bool,

    /// Whether to analyze interactive elements
    pub analyze_interactive_elements: bool,

    /// Whether to analyze forms
    pub analyze_forms: bool,

    /// Whether to extract metadata
    pub extract_metadata: bool,

    /// Maximum depth to analyze
    pub max_depth: Option<usize>,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            analyze_content_structure: true,
            analyze_relationships: true,
            identify_main_content: true,
            analyze_interactive_elements: true,
            analyze_forms: true,
            extract_metadata: true,
            max_depth: None,
        }
    }
}

/// Accessibility analysis options
#[derive(Debug, Clone)]
pub struct AccessibilityOptions {
    /// Whether to check for ARIA attributes
    pub check_aria: bool,

    /// Whether to check for proper heading structure
    pub check_headings: bool,

    /// Whether to check for image alt text
    pub check_alt_text: bool,

    /// Whether to check for form labels
    pub check_form_labels: bool,

    /// Whether to check for keyboard navigation
    pub check_keyboard_navigation: bool,

    /// Whether to check for color contrast
    pub check_color_contrast: bool,

    /// Whether to enhance the DOM with accessibility information
    pub enhance_dom: bool,
}

impl Default for AccessibilityOptions {
    fn default() -> Self {
        Self {
            check_aria: true,
            check_headings: true,
            check_alt_text: true,
            check_form_labels: true,
            check_keyboard_navigation: true,
            check_color_contrast: false, // Requires CSS processing, off by default
            enhance_dom: true,
        }
    }
}

/// DOM processor main struct
#[derive(Clone)]
pub struct DomProcessor {
    /// Configuration for DOM processing
    config: DomProcessingConfig,

    /// DOM preprocessor
    preprocessor: DomPreprocessor,

    /// DOM analyzer
    analyzer: DomAnalyzer,

    /// DOM converter
    converter: DomConverter,

    /// Accessibility analyzer
    accessibility_analyzer: AccessibilityAnalyzer,

    /// HTML to markdown converter
    markdown_converter: HtmlToMarkdown,
}

impl DomProcessor {
    /// Create a new DOM processor with the given configuration
    pub fn new(config: DomProcessingConfig) -> Self {
        Self {
            preprocessor: DomPreprocessor::new(config.preprocessing.clone()),
            analyzer: DomAnalyzer::new(config.analysis.clone()),
            converter: DomConverter::new(config.conversion.clone()),
            accessibility_analyzer: AccessibilityAnalyzer::new(config.accessibility.clone()),
            markdown_converter: HtmlToMarkdown::new(config.markdown.clone()),
            config,
        }
    }

    /// Create a new DOM processor with default configuration
    pub fn default() -> Self {
        Self::new(DomProcessingConfig::default())
    }

    /// Process the HTML content
    ///
    /// This performs the full processing pipeline including preprocessing,
    /// analysis, accessibility checking, and conversion.
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to process
    ///
    /// # Returns
    ///
    /// * `Result<ProcessedDom, DomProcessingError>` - The processed DOM or an error
    pub fn process(&self, html: &str) -> Result<ProcessedDom, crate::error::DOMorpherError> {
        // Preprocess the HTML
        let preprocessed_html = self.preprocessor.preprocess(html)?;

        // Parse the HTML into a DOM
        let dom = self.parse_html(&preprocessed_html)?;

        // Analyze the DOM
        let analysis = self.analyzer.analyze(&dom)?;

        // Perform accessibility analysis
        let accessibility_report = self.accessibility_analyzer.analyze(&dom)?;

        // Convert to markdown
        let markdown = self.markdown_converter.convert(&preprocessed_html)?;

        // Convert to other representations
        let representations = self.converter.convert(&dom)?;

        Ok(ProcessedDom {
            original_html: html.to_string(),
            preprocessed_html,
            dom: Arc::new(dom),
            analysis,
            accessibility_report,
            markdown,
            representations,
        })
    }

    /// Parse HTML into a DOM
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to parse
    ///
    /// # Returns
    ///
    /// * `Result<scraper::Html, DomProcessingError>` - The parsed DOM or an error
    fn parse_html(&self, html: &str) -> Result<scraper::Html, crate::error::DOMorpherError> {
        match scraper::Html::parse_document(html) {
            doc => Ok(doc),
            #[allow(unreachable_patterns)]
            _ => Err(crate::error::DOMorpherError::DomParsingError(
                "Failed to parse HTML document".to_string(),
            )),
        }
    }

    /// Process HTML and convert to semantic markdown
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to convert
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The markdown representation or an error
    pub fn to_markdown(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Preprocess the HTML
        let preprocessed_html = self.preprocessor.preprocess(html)?;

        // Convert to markdown
        self.markdown_converter.convert(&preprocessed_html)
    }

    /// Process HTML and extract the main content
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML content to process
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The main content HTML or an error
    pub fn extract_main_content(&self, html: &str) -> Result<String, crate::error::DOMorpherError> {
        // Preprocess the HTML
        let preprocessed_html = self.preprocessor.preprocess(html)?;

        // Parse the HTML into a DOM
        let dom = self.parse_html(&preprocessed_html)?;

        // Analyze the DOM to find main content
        let analysis = self.analyzer.analyze(&dom)?;

        match analysis.main_content_html {
            Some(content) => Ok(content),
            None => Err(crate::error::DOMorpherError::ExtractionError(
                "Failed to identify main content".to_string(),
            )),
        }
    }
}

/// Processed DOM result
#[derive(Clone)]
pub struct ProcessedDom {
    /// Original HTML content
    pub original_html: String,

    /// Preprocessed HTML content
    pub preprocessed_html: String,

    /// Parsed DOM document
    pub dom: Arc<scraper::Html>,

    /// DOM analysis results
    pub analysis: DomAnalysis,

    /// Accessibility analysis report
    pub accessibility_report: AccessibilityReport,

    /// Markdown representation
    pub markdown: String,

    /// Other DOM representations
    pub representations: Vec<DomRepresentation>,
}

impl ProcessedDom {
    /// Get the DOM representation of the specified type
    ///
    /// # Arguments
    ///
    /// * `rep_type` - The representation type
    ///
    /// # Returns
    ///
    /// * `Option<&DomRepresentation>` - The DOM representation if found
    pub fn get_representation(&self, rep_type: &str) -> Option<&DomRepresentation> {
        self.representations.iter().find(|r| r.rep_type == rep_type)
    }

    /// Get the text content of the main content area
    ///
    /// # Returns
    ///
    /// * `Option<&str>` - The main content text if found
    pub fn get_main_content_text(&self) -> Option<&str> {
        self.analysis.main_content_text.as_deref()
    }

    /// Get the HTML of the main content area
    ///
    /// # Returns
    ///
    /// * `Option<&str>` - The main content HTML if found
    pub fn get_main_content_html(&self) -> Option<&str> {
        self.analysis.main_content_html.as_deref()
    }
}
