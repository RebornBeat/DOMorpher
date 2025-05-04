//! DOM structure analysis module
//!
//! This module provides functionality for analyzing DOM structures to identify
//! content areas, element types, relationships, and other structural information.

use scraper::{ElementRef, Html, Selector};
use selectors::Element;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use url::Url;

/// DOM element types
#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    /// Container element
    Container,

    /// Content element
    Content,

    /// Navigation element
    Navigation,

    /// Form element
    Form,

    /// Media element
    Media,

    /// Interactive element
    Interactive,

    /// Metadata element
    Metadata,

    /// Layout element
    Layout,

    /// Unknown element type
    Unknown,
}

/// DOM element roles
#[derive(Debug, Clone, PartialEq)]
pub enum ElementRole {
    /// Navigation role
    Navigation,

    /// Main content role
    Main,

    /// Header role
    Header,

    /// Footer role
    Footer,

    /// Sidebar role
    Sidebar,

    /// Form role
    Form,

    /// Article role
    Article,

    /// Section role
    Section,

    /// Button role
    Button,

    /// Link role
    Link,

    /// Input role
    Input,

    /// Unknown role
    Unknown,
}

/// DOM node representation
#[derive(Debug, Clone)]
pub struct DomNode {
    /// Node ID
    pub id: String,

    /// Node tag name
    pub tag_name: String,

    /// Node attributes
    pub attributes: HashMap<String, String>,

    /// Node text content
    pub text_content: Option<String>,

    /// Node element type
    pub element_type: ElementType,

    /// Node element role
    pub element_role: ElementRole,

    /// Node parent ID
    pub parent_id: Option<String>,

    /// Node children IDs
    pub children: Vec<String>,

    /// Node depth in the DOM tree
    pub depth: usize,

    /// Node is visible
    pub is_visible: bool,

    /// Node is interactive
    pub is_interactive: bool,

    /// Node position info (if available)
    pub position: Option<ElementPosition>,

    /// Node path from root
    pub path: Vec<String>,
}

/// Element position information
#[derive(Debug, Clone)]
pub struct ElementPosition {
    /// X position
    pub x: f64,

    /// Y position
    pub y: f64,

    /// Width
    pub width: f64,

    /// Height
    pub height: f64,

    /// Is visible
    pub is_visible: bool,
}

/// Element information
#[derive(Debug, Clone)]
pub struct ElementInfo {
    /// Element ID
    pub id: String,

    /// Element tag name
    pub tag_name: String,

    /// Element attributes
    pub attributes: HashMap<String, String>,

    /// Element text content
    pub text_content: Option<String>,

    /// Element CSS classes
    pub classes: Vec<String>,

    /// Element element type
    pub element_type: ElementType,

    /// Element element role
    pub element_role: ElementRole,

    /// Element is visible
    pub is_visible: bool,

    /// Element is interactive
    pub is_interactive: bool,

    /// Element position info (if available)
    pub position: Option<ElementPosition>,

    /// Element selector path
    pub selector_path: String,
}

/// DOM analysis results
#[derive(Debug, Clone)]
pub struct DomAnalysis {
    /// Document title
    pub title: Option<String>,

    /// Document metadata
    pub metadata: HashMap<String, String>,

    /// Main content area selector
    pub main_content_selector: Option<String>,

    /// Main content text
    pub main_content_text: Option<String>,

    /// Main content HTML
    pub main_content_html: Option<String>,

    /// Navigation areas
    pub navigation_areas: Vec<ElementInfo>,

    /// Header area
    pub header: Option<ElementInfo>,

    /// Footer area
    pub footer: Option<ElementInfo>,

    /// Sidebar areas
    pub sidebars: Vec<ElementInfo>,

    /// Forms
    pub forms: Vec<ElementInfo>,

    /// Interactive elements
    pub interactive_elements: Vec<ElementInfo>,

    /// DOM tree structure
    pub dom_tree: HashMap<String, DomNode>,

    /// Total element count
    pub element_count: usize,

    /// Content density score (0-1)
    pub content_density: f64,

    /// Interactive element density score (0-1)
    pub interactive_density: f64,

    /// Element type distribution
    pub element_type_distribution: HashMap<ElementType, usize>,

    /// Page type classification
    pub page_type: Option<String>,

    /// Content structure classification
    pub content_structure: Option<String>,
}

/// Options for DOM analysis
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

/// DOM analyzer
#[derive(Clone)]
pub struct DomAnalyzer {
    /// Options for analysis
    options: AnalysisOptions,

    /// Main content selectors to try
    main_content_selectors: Vec<Selector>,

    /// Navigation selectors to try
    navigation_selectors: Vec<Selector>,

    /// Header selectors to try
    header_selectors: Vec<Selector>,

    /// Footer selectors to try
    footer_selectors: Vec<Selector>,

    /// Sidebar selectors to try
    sidebar_selectors: Vec<Selector>,

    /// Form selectors to try
    form_selectors: Vec<Selector>,

    /// Interactive element selectors to try
    interactive_selectors: Vec<Selector>,
}

impl DomAnalyzer {
    /// Create a new DOM analyzer with the given options
    pub fn new(options: AnalysisOptions) -> Self {
        // Compile selectors
        let main_content_selectors = vec![
            "main",
            "article",
            "#main",
            "#content",
            ".main",
            ".content",
            "#main-content",
            ".main-content",
            "[role=main]",
            ".post",
            ".article",
            ".entry",
            ".post-content",
            ".article-content",
            ".entry-content",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let navigation_selectors = vec![
            "nav",
            "[role=navigation]",
            "#navigation",
            ".navigation",
            "#nav",
            ".nav",
            "#main-nav",
            ".main-nav",
            "#menu",
            ".menu",
            "header ul",
            ".navbar",
            ".menu-container",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let header_selectors = vec![
            "header",
            "#header",
            ".header",
            ".site-header",
            "#site-header",
            "[role=banner]",
            ".masthead",
            "#masthead",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let footer_selectors = vec![
            "footer",
            "#footer",
            ".footer",
            ".site-footer",
            "#site-footer",
            "[role=contentinfo]",
            ".bottom",
            "#bottom",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let sidebar_selectors = vec![
            "aside",
            "#sidebar",
            ".sidebar",
            ".widget-area",
            "#secondary",
            "[role=complementary]",
            ".col-sidebar",
            "#right-sidebar",
            "#left-sidebar",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let form_selectors = vec![
            "form",
            "[role=form]",
            ".form",
            "#form",
            ".contact-form",
            "#contact-form",
            ".search-form",
            "#search-form",
            ".login-form",
            "#login-form",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        let interactive_selectors = vec![
            "a",
            "button",
            "input",
            "select",
            "textarea",
            "[role=button]",
            "[onClick]",
            "[href]",
            "[role=link]",
            "[role=checkbox]",
            "[role=radio]",
            "[role=tab]",
            "[role=menuitem]",
            "[role=combobox]",
            "[role=slider]",
            "details",
        ]
        .into_iter()
        .filter_map(|sel| Selector::parse(sel).ok())
        .collect();

        Self {
            options,
            main_content_selectors,
            navigation_selectors,
            header_selectors,
            footer_selectors,
            sidebar_selectors,
            form_selectors,
            interactive_selectors,
        }
    }

    /// Create a new DOM analyzer with default options
    pub fn default() -> Self {
        Self::new(AnalysisOptions::default())
    }

    /// Analyze the DOM
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Result<DomAnalysis, DomProcessingError>` - The analysis results or an error
    pub fn analyze(&self, document: &Html) -> Result<DomAnalysis, crate::error::DOMorpherError> {
        // Create analysis structure
        let mut analysis = DomAnalysis {
            title: None,
            metadata: HashMap::new(),
            main_content_selector: None,
            main_content_text: None,
            main_content_html: None,
            navigation_areas: Vec::new(),
            header: None,
            footer: None,
            sidebars: Vec::new(),
            forms: Vec::new(),
            interactive_elements: Vec::new(),
            dom_tree: HashMap::new(),
            element_count: 0,
            content_density: 0.0,
            interactive_density: 0.0,
            element_type_distribution: HashMap::new(),
            page_type: None,
            content_structure: None,
        };

        // Extract metadata
        if self.options.extract_metadata {
            self.extract_metadata(document, &mut analysis)?;
        }

        // Extract title
        analysis.title = self.extract_title(document);

        // Identify main content
        if self.options.identify_main_content {
            self.identify_main_content(document, &mut analysis)?;
        }

        // Analyze document structure
        self.analyze_document_structure(document, &mut analysis)?;

        // Analyze navigation areas
        self.identify_navigation_areas(document, &mut analysis)?;

        // Identify header and footer
        self.identify_header_and_footer(document, &mut analysis)?;

        // Identify sidebars
        self.identify_sidebars(document, &mut analysis)?;

        // Analyze forms
        if self.options.analyze_forms {
            self.analyze_forms(document, &mut analysis)?;
        }

        // Analyze interactive elements
        if self.options.analyze_interactive_elements {
            self.analyze_interactive_elements(document, &mut analysis)?;
        }

        // Calculate statistics
        self.calculate_statistics(&mut analysis)?;

        // Classify page type
        self.classify_page_type(&mut analysis)?;

        Ok(analysis)
    }

    /// Extract metadata from the document
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn extract_metadata(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Extract meta tags
        let meta_selector = Selector::parse("meta").unwrap();

        for meta in document.select(&meta_selector) {
            let name = meta
                .value()
                .attr("name")
                .or_else(|| meta.value().attr("property"));
            let content = meta.value().attr("content");

            if let (Some(name), Some(content)) = (name, content) {
                analysis
                    .metadata
                    .insert(name.to_string(), content.to_string());
            }
        }

        // Extract Open Graph metadata
        let og_meta_selector = Selector::parse("meta[property^='og:']").unwrap();

        for meta in document.select(&og_meta_selector) {
            let property = meta.value().attr("property");
            let content = meta.value().attr("content");

            if let (Some(property), Some(content)) = (property, content) {
                analysis
                    .metadata
                    .insert(property.to_string(), content.to_string());
            }
        }

        // Extract JSON-LD structured data
        let script_selector = Selector::parse("script[type='application/ld+json']").unwrap();

        for script in document.select(&script_selector) {
            if let Some(json_text) = script.text().next() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_text) {
                    if let Some(obj) = json.as_object() {
                        for (key, value) in obj {
                            if let Some(str_value) = value.as_str() {
                                analysis
                                    .metadata
                                    .insert(format!("jsonld:{}", key), str_value.to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract the title from the document
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    ///
    /// # Returns
    ///
    /// * `Option<String>` - The title if found
    fn extract_title(&self, document: &Html) -> Option<String> {
        let title_selector = Selector::parse("title").unwrap();

        document
            .select(&title_selector)
            .next()
            .and_then(|element| element.text().next().map(|text| text.trim().to_string()))
    }

    /// Identify the main content area
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn identify_main_content(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Try each main content selector
        for selector in &self.main_content_selectors {
            if let Some(element) = document.select(selector).next() {
                analysis.main_content_selector = Some(selector.to_string());
                analysis.main_content_html = Some(element.html());
                analysis.main_content_text = Some(element.text().collect::<Vec<_>>().join(" "));
                return Ok(());
            }
        }

        // If no selector matched, try a heuristic approach
        self.identify_main_content_heuristic(document, analysis)?;

        Ok(())
    }

    /// Identify the main content area using heuristics
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn identify_main_content_heuristic(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Find the element with the most text content
        let body_selector = Selector::parse("body").unwrap();

        if let Some(body) = document.select(&body_selector).next() {
            // Score each direct child of body
            let mut best_element = None;
            let mut best_score = 0.0;

            for element in body.children() {
                if let Some(element_ref) = ElementRef::wrap(element) {
                    let text_content = element_ref.text().collect::<Vec<_>>().join(" ");
                    let score = self.score_main_content(&element_ref, text_content.len());

                    if score > best_score {
                        best_score = score;
                        best_element = Some(element_ref);
                    }
                }
            }

            if let Some(element) = best_element {
                analysis.main_content_selector = Some(self.generate_selector_path(&element));
                analysis.main_content_html = Some(element.html());
                analysis.main_content_text = Some(element.text().collect::<Vec<_>>().join(" "));
            }
        }

        Ok(())
    }

    /// Score an element as potential main content
    ///
    /// # Arguments
    ///
    /// * `element` - The element to score
    /// * `text_length` - The length of the text content
    ///
    /// # Returns
    ///
    /// * `f64` - The score (higher is better)
    fn score_main_content(&self, element: &ElementRef, text_length: usize) -> f64 {
        let mut score = text_length as f64;

        // Bonus for semantic elements
        match element.value().name.local.as_ref() {
            "main" => score *= 3.0,
            "article" => score *= 2.5,
            "section" => score *= 2.0,
            "div" => score *= 1.0,
            _ => score *= 0.5,
        }

        // Bonus for semantic class names
        let class_value = element.value().attr("class").unwrap_or("");
        let classes = class_value.split_whitespace().collect::<Vec<_>>();

        for class in classes {
            match class {
                "content" | "main" | "main-content" | "article" | "post" => score *= 1.5,
                "sidebar" | "widget" | "nav" | "menu" | "header" | "footer" => score *= 0.2,
                _ => {}
            }
        }

        // Bonus for role attribute
        if let Some(role) = element.value().attr("role") {
            match role {
                "main" => score *= 3.0,
                "article" => score *= 2.5,
                "complementary" | "navigation" | "banner" | "contentinfo" => score *= 0.2,
                _ => {}
            }
        }

        // Penalty for very short content
        if text_length < 100 {
            score *= 0.5;
        }

        score
    }

    /// Analyze the document structure
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn analyze_document_structure(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Start with the body element
        let body_selector = Selector::parse("body").unwrap();

        if let Some(body) = document.select(&body_selector).next() {
            // Build DOM tree
            self.build_dom_tree(body, None, 0, &mut analysis.dom_tree)?;

            // Count elements
            analysis.element_count = analysis.dom_tree.len();
        }

        Ok(())
    }

    /// Build the DOM tree representation
    ///
    /// # Arguments
    ///
    /// * `element` - The current element
    /// * `parent_id` - The parent element ID
    /// * `depth` - The current depth
    /// * `dom_tree` - The DOM tree map to update
    ///
    /// # Returns
    ///
    /// * `Result<String, DomProcessingError>` - The element ID or an error
    fn build_dom_tree(
        &self,
        element: ElementRef,
        parent_id: Option<String>,
        depth: usize,
        dom_tree: &mut HashMap<String, DomNode>,
    ) -> Result<String, crate::error::DOMorpherError> {
        // Check if we've reached the maximum depth
        if let Some(max_depth) = self.options.max_depth {
            if depth > max_depth {
                return Ok(String::new());
            }
        }

        // Generate unique ID for this element
        let id = format!("e{}", dom_tree.len());

        // Get tag name
        let tag_name = element.value().name.local.to_string();

        // Get attributes
        let mut attributes = HashMap::new();
        for attr in element.value().attrs() {
            attributes.insert(attr.0.local.to_string(), attr.1.to_string());
        }

        // Get text content
        let text_content = Some(
            element
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string(),
        )
        .filter(|s| !s.is_empty());

        // Determine element type and role
        let element_type = self.determine_element_type(&element);
        let element_role = self.determine_element_role(&element);

        // Determine if element is visible (basic heuristic)
        let is_visible = !attributes.contains_key("hidden")
            && !attributes.get("style").map_or(false, |s| {
                s.contains("display: none") || s.contains("visibility: hidden")
            });

        // Determine if element is interactive
        let is_interactive = element_type == ElementType::Interactive
            || attributes.contains_key("onclick")
            || matches!(
                tag_name.as_str(),
                "a" | "button" | "input" | "select" | "textarea"
            );

        // Build path
        let mut path = Vec::new();
        if let Some(ref parent_id) = parent_id {
            if let Some(parent_node) = dom_tree.get(parent_id) {
                path.extend(parent_node.path.clone());
            }
        }
        path.push(tag_name.clone());

        // Create node
        let node = DomNode {
            id: id.clone(),
            tag_name,
            attributes,
            text_content,
            element_type,
            element_role,
            parent_id: parent_id.clone(),
            children: Vec::new(),
            depth,
            is_visible,
            is_interactive,
            position: None, // Would require JavaScript execution to get accurate position
            path,
        };

        // Add to tree
        dom_tree.insert(id.clone(), node);

        // Process children
        for child in element.children() {
            if let Some(child_ref) = ElementRef::wrap(child) {
                let child_id =
                    self.build_dom_tree(child_ref, Some(id.clone()), depth + 1, dom_tree)?;

                if !child_id.is_empty() {
                    if let Some(node) = dom_tree.get_mut(&id) {
                        node.children.push(child_id);
                    }
                }
            }
        }

        Ok(id)
    }

    /// Determine the element type
    ///
    /// # Arguments
    ///
    /// * `element` - The element
    ///
    /// # Returns
    ///
    /// * `ElementType` - The element type
    fn determine_element_type(&self, element: &ElementRef) -> ElementType {
        match element.value().name.local.as_ref() {
            "div" | "span" | "section" | "article" | "main" | "aside" | "header" | "footer"
            | "nav" => ElementType::Container,
            "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote" | "pre" | "code"
            | "ul" | "ol" | "li" => ElementType::Content,
            "a" | "button" | "input" | "select" | "textarea" | "label" | "option" => {
                ElementType::Interactive
            }
            "img" | "video" | "audio" | "canvas" | "svg" => ElementType::Media,
            "form" | "fieldset" | "legend" => ElementType::Form,
            "meta" | "link" | "title" | "style" | "script" => ElementType::Metadata,
            "table" | "tr" | "td" | "th" | "thead" | "tbody" | "tfoot" => ElementType::Layout,
            _ => {
                // Check for ARIA roles
                if let Some(role) = element.value().attr("role") {
                    match role {
                        "button" | "link" | "checkbox" | "radio" | "tab" | "menuitem"
                        | "combobox" => ElementType::Interactive,
                        "form" => ElementType::Form,
                        "navigation" | "menu" => ElementType::Navigation,
                        "img" | "banner" => ElementType::Media,
                        _ => ElementType::Unknown,
                    }
                } else {
                    ElementType::Unknown
                }
            }
        }
    }

    /// Determine the element role
    ///
    /// # Arguments
    ///
    /// * `element` - The element
    ///
    /// # Returns
    ///
    /// * `ElementRole` - The element role
    fn determine_element_role(&self, element: &ElementRef) -> ElementRole {
        // First check explicit role attribute
        if let Some(role) = element.value().attr("role") {
            match role {
                "navigation" => return ElementRole::Navigation,
                "main" => return ElementRole::Main,
                "banner" => return ElementRole::Header,
                "contentinfo" => return ElementRole::Footer,
                "complementary" => return ElementRole::Sidebar,
                "form" => return ElementRole::Form,
                "article" => return ElementRole::Article,
                "button" => return ElementRole::Button,
                "link" => return ElementRole::Link,
                "textbox" | "searchbox" => return ElementRole::Input,
                _ => {}
            }
        }

        // Then check tag name
        match element.value().name.local.as_ref() {
            "nav" => ElementRole::Navigation,
            "main" => ElementRole::Main,
            "header" => ElementRole::Header,
            "footer" => ElementRole::Footer,
            "aside" => ElementRole::Sidebar,
            "form" => ElementRole::Form,
            "article" => ElementRole::Article,
            "section" => ElementRole::Section,
            "button" => ElementRole::Button,
            "a" => ElementRole::Link,
            "input" | "textarea" | "select" => ElementRole::Input,
            _ => {
                // Check classes for hints
                let class_value = element.value().attr("class").unwrap_or("");
                let classes = class_value.split_whitespace().collect::<Vec<_>>();

                for class in classes {
                    match class {
                        "nav" | "navigation" | "menu" | "navbar" => return ElementRole::Navigation,
                        "main" | "content" | "main-content" => return ElementRole::Main,
                        "header" | "site-header" | "page-header" => return ElementRole::Header,
                        "footer" | "site-footer" | "page-footer" => return ElementRole::Footer,
                        "sidebar" | "widget-area" => return ElementRole::Sidebar,
                        "form" | "contact-form" | "search-form" => return ElementRole::Form,
                        "article" | "post" | "entry" => return ElementRole::Article,
                        "section" => return ElementRole::Section,
                        "btn" | "button" => return ElementRole::Button,
                        "link" => return ElementRole::Link,
                        "input" | "field" => return ElementRole::Input,
                        _ => {}
                    }
                }

                // Check ID for hints
                if let Some(id) = element.value().attr("id") {
                    match id {
                        "nav" | "navigation" | "menu" | "navbar" => return ElementRole::Navigation,
                        "main" | "content" | "main-content" => return ElementRole::Main,
                        "header" | "site-header" | "page-header" => return ElementRole::Header,
                        "footer" | "site-footer" | "page-footer" => return ElementRole::Footer,
                        "sidebar" | "widget-area" => return ElementRole::Sidebar,
                        _ => {}
                    }
                }

                ElementRole::Unknown
            }
        }
    }

    /// Identify navigation areas
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn identify_navigation_areas(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        for selector in &self.navigation_selectors {
            for element in document.select(selector) {
                let element_info = self.create_element_info(
                    &element,
                    ElementType::Navigation,
                    ElementRole::Navigation,
                );
                analysis.navigation_areas.push(element_info);
            }
        }

        Ok(())
    }

    /// Identify header and footer
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn identify_header_and_footer(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Try to find header
        for selector in &self.header_selectors {
            if let Some(element) = document.select(selector).next() {
                let element_info =
                    self.create_element_info(&element, ElementType::Container, ElementRole::Header);
                analysis.header = Some(element_info);
                break;
            }
        }

        // Try to find footer
        for selector in &self.footer_selectors {
            if let Some(element) = document.select(selector).next() {
                let element_info =
                    self.create_element_info(&element, ElementType::Container, ElementRole::Footer);
                analysis.footer = Some(element_info);
                break;
            }
        }

        Ok(())
    }

    /// Identify sidebars
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn identify_sidebars(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        for selector in &self.sidebar_selectors {
            for element in document.select(selector) {
                let element_info = self.create_element_info(
                    &element,
                    ElementType::Container,
                    ElementRole::Sidebar,
                );
                analysis.sidebars.push(element_info);
            }
        }

        Ok(())
    }

    /// Analyze forms
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn analyze_forms(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        for selector in &self.form_selectors {
            for element in document.select(selector) {
                let element_info =
                    self.create_element_info(&element, ElementType::Form, ElementRole::Form);
                analysis.forms.push(element_info);
            }
        }

        Ok(())
    }

    /// Analyze interactive elements
    ///
    /// # Arguments
    ///
    /// * `document` - The parsed HTML document
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn analyze_interactive_elements(
        &self,
        document: &Html,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        for selector in &self.interactive_selectors {
            for element in document.select(selector) {
                // Determine correct element type and role based on the element
                let element_type = self.determine_element_type(&element);
                let element_role = self.determine_element_role(&element);

                let element_info = self.create_element_info(&element, element_type, element_role);
                analysis.interactive_elements.push(element_info);
            }
        }

        Ok(())
    }

    /// Calculate statistics
    ///
    /// # Arguments
    ///
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn calculate_statistics(
        &self,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Count elements by type
        let mut content_elements = 0;
        let mut interactive_elements = 0;

        for node in analysis.dom_tree.values() {
            let count = analysis
                .element_type_distribution
                .entry(node.element_type.clone())
                .or_insert(0);
            *count += 1;

            if node.element_type == ElementType::Content {
                content_elements += 1;
            }

            if node.is_interactive {
                interactive_elements += 1;
            }
        }

        // Calculate densities
        if analysis.element_count > 0 {
            analysis.content_density = content_elements as f64 / analysis.element_count as f64;
            analysis.interactive_density =
                interactive_elements as f64 / analysis.element_count as f64;
        }

        Ok(())
    }

    /// Classify page type
    ///
    /// # Arguments
    ///
    /// * `analysis` - The analysis results to update
    ///
    /// # Returns
    ///
    /// * `Result<(), DomProcessingError>` - Success or an error
    fn classify_page_type(
        &self,
        analysis: &mut DomAnalysis,
    ) -> Result<(), crate::error::DOMorpherError> {
        // Basic page type classification based on content structure

        // Check for e-commerce patterns
        if analysis.interactive_elements.iter().any(|e| {
            e.attributes.get("class").map_or(false, |c| {
                c.contains("add-to-cart") || c.contains("buy-now")
            }) || e.text_content.as_ref().map_or(false, |t| {
                t.contains("Add to Cart") || t.contains("Buy Now")
            })
        }) {
            analysis.page_type = Some("e-commerce".to_string());
            return Ok(());
        }

        // Check for blog/article pattern
        if analysis.main_content_html.is_some()
            && (analysis.dom_tree.values().any(|n| n.tag_name == "article")
                || analysis.metadata.contains_key("article:published_time"))
        {
            analysis.page_type = Some("article".to_string());
            return Ok(());
        }

        // Check for search results
        if analysis
            .title
            .as_ref()
            .map_or(false, |t| t.to_lowercase().contains("search"))
            || analysis.interactive_elements.iter().any(|e| {
                e.attributes
                    .get("class")
                    .map_or(false, |c| c.contains("search-result"))
            })
        {
            analysis.page_type = Some("search-results".to_string());
            return Ok(());
        }

        // Check for forms
        if !analysis.forms.is_empty() {
            if analysis.forms.iter().any(|f| {
                f.attributes
                    .get("class")
                    .map_or(false, |c| c.contains("contact-form"))
                    || f.attributes
                        .get("id")
                        .map_or(false, |c| c.contains("contact-form"))
            }) {
                analysis.page_type = Some("contact-form".to_string());
                return Ok(());
            }

            analysis.page_type = Some("form".to_string());
            return Ok(());
        }

        // Default to generic content page
        analysis.page_type = Some("content".to_string());

        Ok(())
    }

    /// Create element info
    ///
    /// # Arguments
    ///
    /// * `element` - The element
    /// * `element_type` - The element type
    /// * `element_role` - The element role
    ///
    /// # Returns
    ///
    /// * `ElementInfo` - The element info
    fn create_element_info(
        &self,
        element: &ElementRef,
        element_type: ElementType,
        element_role: ElementRole,
    ) -> ElementInfo {
        // Generate unique ID
        let id = format!("ei{}", rand::random::<u32>());

        // Get tag name
        let tag_name = element.value().name.local.to_string();

        // Get attributes
        let mut attributes = HashMap::new();
        for attr in element.value().attrs() {
            attributes.insert(attr.0.local.to_string(), attr.1.to_string());
        }

        // Get classes
        let classes = element
            .value()
            .attr("class")
            .map(|c| c.split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_else(Vec::new);

        // Get text content
        let text_content = Some(
            element
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string(),
        )
        .filter(|s| !s.is_empty());

        // Determine if element is visible (basic heuristic)
        let is_visible = !attributes.contains_key("hidden")
            && !attributes.get("style").map_or(false, |s| {
                s.contains("display: none") || s.contains("visibility: hidden")
            });

        // Determine if element is interactive
        let is_interactive = element_type == ElementType::Interactive
            || attributes.contains_key("onclick")
            || matches!(
                tag_name.as_str(),
                "a" | "button" | "input" | "select" | "textarea"
            );

        // Generate selector path
        let selector_path = self.generate_selector_path(element);

        ElementInfo {
            id,
            tag_name,
            attributes,
            text_content,
            classes,
            element_type,
            element_role,
            is_visible,
            is_interactive,
            position: None,
            selector_path,
        }
    }

    /// Generate a selector path for an element
    ///
    /// # Arguments
    ///
    /// * `element` - The element
    ///
    /// # Returns
    ///
    /// * `String` - The selector path
    fn generate_selector_path(&self, element: &ElementRef) -> String {
        // Try to generate a unique selector path for the element

        // Check for ID
        if let Some(id) = element.value().attr("id") {
            return format!("#{}", id);
        }

        // Build path of tags and classes
        let mut path = Vec::new();
        let mut current = Some(*element);

        while let Some(element) = current {
            let mut selector = element.value().name.local.to_string();

            if let Some(id) = element.value().attr("id") {
                selector = format!("{}#{}", selector, id);
            } else if let Some(class) = element.value().attr("class") {
                let first_class = class.split_whitespace().next();
                if let Some(class_name) = first_class {
                    selector = format!("{}.{}", selector, class_name);
                }
            }

            path.push(selector);

            current = element.parent().and_then(ElementRef::wrap);
        }

        // Reverse path and join
        path.reverse();
        path.join(" > ")
    }
}
