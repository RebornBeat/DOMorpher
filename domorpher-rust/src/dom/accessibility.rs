//! Accessibility analysis module for DOMorpher
//!
//! This module provides functionality for analyzing web content for accessibility
//! issues, generating reports, and enhancing DOM with accessibility information.
//! It supports various WCAG guidelines and can provide recommendations for improving
//! accessibility.

use cssparser::{Parser as CSSParser, ParserInput, RGBA};
use scraper::{ElementRef, Html, Selector};
use selectors::Element;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::dom::preprocessor::PreprocessingOptions;
use crate::error::DOMorpherError;

/// Accessibility feature types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccessibilityFeature {
    /// ARIA attributes for enhanced semantics
    Aria,
    /// Proper heading structure
    Headings,
    /// Alt text for images
    AltText,
    /// Labels for form elements
    FormLabels,
    /// Keyboard navigation support
    KeyboardNavigation,
    /// Proper color contrast
    ColorContrast,
    /// Semantic HTML structure
    SemanticStructure,
    /// Link text quality
    LinkText,
    /// Table accessibility features
    TableAccessibility,
    /// Focus management
    FocusManagement,
    /// Language specification
    LanguageSpecification,
    /// Skip navigation links
    SkipNavigation,
    /// Document title
    DocumentTitle,
    /// Frame titles
    FrameTitles,
    /// Audio/video controls
    MediaControls,
    /// Form validation
    FormValidation,
    /// Error identification
    ErrorIdentification,
    /// ARIA landmark roles
    LandmarkRoles,
}

/// Accessibility issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessibilitySeverity {
    /// Info-level issue (suggestion)
    Info,
    /// Warning-level issue (should be addressed)
    Warning,
    /// Error-level issue (must be addressed)
    Error,
    /// Critical-level issue (major barrier)
    Critical,
}

/// Accessibility issue
#[derive(Debug, Clone)]
pub struct AccessibilityIssue {
    /// Feature category of the issue
    pub feature: AccessibilityFeature,
    /// Severity level
    pub severity: AccessibilitySeverity,
    /// Description of the issue
    pub description: String,
    /// Element selector path
    pub element_path: String,
    /// WCAG success criterion reference
    pub wcag_reference: Option<String>,
    /// Suggested fix
    pub suggestion: String,
}

/// Accessibility success
#[derive(Debug, Clone)]
pub struct AccessibilitySuccess {
    /// Feature category of the success
    pub feature: AccessibilityFeature,
    /// Description of the success
    pub description: String,
    /// Element count or reference
    pub element_reference: String,
}

/// Accessibility report
#[derive(Debug, Clone)]
pub struct AccessibilityReport {
    /// Identified issues
    pub issues: Vec<AccessibilityIssue>,
    /// Successful implementations
    pub successes: Vec<AccessibilitySuccess>,
    /// Overall score (0-100)
    pub score: f64,
    /// Feature-specific scores
    pub feature_scores: HashMap<AccessibilityFeature, f64>,
    /// Elements with accessibility enhancements
    pub enhanced_elements: HashSet<String>,
    /// Report generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Metadata about the analyzed document
    pub document_metadata: HashMap<String, String>,
}

impl AccessibilityReport {
    /// Create a new empty accessibility report
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            successes: Vec::new(),
            score: 0.0,
            feature_scores: HashMap::new(),
            enhanced_elements: HashSet::new(),
            timestamp: chrono::Utc::now(),
            document_metadata: HashMap::new(),
        }
    }

    /// Add an issue to the report
    pub fn add_issue(&mut self, issue: AccessibilityIssue) {
        self.issues.push(issue);
    }

    /// Add a success to the report
    pub fn add_success(&mut self, success: AccessibilitySuccess) {
        self.successes.push(success);
    }

    /// Get issues by severity
    pub fn get_issues_by_severity(
        &self,
        severity: AccessibilitySeverity,
    ) -> Vec<&AccessibilityIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Get issues by feature
    pub fn get_issues_by_feature(
        &self,
        feature: &AccessibilityFeature,
    ) -> Vec<&AccessibilityIssue> {
        self.issues
            .iter()
            .filter(|i| i.feature == *feature)
            .collect()
    }

    /// Calculate and update the overall score
    pub fn calculate_score(&mut self) {
        // Base score starts at 100
        let mut base_score = 100.0;

        // Calculate penalties by severity
        let critical_count = self
            .get_issues_by_severity(AccessibilitySeverity::Critical)
            .len();
        let error_count = self
            .get_issues_by_severity(AccessibilitySeverity::Error)
            .len();
        let warning_count = self
            .get_issues_by_severity(AccessibilitySeverity::Warning)
            .len();
        let info_count = self
            .get_issues_by_severity(AccessibilitySeverity::Info)
            .len();

        // Apply penalties
        base_score -= (critical_count as f64) * 10.0; // -10 points per critical issue
        base_score -= (error_count as f64) * 5.0; // -5 points per error
        base_score -= (warning_count as f64) * 2.0; // -2 points per warning
        base_score -= (info_count as f64) * 0.5; // -0.5 points per info item

        // Clamp score between 0 and 100
        self.score = base_score.max(0.0).min(100.0);

        // Calculate feature-specific scores
        for feature in [
            AccessibilityFeature::Aria,
            AccessibilityFeature::Headings,
            AccessibilityFeature::AltText,
            AccessibilityFeature::FormLabels,
            AccessibilityFeature::KeyboardNavigation,
            AccessibilityFeature::ColorContrast,
            AccessibilityFeature::SemanticStructure,
            AccessibilityFeature::LinkText,
            AccessibilityFeature::TableAccessibility,
            AccessibilityFeature::LandmarkRoles,
        ]
        .iter()
        {
            // Count issues by severity for this feature
            let feature_critical = self
                .issues
                .iter()
                .filter(|i| i.feature == *feature && i.severity == AccessibilitySeverity::Critical)
                .count();
            let feature_error = self
                .issues
                .iter()
                .filter(|i| i.feature == *feature && i.severity == AccessibilitySeverity::Error)
                .count();
            let feature_warning = self
                .issues
                .iter()
                .filter(|i| i.feature == *feature && i.severity == AccessibilitySeverity::Warning)
                .count();

            // Count successes for this feature
            let feature_success = self
                .successes
                .iter()
                .filter(|s| s.feature == *feature)
                .count();

            // Calculate feature score
            let mut feature_score = 100.0;
            feature_score -= (feature_critical as f64) * 20.0;
            feature_score -= (feature_error as f64) * 10.0;
            feature_score -= (feature_warning as f64) * 5.0;
            feature_score += (feature_success as f64) * 2.0;

            // Clamp feature score
            let clamped_score = feature_score.max(0.0).min(100.0);
            self.feature_scores.insert(feature.clone(), clamped_score);
        }
    }

    /// Get an overall accessibility grade (A, AA, AAA)
    pub fn get_grade(&self) -> &str {
        if self.score >= 95.0
            && self
                .get_issues_by_severity(AccessibilitySeverity::Critical)
                .is_empty()
        {
            "AAA"
        } else if self.score >= 85.0
            && self
                .get_issues_by_severity(AccessibilitySeverity::Critical)
                .is_empty()
        {
            "AA"
        } else if self.score >= 70.0 {
            "A"
        } else {
            "Failing"
        }
    }

    /// Generate a summary of the report
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!(
            "Accessibility Score: {:.1}% (Grade: {})\n\n",
            self.score,
            self.get_grade()
        ));

        summary.push_str("Issues by Severity:\n");
        summary.push_str(&format!(
            "- Critical: {}\n",
            self.get_issues_by_severity(AccessibilitySeverity::Critical)
                .len()
        ));
        summary.push_str(&format!(
            "- Error: {}\n",
            self.get_issues_by_severity(AccessibilitySeverity::Error)
                .len()
        ));
        summary.push_str(&format!(
            "- Warning: {}\n",
            self.get_issues_by_severity(AccessibilitySeverity::Warning)
                .len()
        ));
        summary.push_str(&format!(
            "- Info: {}\n",
            self.get_issues_by_severity(AccessibilitySeverity::Info)
                .len()
        ));

        summary.push_str("\nFeature Scores:\n");
        let mut sorted_features: Vec<_> = self.feature_scores.iter().collect();
        sorted_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (feature, score) in sorted_features {
            summary.push_str(&format!("- {:?}: {:.1}%\n", feature, score));
        }

        summary
    }
}

/// Accessibility analyzer configuration options
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

    /// Whether to check for semantic HTML structure
    pub check_semantic_structure: bool,

    /// Whether to check link text quality
    pub check_link_text: bool,

    /// Whether to check table accessibility
    pub check_table_accessibility: bool,

    /// Whether to check for landmark roles
    pub check_landmark_roles: bool,

    /// Whether to check document title
    pub check_document_title: bool,

    /// Whether to check language specification
    pub check_language: bool,

    /// Minimum color contrast ratio (WCAG AA: 4.5:1, AAA: 7:1)
    pub min_contrast_ratio: f64,

    /// Minimum large text color contrast ratio (WCAG AA: 3:1, AAA: 4.5:1)
    pub min_large_text_contrast_ratio: f64,
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
            check_semantic_structure: true,
            check_link_text: true,
            check_table_accessibility: true,
            check_landmark_roles: true,
            check_document_title: true,
            check_language: true,
            min_contrast_ratio: 4.5,
            min_large_text_contrast_ratio: 3.0,
        }
    }
}

/// Accessibility analyzer for HTML documents
#[derive(Clone)]
pub struct AccessibilityAnalyzer {
    /// Configuration options
    options: AccessibilityOptions,
    /// Cached selectors
    selectors: AccessibilitySelectors,
}

/// Cached selectors for accessibility analysis
#[derive(Clone)]
struct AccessibilitySelectors {
    images: Selector,
    headings: Selector,
    forms: Selector,
    inputs: Selector,
    buttons: Selector,
    links: Selector,
    tables: Selector,
    interactive: Selector,
    landmarks: Selector,
    aria_elements: Selector,
}

impl Default for AccessibilitySelectors {
    fn default() -> Self {
        Self {
            images: Selector::parse("img").unwrap(),
            headings: Selector::parse("h1, h2, h3, h4, h5, h6").unwrap(),
            forms: Selector::parse("form").unwrap(),
            inputs: Selector::parse("input, select, textarea").unwrap(),
            buttons: Selector::parse("button, [role='button']").unwrap(),
            links: Selector::parse("a").unwrap(),
            tables: Selector::parse("table").unwrap(),
            interactive: Selector::parse("a, button, [role='button'], input, select, textarea, [tabindex]").unwrap(),
            landmarks: Selector::parse("[role='banner'], [role='navigation'], [role='main'], [role='complementary'], [role='contentinfo'], nav, main, header, footer, aside").unwrap(),
            aria_elements: Selector::parse("[aria-*]").unwrap(),
        }
    }
}

impl AccessibilityAnalyzer {
    /// Create a new accessibility analyzer with the given options
    pub fn new(options: AccessibilityOptions) -> Self {
        Self {
            options,
            selectors: AccessibilitySelectors::default(),
        }
    }

    /// Create a new accessibility analyzer with default options
    pub fn default() -> Self {
        Self::new(AccessibilityOptions::default())
    }

    /// Analyze a DOM document for accessibility issues
    ///
    /// # Arguments
    ///
    /// * `dom` - The HTML DOM to analyze
    ///
    /// # Returns
    ///
    /// * `Result<AccessibilityReport, DOMorpherError>` - The accessibility report or an error
    pub fn analyze(&self, dom: &Html) -> Result<AccessibilityReport, DOMorpherError> {
        let mut report = AccessibilityReport::new();

        // Extract basic document metadata
        self.extract_document_metadata(dom, &mut report)?;

        // Run enabled checks
        if self.options.check_document_title {
            self.check_document_title(dom, &mut report)?;
        }

        if self.options.check_language {
            self.check_language(dom, &mut report)?;
        }

        if self.options.check_headings {
            self.check_headings(dom, &mut report)?;
        }

        if self.options.check_alt_text {
            self.check_alt_text(dom, &mut report)?;
        }

        if self.options.check_form_labels {
            self.check_form_labels(dom, &mut report)?;
        }

        if self.options.check_aria {
            self.check_aria(dom, &mut report)?;
        }

        if self.options.check_link_text {
            self.check_link_text(dom, &mut report)?;
        }

        if self.options.check_table_accessibility {
            self.check_table_accessibility(dom, &mut report)?;
        }

        if self.options.check_landmark_roles {
            self.check_landmark_roles(dom, &mut report)?;
        }

        if self.options.check_semantic_structure {
            self.check_semantic_structure(dom, &mut report)?;
        }

        if self.options.check_keyboard_navigation {
            self.check_keyboard_navigation(dom, &mut report)?;
        }

        if self.options.check_color_contrast {
            self.check_color_contrast(dom, &mut report)?;
        }

        // Calculate overall score
        report.calculate_score();

        Ok(report)
    }

    /// Extract document metadata
    fn extract_document_metadata(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        // Extract title
        if let Some(title_selector) = Selector::parse("title").ok() {
            if let Some(title_el) = dom.select(&title_selector).next() {
                report
                    .document_metadata
                    .insert("title".to_string(), title_el.text().collect::<String>());
            }
        }

        // Extract language
        if let Some(html_selector) = Selector::parse("html").ok() {
            if let Some(html_el) = dom.select(&html_selector).next() {
                if let Some(lang) = html_el.value().attr("lang") {
                    report
                        .document_metadata
                        .insert("language".to_string(), lang.to_string());
                }
            }
        }

        // Extract meta description
        if let Some(meta_selector) = Selector::parse("meta[name='description']").ok() {
            if let Some(meta_el) = dom.select(&meta_selector).next() {
                if let Some(content) = meta_el.value().attr("content") {
                    report
                        .document_metadata
                        .insert("description".to_string(), content.to_string());
                }
            }
        }

        // Count elements
        if let Some(body_selector) = Selector::parse("body").ok() {
            if let Some(body_el) = dom.select(&body_selector).next() {
                // Count total elements
                let element_count = body_el.children().count();
                report
                    .document_metadata
                    .insert("element_count".to_string(), element_count.to_string());

                // Count interactive elements
                let interactive_count = dom.select(&self.selectors.interactive).count();
                report.document_metadata.insert(
                    "interactive_element_count".to_string(),
                    interactive_count.to_string(),
                );

                // Count images
                let image_count = dom.select(&self.selectors.images).count();
                report
                    .document_metadata
                    .insert("image_count".to_string(), image_count.to_string());

                // Count forms
                let form_count = dom.select(&self.selectors.forms).count();
                report
                    .document_metadata
                    .insert("form_count".to_string(), form_count.to_string());
            }
        }

        Ok(())
    }

    /// Check document title
    fn check_document_title(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        if let Some(title_selector) = Selector::parse("title").ok() {
            let title_elements = dom.select(&title_selector).collect::<Vec<_>>();

            if title_elements.is_empty() {
                // Missing title
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::DocumentTitle,
                    severity: AccessibilitySeverity::Error,
                    description: "Document is missing a title".to_string(),
                    element_path: "html > head".to_string(),
                    wcag_reference: Some("2.4.2 Page Titled (Level A)".to_string()),
                    suggestion: "Add a descriptive <title> element in the document head"
                        .to_string(),
                });
            } else if title_elements.len() > 1 {
                // Multiple titles
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::DocumentTitle,
                    severity: AccessibilitySeverity::Warning,
                    description: "Document has multiple title elements".to_string(),
                    element_path: "html > head".to_string(),
                    wcag_reference: Some("2.4.2 Page Titled (Level A)".to_string()),
                    suggestion: "Keep only one <title> element in the document head".to_string(),
                });
            } else {
                // Check title quality
                let title_text = title_elements[0].text().collect::<String>();

                if title_text.trim().is_empty() {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::DocumentTitle,
                        severity: AccessibilitySeverity::Error,
                        description: "Document title is empty".to_string(),
                        element_path: "html > head > title".to_string(),
                        wcag_reference: Some("2.4.2 Page Titled (Level A)".to_string()),
                        suggestion: "Add descriptive content to the title element".to_string(),
                    });
                } else if title_text.trim().len() < 5 {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::DocumentTitle,
                        severity: AccessibilitySeverity::Warning,
                        description: "Document title is very short and may not be descriptive"
                            .to_string(),
                        element_path: "html > head > title".to_string(),
                        wcag_reference: Some("2.4.2 Page Titled (Level A)".to_string()),
                        suggestion: "Use a more descriptive title that identifies the page content"
                            .to_string(),
                    });
                } else {
                    report.add_success(AccessibilitySuccess {
                        feature: AccessibilityFeature::DocumentTitle,
                        description: "Document has a descriptive title".to_string(),
                        element_reference: format!("Title: \"{}\"", title_text.trim()),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check HTML language attribute
    fn check_language(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        if let Some(html_selector) = Selector::parse("html").ok() {
            if let Some(html_el) = dom.select(&html_selector).next() {
                if let Some(lang) = html_el.value().attr("lang") {
                    if lang.trim().is_empty() {
                        report.add_issue(AccessibilityIssue {
                            feature: AccessibilityFeature::LanguageSpecification,
                            severity: AccessibilitySeverity::Warning,
                            description: "HTML element has empty lang attribute".to_string(),
                            element_path: "html".to_string(),
                            wcag_reference: Some("3.1.1 Language of Page (Level A)".to_string()),
                            suggestion: "Set a valid language code in the lang attribute"
                                .to_string(),
                        });
                    } else {
                        // Very basic language code validation
                        if lang.len() < 2 || lang.len() > 35 {
                            report.add_issue(AccessibilityIssue {
                                feature: AccessibilityFeature::LanguageSpecification,
                                severity: AccessibilitySeverity::Warning,
                                description: format!(
                                    "HTML element has potentially invalid lang attribute: '{}'",
                                    lang
                                ),
                                element_path: "html".to_string(),
                                wcag_reference: Some(
                                    "3.1.1 Language of Page (Level A)".to_string(),
                                ),
                                suggestion: "Use a valid BCP 47 language code (e.g. 'en', 'es-MX')"
                                    .to_string(),
                            });
                        } else {
                            report.add_success(AccessibilitySuccess {
                                feature: AccessibilityFeature::LanguageSpecification,
                                description: "Document language is specified".to_string(),
                                element_reference: format!("Language: {}", lang),
                            });
                        }
                    }
                } else {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::LanguageSpecification,
                        severity: AccessibilitySeverity::Error,
                        description: "HTML element is missing lang attribute".to_string(),
                        element_path: "html".to_string(),
                        wcag_reference: Some("3.1.1 Language of Page (Level A)".to_string()),
                        suggestion: "Add a lang attribute to the HTML element that identifies the document's language".to_string(),
                    });
                }
            }
        }

        // Check for content in different languages
        if let Some(lang_selector) = Selector::parse("[lang]:not(html)").ok() {
            let lang_elements = dom.select(&lang_selector).collect::<Vec<_>>();
            if !lang_elements.is_empty() {
                report.add_success(AccessibilitySuccess {
                    feature: AccessibilityFeature::LanguageSpecification,
                    description: "Document uses lang attributes for content in different languages"
                        .to_string(),
                    element_reference: format!(
                        "{} elements with language specifications",
                        lang_elements.len()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Check heading structure
    fn check_headings(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let headings = dom.select(&self.selectors.headings).collect::<Vec<_>>();

        if headings.is_empty() {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::Headings,
                severity: AccessibilitySeverity::Warning,
                description: "Document has no headings".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some(
                    "1.3.1 Info and Relationships (Level A), 2.4.6 Headings and Labels (Level AA)"
                        .to_string(),
                ),
                suggestion: "Add headings to structure the content and improve navigation"
                    .to_string(),
            });
            return Ok(());
        }

        // Check for H1
        let h1_count = dom.select(&Selector::parse("h1").unwrap()).count();

        if h1_count == 0 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::Headings,
                severity: AccessibilitySeverity::Error,
                description: "Document has no main heading (h1)".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some(
                    "1.3.1 Info and Relationships (Level A), 2.4.6 Headings and Labels (Level AA)"
                        .to_string(),
                ),
                suggestion: "Add an h1 element as the main heading of the document".to_string(),
            });
        } else if h1_count > 1 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::Headings,
                severity: AccessibilitySeverity::Warning,
                description: format!("Document has multiple main headings ({})", h1_count),
                element_path: "h1".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Consider using only one h1 as the main heading for the document"
                    .to_string(),
            });
        } else {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::Headings,
                description: "Document has a single main heading (h1)".to_string(),
                element_reference: "h1".to_string(),
            });
        }

        // Check heading hierarchy
        let mut current_level = 0;
        let mut hierarchy_issues = 0;

        for heading in &headings {
            let level = match heading.value().name() {
                "h1" => 1,
                "h2" => 2,
                "h3" => 3,
                "h4" => 4,
                "h5" => 5,
                "h6" => 6,
                _ => 0, // Shouldn't happen given the selector
            };

            // Check for skipped levels (e.g., h1 to h3 without h2)
            if level > current_level + 1 && current_level > 0 {
                let path = self.get_element_path(heading);
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::Headings,
                    severity: AccessibilitySeverity::Warning,
                    description: format!(
                        "Heading level skipped from h{} to h{}",
                        current_level, level
                    ),
                    element_path: path,
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: format!(
                        "Ensure heading hierarchy doesn't skip levels. Use h{} before h{}",
                        current_level + 1,
                        level
                    ),
                });
                hierarchy_issues += 1;
            }

            // Check for empty headings
            let heading_text = heading.text().collect::<String>().trim().to_string();
            if heading_text.is_empty() {
                let path = self.get_element_path(heading);
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::Headings,
                    severity: AccessibilitySeverity::Error,
                    description: "Empty heading found".to_string(),
                    element_path: path,
                    wcag_reference: Some("2.4.6 Headings and Labels (Level AA)".to_string()),
                    suggestion: "Add descriptive text to the heading or remove it".to_string(),
                });
            }

            // Update current level if this is a higher level than before
            if level > current_level || current_level == 0 {
                current_level = level;
            }
        }

        if hierarchy_issues == 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::Headings,
                description: "Document has a proper heading hierarchy without skipped levels"
                    .to_string(),
                element_reference: format!("{} headings", headings.len()),
            });
        }

        // Check for landmark roles with headings
        if self.options.check_landmark_roles {
            let main_selector = Selector::parse("main, [role='main']").unwrap();
            if let Some(main_el) = dom.select(&main_selector).next() {
                let main_headings = main_el.select(&self.selectors.headings).count();
                if main_headings == 0 {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::Headings,
                        severity: AccessibilitySeverity::Warning,
                        description: "Main content area has no headings".to_string(),
                        element_path: "main".to_string(),
                        wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                        suggestion: "Add headings to structure the main content area".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check image alt text
    fn check_alt_text(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let images = dom.select(&self.selectors.images).collect::<Vec<_>>();

        if images.is_empty() {
            // No images to check
            return Ok(());
        }

        let mut missing_alt = 0;
        let mut empty_alt = 0;
        let mut decorative_images = 0;
        let mut good_alt = 0;

        for img in images {
            let path = self.get_element_path(&img);

            if !img.value().has_attr("alt") {
                missing_alt += 1;
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::AltText,
                    severity: AccessibilitySeverity::Error,
                    description: "Image is missing alt attribute".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.1.1 Non-text Content (Level A)".to_string()),
                    suggestion: "Add an alt attribute that describes the image content or purpose"
                        .to_string(),
                });
            } else {
                let alt_text = img.value().attr("alt").unwrap_or("").trim();

                if alt_text.is_empty() {
                    // Empty alt is ok for decorative images
                    empty_alt += 1;

                    // Check if image is truly decorative
                    if img.value().has_attr("src") {
                        let src = img.value().attr("src").unwrap_or("");
                        if src.contains("logo") || src.contains("banner") || src.contains("icon") {
                            report.add_issue(AccessibilityIssue {
                                feature: AccessibilityFeature::AltText,
                                severity: AccessibilitySeverity::Warning,
                                description: "Potentially meaningful image has empty alt text"
                                    .to_string(),
                                element_path: path,
                                wcag_reference: Some(
                                    "1.1.1 Non-text Content (Level A)".to_string(),
                                ),
                                suggestion:
                                    "If this image conveys information, add descriptive alt text"
                                        .to_string(),
                            });
                        } else {
                            decorative_images += 1;
                        }
                    }
                } else if alt_text.to_lowercase().contains("image of")
                    || alt_text.to_lowercase().contains("picture of")
                    || alt_text.to_lowercase() == "image"
                    || alt_text.to_lowercase() == "photo"
                {
                    // Generic alt text
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::AltText,
                        severity: AccessibilitySeverity::Warning,
                        description: format!("Image has generic alt text: '{}'", alt_text),
                        element_path: path,
                        wcag_reference: Some("1.1.1 Non-text Content (Level A)".to_string()),
                        suggestion:
                            "Use more specific alt text that describes the image content or purpose"
                                .to_string(),
                    });
                } else if alt_text.len() > 100 {
                    // Alt text too long
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::AltText,
                        severity: AccessibilitySeverity::Warning,
                        description: "Image has very long alt text".to_string(),
                        element_path: path,
                        wcag_reference: Some("1.1.1 Non-text Content (Level A)".to_string()),
                        suggestion: "Keep alt text concise while still describing the image content or purpose".to_string(),
                    });
                } else {
                    good_alt += 1;
                }
            }
        }

        // Add summary to report
        if missing_alt == 0 && good_alt > 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::AltText,
                description: "All images have alt attributes".to_string(),
                element_reference: format!(
                    "{} images with good alt text, {} decorative images",
                    good_alt, decorative_images
                ),
            });
        }

        Ok(())
    }

    /// Check form labels
    fn check_form_labels(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let inputs = dom.select(&self.selectors.inputs).collect::<Vec<_>>();

        if inputs.is_empty() {
            // No form inputs to check
            return Ok(());
        }

        let mut missing_labels = 0;
        let mut good_labels = 0;

        // Check for <label> elements
        let label_selector = Selector::parse("label").unwrap();
        let labels = dom.select(&label_selector).collect::<Vec<_>>();
        let label_for_ids: HashSet<String> = labels
            .iter()
            .filter_map(|l| l.value().attr("for").map(|f| f.to_string()))
            .collect();

        for input in inputs {
            let path = self.get_element_path(&input);
            let input_type = input.value().attr("type").unwrap_or("text");

            // Skip hidden inputs and certain types
            if input_type == "hidden"
                || input_type == "submit"
                || input_type == "button"
                || input_type == "reset"
                || input_type == "image"
            {
                continue;
            }

            // Check for id attribute that can be matched with label
            if let Some(id) = input.value().attr("id") {
                if label_for_ids.contains(id) {
                    good_labels += 1;
                    continue;
                }
            }

            // Check for aria-label
            if let Some(aria_label) = input.value().attr("aria-label") {
                if !aria_label.trim().is_empty() {
                    good_labels += 1;
                    continue;
                }
            }

            // Check for aria-labelledby
            if let Some(_) = input.value().attr("aria-labelledby") {
                good_labels += 1;
                continue;
            }

            // Check for placeholder (not sufficient but better than nothing)
            if let Some(placeholder) = input.value().attr("placeholder") {
                if !placeholder.trim().is_empty() {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::FormLabels,
                        severity: AccessibilitySeverity::Warning,
                        description: "Input relies only on placeholder for labeling".to_string(),
                        element_path: path,
                        wcag_reference: Some("1.3.1 Info and Relationships (Level A), 3.3.2 Labels or Instructions (Level A)".to_string()),
                        suggestion: "Add a proper label element or aria-label attribute in addition to the placeholder".to_string(),
                    });
                    continue;
                }
            }

            // Check for title (not sufficient but better than nothing)
            if let Some(title) = input.value().attr("title") {
                if !title.trim().is_empty() {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::FormLabels,
                        severity: AccessibilitySeverity::Warning,
                        description: "Input relies only on title attribute for labeling".to_string(),
                        element_path: path,
                        wcag_reference: Some("1.3.1 Info and Relationships (Level A), 3.3.2 Labels or Instructions (Level A)".to_string()),
                        suggestion: "Add a proper label element or aria-label attribute in addition to the title".to_string(),
                    });
                    continue;
                }
            }

            // No label found
            missing_labels += 1;
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::FormLabels,
                severity: AccessibilitySeverity::Error,
                description: format!("Input element ({} type) has no associated label", input_type),
                element_path: path,
                wcag_reference: Some("1.3.1 Info and Relationships (Level A), 3.3.2 Labels or Instructions (Level A)".to_string()),
                suggestion: "Add a label element with matching 'for' attribute, or aria-label/aria-labelledby attributes".to_string(),
            });
        }

        // Check for required indicators
        let required_inputs = dom
            .select(&Selector::parse("input[required], [aria-required='true']").unwrap())
            .count();
        if required_inputs > 0 {
            let required_explained = dom
                .select(
                    &Selector::parse(
                        "form .required, form .mandatory, form .asterisk, form:contains('*')",
                    )
                    .unwrap(),
                )
                .count()
                > 0;
            if !required_explained {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::FormLabels,
                    severity: AccessibilitySeverity::Warning,
                    description: "Form has required fields but no explanation of required field indicators".to_string(),
                    element_path: "form".to_string(),
                    wcag_reference: Some("3.3.2 Labels or Instructions (Level A)".to_string()),
                    suggestion: "Add an explanation of how required fields are indicated (e.g., 'Fields marked with * are required')".to_string(),
                });
            }
        }

        // Add summary to report
        if missing_labels == 0 && good_labels > 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::FormLabels,
                description: "All form inputs have associated labels".to_string(),
                element_reference: format!("{} inputs with labels", good_labels),
            });
        }

        Ok(())
    }

    /// Check ARIA attributes
    fn check_aria(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let aria_elements = dom
            .select(&self.selectors.aria_elements)
            .collect::<Vec<_>>();

        if aria_elements.is_empty() {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::Aria,
                severity: AccessibilitySeverity::Info,
                description: "Document does not use ARIA attributes".to_string(),
                element_path: "document".to_string(),
                wcag_reference: None,
                suggestion:
                    "Consider using ARIA attributes to enhance accessibility where appropriate"
                        .to_string(),
            });
            return Ok(());
        }

        // Check for common ARIA issues
        let mut good_aria = 0;

        for el in aria_elements {
            let path = self.get_element_path(&el);

            // Check for aria-hidden on focusable elements
            if let Some(aria_hidden) = el.value().attr("aria-hidden") {
                if aria_hidden == "true" {
                    // Check if element is focusable
                    let tag_name = el.value().name();
                    let is_focusable = tag_name == "a"
                        || tag_name == "button"
                        || tag_name == "input"
                        || tag_name == "select"
                        || tag_name == "textarea"
                        || el.value().attr("tabindex").is_some();

                    if is_focusable {
                        report.add_issue(AccessibilityIssue {
                            feature: AccessibilityFeature::Aria,
                            severity: AccessibilitySeverity::Error,
                            description: "aria-hidden='true' used on focusable element".to_string(),
                            element_path: path,
                            wcag_reference: Some("4.1.2 Name, Role, Value (Level A)".to_string()),
                            suggestion: "Remove aria-hidden='true' from focusable elements or make them unfocusable".to_string(),
                        });
                        continue;
                    }
                }
            }

            // Check for invalid ARIA role
            if let Some(role) = el.value().attr("role") {
                let valid_roles = vec![
                    "alert",
                    "alertdialog",
                    "application",
                    "article",
                    "banner",
                    "button",
                    "cell",
                    "checkbox",
                    "columnheader",
                    "combobox",
                    "complementary",
                    "contentinfo",
                    "definition",
                    "dialog",
                    "directory",
                    "document",
                    "feed",
                    "figure",
                    "form",
                    "grid",
                    "gridcell",
                    "group",
                    "heading",
                    "img",
                    "link",
                    "list",
                    "listbox",
                    "listitem",
                    "log",
                    "main",
                    "marquee",
                    "math",
                    "menu",
                    "menubar",
                    "menuitem",
                    "menuitemcheckbox",
                    "menuitemradio",
                    "navigation",
                    "none",
                    "note",
                    "option",
                    "presentation",
                    "progressbar",
                    "radio",
                    "radiogroup",
                    "region",
                    "row",
                    "rowgroup",
                    "rowheader",
                    "scrollbar",
                    "search",
                    "searchbox",
                    "separator",
                    "slider",
                    "spinbutton",
                    "status",
                    "switch",
                    "tab",
                    "table",
                    "tablist",
                    "tabpanel",
                    "term",
                    "textbox",
                    "timer",
                    "toolbar",
                    "tooltip",
                    "tree",
                    "treegrid",
                    "treeitem",
                ];

                if !valid_roles.contains(&role) {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::Aria,
                        severity: AccessibilitySeverity::Error,
                        description: format!("Invalid ARIA role: '{}'", role),
                        element_path: path,
                        wcag_reference: Some("4.1.2 Name, Role, Value (Level A)".to_string()),
                        suggestion: "Use a valid ARIA role value".to_string(),
                    });
                    continue;
                }

                // Check for redundant roles
                let tag_name = el.value().name();
                let redundant = match (tag_name, role) {
                    ("button", "button") => true,
                    ("a", "link") => true,
                    ("input", "textbox") if el.value().attr("type").unwrap_or("text") == "text" => {
                        true
                    }
                    ("input", "checkbox")
                        if el.value().attr("type").unwrap_or("") == "checkbox" =>
                    {
                        true
                    }
                    ("input", "radio") if el.value().attr("type").unwrap_or("") == "radio" => true,
                    ("h1" | "h2" | "h3" | "h4" | "h5" | "h6", "heading") => true,
                    ("img", "img") => true,
                    ("ul" | "ol", "list") => true,
                    ("li", "listitem") => true,
                    ("nav", "navigation") => true,
                    ("main", "main") => true,
                    ("aside", "complementary") => true,
                    ("header", "banner") => true,
                    ("footer", "contentinfo") => true,
                    _ => false,
                };

                if redundant {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::Aria,
                        severity: AccessibilitySeverity::Warning,
                        description: format!(
                            "Redundant ARIA role '{}' on <{}> element",
                            role, tag_name
                        ),
                        element_path: path,
                        wcag_reference: None,
                        suggestion:
                            "Remove redundant role that duplicates the element's implicit role"
                                .to_string(),
                    });
                }
            }

            // Check for missing required attributes for certain roles
            if let Some(role) = el.value().attr("role") {
                let required_attrs = match role {
                    "checkbox" | "switch" => vec!["aria-checked"],
                    "combobox" => vec!["aria-expanded"],
                    "slider" => vec!["aria-valuemin", "aria-valuemax", "aria-valuenow"],
                    "spinbutton" => vec!["aria-valuemin", "aria-valuemax", "aria-valuenow"],
                    _ => vec![],
                };

                for attr in &required_attrs {
                    if el.value().attr(attr).is_none() {
                        report.add_issue(AccessibilityIssue {
                            feature: AccessibilityFeature::Aria,
                            severity: AccessibilitySeverity::Error,
                            description: format!(
                                "Missing required ARIA attribute '{}' for role '{}'",
                                attr, role
                            ),
                            element_path: path,
                            wcag_reference: Some("4.1.2 Name, Role, Value (Level A)".to_string()),
                            suggestion: format!(
                                "Add the '{}' attribute to elements with role '{}'",
                                attr, role
                            ),
                        });
                    }
                }
            }

            good_aria += 1;
        }

        // Add summary to report
        if good_aria > 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::Aria,
                description: "Document uses ARIA attributes to enhance accessibility".to_string(),
                element_reference: format!("{} elements with ARIA attributes", good_aria),
            });
        }

        Ok(())
    }

    /// Check link text quality
    fn check_link_text(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let links = dom.select(&self.selectors.links).collect::<Vec<_>>();

        if links.is_empty() {
            // No links to check
            return Ok(());
        }

        let mut empty_links = 0;
        let mut generic_links = 0;
        let mut good_links = 0;

        // Generic link text phrases
        let generic_phrases = vec![
            "click here",
            "click",
            "here",
            "more",
            "read more",
            "details",
            "link",
            "this link",
            "this page",
            "learn more",
            "more information",
            "info",
        ];

        for link in links {
            let path = self.get_element_path(&link);
            let link_text = link.text().collect::<String>().trim().to_string();

            // Skip links that are likely meant to be buttons
            if link.value().attr("role") == Some("button") {
                continue;
            }

            if link_text.is_empty() {
                // Check for aria-label or title
                if let Some(aria_label) = link.value().attr("aria-label") {
                    if !aria_label.trim().is_empty() {
                        good_links += 1;
                        continue;
                    }
                } else if let Some(title) = link.value().attr("title") {
                    if !title.trim().is_empty() {
                        good_links += 1;
                        continue;
                    }
                } else if link.select(&Selector::parse("img[alt]").unwrap()).count() > 0 {
                    // Link contains an image with alt text
                    good_links += 1;
                    continue;
                } else {
                    empty_links += 1;
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::LinkText,
                        severity: AccessibilitySeverity::Error,
                        description: "Link has no text content or accessible name".to_string(),
                        element_path: path,
                        wcag_reference: Some("2.4.4 Link Purpose (In Context) (Level A)".to_string()),
                        suggestion: "Add descriptive text content, aria-label, or aria-labelledby to the link".to_string(),
                    });
                }
            } else {
                // Check for generic link text
                let lower_text = link_text.to_lowercase();
                if generic_phrases.iter().any(|p| lower_text == *p) {
                    generic_links += 1;
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::LinkText,
                        severity: AccessibilitySeverity::Warning,
                        description: format!("Link has generic text: '{}'", link_text),
                        element_path: path,
                        wcag_reference: Some("2.4.4 Link Purpose (In Context) (Level A), 2.4.9 Link Purpose (Link Only) (Level AAA)".to_string()),
                        suggestion: "Use descriptive link text that identifies the link's purpose".to_string(),
                    });
                } else {
                    good_links += 1;
                }

                // Check for links with identical text but different destinations
                if links
                    .iter()
                    .filter(|l| l.text().collect::<String>() == link_text)
                    .count()
                    > 1
                {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::LinkText,
                        severity: AccessibilitySeverity::Warning,
                        description: format!("Multiple links with identical text: '{}'", link_text),
                        element_path: path,
                        wcag_reference: Some("2.4.4 Link Purpose (In Context) (Level A)".to_string()),
                        suggestion: "Ensure that links with the same text go to the same destination, or make the link text more specific".to_string(),
                    });
                }
            }
        }

        // Add summary to report
        if empty_links == 0 && generic_links == 0 && good_links > 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::LinkText,
                description: "All links have descriptive text".to_string(),
                element_reference: format!("{} links with descriptive text", good_links),
            });
        }

        Ok(())
    }

    /// Check table accessibility
    fn check_table_accessibility(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let tables = dom.select(&self.selectors.tables).collect::<Vec<_>>();

        if tables.is_empty() {
            // No tables to check
            return Ok(());
        }

        let mut good_tables = 0;

        for table in tables {
            let path = self.get_element_path(&table);

            // Check for data vs. layout table
            let is_layout_table = table.value().attr("role") == Some("presentation")
                || table.value().attr("role") == Some("none");

            // Skip further checks for layout tables
            if is_layout_table {
                continue;
            }

            // Check for table headers
            let has_th = table.select(&Selector::parse("th").unwrap()).count() > 0;
            let has_thead = table.select(&Selector::parse("thead").unwrap()).count() > 0;

            if !has_th && !has_thead {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::TableAccessibility,
                    severity: AccessibilitySeverity::Error,
                    description: "Table has no header cells (th) or header section (thead)".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Add th elements for column and/or row headers, or a thead section with header cells".to_string(),
                });
            }

            // Check for scope attribute on headers
            let headers_without_scope = table
                .select(&Selector::parse("th:not([scope])").unwrap())
                .count();
            if headers_without_scope > 0 && has_th {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::TableAccessibility,
                    severity: AccessibilitySeverity::Warning,
                    description: "Table has header cells (th) without scope attribute".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Add scope='col' or scope='row' to th elements to clarify their relationship to data cells".to_string(),
                });
            }

            // Check for caption
            let has_caption = table.select(&Selector::parse("caption").unwrap()).count() > 0;
            if !has_caption {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::TableAccessibility,
                    severity: AccessibilitySeverity::Warning,
                    description: "Table is missing a caption".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Add a caption element to provide a brief description of the table's purpose or content".to_string(),
                });
            }

            // Check for summary (HTML4 attribute, not in HTML5 but still useful for accessibility)
            let has_summary = table.value().attr("summary").is_some();
            let has_aria_description = table.value().attr("aria-describedby").is_some();

            if !has_summary && !has_aria_description && !has_caption {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::TableAccessibility,
                    severity: AccessibilitySeverity::Info,
                    description: "Table has no description (caption, summary, or aria-describedby)".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Add a caption element or aria-describedby attribute to describe the table's purpose or structure".to_string(),
                });
            }

            // Check for complex tables that might need additional accessibility features
            let is_complex = table
                .select(
                    &Selector::parse("th[rowspan], th[colspan], td[rowspan], td[colspan]").unwrap(),
                )
                .count()
                > 0;
            if is_complex {
                let uses_headers_id = table
                    .select(&Selector::parse("th[id], td[headers]").unwrap())
                    .count()
                    > 0;
                if !uses_headers_id {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::TableAccessibility,
                        severity: AccessibilitySeverity::Warning,
                        description: "Complex table (with spanning cells) does not use id/headers attributes".to_string(),
                        element_path: path,
                        wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                        suggestion: "Add id attributes to th elements and headers attributes to td elements to explicitly associate data cells with headers".to_string(),
                    });
                }
            }

            // If table passes all checks
            if has_th
                && (has_caption || has_summary || has_aria_description)
                && (!is_complex
                    || (is_complex
                        && table
                            .select(&Selector::parse("th[id], td[headers]").unwrap())
                            .count()
                            > 0))
            {
                good_tables += 1;
            }
        }

        // Add summary to report
        if good_tables > 0 {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::TableAccessibility,
                description: "Document has properly structured tables with accessibility features"
                    .to_string(),
                element_reference: format!("{} accessible tables", good_tables),
            });
        }

        Ok(())
    }

    /// Check landmark roles
    fn check_landmark_roles(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        let landmarks = dom.select(&self.selectors.landmarks).collect::<Vec<_>>();

        if landmarks.is_empty() {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Warning,
                description: "Document does not use landmark roles or semantic elements".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A), 2.4.1 Bypass Blocks (Level A)".to_string()),
                suggestion: "Use landmark roles (e.g., banner, navigation, main) or semantic HTML elements (e.g., header, nav, main) to define page regions".to_string(),
            });
            return Ok(());
        }

        // Check for required landmarks
        let has_main = dom
            .select(&Selector::parse("main, [role='main']").unwrap())
            .count()
            > 0;
        if !has_main {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Warning,
                description: "Document is missing a main landmark".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A), 2.4.1 Bypass Blocks (Level A)".to_string()),
                suggestion: "Add a main element or an element with role='main' to identify the main content area".to_string(),
            });
        }

        let has_navigation = dom
            .select(&Selector::parse("nav, [role='navigation']").unwrap())
            .count()
            > 0;
        if !has_navigation {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Info,
                description: "Document does not have a navigation landmark".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "If the page includes navigation links, wrap them in a nav element or an element with role='navigation'".to_string(),
            });
        }

        // Check for duplicate landmarks
        let banner_count = dom
            .select(&Selector::parse("header[role='banner'], [role='banner']").unwrap())
            .count();
        if banner_count > 1 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Warning,
                description: format!("Document has multiple banner landmarks ({})", banner_count),
                element_path: "header[role='banner'], [role='banner']".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Use only one banner landmark to identify the site header".to_string(),
            });
        }

        let main_count = dom
            .select(&Selector::parse("main, [role='main']").unwrap())
            .count();
        if main_count > 1 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Warning,
                description: format!("Document has multiple main landmarks ({})", main_count),
                element_path: "main, [role='main']".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Use only one main landmark to identify the main content area"
                    .to_string(),
            });
        }

        let contentinfo_count = dom
            .select(&Selector::parse("footer[role='contentinfo'], [role='contentinfo']").unwrap())
            .count();
        if contentinfo_count > 1 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::LandmarkRoles,
                severity: AccessibilitySeverity::Warning,
                description: format!(
                    "Document has multiple contentinfo landmarks ({})",
                    contentinfo_count
                ),
                element_path: "footer[role='contentinfo'], [role='contentinfo']".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Use only one contentinfo landmark to identify the site footer"
                    .to_string(),
            });
        }

        // Add summary to report
        if has_main && has_navigation {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::LandmarkRoles,
                description: "Document uses landmark roles to define page regions".to_string(),
                element_reference: format!("{} landmarks identified", landmarks.len()),
            });
        }

        Ok(())
    }

    /// Check semantic structure
    fn check_semantic_structure(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        // Check for semantic HTML5 elements
        let uses_semantic_elements = dom
            .select(
                &Selector::parse(
                    "article, section, aside, figure, figcaption, time, mark, details, summary",
                )
                .unwrap(),
            )
            .count()
            > 0;

        if !uses_semantic_elements {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::SemanticStructure,
                severity: AccessibilitySeverity::Info,
                description: "Document does not use HTML5 semantic elements".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Consider using semantic HTML5 elements (article, section, aside, etc.) to better structure content".to_string(),
            });
        } else {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::SemanticStructure,
                description: "Document uses HTML5 semantic elements".to_string(),
                element_reference: "article, section, aside, etc.".to_string(),
            });
        }

        // Check for lists
        let unordered_lists = dom.select(&Selector::parse("ul").unwrap()).count();
        let ordered_lists = dom.select(&Selector::parse("ol").unwrap()).count();
        let definition_lists = dom.select(&Selector::parse("dl").unwrap()).count();

        let total_lists = unordered_lists + ordered_lists + definition_lists;

        if total_lists > 0 {
            // Check for empty lists
            let empty_lists = dom
                .select(&Selector::parse("ul:empty, ol:empty, dl:empty").unwrap())
                .count();
            if empty_lists > 0 {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::SemanticStructure,
                    severity: AccessibilitySeverity::Warning,
                    description: format!("Document has {} empty list elements", empty_lists),
                    element_path: "ul:empty, ol:empty, dl:empty".to_string(),
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Add list items to empty lists or remove them".to_string(),
                });
            }

            // Check for list items outside of lists
            let orphaned_list_items = dom
                .select(&Selector::parse("body > li, div > li, p > li").unwrap())
                .count();
            if orphaned_list_items > 0 {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::SemanticStructure,
                    severity: AccessibilitySeverity::Error,
                    description: format!("Document has {} list items outside of a list container", orphaned_list_items),
                    element_path: "body > li, div > li, p > li".to_string(),
                    wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                    suggestion: "Ensure all list items (li) are contained within a proper list element (ul, ol, or menu)".to_string(),
                });
            } else {
                report.add_success(AccessibilitySuccess {
                    feature: AccessibilityFeature::SemanticStructure,
                    description: "Document uses proper list structures".to_string(),
                    element_reference: format!(
                        "{} lists (UL: {}, OL: {}, DL: {})",
                        total_lists, unordered_lists, ordered_lists, definition_lists
                    ),
                });
            }
        }

        // Check for definition list structure
        if definition_lists > 0 {
            let dl_elements = dom
                .select(&Selector::parse("dl").unwrap())
                .collect::<Vec<_>>();

            for dl in dl_elements {
                let path = self.get_element_path(&dl);
                let has_dt = dl.select(&Selector::parse("dt").unwrap()).count() > 0;
                let has_dd = dl.select(&Selector::parse("dd").unwrap()).count() > 0;

                if !has_dt || !has_dd {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::SemanticStructure,
                        severity: AccessibilitySeverity::Warning,
                        description: "Definition list is missing term (dt) or description (dd) elements".to_string(),
                        element_path: path,
                        wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                        suggestion: "Ensure definition lists contain both dt (term) and dd (description) elements".to_string(),
                    });
                }
            }
        }

        // Check for paragraphs and text structure
        let paragraphs = dom.select(&Selector::parse("p").unwrap()).count();

        if paragraphs == 0 {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::SemanticStructure,
                severity: AccessibilitySeverity::Warning,
                description: "Document does not use paragraph elements".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("1.3.1 Info and Relationships (Level A)".to_string()),
                suggestion: "Use paragraph (p) elements to structure text content".to_string(),
            });
        } else {
            // Check for empty paragraphs
            let empty_paragraphs = dom.select(&Selector::parse("p:empty").unwrap()).count();
            if empty_paragraphs > 0 {
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::SemanticStructure,
                    severity: AccessibilitySeverity::Info,
                    description: format!(
                        "Document has {} empty paragraph elements",
                        empty_paragraphs
                    ),
                    element_path: "p:empty".to_string(),
                    wcag_reference: None,
                    suggestion: "Remove empty paragraph elements or add content to them"
                        .to_string(),
                });
            }
        }

        Ok(())
    }

    /// Check keyboard navigation
    fn check_keyboard_navigation(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        // Check for positive tabindex values
        let positive_tabindex_elements = dom
            .select(&Selector::parse("[tabindex]").unwrap())
            .filter(|el| {
                if let Some(tabindex) = el.value().attr("tabindex") {
                    if let Ok(value) = tabindex.parse::<i32>() {
                        return value > 0;
                    }
                }
                false
            })
            .collect::<Vec<_>>();

        if !positive_tabindex_elements.is_empty() {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::KeyboardNavigation,
                severity: AccessibilitySeverity::Warning,
                description: format!("Document has {} elements with positive tabindex values", positive_tabindex_elements.len()),
                element_path: "[tabindex]".to_string(),
                wcag_reference: Some("2.4.3 Focus Order (Level A)".to_string()),
                suggestion: "Avoid using positive tabindex values as they disrupt the natural tab order. Use structural order and 0 or -1 values instead".to_string(),
            });
        }

        // Check for keyboard traps
        let has_keyboard_trap_risk = dom
            .select(&Selector::parse("object, embed, applet").unwrap())
            .count()
            > 0;
        if has_keyboard_trap_risk {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::KeyboardNavigation,
                severity: AccessibilitySeverity::Warning,
                description: "Document contains embedded objects that may create keyboard traps".to_string(),
                element_path: "object, embed, applet".to_string(),
                wcag_reference: Some("2.1.2 No Keyboard Trap (Level A)".to_string()),
                suggestion: "Ensure embedded content does not trap keyboard focus. Test keyboard navigation through these elements".to_string(),
            });
        }

        // Check interactive elements with event handlers
        let onclick_elements = dom
            .select(&Selector::parse("[onclick]").unwrap())
            .collect::<Vec<_>>();
        let onkeydown_elements = dom
            .select(&Selector::parse("[onkeydown]").unwrap())
            .collect::<Vec<_>>();
        let onkeypress_elements = dom
            .select(&Selector::parse("[onkeypress]").unwrap())
            .collect::<Vec<_>>();
        let onkeyup_elements = dom
            .select(&Selector::parse("[onkeyup]").unwrap())
            .collect::<Vec<_>>();

        // Elements with click but no keyboard event handlers
        let mut click_only_elements = 0;

        for el in onclick_elements {
            let has_keyboard = el.value().attr("onkeydown").is_some()
                || el.value().attr("onkeypress").is_some()
                || el.value().attr("onkeyup").is_some();

            let is_naturally_keyboard_accessible = match el.value().name() {
                "a" | "button" | "input" | "select" | "textarea" => true,
                _ => false,
            };

            if !has_keyboard && !is_naturally_keyboard_accessible {
                click_only_elements += 1;
                let path = self.get_element_path(&el);

                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::KeyboardNavigation,
                    severity: AccessibilitySeverity::Error,
                    description: "Element has onclick event but no keyboard event handlers".to_string(),
                    element_path: path,
                    wcag_reference: Some("2.1.1 Keyboard (Level A)".to_string()),
                    suggestion: "Add keyboard event handlers (onkeydown, onkeypress, onkeyup) or use naturally keyboard-accessible elements".to_string(),
                });
            }
        }

        // Check for skip navigation link
        let has_skip_link = dom
            .select(
                &Selector::parse("a[href^='#']:first-child, a[href^='#']:first-of-type").unwrap(),
            )
            .any(|el| {
                let text = el.text().collect::<String>().to_lowercase();
                text.contains("skip") || text.contains("jump") || text.contains("main content")
            });

        if !has_skip_link
            && dom
                .select(&Selector::parse("nav, [role='navigation']").unwrap())
                .count()
                > 0
        {
            report.add_issue(AccessibilityIssue {
                feature: AccessibilityFeature::SkipNavigation,
                severity: AccessibilitySeverity::Warning,
                description: "Document has navigation but no skip navigation link".to_string(),
                element_path: "body".to_string(),
                wcag_reference: Some("2.4.1 Bypass Blocks (Level A)".to_string()),
                suggestion: "Add a 'Skip to main content' link at the beginning of the page that leads to the main content area".to_string(),
            });
        } else if has_skip_link {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::KeyboardNavigation,
                description: "Document has a skip navigation link".to_string(),
                element_reference: "Skip to content link".to_string(),
            });
        }

        // Add summary to report
        if click_only_elements == 0
            && !positive_tabindex_elements.is_empty() == false
            && !has_keyboard_trap_risk
        {
            report.add_success(AccessibilitySuccess {
                feature: AccessibilityFeature::KeyboardNavigation,
                description: "Document has good keyboard navigation support".to_string(),
                element_reference: format!(
                    "{} keyboard-accessible interactive elements",
                    onclick_elements.len()
                        + onkeydown_elements.len()
                        + onkeypress_elements.len()
                        + onkeyup_elements.len()
                ),
            });
        }

        Ok(())
    }

    /// Check color contrast
    fn check_color_contrast(
        &self,
        dom: &Html,
        report: &mut AccessibilityReport,
    ) -> Result<(), DOMorpherError> {
        // This is a simplified contrast check since actual computation would require
        // fully rendering the page and analyzing computed styles

        // Check for inline styles with color information
        let elements_with_color = dom
            .select(&Selector::parse("[style*='color']").unwrap())
            .collect::<Vec<_>>();

        for el in elements_with_color {
            let path = self.get_element_path(&el);
            let style = el.value().attr("style").unwrap_or("");

            // Try to extract foreground and background colors
            let fg_color = self.extract_color_from_style(style, "color");
            let bg_color = self.extract_color_from_style(style, "background-color");

            // If we have both colors, check contrast
            if let (Some(fg), Some(bg)) = (fg_color, bg_color) {
                let contrast_ratio = self.calculate_contrast_ratio(fg, bg);

                // Determine if this is large text
                let is_large_text = self.is_large_text(el);
                let min_ratio = if is_large_text {
                    self.options.min_large_text_contrast_ratio
                } else {
                    self.options.min_contrast_ratio
                };

                if contrast_ratio < min_ratio {
                    report.add_issue(AccessibilityIssue {
                        feature: AccessibilityFeature::ColorContrast,
                        severity: AccessibilitySeverity::Warning,
                        description: format!("Element has insufficient contrast ratio: {:.2}:1 (minimum should be {:.1}:1)", contrast_ratio, min_ratio),
                        element_path: path,
                        wcag_reference: Some("1.4.3 Contrast (Minimum) (Level AA)".to_string()),
                        suggestion: "Increase the contrast between foreground and background colors".to_string(),
                    });
                }
            }
        }

        // Check for color-only indicators (simplified check)
        let elements_with_indicators = dom.select(&Selector::parse("[class*='error'], [class*='required'], [class*='invalid'], [class*='success'], [class*='warning']").unwrap()).collect::<Vec<_>>();

        for el in elements_with_indicators {
            let class_attr = el.value().attr("class").unwrap_or("");

            // Check if there's a non-color indicator
            let has_icon = el.select(&Selector::parse("i, svg, img").unwrap()).count() > 0;
            let has_text_marker = el.text().collect::<String>().contains("*")
                || el.text().collect::<String>().contains("required")
                || el.text().collect::<String>().contains("error");

            if !has_icon && !has_text_marker {
                // Likely using color alone for indication
                let path = self.get_element_path(&el);
                report.add_issue(AccessibilityIssue {
                    feature: AccessibilityFeature::ColorContrast,
                    severity: AccessibilitySeverity::Warning,
                    description: "Element may be using color alone as an indicator".to_string(),
                    element_path: path,
                    wcag_reference: Some("1.4.1 Use of Color (Level A)".to_string()),
                    suggestion: "Don't rely on color alone to convey meaning. Add icons, patterns, or text to supplement color indicators".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Extract color from inline style
    fn extract_color_from_style(&self, style: &str, property: &str) -> Option<RGBA> {
        let style_lower = style.to_lowercase();
        if !style_lower.contains(property) {
            return None;
        }

        // Find the property in the style string
        if let Some(start) = style_lower.find(property) {
            let property_portion = &style[start..];
            let value_start = property_portion.find(':')? + 1;
            let mut value_end = property_portion[value_start..]
                .find(';')
                .unwrap_or(property_portion.len() - value_start);
            value_end += value_start;

            let color_value = property_portion[value_start..value_end].trim();

            // Parse the color value
            return self.parse_color(color_value);
        }

        None
    }

    /// Parse color value to RGBA
    fn parse_color(&self, color: &str) -> Option<RGBA> {
        // Simplified color parsing for common formats
        if color.starts_with('#') {
            // Hex format
            let hex = color.trim_start_matches('#');
            match hex.len() {
                3 => {
                    // #RGB format
                    let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
                    let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
                    let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
                    return Some(RGBA::new(r, g, b, 255));
                }
                6 => {
                    // #RRGGBB format
                    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                    return Some(RGBA::new(r, g, b, 255));
                }
                8 => {
                    // #RRGGBBAA format
                    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                    let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                    return Some(RGBA::new(r, g, b, a));
                }
                _ => return None,
            }
        } else if color.starts_with("rgb(") || color.starts_with("rgba(") {
            // RGB or RGBA format
            let inner = color
                .trim_start_matches("rgb(")
                .trim_start_matches("rgba(")
                .trim_end_matches(')');

            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() >= 3 {
                let r = parts[0].trim().parse::<u8>().ok()?;
                let g = parts[1].trim().parse::<u8>().ok()?;
                let b = parts[2].trim().parse::<u8>().ok()?;
                let a = if parts.len() > 3 {
                    let a_val = parts[3].trim().parse::<f32>().ok()?;
                    (a_val * 255.0) as u8
                } else {
                    255
                };
                return Some(RGBA::new(r, g, b, a));
            }
        } else if color == "transparent" {
            return Some(RGBA::new(0, 0, 0, 0));
        } else if color == "white" {
            return Some(RGBA::new(255, 255, 255, 255));
        } else if color == "black" {
            return Some(RGBA::new(0, 0, 0, 255));
        }

        // Basic named colors could be added here
        // For a production system, a full CSS color parser would be necessary

        None
    }

    /// Calculate contrast ratio between two colors
    fn calculate_contrast_ratio(&self, fg: RGBA, bg: RGBA) -> f64 {
        // Calculate relative luminance for each color
        let fg_luminance = self.relative_luminance(fg);
        let bg_luminance = self.relative_luminance(bg);

        // Calculate contrast ratio
        let (lighter, darker) = if fg_luminance > bg_luminance {
            (fg_luminance, bg_luminance)
        } else {
            (bg_luminance, fg_luminance)
        };

        (lighter + 0.05) / (darker + 0.05)
    }

    /// Calculate relative luminance of a color
    fn relative_luminance(&self, color: RGBA) -> f64 {
        // Convert RGB to relative luminance using the formula from WCAG 2.0
        let r_srgb = color.red as f64 / 255.0;
        let g_srgb = color.green as f64 / 255.0;
        let b_srgb = color.blue as f64 / 255.0;

        let r = if r_srgb <= 0.03928 {
            r_srgb / 12.92
        } else {
            ((r_srgb + 0.055) / 1.055).powf(2.4)
        };

        let g = if g_srgb <= 0.03928 {
            g_srgb / 12.92
        } else {
            ((g_srgb + 0.055) / 1.055).powf(2.4)
        };

        let b = if b_srgb <= 0.03928 {
            b_srgb / 12.92
        } else {
            ((b_srgb + 0.055) / 1.055).powf(2.4)
        };

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Determine if an element contains large text
    fn is_large_text(&self, el: ElementRef) -> bool {
        // Check for font-size in inline style
        if let Some(style) = el.value().attr("style") {
            let style_lower = style.to_lowercase();
            if style_lower.contains("font-size") {
                // Extract font size
                if let Some(start) = style_lower.find("font-size") {
                    let size_portion = &style[start..];
                    if let Some(value_start) = size_portion.find(':') {
                        let value_start = value_start + 1;
                        let value_end = size_portion[value_start..]
                            .find(';')
                            .unwrap_or(size_portion.len() - value_start);
                        let font_size = size_portion[value_start..value_start + value_end].trim();

                        // Check if font size is large (approximately 18pt/24px or larger)
                        if font_size.contains("pt") {
                            if let Some(pt_value) =
                                font_size.trim_end_matches("pt").parse::<f64>().ok()
                            {
                                return pt_value >= 18.0;
                            }
                        } else if font_size.contains("px") {
                            if let Some(px_value) =
                                font_size.trim_end_matches("px").parse::<f64>().ok()
                            {
                                return px_value >= 24.0;
                            }
                        } else if font_size.contains("em") {
                            if let Some(em_value) =
                                font_size.trim_end_matches("em").parse::<f64>().ok()
                            {
                                return em_value >= 1.5;
                            }
                        } else if font_size.contains("%") {
                            if let Some(pct_value) =
                                font_size.trim_end_matches("%").parse::<f64>().ok()
                            {
                                return pct_value >= 150.0;
                            }
                        } else if font_size == "large"
                            || font_size == "x-large"
                            || font_size == "xx-large"
                        {
                            return true;
                        }
                    }
                }
            }
        }

        // Check element tag for headers
        match el.value().name() {
            "h1" | "h2" => true,
            _ => false,
        }
    }

    /// Get element path for reporting
    fn get_element_path(&self, el: &ElementRef) -> String {
        let mut path = String::new();
        path.push_str(el.value().name());

        // Add id if present
        if let Some(id) = el.value().attr("id") {
            path.push_str(&format!("#{}", id));
        }
        // Add class if present
        else if let Some(class) = el.value().attr("class") {
            if !class.is_empty() {
                let first_class = class.split_whitespace().next().unwrap_or("");
                if !first_class.is_empty() {
                    path.push_str(&format!(".{}", first_class));
                }
            }
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_report() {
        let report = AccessibilityReport::new();
        assert_eq!(report.issues.len(), 0);
        assert_eq!(report.successes.len(), 0);
        assert_eq!(report.score, 0.0);
    }

    #[test]
    fn test_add_issue() {
        let mut report = AccessibilityReport::new();
        report.add_issue(AccessibilityIssue {
            feature: AccessibilityFeature::AltText,
            severity: AccessibilitySeverity::Error,
            description: "Missing alt text".to_string(),
            element_path: "img".to_string(),
            wcag_reference: Some("1.1.1".to_string()),
            suggestion: "Add alt attribute".to_string(),
        });

        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].feature, AccessibilityFeature::AltText);
        assert_eq!(report.issues[0].severity, AccessibilitySeverity::Error);
    }

    #[test]
    fn test_calculate_score() {
        let mut report = AccessibilityReport::new();

        // Add a critical issue
        report.add_issue(AccessibilityIssue {
            feature: AccessibilityFeature::AltText,
            severity: AccessibilitySeverity::Critical,
            description: "Missing alt text".to_string(),
            element_path: "img".to_string(),
            wcag_reference: Some("1.1.1".to_string()),
            suggestion: "Add alt attribute".to_string(),
        });

        // Add an error
        report.add_issue(AccessibilityIssue {
            feature: AccessibilityFeature::FormLabels,
            severity: AccessibilitySeverity::Error,
            description: "Missing label".to_string(),
            element_path: "input".to_string(),
            wcag_reference: Some("1.3.1".to_string()),
            suggestion: "Add label".to_string(),
        });

        // Calculate score
        report.calculate_score();

        // Score should be 100 - 10 (critical) - 5 (error) = 85
        assert_eq!(report.score, 85.0);
    }
}
