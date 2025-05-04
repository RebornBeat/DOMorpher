//! # Analysis Prompts Module
//!
//! This module provides specialized prompt templates for analyzing DOM structure.
//! These templates guide LLMs in understanding page organization, identifying key components,
//! and providing insights about the page content.
//!
//! ## Features
//!
//! - Structural analysis for understanding page organization
//! - Main content identification
//! - Interactive element analysis
//! - Form structure analysis
//! - Tabular data analysis

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    PromptComponents, PromptExample, PromptTemplate, TemplateVariation,
    create_standard_prompt, format_examples, format_system_instruction,
    system_instructions, truncate_html_for_context
};

/// Template for DOM structure analysis
#[derive(Clone, Debug)]
pub struct AnalysisPromptTemplate {
    /// HTML content to analyze
    pub html: String,

    /// Analysis objective
    pub objective: String,

    /// Analysis type for specialized prompts
    pub analysis_type: AnalysisType,

    /// Shared prompt components
    pub components: PromptComponents,

    /// Template variation for different LLM providers
    pub variation: TemplateVariation,

    /// Maximum tokens to allow for the HTML content
    pub max_html_tokens: usize,

    /// Additional context
    pub additional_context: Option<String>,

    /// Expected output format
    pub output_format: OutputFormat,
}

/// Types of analysis for specialized prompts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnalysisType {
    /// General structure analysis
    GeneralStructure,

    /// Main content identification
    MainContent,

    /// Interactive element analysis
    InteractiveElements,

    /// Form structure analysis
    FormStructure,

    /// Table structure analysis
    TableStructure,
}

/// Output format for analysis results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON output
    Json,

    /// Markdown output
    Markdown,

    /// Plain text output
    Text,
}

impl Default for AnalysisPromptTemplate {
    fn default() -> Self {
        Self {
            html: String::new(),
            objective: String::new(),
            analysis_type: AnalysisType::GeneralStructure,
            components: PromptComponents {
                system_instruction: system_instructions::DOM_ANALYSIS_SYSTEM.to_string(),
                examples: Vec::new(),
                additional_context: None,
            },
            variation: TemplateVariation::Claude,
            max_html_tokens: 8000,
            additional_context: None,
            output_format: OutputFormat::Json,
        }
    }
}

impl AnalysisPromptTemplate {
    /// Create a new analysis prompt template
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTML content
    pub fn with_html(mut self, html: &str) -> Self {
        self.html = html.to_string();
        self
    }

    /// Set the analysis objective
    pub fn with_objective(mut self, objective: &str) -> Self {
        self.objective = objective.to_string();
        self
    }

    /// Set the analysis type
    pub fn with_analysis_type(mut self, analysis_type: AnalysisType) -> Self {
        self.analysis_type = analysis_type;

        // Update system instruction based on analysis type
        match analysis_type {
            AnalysisType::MainContent => {
                self.components.system_instruction = format!(
                    "{} Focus on identifying the main content area of the page, distinguishing it from navigation, headers, footers, and sidebars.",
                    self.components.system_instruction
                );
            }
            AnalysisType::InteractiveElements => {
                self.components.system_instruction = format!(
                    "{} Focus on identifying all interactive elements on the page, such as buttons, links, inputs, and other controls.",
                    self.components.system_instruction
                );
            }
            AnalysisType::FormStructure => {
                self.components.system_instruction = format!(
                    "{} Focus on analyzing form structures, including field types, labels, required fields, and validation rules.",
                    self.components.system_instruction
                );
            }
            AnalysisType::TableStructure => {
                self.components.system_instruction = format!(
                    "{} Focus on analyzing table structures, including headers, data cells, and relationships between columns and rows.",
                    self.components.system_instruction
                );
            }
            _ => {}
        }

        self
    }

    /// Set the template variation
    pub fn with_variation(mut self, variation: TemplateVariation) -> Self {
        self.variation = variation;

        // Update system instruction based on variation
        match variation {
            TemplateVariation::Claude => {
                self.components.system_instruction = system_instructions::DOM_ANALYSIS_SYSTEM.to_string();
            }
            TemplateVariation::Gpt => {
                self.components.system_instruction = system_instructions::DOM_ANALYSIS_SYSTEM.to_string();
            }
            _ => {}
        }

        self
    }

    /// Add an example to the prompt
    pub fn with_example(mut self, input: &str, output: &str, explanation: Option<&str>) -> Self {
        let mut example = PromptExample::new(input, output);

        if let Some(explanation) = explanation {
            example = example.with_explanation(explanation);
        }

        self.components.examples.push(example);
        self
    }

    /// Set the maximum tokens for HTML content
    pub fn with_max_html_tokens(mut self, max_tokens: usize) -> Self {
        self.max_html_tokens = max_tokens;
        self
    }

    /// Set additional context
    pub fn with_additional_context(mut self, context: &str) -> Self {
        self.additional_context = Some(context.to_string());
        self
    }

    /// Set the output format
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Build the final prompt
    pub fn build(self) -> Self {
        self
    }

    /// Generate analysis-specific guidelines
    fn generate_guidelines(&self) -> String {
        let mut guidelines = vec![
            "1. Analyze the DOM structure and organization".to_string(),
            "2. Identify key components and their relationships".to_string(),
            "3. Provide detailed insights about the page structure".to_string(),
        ];

        match self.analysis_type {
            AnalysisType::MainContent => {
                guidelines.push("4. Identify the main content area of the page".to_string());
                guidelines.push("5. Distinguish main content from navigation, headers, footers, and sidebars".to_string());
                guidelines.push("6. Provide selectors that can be used to extract the main content".to_string());
            }
            AnalysisType::InteractiveElements => {
                guidelines.push("4. Identify all interactive elements on the page".to_string());
                guidelines.push("5. Categorize elements by type (button, link, input, etc.)".to_string());
                guidelines.push("6. Provide selectors and details for each interactive element".to_string());
            }
            AnalysisType::FormStructure => {
                guidelines.push("4. Analyze all forms on the page".to_string());
                guidelines.push("5. Identify field types, labels, required fields, and validation rules".to_string());
                guidelines.push("6. Map out the form submission process and any client-side validation".to_string());
            }
            AnalysisType::TableStructure => {
                guidelines.push("4. Analyze all tables on the page".to_string());
                guidelines.push("5. Identify headers, data cells, and relationships between columns and rows".to_string());
                guidelines.push("6. Provide insights about the table structure and content organization".to_string());
            }
            AnalysisType::GeneralStructure => {
                guidelines.push("4. Provide a comprehensive analysis of the page structure".to_string());
                guidelines.push("5. Identify key sections, content areas, and their relationships".to_string());
                guidelines.push("6. Include insights about the overall page organization and hierarchy".to_string());
            }
        }

        guidelines.join("\n")
    }

    /// Generate format instructions based on output format
    fn generate_format_instructions(&self) -> String {
        match self.output_format {
            OutputFormat::Json => {
                format!(
                    "Provide your analysis as a JSON object with the following structure:\n{}",
                    self.json_structure_for_analysis_type()
                )
            }
            OutputFormat::Markdown => {
                "Provide your analysis as a detailed Markdown document with sections for each major finding. Use headers, lists, and code blocks for structure and clarity.".to_string()
            }
            OutputFormat::Text => {
                "Provide your analysis as a detailed plain text document with clear sections and formatting.".to_string()
            }
        }
    }

    /// Generate JSON structure based on analysis type
    fn json_structure_for_analysis_type(&self) -> String {
        match self.analysis_type {
            AnalysisType::MainContent => {
                r#"{
  "mainContentAreas": [
    {
      "description": "Primary article content",
      "selector": "article.main-content",
      "confidence": 0.95,
      "containsText": true,
      "containsImages": true
    }
  ],
  "otherContentAreas": [
    {
      "type": "navigation",
      "selector": "nav.main-nav",
      "position": "top"
    },
    // Other non-main content areas
  ],
  "recommendedExtractionSelector": "article.main-content"
}"#.to_string()
            }
            AnalysisType::InteractiveElements => {
                r#"{
  "interactiveElements": [
    {
      "type": "button",
      "text": "Submit",
      "selector": "button#submit",
      "purpose": "Form submission",
      "isEnabled": true,
      "position": {"x": 100, "y": 200}
    },
    // Other interactive elements
  ],
  "elementCategories": {
    "buttons": ["button#submit", "a.btn-primary"],
    "inputs": ["input#email", "input#password"],
    "links": ["a.nav-link", "a.footer-link"]
  },
  "primaryActionElements": ["button#submit", "a.cta-button"]
}"#.to_string()
            }
            AnalysisType::FormStructure => {
                r#"{
  "forms": [
    {
      "id": "login-form",
      "action": "/login",
      "method": "POST",
      "fields": [
        {
          "name": "email",
          "type": "email",
          "label": "Email Address",
          "required": true,
          "validationRules": ["email", "required"]
        },
        // Other form fields
      ],
      "submitButton": "button[type='submit']",
      "clientSideValidation": true
    }
  ],
  "requiredFields": ["input#email", "input#password"],
  "validationPatterns": {
    "email": "email format validation",
    "password": "minimum 8 characters"
  }
}"#.to_string()
            }
            AnalysisType::TableStructure => {
                r#"{
  "tables": [
    {
      "selector": "table.data-table",
      "headerRow": true,
      "columns": [
        {
          "index": 0,
          "name": "Product",
          "type": "string"
        },
        // Other columns
      ],
      "rowCount": 10,
      "columnCount": 4,
      "hasMergedCells": false
    }
  ],
  "tableRelationships": {
    "parentChild": ["table.parent-table", "table.child-table"],
    "related": ["table.prices", "table.inventory"]
  },
  "extractionSelectors": {
    "headers": "table.data-table > thead > tr > th",
    "rows": "table.data-table > tbody > tr",
    "cells": "table.data-table > tbody > tr > td"
  }
}"#.to_string()
            }
            AnalysisType::GeneralStructure => {
                r#"{
  "pageStructure": {
    "header": {
      "selector": "header",
      "components": ["logo", "navigation", "search"]
    },
    "mainContent": {
      "selector": "main",
      "components": ["article", "sidebar", "comments"]
    },
    "footer": {
      "selector": "footer",
      "components": ["links", "copyright", "social"]
    }
  },
  "contentAreas": [
    {
      "type": "main",
      "selector": "main article",
      "importance": "primary"
    },
    // Other content areas
  ],
  "navigation": {
    "primary": "nav.main-nav",
    "secondary": "nav.footer-nav",
    "breadcrumbs": "nav.breadcrumbs"
  },
  "interactiveAreas": [
    {
      "type": "form",
      "selector": "form#contact",
      "purpose": "Contact form"
    },
    // Other interactive areas
  ]
}"#.to_string()
            }
        }
    }
}

impl fmt::Display for AnalysisPromptTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Truncate HTML if needed
        let truncated_html = truncate_html_for_context(&self.html, self.max_html_tokens);

        // Format guidelines
        let guidelines = self.generate_guidelines();

        // Format output instructions
        let format_instructions = self.generate_format_instructions();

        // Format examples if any
        let examples = format_examples(&self.components.examples, &self.variation);

        // Additional context if any
        let additional_context = self.additional_context
            .as_ref()
            .map(|ctx| format!("Additional Context:\n{}", ctx))
            .unwrap_or_default();

        // Build components map
        let mut components = HashMap::new();
        components.insert("system", self.components.system_instruction.clone());
        components.insert("html", format!("HTML Content:\n```html\n{}\n```", truncated_html));
        components.insert("objective", format!("Analysis Objective:\n{}", self.objective));
        components.insert("guidelines", format!("Analysis Guidelines:\n{}", guidelines));
        components.insert("format", format_instructions);

        // Add additional context if not empty
        if !additional_context.is_empty() {
            components.insert("context", additional_context);
        }

        // Add examples if not empty
        if !examples.is_empty() {
            components.insert("examples", examples);
        }

        // Create the standard prompt
        let prompt = create_standard_prompt(&components, &self.variation);
        write!(f, "{}", prompt)
    }
}

impl PromptTemplate for AnalysisPromptTemplate {
    fn as_string(&self) -> String {
        self.to_string()
    }

    fn estimate_token_count(&self) -> usize {
        let html_token_count = self.html.len() / 4;
        let objective_token_count = self.objective.len() / 4;
        let system_token_count = self.components.system_instruction.len() / 4;

        let additional_token_count = match &self.additional_context {
            Some(ctx) => ctx.len() / 4,
            None => 0,
        };

        let examples_token_count = self.components.examples.iter()
            .map(|ex| (ex.input.len() + ex.output.len()) / 4)
            .sum::<usize>();

        // Sum all token counts with some overhead for formatting
        html_token_count + objective_token_count + system_token_count +
        additional_token_count + examples_token_count + 300
    }

    fn clone_template(&self) -> Box<dyn PromptTemplate> {
        Box::new(self.clone())
    }
}

/// Helper function to create a general structure analysis prompt
pub fn create_structure_analysis_prompt(
    html: &str,
    objective: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    AnalysisPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_analysis_type(AnalysisType::GeneralStructure)
        .with_variation(variation)
        .with_output_format(OutputFormat::Json)
        .build()
        .to_string()
}

/// Helper function to create a main content identification prompt
pub fn create_main_content_analysis_prompt(
    html: &str,
    objective: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    AnalysisPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_analysis_type(AnalysisType::MainContent)
        .with_variation(variation)
        .with_output_format(OutputFormat::Json)
        .build()
        .to_string()
}

/// Helper function to create an interactive elements analysis prompt
pub fn create_interactive_elements_analysis_prompt(
    html: &str,
    objective: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    AnalysisPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_analysis_type(AnalysisType::InteractiveElements)
        .with_variation(variation)
        .with_output_format(OutputFormat::Json)
        .build()
        .to_string()
}

/// Helper function to create a form structure analysis prompt
pub fn create_form_analysis_prompt(
    html: &str,
    objective: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    AnalysisPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_analysis_type(AnalysisType::FormStructure)
        .with_variation(variation)
        .with_output_format(OutputFormat::Json)
        .build()
        .to_string()
}

/// Helper function to create a table structure analysis prompt
pub fn create_table_analysis_prompt(
    html: &str,
    objective: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    AnalysisPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_analysis_type(AnalysisType::TableStructure)
        .with_variation(variation)
        .with_output_format(OutputFormat::Json)
        .build()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_analysis_prompt() {
        let html = "<div><header>Header</header><main>Content</main><footer>Footer</footer></div>";
        let objective = "Analyze the page structure";

        let prompt = create_structure_analysis_prompt(html, objective, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("pageStructure"));
    }

    #[test]
    fn test_main_content_analysis_prompt() {
        let html = "<div><header>Header</header><main>Content</main><footer>Footer</footer></div>";
        let objective = "Identify the main content area";

        let prompt = create_main_content_analysis_prompt(html, objective, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("mainContentAreas"));
        assert!(prompt.contains("main content area"));
    }

    #[test]
    fn test_interactive_elements_analysis_prompt() {
        let html = "<div><button>Click</button><a href='#'>Link</a></div>";
        let objective = "Identify all interactive elements";

        let prompt = create_interactive_elements_analysis_prompt(html, objective, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("interactiveElements"));
        assert!(prompt.contains("interactive elements"));
    }

    #[test]
    fn test_form_analysis_prompt() {
        let html = "<form><input type='text'><button type='submit'>Submit</button></form>";
        let objective = "Analyze the form structure";

        let prompt = create_form_analysis_prompt(html, objective, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("forms"));
        assert!(prompt.contains("form structures"));
    }

    #[test]
    fn test_table_analysis_prompt() {
        let html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>";
        let objective = "Analyze the table structure";

        let prompt = create_table_analysis_prompt(html, objective, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("tables"));
        assert!(prompt.contains("table structures"));
    }

    #[test]
    fn test_analysis_prompt_template_build() {
        let template = AnalysisPromptTemplate::new()
            .with_html("<div>Test</div>")
            .with_objective("Analyze the structure")
            .with_analysis_type(AnalysisType::GeneralStructure)
            .with_output_format(OutputFormat::Markdown)
            .build();

        let prompt = template.to_string();

        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Analyze the structure"));
        assert!(prompt.contains("Markdown"));
    }

    #[test]
    fn test_prompt_template_trait_implementation() {
        let template: Box<dyn PromptTemplate> = Box::new(
            AnalysisPromptTemplate::new()
                .with_html("<div>Test</div>")
                .with_objective("Analyze the structure")
                .build()
        );

        let prompt = template.as_string();
        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Analyze the structure"));

        let token_count = template.estimate_token_count();
        assert!(token_count > 0);

        let cloned = template.clone_template();
        let cloned_prompt = cloned.as_string();
        assert_eq!(prompt, cloned_prompt);
    }
}
