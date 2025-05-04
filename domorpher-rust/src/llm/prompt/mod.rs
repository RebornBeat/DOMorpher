//! # LLM Prompts Module
//!
//! This module provides specialized prompt templates for different extraction and navigation tasks.
//! The templates are designed to optimize LLM performance for specific DOMorpher operations.
//!
//! ## Available Prompt Templates
//!
//! - **Extraction Prompts**: Templates for data extraction from HTML content
//! - **Navigation Prompts**: Templates for autonomous web navigation and interaction
//! - **Analysis Prompts**: Templates for DOM structure analysis and understanding
//!
//! ## Usage Example
//!
//! ```rust
//! use domorpher::llm::prompts::extraction::ExtractionPromptTemplate;
//!
//! let extraction_prompt = ExtractionPromptTemplate::new()
//!     .with_html("<div>Product information</div>")
//!     .with_instruction("Extract the product name and price")
//!     .with_format("json")
//!     .build();
//!
//! println!("Prompt text: {}", extraction_prompt);
//! ```

use std::collections::HashMap;
use std::fmt;

pub mod analysis;
pub mod extraction;
pub mod navigation;

/// Re-export key types
pub use analysis::AnalysisPromptTemplate;
pub use extraction::ExtractionPromptTemplate;
pub use navigation::NavigationPromptTemplate;

/// Base trait for all prompt templates
pub trait PromptTemplate: fmt::Display {
    /// Returns the template as a string
    fn as_string(&self) -> String;

    /// Returns the estimated token count for this prompt
    fn estimate_token_count(&self) -> usize;

    /// Returns a cloned version of the template
    fn clone_template(&self) -> Box<dyn PromptTemplate>;
}

/// Common prompt components shared across different templates
#[derive(Clone, Debug)]
pub struct PromptComponents {
    /// System instruction that sets the context and role
    pub system_instruction: String,

    /// Optional examples to include for few-shot learning
    pub examples: Vec<PromptExample>,

    /// Optional additional context
    pub additional_context: Option<String>,
}

impl Default for PromptComponents {
    fn default() -> Self {
        Self {
            system_instruction: String::new(),
            examples: Vec::new(),
            additional_context: None,
        }
    }
}

/// Example for few-shot learning
#[derive(Clone, Debug)]
pub struct PromptExample {
    /// Input to the model
    pub input: String,

    /// Expected output from the model
    pub output: String,

    /// Optional explanation of the example
    pub explanation: Option<String>,
}

impl PromptExample {
    /// Create a new prompt example
    pub fn new(input: &str, output: &str) -> Self {
        Self {
            input: input.to_string(),
            output: output.to_string(),
            explanation: None,
        }
    }

    /// Add an explanation to the example
    pub fn with_explanation(mut self, explanation: &str) -> Self {
        self.explanation = Some(explanation.to_string());
        self
    }

    /// Format the example as a string
    pub fn format(&self) -> String {
        let mut formatted = format!("Input:\n{}\n\nOutput:\n{}", self.input, self.output);

        if let Some(explanation) = &self.explanation {
            formatted.push_str(&format!("\n\nExplanation:\n{}", explanation));
        }

        formatted
    }
}

/// Common system instructions for different providers
pub mod system_instructions {
    /// System instruction for Anthropic Claude
    pub const CLAUDE_SYSTEM: &str = "You are Claude, an AI assistant that specializes in web data extraction. You are helping with extracting structured data from HTML content according to specific instructions. Your task is to identify the relevant information from the HTML and return it in the requested format.";

    /// System instruction for OpenAI GPT
    pub const GPT_SYSTEM: &str = "You are a specialized web data extraction assistant. Your job is to extract structured data from HTML content according to specific instructions. Identify the relevant information from the HTML and return it in the requested format.";

    /// System instruction for DOM analysis
    pub const DOM_ANALYSIS_SYSTEM: &str = "You are an expert web DOM analyzer. Your task is to analyze the structure of HTML content, identify key components, and provide insights about the page organization, main content areas, and interactive elements.";

    /// System instruction for web navigation
    pub const WEB_NAVIGATION_SYSTEM: &str = "You are an autonomous web navigation assistant. Your job is to determine the next steps for navigating a website to accomplish a specific goal. Based on the current page state and objective, decide on the most appropriate action to take next.";

    /// System instruction for extraction with schema
    pub const SCHEMA_EXTRACTION_SYSTEM: &str = "You are a precise data extraction assistant. Your task is to extract structured data from HTML content according to a specific schema. You must adhere strictly to the schema, ensuring all required fields are present and formatted correctly.";
}

/// Template variations for different LLM providers
#[derive(Clone, Debug)]
pub enum TemplateVariation {
    /// Anthropic Claude optimized template
    Claude,

    /// OpenAI GPT optimized template
    Gpt,

    /// Local model optimized template
    Local,

    /// Generic template that works with most providers
    Generic,
}

impl Default for TemplateVariation {
    fn default() -> Self {
        Self::Generic
    }
}

/// Helper function to truncate HTML content to fit within token limits
pub fn truncate_html_for_context(html: &str, max_tokens: usize) -> String {
    // Simple estimation: 1 token ≈ 4 characters
    let max_chars = max_tokens * 4;

    if html.len() <= max_chars {
        return html.to_string();
    }

    // Try to find a clean break point
    let truncated = &html[..max_chars];

    // Find the last closing tag
    if let Some(last_closing) = truncated.rfind('>') {
        let truncated = &truncated[..=last_closing];

        // Add closing tags for any unclosed tags
        let mut unclosed_tags = Vec::new();
        let mut tag_stack = Vec::new();

        let re = regex::Regex::new(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*>").unwrap();
        for cap in re.captures_iter(truncated) {
            let is_closing = &cap[1] == "/";
            let tag_name = &cap[2];

            if is_closing {
                if let Some(last_tag) = tag_stack.last() {
                    if last_tag == tag_name {
                        tag_stack.pop();
                    }
                }
            } else {
                // Check if it's a self-closing tag
                let tag_match = &cap[0];
                if !tag_match.ends_with("/>") {
                    tag_stack.push(tag_name.to_string());
                }
            }
        }

        // Add closing tags in reverse order
        let mut result = truncated.to_string();
        for tag in tag_stack.iter().rev() {
            result.push_str(&format!("</{}>", tag));
        }

        result.push_str("\n<!-- HTML content truncated to fit context window -->");
        return result;
    }

    // Fallback: simple truncation with warning
    format!("{}<!-- HTML content truncated -->", &html[..max_chars])
}

/// Helper function to format a system instruction based on template variation
pub fn format_system_instruction(instruction: &str, variation: &TemplateVariation) -> String {
    match variation {
        TemplateVariation::Claude => {
            format!("I need you to help with the following:\n\n{}", instruction)
        }
        TemplateVariation::Gpt => format!("System: {}", instruction),
        TemplateVariation::Local => instruction.to_string(),
        TemplateVariation::Generic => instruction.to_string(),
    }
}

/// Helper function to format examples based on template variation
pub fn format_examples(examples: &[PromptExample], variation: &TemplateVariation) -> String {
    if examples.is_empty() {
        return String::new();
    }

    let formatted_examples: Vec<String> = examples.iter().map(|ex| ex.format()).collect();

    match variation {
        TemplateVariation::Claude => {
            format!(
                "\n\nHere are some examples to help guide your response:\n\n{}",
                formatted_examples.join("\n\n---\n\n")
            )
        }
        TemplateVariation::Gpt => {
            format!(
                "\n\nExamples:\n\n{}",
                formatted_examples.join("\n\n---\n\n")
            )
        }
        _ => {
            format!(
                "\n\nExamples:\n\n{}",
                formatted_examples.join("\n\n---\n\n")
            )
        }
    }
}

/// Helper function to create a standardized template for all providers
pub fn create_standard_prompt(
    components: &HashMap<&str, String>,
    variation: &TemplateVariation,
) -> String {
    let mut parts = Vec::new();

    // Add system instruction if present
    if let Some(system) = components.get("system") {
        parts.push(format_system_instruction(system, variation));
    }

    // Add other components in a standardized order
    for key in &[
        "context",
        "html",
        "instruction",
        "format",
        "schema",
        "examples",
        "additional",
    ] {
        if let Some(value) = components.get(key) {
            if !value.is_empty() {
                match *key {
                    "html" => parts.push(format!("HTML Content:\n```html\n{}\n```", value)),
                    "instruction" => parts.push(format!("Instructions:\n{}", value)),
                    "format" => parts.push(format!("Output Format:\n{}", value)),
                    "schema" => parts.push(format!("Schema:\n```json\n{}\n```", value)),
                    "context" => parts.push(format!("Context:\n{}", value)),
                    "examples" => parts.push(value.clone()),
                    "additional" => parts.push(value.clone()),
                    _ => parts.push(value.clone()),
                }
            }
        }
    }

    parts.join("\n\n")
}

/// Parse provider information from a prompt template variation
pub fn get_provider_from_variation(
    variation: &TemplateVariation,
) -> crate::llm::provider::LlmProvider {
    match variation {
        TemplateVariation::Claude => crate::llm::provider::LlmProvider::Anthropic,
        TemplateVariation::Gpt => crate::llm::provider::LlmProvider::OpenAI,
        TemplateVariation::Local => crate::llm::provider::LlmProvider::Local,
        TemplateVariation::Generic => crate::llm::provider::LlmProvider::Anthropic, // Default to Anthropic for generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_example_format() {
        let example = PromptExample::new(
            "Extract the product name",
            "{ \"name\": \"Ergonomic Chair\" }",
        )
        .with_explanation("This example shows how to extract a product name");

        let formatted = example.format();
        assert!(formatted.contains("Input:"));
        assert!(formatted.contains("Output:"));
        assert!(formatted.contains("Explanation:"));
    }

    #[test]
    fn test_truncate_html_for_context() {
        let html = "<div><p>Test paragraph</p><span>Text</span></div>";

        // Test without truncation (limit higher than content)
        let result = truncate_html_for_context(html, 100);
        assert_eq!(result, html);

        // Test with truncation
        let result = truncate_html_for_context(html, 5); // Very small limit
        assert!(result.contains("</div>"));
        assert!(result.contains("<!-- HTML content truncated"));
    }

    #[test]
    fn test_format_system_instruction() {
        let instruction = "Extract data from HTML";

        let claude = format_system_instruction(instruction, &TemplateVariation::Claude);
        assert!(claude.contains("I need you to help with"));

        let gpt = format_system_instruction(instruction, &TemplateVariation::Gpt);
        assert!(gpt.contains("System:"));

        let generic = format_system_instruction(instruction, &TemplateVariation::Generic);
        assert_eq!(generic, instruction);
    }

    #[test]
    fn test_create_standard_prompt() {
        let mut components = HashMap::new();
        components.insert("system", "You are a data extraction assistant".to_string());
        components.insert("html", "<div>Product</div>".to_string());
        components.insert("instruction", "Extract the product name".to_string());

        let prompt = create_standard_prompt(&components, &TemplateVariation::Generic);

        assert!(prompt.contains("You are a data extraction assistant"));
        assert!(prompt.contains("<div>Product</div>"));
        assert!(prompt.contains("Extract the product name"));
    }
}
