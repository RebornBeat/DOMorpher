//! # Extraction Prompts Module
//!
//! This module provides specialized prompt templates for extracting structured data from HTML content.
//! These templates are optimized to guide LLMs in accurate extraction based on natural language instructions.
//!
//! ## Features
//!
//! - Schema-based extraction prompts
//! - Hierarchical extraction for nested data
//! - Multi-item extraction for lists and collections
//! - Table extraction specialization
//! - Text-heavy content extraction

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    PromptComponents, PromptExample, PromptTemplate, TemplateVariation, create_standard_prompt,
    format_examples, format_system_instruction, system_instructions, truncate_html_for_context,
};

/// Template for extracting structured data from HTML content
#[derive(Clone, Debug)]
pub struct ExtractionPromptTemplate {
    /// HTML content to extract from
    pub html: String,

    /// Extraction instruction
    pub instruction: String,

    /// Expected output format (e.g., "json", "csv", "yaml")
    pub format: String,

    /// Optional schema for the extraction
    pub schema: Option<String>,

    /// Shared prompt components
    pub components: PromptComponents,

    /// Template variation for different LLM providers
    pub variation: TemplateVariation,

    /// Extraction type for specialized prompts
    pub extraction_type: ExtractionType,

    /// Maximum tokens to allow for the HTML content
    pub max_html_tokens: usize,
}

/// Types of extraction for specialized prompts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExtractionType {
    /// Standard extraction
    Standard,

    /// Hierarchical extraction for nested data
    Hierarchical,

    /// Multi-item extraction for lists
    MultiItem,

    /// Table extraction
    Table,

    /// Text-heavy content extraction
    TextHeavy,
}

impl Default for ExtractionPromptTemplate {
    fn default() -> Self {
        Self {
            html: String::new(),
            instruction: String::new(),
            format: "json".to_string(),
            schema: None,
            components: PromptComponents {
                system_instruction: system_instructions::CLAUDE_SYSTEM.to_string(),
                examples: Vec::new(),
                additional_context: None,
            },
            variation: TemplateVariation::Claude,
            extraction_type: ExtractionType::Standard,
            max_html_tokens: 8000,
        }
    }
}

impl ExtractionPromptTemplate {
    /// Create a new extraction prompt template
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTML content
    pub fn with_html(mut self, html: &str) -> Self {
        self.html = html.to_string();
        self
    }

    /// Set the extraction instruction
    pub fn with_instruction(mut self, instruction: &str) -> Self {
        self.instruction = instruction.to_string();
        self
    }

    /// Set the output format
    pub fn with_format(mut self, format: &str) -> Self {
        self.format = format.to_string();
        self
    }

    /// Set the extraction schema
    pub fn with_schema(mut self, schema: &str) -> Self {
        self.schema = Some(schema.to_string());
        self
    }

    /// Set the template variation
    pub fn with_variation(mut self, variation: TemplateVariation) -> Self {
        self.variation = variation;

        // Update system instruction based on variation
        match variation {
            TemplateVariation::Claude => {
                self.components.system_instruction = system_instructions::CLAUDE_SYSTEM.to_string();
            }
            TemplateVariation::Gpt => {
                self.components.system_instruction = system_instructions::GPT_SYSTEM.to_string();
            }
            _ => {}
        }

        self
    }

    /// Set the extraction type
    pub fn with_extraction_type(mut self, extraction_type: ExtractionType) -> Self {
        self.extraction_type = extraction_type;

        // Update system instruction based on extraction type
        if let ExtractionType::Hierarchical = extraction_type {
            self.components.system_instruction = format!(
                "{} Pay special attention to hierarchical relationships between elements.",
                self.components.system_instruction
            );
        }

        if let ExtractionType::Table = extraction_type {
            self.components.system_instruction = format!(
                "{} Focus on extracting tabular data, preserving row and column relationships.",
                self.components.system_instruction
            );
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

    /// Add additional context to the prompt
    pub fn with_additional_context(mut self, context: &str) -> Self {
        self.components.additional_context = Some(context.to_string());
        self
    }

    /// Build the final prompt
    pub fn build(self) -> Self {
        self
    }

    /// Format the expected structure based on schema or instruction
    fn format_expected_structure(&self) -> String {
        if let Some(schema) = &self.schema {
            return schema.clone();
        }

        // If no schema provided, generate a simple structure based on the instruction
        match self.extraction_type {
            ExtractionType::MultiItem => r#"[
  {
    "item1": "value1",
    "item2": "value2",
    // Additional fields as needed
  },
  // Additional items as needed
]"#
            .to_string(),
            ExtractionType::Table => r#"[
  {
    "column1": "row1_value1",
    "column2": "row1_value2",
    // Additional columns as needed
  },
  // Additional rows as needed
]"#
            .to_string(),
            ExtractionType::Hierarchical => r#"{
  "parent1": {
    "child1": "value1",
    "child2": {
      "grandchild1": "value2"
    }
  },
  // Additional hierarchical data as needed
}"#
            .to_string(),
            _ => r#"{
  "field1": "value1",
  "field2": "value2",
  // Additional fields as needed
}"#
            .to_string(),
        }
    }

    /// Generate extraction-specific guidelines
    fn generate_guidelines(&self) -> String {
        let mut guidelines = vec![
            "1. Be precise and accurate in your extraction".to_string(),
            "2. Follow the exact structure requested".to_string(),
            "3. Return null for any fields you cannot find".to_string(),
        ];

        match self.extraction_type {
            ExtractionType::Hierarchical => {
                guidelines.push(
                    "4. Preserve parent-child relationships in the extracted data".to_string(),
                );
                guidelines.push(
                    "5. Ensure nested objects maintain their proper hierarchical structure"
                        .to_string(),
                );
            }
            ExtractionType::MultiItem => {
                guidelines
                    .push("4. Extract all matching items, not just the first one".to_string());
                guidelines.push(
                    "5. Ensure consistency in the structure across all extracted items".to_string(),
                );
            }
            ExtractionType::Table => {
                guidelines.push("4. Preserve the tabular structure of the data".to_string());
                guidelines.push(
                    "5. Handle merged cells by duplicating values as appropriate".to_string(),
                );
                guidelines.push("6. Identify and use proper column headers".to_string());
            }
            ExtractionType::TextHeavy => {
                guidelines.push(
                    "4. Preserve paragraph breaks and text formatting where relevant".to_string(),
                );
                guidelines.push("5. Extract complete sentences and maintain context".to_string());
            }
            _ => {
                guidelines.push(
                    "4. Extract all relevant data as specified in the instruction".to_string(),
                );
            }
        }

        guidelines.join("\n")
    }

    /// Generate type-specific additional instructions
    fn type_specific_instructions(&self) -> String {
        match self.extraction_type {
            ExtractionType::Hierarchical => {
                "Focus on correctly identifying parent-child relationships in the HTML structure. Pay attention to nested elements and how they relate to each other. Make sure the output maintains these hierarchical relationships."
            }
            ExtractionType::MultiItem => {
                "Extract all items that match the criteria, not just the first one. Each item should have consistent fields, even if some values are null. Make sure to identify repeating patterns in the HTML structure."
            }
            ExtractionType::Table => {
                "Extract the data as a table, preserving row and column relationships. Table headers should become field names. Handle merged cells by duplicating values across rows or columns as appropriate. Complex tables should maintain their structure."
            }
            ExtractionType::TextHeavy => {
                "Focus on extracting and preserving text content, including paragraph structure, formatting, and context. For long text, ensure you capture the complete content while maintaining its organization."
            }
            _ => ""
        }.to_string()
    }
}

impl fmt::Display for ExtractionPromptTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Truncate HTML if needed
        let truncated_html = truncate_html_for_context(&self.html, self.max_html_tokens);

        // Get additional instructions for this extraction type
        let type_specific = self.type_specific_instructions();

        // Format the expected output structure
        let expected_structure = self.format_expected_structure();

        // Format the guidelines
        let guidelines = self.generate_guidelines();

        // Format examples if any
        let examples = format_examples(&self.components.examples, &self.variation);

        // Additional context if any
        let additional_context = self
            .components
            .additional_context
            .as_ref()
            .map(|ctx| format!("Additional Context:\n{}", ctx))
            .unwrap_or_default();

        // Build components map
        let mut components = HashMap::new();
        components.insert("system", self.components.system_instruction.clone());
        components.insert("html", truncated_html);
        components.insert("instruction", self.instruction.clone());
        components.insert("format", format!("Extract the requested information and return it as a valid {} object with the following structure:\n{}", self.format, expected_structure));
        components.insert(
            "additional",
            format!("Additional Guidelines:\n{}", guidelines),
        );

        // Add schema if present
        if let Some(schema) = &self.schema {
            components.insert("schema", schema.clone());
        }

        // Add type-specific instructions if not empty
        if !type_specific.is_empty() {
            components.insert(
                "type_specific",
                format!("Specific Instructions:\n{}", type_specific),
            );
        }

        // Add examples if not empty
        if !examples.is_empty() {
            components.insert("examples", examples);
        }

        // Add additional context if not empty
        if !additional_context.is_empty() {
            components.insert("context", additional_context);
        }

        // Create the standard prompt
        let prompt = create_standard_prompt(&components, &self.variation);
        write!(f, "{}", prompt)
    }
}

impl PromptTemplate for ExtractionPromptTemplate {
    fn as_string(&self) -> String {
        self.to_string()
    }

    fn estimate_token_count(&self) -> usize {
        let html_token_count = self.html.len() / 4;
        let instruction_token_count = self.instruction.len() / 4;
        let system_token_count = self.components.system_instruction.len() / 4;
        let examples_token_count = self
            .components
            .examples
            .iter()
            .map(|ex| (ex.input.len() + ex.output.len()) / 4)
            .sum::<usize>();

        let additional_token_count = match &self.components.additional_context {
            Some(ctx) => ctx.len() / 4,
            None => 0,
        };

        let schema_token_count = match &self.schema {
            Some(schema) => schema.len() / 4,
            None => 0,
        };

        // Sum all token counts with some overhead for formatting
        html_token_count
            + instruction_token_count
            + system_token_count
            + examples_token_count
            + additional_token_count
            + schema_token_count
            + 200
    }

    fn clone_template(&self) -> Box<dyn PromptTemplate> {
        Box::new(self.clone())
    }
}

/// Helper function to create a basic extraction prompt
pub fn create_basic_extraction_prompt(
    html: &str,
    instruction: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    ExtractionPromptTemplate::new()
        .with_html(html)
        .with_instruction(instruction)
        .with_variation(variation)
        .build()
        .to_string()
}

/// Helper function to create a schema-based extraction prompt
pub fn create_schema_extraction_prompt(
    html: &str,
    instruction: &str,
    schema: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    ExtractionPromptTemplate::new()
        .with_html(html)
        .with_instruction(instruction)
        .with_schema(schema)
        .with_variation(variation)
        .with_extraction_type(ExtractionType::Standard)
        .build()
        .to_string()
}

/// Helper function to create a table extraction prompt
pub fn create_table_extraction_prompt(
    html: &str,
    instruction: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    ExtractionPromptTemplate::new()
        .with_html(html)
        .with_instruction(instruction)
        .with_variation(variation)
        .with_extraction_type(ExtractionType::Table)
        .build()
        .to_string()
}

/// Helper function to create a hierarchical extraction prompt
pub fn create_hierarchical_extraction_prompt(
    html: &str,
    instruction: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    ExtractionPromptTemplate::new()
        .with_html(html)
        .with_instruction(instruction)
        .with_variation(variation)
        .with_extraction_type(ExtractionType::Hierarchical)
        .build()
        .to_string()
}

/// Helper function to create a multi-item extraction prompt
pub fn create_multi_item_extraction_prompt(
    html: &str,
    instruction: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    ExtractionPromptTemplate::new()
        .with_html(html)
        .with_instruction(instruction)
        .with_variation(variation)
        .with_extraction_type(ExtractionType::MultiItem)
        .build()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extraction_prompt() {
        let html = "<div><p>Product: Widget</p><p>Price: $19.99</p></div>";
        let instruction = "Extract the product name and price";

        let prompt = create_basic_extraction_prompt(html, instruction, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(instruction));
        assert!(prompt.contains("Extract the requested information"));
    }

    #[test]
    fn test_schema_extraction_prompt() {
        let html = "<div><p>Product: Widget</p><p>Price: $19.99</p></div>";
        let instruction = "Extract the product name and price";
        let schema = r#"{"name": "string", "price": "number"}"#;

        let prompt = create_schema_extraction_prompt(html, instruction, schema, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(instruction));
        assert!(prompt.contains(schema));
    }

    #[test]
    fn test_table_extraction_prompt() {
        let html = "<table><tr><th>Name</th><th>Price</th></tr><tr><td>Widget</td><td>$19.99</td></tr></table>";
        let instruction = "Extract the product table";

        let prompt = create_table_extraction_prompt(html, instruction, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(instruction));
        assert!(prompt.contains("extracting tabular data"));
    }

    #[test]
    fn test_extraction_prompt_template_build() {
        let template = ExtractionPromptTemplate::new()
            .with_html("<div>Test</div>")
            .with_instruction("Extract test")
            .with_format("yaml")
            .with_extraction_type(ExtractionType::Hierarchical)
            .build();

        let prompt = template.to_string();
        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Extract test"));
        assert!(prompt.contains("yaml"));
        assert!(prompt.contains("hierarchical"));
    }

    #[test]
    fn test_prompt_template_trait_implementation() {
        let template: Box<dyn PromptTemplate> = Box::new(
            ExtractionPromptTemplate::new()
                .with_html("<div>Test</div>")
                .with_instruction("Extract test")
                .build(),
        );

        let prompt = template.as_string();
        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Extract test"));

        let token_count = template.estimate_token_count();
        assert!(token_count > 0);

        let cloned = template.clone_template();
        let cloned_prompt = cloned.as_string();
        assert_eq!(prompt, cloned_prompt);
    }
}
