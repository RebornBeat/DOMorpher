//! # Navigation Prompts Module
//!
//! This module provides specialized prompt templates for autonomous web navigation.
//! These templates guide LLMs in deciding which actions to take on a web page to
//! accomplish specific objectives.
//!
//! ## Features
//!
//! - Goal-oriented navigation prompts
//! - Context-aware decision making
//! - Element interaction guidance
//! - Multi-step planning
//! - Form filling assistance

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    PromptComponents, PromptExample, PromptTemplate, TemplateVariation, create_standard_prompt,
    format_examples, format_system_instruction, system_instructions, truncate_html_for_context,
};

/// Template for web navigation decisions
#[derive(Clone, Debug)]
pub struct NavigationPromptTemplate {
    /// Current HTML state
    pub html: String,

    /// Navigation objective
    pub objective: String,

    /// Current navigation context (history, previous actions, etc.)
    pub context: String,

    /// Interactive elements identified on the page
    pub interactive_elements: Vec<InteractiveElement>,

    /// Navigation type for specialized prompts
    pub navigation_type: NavigationType,

    /// Shared prompt components
    pub components: PromptComponents,

    /// Template variation for different LLM providers
    pub variation: TemplateVariation,

    /// Maximum tokens to allow for the HTML content
    pub max_html_tokens: usize,

    /// Previous actions taken
    pub previous_actions: Vec<NavigationAction>,

    /// Current page URL
    pub current_url: Option<String>,

    /// Progress towards objective (0.0 to 1.0)
    pub progress: Option<f32>,
}

/// Types of navigation for specialized prompts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NavigationType {
    /// Standard navigation
    Standard,

    /// Form filling navigation
    FormFilling,

    /// Search and filter navigation
    SearchAndFilter,

    /// Checkout process navigation
    Checkout,

    /// Data extraction navigation
    DataExtraction,
}

/// Representation of an interactive element on the page
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractiveElement {
    /// Element index
    pub index: usize,

    /// Element tag (e.g., "button", "a", "input")
    pub tag: String,

    /// Element text content
    pub text: Option<String>,

    /// Element types (for inputs)
    pub input_type: Option<String>,

    /// Element placeholder
    pub placeholder: Option<String>,

    /// Element ID
    pub id: Option<String>,

    /// Element classes
    pub classes: Vec<String>,

    /// Whether the element is visible
    pub visible: bool,

    /// Element position information
    pub position: ElementPosition,
}

/// Element position on the page
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElementPosition {
    /// X coordinate (from left)
    pub x: f32,

    /// Y coordinate (from top)
    pub y: f32,

    /// Width of the element
    pub width: f32,

    /// Height of the element
    pub height: f32,
}

/// Navigation action
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavigationAction {
    /// Action type
    pub action_type: String,

    /// Target element index or description
    pub target: String,

    /// Action value (e.g., text to input)
    pub value: Option<String>,

    /// Action result
    pub result: Option<String>,

    /// Timestamp
    pub timestamp: Option<String>,
}

impl Default for NavigationPromptTemplate {
    fn default() -> Self {
        Self {
            html: String::new(),
            objective: String::new(),
            context: String::new(),
            interactive_elements: Vec::new(),
            navigation_type: NavigationType::Standard,
            components: PromptComponents {
                system_instruction: system_instructions::WEB_NAVIGATION_SYSTEM.to_string(),
                examples: Vec::new(),
                additional_context: None,
            },
            variation: TemplateVariation::Claude,
            max_html_tokens: 8000,
            previous_actions: Vec::new(),
            current_url: None,
            progress: None,
        }
    }
}

impl NavigationPromptTemplate {
    /// Create a new navigation prompt template
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTML content
    pub fn with_html(mut self, html: &str) -> Self {
        self.html = html.to_string();
        self
    }

    /// Set the navigation objective
    pub fn with_objective(mut self, objective: &str) -> Self {
        self.objective = objective.to_string();
        self
    }

    /// Set the navigation context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = context.to_string();
        self
    }

    /// Set the interactive elements
    pub fn with_interactive_elements(mut self, elements: Vec<InteractiveElement>) -> Self {
        self.interactive_elements = elements;
        self
    }

    /// Add an interactive element
    pub fn add_interactive_element(mut self, element: InteractiveElement) -> Self {
        self.interactive_elements.push(element);
        self
    }

    /// Set the navigation type
    pub fn with_navigation_type(mut self, navigation_type: NavigationType) -> Self {
        self.navigation_type = navigation_type;

        // Update system instruction based on navigation type
        match navigation_type {
            NavigationType::FormFilling => {
                self.components.system_instruction = format!(
                    "{} Focus on accurately filling out forms with the provided information.",
                    self.components.system_instruction
                );
            }
            NavigationType::SearchAndFilter => {
                self.components.system_instruction = format!(
                    "{} Focus on using search and filter controls to find specific content.",
                    self.components.system_instruction
                );
            }
            NavigationType::Checkout => {
                self.components.system_instruction = format!(
                    "{} Focus on completing a checkout process with the provided information.",
                    self.components.system_instruction
                );
            }
            NavigationType::DataExtraction => {
                self.components.system_instruction = format!(
                    "{} Focus on navigating to find specific data for extraction.",
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
                self.components.system_instruction =
                    system_instructions::WEB_NAVIGATION_SYSTEM.to_string();
            }
            TemplateVariation::Gpt => {
                self.components.system_instruction =
                    system_instructions::WEB_NAVIGATION_SYSTEM.to_string();
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

    /// Set the previous actions
    pub fn with_previous_actions(mut self, actions: Vec<NavigationAction>) -> Self {
        self.previous_actions = actions;
        self
    }

    /// Add a previous action
    pub fn add_previous_action(mut self, action: NavigationAction) -> Self {
        self.previous_actions.push(action);
        self
    }

    /// Set the current URL
    pub fn with_current_url(mut self, url: &str) -> Self {
        self.current_url = Some(url.to_string());
        self
    }

    /// Set the progress towards objective
    pub fn with_progress(mut self, progress: f32) -> Self {
        self.progress = Some(progress.max(0.0).min(1.0));
        self
    }

    /// Build the final prompt
    pub fn build(self) -> Self {
        self
    }

    /// Format the interactive elements as a string
    fn format_interactive_elements(&self) -> String {
        if self.interactive_elements.is_empty() {
            return "No interactive elements identified on the page.".to_string();
        }

        let mut result = String::from("Interactive elements on the page:\n\n");

        for element in &self.interactive_elements {
            let text = element.text.clone().unwrap_or_default();
            let placeholder = element.placeholder.clone().unwrap_or_default();
            let id = element.id.clone().unwrap_or_default();
            let classes = element.classes.join(", ");
            let visibility = if element.visible {
                "visible"
            } else {
                "not visible"
            };

            result.push_str(&format!(
                "Element {}: {} - Text: '{}' - Placeholder: '{}' - ID: '{}' - Classes: [{}] - {} - Position: (x:{}, y:{})\n",
                element.index,
                element.tag,
                text,
                placeholder,
                id,
                classes,
                visibility,
                element.position.x,
                element.position.y
            ));
        }

        result
    }

    /// Format the previous actions as a string
    fn format_previous_actions(&self) -> String {
        if self.previous_actions.is_empty() {
            return "No previous actions taken.".to_string();
        }

        let mut result = String::from("Previous actions taken:\n\n");

        for (i, action) in self.previous_actions.iter().enumerate() {
            let action_result = action.result.clone().unwrap_or_default();

            result.push_str(&format!(
                "Action {}: {} on target '{}' ",
                i + 1,
                action.action_type,
                action.target
            ));

            if let Some(value) = &action.value {
                result.push_str(&format!("with value '{}' ", value));
            }

            if !action_result.is_empty() {
                result.push_str(&format!("- Result: {}", action_result));
            }

            result.push('\n');
        }

        result
    }

    /// Generate navigation-specific guidelines
    fn generate_guidelines(&self) -> String {
        let mut guidelines = vec![
            "1. Analyze the current page state and interactive elements".to_string(),
            "2. Consider the objective and previous actions".to_string(),
            "3. Determine the most appropriate next action".to_string(),
            "4. Provide your reasoning for the chosen action".to_string(),
        ];

        match self.navigation_type {
            NavigationType::FormFilling => {
                guidelines.push(
                    "5. Focus on accurately filling form fields with the correct information"
                        .to_string(),
                );
                guidelines.push(
                    "6. Ensure all required fields are completed before submission".to_string(),
                );
            }
            NavigationType::SearchAndFilter => {
                guidelines.push("5. Identify search and filter controls on the page".to_string());
                guidelines.push(
                    "6. Use specific search terms and filter options to narrow down results"
                        .to_string(),
                );
            }
            NavigationType::Checkout => {
                guidelines
                    .push("5. Focus on completing each step of the checkout process".to_string());
                guidelines.push(
                    "6. Enter information accurately and verify details before final submission"
                        .to_string(),
                );
            }
            NavigationType::DataExtraction => {
                guidelines.push("5. Navigate to pages containing the required data".to_string());
                guidelines.push(
                    "6. Identify when the goal has been reached and data can be extracted"
                        .to_string(),
                );
            }
            _ => {
                guidelines
                    .push("5. Choose actions that make progress toward the objective".to_string());
                guidelines
                    .push("6. Consider the most efficient path to achieve the goal".to_string());
            }
        }

        guidelines.join("\n")
    }

    /// Generate the expected action format instructions
    fn generate_action_format(&self) -> String {
        r#"Provide your reasoning in detail, then specify the exact action in this format:

ACTION: [action type - click, input, select, scroll, wait, navigate, extract, complete]
TARGET: [element index or description]
VALUE: [any additional parameters needed]

For example:
ACTION: click
TARGET: Element 3
VALUE:

Or:
ACTION: input
TARGET: Element 5
VALUE: search term

Or:
ACTION: complete
TARGET: task complete
VALUE: objective achieved"#
            .to_string()
    }
}

impl fmt::Display for NavigationPromptTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Truncate HTML if needed
        let truncated_html = truncate_html_for_context(&self.html, self.max_html_tokens);

        // Format interactive elements
        let interactive_elements = self.format_interactive_elements();

        // Format previous actions
        let previous_actions = self.format_previous_actions();

        // Format guidelines
        let guidelines = self.generate_guidelines();

        // Format action format instructions
        let action_format = self.generate_action_format();

        // Format examples if any
        let examples = format_examples(&self.components.examples, &self.variation);

        // Format progress if available
        let progress = if let Some(p) = self.progress {
            format!("Current progress: {}%", (p * 100.0) as usize)
        } else {
            String::new()
        };

        // Format current URL if available
        let current_url = if let Some(url) = &self.current_url {
            format!("Current URL: {}", url)
        } else {
            String::new()
        };

        // Build components map
        let mut components = HashMap::new();
        components.insert("system", self.components.system_instruction.clone());

        // Create context section combining various information
        let mut context_parts = Vec::new();
        if !self.context.is_empty() {
            context_parts.push(format!("Navigation Context:\n{}", self.context));
        }
        if !previous_actions.is_empty() {
            context_parts.push(previous_actions);
        }
        if !current_url.is_empty() {
            context_parts.push(current_url);
        }
        if !progress.is_empty() {
            context_parts.push(progress);
        }

        let combined_context = if !context_parts.is_empty() {
            context_parts.join("\n\n")
        } else {
            "No previous context available.".to_string()
        };

        components.insert("context", combined_context);
        components.insert("objective", format!("Objective:\n{}", self.objective));
        components.insert(
            "html",
            format!("Current Page HTML:\n```html\n{}\n```", truncated_html),
        );
        components.insert("elements", interactive_elements);
        components.insert("guidelines", format!("Guidelines:\n{}", guidelines));
        components.insert("action_format", action_format);

        // Add examples if not empty
        if !examples.is_empty() {
            components.insert("examples", examples);
        }

        // Create the standard prompt
        let prompt = create_standard_prompt(&components, &self.variation);
        write!(f, "{}", prompt)
    }
}

impl PromptTemplate for NavigationPromptTemplate {
    fn as_string(&self) -> String {
        self.to_string()
    }

    fn estimate_token_count(&self) -> usize {
        let html_token_count = self.html.len() / 4;
        let objective_token_count = self.objective.len() / 4;
        let context_token_count = self.context.len() / 4;
        let system_token_count = self.components.system_instruction.len() / 4;

        let elements_token_count = self.interactive_elements.len() * 50; // Rough estimate
        let actions_token_count = self.previous_actions.len() * 50; // Rough estimate

        let examples_token_count = self
            .components
            .examples
            .iter()
            .map(|ex| (ex.input.len() + ex.output.len()) / 4)
            .sum::<usize>();

        // Sum all token counts with some overhead for formatting
        html_token_count
            + objective_token_count
            + context_token_count
            + system_token_count
            + elements_token_count
            + actions_token_count
            + examples_token_count
            + 300
    }

    fn clone_template(&self) -> Box<dyn PromptTemplate> {
        Box::new(self.clone())
    }
}

/// Helper function to create a basic navigation prompt
pub fn create_basic_navigation_prompt(
    html: &str,
    objective: &str,
    interactive_elements: &[InteractiveElement],
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    NavigationPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_interactive_elements(interactive_elements.to_vec())
        .with_variation(variation)
        .build()
        .to_string()
}

/// Helper function to create a form filling navigation prompt
pub fn create_form_filling_prompt(
    html: &str,
    objective: &str,
    interactive_elements: &[InteractiveElement],
    form_data: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    NavigationPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_interactive_elements(interactive_elements.to_vec())
        .with_context(format!("Form data to enter:\n{}", form_data))
        .with_navigation_type(NavigationType::FormFilling)
        .with_variation(variation)
        .build()
        .to_string()
}

/// Helper function to create a checkout navigation prompt
pub fn create_checkout_prompt(
    html: &str,
    objective: &str,
    interactive_elements: &[InteractiveElement],
    checkout_info: &str,
    previous_actions: &[NavigationAction],
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    NavigationPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_interactive_elements(interactive_elements.to_vec())
        .with_context(format!("Checkout information:\n{}", checkout_info))
        .with_navigation_type(NavigationType::Checkout)
        .with_previous_actions(previous_actions.to_vec())
        .with_variation(variation)
        .build()
        .to_string()
}

/// Helper function to create a search and filter navigation prompt
pub fn create_search_filter_prompt(
    html: &str,
    objective: &str,
    interactive_elements: &[InteractiveElement],
    search_criteria: &str,
    variation: Option<TemplateVariation>,
) -> String {
    let variation = variation.unwrap_or(TemplateVariation::Claude);

    NavigationPromptTemplate::new()
        .with_html(html)
        .with_objective(objective)
        .with_interactive_elements(interactive_elements.to_vec())
        .with_context(format!("Search criteria:\n{}", search_criteria))
        .with_navigation_type(NavigationType::SearchAndFilter)
        .with_variation(variation)
        .build()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_interactive_element(index: usize) -> InteractiveElement {
        InteractiveElement {
            index,
            tag: "button".to_string(),
            text: Some("Click me".to_string()),
            input_type: None,
            placeholder: None,
            id: Some(format!("button-{}", index)),
            classes: vec!["btn".to_string(), "primary".to_string()],
            visible: true,
            position: ElementPosition {
                x: 100.0,
                y: 100.0,
                width: 50.0,
                height: 30.0,
            },
        }
    }

    fn create_test_action() -> NavigationAction {
        NavigationAction {
            action_type: "click".to_string(),
            target: "Element 1".to_string(),
            value: None,
            result: Some("Page loaded".to_string()),
            timestamp: Some("2023-01-01T12:00:00Z".to_string()),
        }
    }

    #[test]
    fn test_basic_navigation_prompt() {
        let html = "<div><button id='btn1'>Click me</button></div>";
        let objective = "Find and click the button";
        let elements = vec![create_test_interactive_element(0)];

        let prompt = create_basic_navigation_prompt(html, objective, &elements, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains("Element 0"));
        assert!(prompt.contains("ACTION:"));
    }

    #[test]
    fn test_form_filling_prompt() {
        let html = "<div><input type='text' id='name'></div>";
        let objective = "Fill out the form with the name 'John'";
        let elements = vec![create_test_interactive_element(0)];
        let form_data = "Name: John";

        let prompt = create_form_filling_prompt(html, objective, &elements, form_data, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains(form_data));
        assert!(prompt.contains("filling out forms"));
    }

    #[test]
    fn test_checkout_prompt() {
        let html = "<div><button id='checkout'>Checkout</button></div>";
        let objective = "Complete the checkout process";
        let elements = vec![create_test_interactive_element(0)];
        let checkout_info = "Address: 123 Main St\nPayment: Credit Card";
        let actions = vec![create_test_action()];

        let prompt =
            create_checkout_prompt(html, objective, &elements, checkout_info, &actions, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains(checkout_info));
        assert!(prompt.contains("completing a checkout process"));
        assert!(prompt.contains("Previous actions"));
    }

    #[test]
    fn test_search_filter_prompt() {
        let html = "<div><input type='text' id='search'></div>";
        let objective = "Search for products under $100";
        let elements = vec![create_test_interactive_element(0)];
        let search_criteria = "Price: < $100\nCategory: Electronics";

        let prompt = create_search_filter_prompt(html, objective, &elements, search_criteria, None);

        assert!(prompt.contains(html));
        assert!(prompt.contains(objective));
        assert!(prompt.contains(search_criteria));
        assert!(prompt.contains("search and filter"));
    }

    #[test]
    fn test_navigation_prompt_template_build() {
        let template = NavigationPromptTemplate::new()
            .with_html("<div>Test</div>")
            .with_objective("Find a button")
            .with_interactive_elements(vec![create_test_interactive_element(0)])
            .with_navigation_type(NavigationType::Standard)
            .with_previous_actions(vec![create_test_action()])
            .with_current_url("https://example.com")
            .with_progress(0.5)
            .build();

        let prompt = template.to_string();

        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Find a button"));
        assert!(prompt.contains("Element 0"));
        assert!(prompt.contains("Previous actions"));
        assert!(prompt.contains("https://example.com"));
        assert!(prompt.contains("50%"));
    }

    #[test]
    fn test_prompt_template_trait_implementation() {
        let template: Box<dyn PromptTemplate> = Box::new(
            NavigationPromptTemplate::new()
                .with_html("<div>Test</div>")
                .with_objective("Find a button")
                .build(),
        );

        let prompt = template.as_string();
        assert!(prompt.contains("<div>Test</div>"));
        assert!(prompt.contains("Find a button"));

        let token_count = template.estimate_token_count();
        assert!(token_count > 0);

        let cloned = template.clone_template();
        let cloned_prompt = cloned.as_string();
        assert_eq!(prompt, cloned_prompt);
    }
}
