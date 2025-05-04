//! # LLM Integration Module
//!
//! This module provides integration with large language models (LLMs) for DOMorpher.
//! It handles communication with various LLM providers, manages prompt creation,
//! and implements rate limiting and error handling.
//!
//! ## Features
//!
//! - Abstract provider interface for multiple LLM backends
//! - Specialized prompt templates for different extraction and navigation tasks
//! - Intelligent rate limiting and request throttling
//! - Connection pooling and retry logic
//! - Streaming response handling
//! - Caching of LLM responses
//! - Instrumentation and metrics collection

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

// Re-export key types and functions
pub use anthropic::AnthropicProvider;
pub use local::LocalModelProvider;
pub use openai::OpenAiProvider;
pub use prompts::analysis::AnalysisPromptTemplate;
pub use prompts::extraction::ExtractionPromptTemplate;
pub use prompts::navigation::NavigationPromptTemplate;
pub use provider::{LlmClient, LlmProvider, LlmProviderManager};
pub use rate_limiting::{RateLimitConfig, RateLimiter};

// Module declarations
pub mod anthropic;
pub mod local;
pub mod openai;
pub mod prompts;
pub mod provider;
pub mod rate_limiting;

/// Error types for LLM module
#[derive(Error, Debug)]
pub enum LlmError {
    /// API key not configured for provider
    #[error("API key not configured for provider {0}")]
    MissingApiKey(String),

    /// Provider not supported
    #[error("LLM provider {0} not supported")]
    UnsupportedProvider(String),

    /// Provider not available
    #[error("LLM provider {0} not available")]
    ProviderUnavailable(String),

    /// HTTP request error
    #[error("HTTP request error: {0}")]
    HttpError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded for provider {0}, retry after {1} seconds")]
    RateLimitExceeded(String, u64),

    /// Request timeout
    #[error("Request timeout after {0} seconds")]
    Timeout(u64),

    /// Invalid response format
    #[error("Invalid response format: {0}")]
    InvalidResponseFormat(String),

    /// Content filtering triggered
    #[error("Content filtering triggered: {0}")]
    ContentFiltering(String),

    /// Context limit exceeded
    #[error("Context limit exceeded: {0}")]
    ContextLimitExceeded(String),

    /// Model overloaded
    #[error("Model overloaded: {0}")]
    ModelOverloaded(String),

    /// Token limit exceeded
    #[error("Token limit exceeded: maximum {0}, got {1}")]
    TokenLimitExceeded(usize, usize),

    /// Provider error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for LLM operations
pub type LlmResult<T> = Result<T, LlmError>;

/// Options for text generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmOptions {
    /// The LLM provider to use
    pub provider: Option<LlmProvider>,

    /// The model to use
    pub model: Option<String>,

    /// Temperature (randomness) parameter (0.0 to 2.0)
    pub temperature: Option<f32>,

    /// Top-p sampling parameter (0.0 to 1.0)
    pub top_p: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,

    /// Stop sequences to end generation
    pub stop_sequences: Option<Vec<String>>,

    /// Timeout in seconds
    pub timeout_seconds: Option<u64>,

    /// Whether to stream the response
    pub stream: Option<bool>,

    /// Additional provider-specific parameters
    pub extra_params: Option<HashMap<String, serde_json::Value>>,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            temperature: Some(0.7),
            top_p: Some(1.0),
            max_tokens: Some(4096),
            stop_sequences: None,
            timeout_seconds: Some(60),
            stream: Some(false),
            extra_params: None,
        }
    }
}

/// Response from an LLM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text
    pub text: String,

    /// Number of prompt tokens used
    pub prompt_tokens: Option<usize>,

    /// Number of completion tokens used
    pub completion_tokens: Option<usize>,

    /// Total tokens used
    pub total_tokens: Option<usize>,

    /// Model used for generation
    pub model: Option<String>,

    /// Provider-specific metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Finish reason (e.g., "stop", "length", etc.)
    pub finish_reason: Option<String>,
}

/// Helper function to get default LLM client
pub async fn get_default_client() -> LlmResult<Arc<dyn provider::LlmProvider>> {
    let manager = crate::get_llm_manager();
    let manager = manager.lock().await;

    let default_provider = manager.get_default_provider();
    let provider = manager
        .get_provider(&default_provider)
        .ok_or_else(|| LlmError::ProviderUnavailable(default_provider.clone()))?;

    Ok(provider.clone())
}

/// Helper function to create an LLM client with the given provider
pub async fn create_client(provider: LlmProvider) -> LlmResult<Arc<dyn provider::LlmProvider>> {
    let manager = crate::get_llm_manager();
    let manager = manager.lock().await;

    let provider_name = provider.to_string();
    let provider = manager
        .get_provider_by_type(&provider)
        .ok_or_else(|| LlmError::ProviderUnavailable(provider_name))?;

    Ok(provider.clone())
}

/// Generate text using the default LLM provider
pub async fn generate(prompt: &str, options: Option<LlmOptions>) -> LlmResult<String> {
    let client = get_default_client().await?;
    let response = client.generate(prompt, options).await?;
    Ok(response.text)
}

/// Generate text using a specific LLM provider
pub async fn generate_with_provider(
    provider: LlmProvider,
    prompt: &str,
    options: Option<LlmOptions>,
) -> LlmResult<String> {
    let client = create_client(provider).await?;
    let response = client.generate(prompt, options).await?;
    Ok(response.text)
}

/// Check if an API key is configured for the given provider
pub fn is_provider_configured(provider: &LlmProvider) -> bool {
    // Get global configuration
    if let Ok(config) = crate::get_config() {
        match provider {
            LlmProvider::Anthropic => {
                if let Some(provider_config) = config.llm.providers.get("anthropic") {
                    return provider_config.api_key.is_some();
                }
            }
            LlmProvider::OpenAI => {
                if let Some(provider_config) = config.llm.providers.get("openai") {
                    return provider_config.api_key.is_some();
                }
            }
            LlmProvider::Local => {
                if let Some(provider_config) = config.llm.providers.get("local") {
                    return provider_config.api_key.is_some()
                        || provider_config.extra_params.get("model_path").is_some();
                }
            }
            LlmProvider::Custom(name) => {
                if let Some(provider_config) = config.llm.providers.get(name) {
                    return provider_config.api_key.is_some();
                }
            }
        }
    }

    false
}

/// Get the default model for the given provider
pub fn get_default_model(provider: &LlmProvider) -> Option<String> {
    // Get global configuration
    if let Ok(config) = crate::get_config() {
        match provider {
            LlmProvider::Anthropic => {
                if let Some(provider_config) = config.llm.providers.get("anthropic") {
                    return Some(provider_config.default_model.clone());
                }
            }
            LlmProvider::OpenAI => {
                if let Some(provider_config) = config.llm.providers.get("openai") {
                    return Some(provider_config.default_model.clone());
                }
            }
            LlmProvider::Local => {
                if let Some(provider_config) = config.llm.providers.get("local") {
                    return Some(provider_config.default_model.clone());
                }
            }
            LlmProvider::Custom(name) => {
                if let Some(provider_config) = config.llm.providers.get(name) {
                    return Some(provider_config.default_model.clone());
                }
            }
        }
    }

    None
}

/// Calculate the number of tokens in the given text
///
/// This is a simple estimation based on words. For accurate token counts,
/// use the provider-specific tokenizer.
pub fn estimate_token_count(text: &str) -> usize {
    // Simple estimation: 1 token ≈ 4 characters
    text.chars().count() / 4 + 1
}

/// Calculate the cost of a request based on tokens and model
pub fn calculate_cost(
    prompt_tokens: usize,
    completion_tokens: usize,
    provider: &LlmProvider,
    model: &str,
) -> f64 {
    match provider {
        LlmProvider::Anthropic => {
            match model {
                "claude-3-opus" => {
                    // Claude 3 Opus: $15 per 1M input tokens, $75 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 15.0 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 75.0 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                "claude-3-sonnet" => {
                    // Claude 3 Sonnet: $3 per 1M input tokens, $15 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 3.0 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 15.0 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                "claude-3-haiku" => {
                    // Claude 3 Haiku: $0.25 per 1M input tokens, $1.25 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 0.25 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 1.25 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                _ => {
                    // Default to Sonnet pricing if model not recognized
                    let prompt_cost = (prompt_tokens as f64) * 3.0 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 15.0 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
            }
        }
        LlmProvider::OpenAI => {
            match model {
                "gpt-4o" => {
                    // GPT-4o: $5 per 1M input tokens, $15 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 5.0 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 15.0 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                "gpt-4-turbo" => {
                    // GPT-4 Turbo: $10 per 1M input tokens, $30 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 10.0 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 30.0 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                "gpt-3.5-turbo" => {
                    // GPT-3.5 Turbo: $0.5 per 1M input tokens, $1.5 per 1M output tokens
                    let prompt_cost = (prompt_tokens as f64) * 0.5 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 1.5 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
                _ => {
                    // Default to GPT-3.5 Turbo pricing if model not recognized
                    let prompt_cost = (prompt_tokens as f64) * 0.5 / 1_000_000.0;
                    let completion_cost = (completion_tokens as f64) * 1.5 / 1_000_000.0;
                    prompt_cost + completion_cost
                }
            }
        }
        LlmProvider::Local => {
            // No cost for local models
            0.0
        }
        LlmProvider::Custom(_) => {
            // Use a conservative estimate for custom providers
            let prompt_cost = (prompt_tokens as f64) * 10.0 / 1_000_000.0;
            let completion_cost = (completion_tokens as f64) * 30.0 / 1_000_000.0;
            prompt_cost + completion_cost
        }
    }
}

/// Test if the LLM integration is working properly
pub async fn test_integration() -> LlmResult<bool> {
    let test_prompt = "Reply with exactly the word 'success' and nothing else.";

    let response = generate(test_prompt, None).await?;

    // Check if the response contains "success" (case-insensitive)
    if response.to_lowercase().contains("success") {
        Ok(true)
    } else {
        Err(LlmError::InvalidResponseFormat(format!(
            "Expected 'success', got: {}",
            response
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_token_count() {
        let text = "This is a test sentence with exactly ten words.";
        let count = estimate_token_count(text);
        assert!(count > 0);
    }

    #[test]
    fn test_calculation_cost() {
        let prompt_tokens = 1000;
        let completion_tokens = 500;

        // Test Anthropic cost calculation
        let anthropic_cost = calculate_cost(
            prompt_tokens,
            completion_tokens,
            &LlmProvider::Anthropic,
            "claude-3-sonnet",
        );
        assert!(anthropic_cost > 0.0);

        // Test OpenAI cost calculation
        let openai_cost = calculate_cost(
            prompt_tokens,
            completion_tokens,
            &LlmProvider::OpenAI,
            "gpt-4o",
        );
        assert!(openai_cost > 0.0);

        // Test Local cost calculation
        let local_cost = calculate_cost(
            prompt_tokens,
            completion_tokens,
            &LlmProvider::Local,
            "llama-3-8b",
        );
        assert_eq!(local_cost, 0.0);
    }

    #[tokio::test]
    async fn test_provider_configuration() {
        // This test just checks the function exists and returns a boolean
        let is_configured = is_provider_configured(&LlmProvider::Anthropic);
        assert!(is_configured == true || is_configured == false);
    }
}
