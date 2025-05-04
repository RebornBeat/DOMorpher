//! # LLM Provider Interface
//!
//! This module defines the interface for LLM providers used by DOMorpher.
//! It provides abstract traits and implementations for interacting with
//! different LLM backends, managing providers, and handling requests.
//!
//! ## Provider Traits
//!
//! The main trait is `LlmProvider`, which defines the interface that
//! all LLM providers must implement. This includes methods for:
//! - Generating text completions
//! - Managing rate limits
//! - Handling errors
//! - Testing connections
//!
//! ## Provider Manager
//!
//! The `LlmProviderManager` manages multiple LLM providers and provides
//! a unified interface for selecting and using them.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::sleep;

use super::{LlmError, LlmOptions, LlmResponse, LlmResult};

/// Available LLM providers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Anthropic Claude models
    Anthropic,

    /// OpenAI models (GPT series)
    OpenAI,

    /// Local models (e.g., llama.cpp, Phi models)
    Local,

    /// Custom provider
    Custom(String),
}

impl fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmProvider::Anthropic => write!(f, "anthropic"),
            LlmProvider::OpenAI => write!(f, "openai"),
            LlmProvider::Local => write!(f, "local"),
            LlmProvider::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

impl LlmProvider {
    /// Parse a provider from a string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "anthropic" => Some(Self::Anthropic),
            "claude" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAI),
            "gpt" => Some(Self::OpenAI),
            "local" => Some(Self::Local),
            _ => {
                if s.starts_with("custom:") {
                    let name = s.splitn(2, ':').nth(1)?.to_string();
                    Some(Self::Custom(name))
                } else {
                    None
                }
            }
        }
    }

    /// Get the provider name
    pub fn name(&self) -> String {
        match self {
            Self::Anthropic => "anthropic".to_string(),
            Self::OpenAI => "openai".to_string(),
            Self::Local => "local".to_string(),
            Self::Custom(name) => format!("custom:{}", name),
        }
    }

    /// Check if the provider is Anthropic
    pub fn is_anthropic(&self) -> bool {
        matches!(self, Self::Anthropic)
    }

    /// Check if the provider is OpenAI
    pub fn is_openai(&self) -> bool {
        matches!(self, Self::OpenAI)
    }

    /// Check if the provider is a local model
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local)
    }

    /// Check if the provider is a custom provider
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Get the default model for this provider
    pub fn default_model(&self) -> &'static str {
        match self {
            Self::Anthropic => "claude-3-sonnet",
            Self::OpenAI => "gpt-4o",
            Self::Local => "llama-3-8b",
            Self::Custom(_) => "default",
        }
    }
}

/// Trait for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate text using the provider
    async fn generate(&self, prompt: &str, options: Option<LlmOptions>) -> LlmResult<LlmResponse>;

    /// Generate text with streaming response
    async fn generate_streaming(
        &self,
        prompt: &str,
        options: Option<LlmOptions>,
        callback: impl FnMut(String) -> Result<(), Box<dyn std::error::Error>> + Send + 'static,
    ) -> LlmResult<LlmResponse> {
        // Default implementation falls back to non-streaming
        debug!("Streaming not implemented for this provider, falling back to standard generation");
        self.generate(prompt, options).await
    }

    /// Get the provider type
    fn provider_type(&self) -> LlmProvider;

    /// Get the default model
    fn default_model(&self) -> String;

    /// Get the number of tokens in a text
    fn count_tokens(&self, text: &str) -> usize {
        // Default implementation uses character-based estimation
        // Providers should override this with more accurate counting
        (text.len() as f64 / 4.0).ceil() as usize
    }

    /// Test the connection to the provider
    async fn test_connection(&self) -> LlmResult<()> {
        // Default test: generate a simple response
        let test_prompt = "Reply with the word 'success' only.";
        let options = LlmOptions {
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        };

        let response = self.generate(test_prompt, Some(options)).await?;

        // Check if response contains "success"
        if response.text.to_lowercase().contains("success") {
            Ok(())
        } else {
            Err(LlmError::ProviderError(format!(
                "Test response does not contain 'success': {}",
                response.text
            )))
        }
    }

    /// Get the provider name
    fn name(&self) -> String {
        self.provider_type().name()
    }

    /// Check if the provider supports streaming
    fn supports_streaming(&self) -> bool {
        false // Default implementation
    }

    /// Get the API endpoint
    fn get_api_endpoint(&self) -> String;

    /// Get the API key
    fn get_api_key(&self) -> Option<String>;

    /// Get the provider's capabilities
    fn capabilities(&self) -> ProviderCapabilities {
        // Default implementation
        ProviderCapabilities {
            streaming: false,
            function_calling: false,
            vision: false,
            embeddings: false,
            max_tokens: 8192,
            supports_system_messages: true,
        }
    }
}

/// Wrapper for accessing LLM providers
pub struct LlmClient {
    provider: Arc<dyn LlmProvider>,
}

impl LlmClient {
    /// Create a new LLM client with the given provider
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self { provider }
    }

    /// Generate text using the provider
    pub async fn generate(
        &self,
        prompt: &str,
        options: Option<LlmOptions>,
    ) -> LlmResult<LlmResponse> {
        self.provider.generate(prompt, options).await
    }

    /// Generate text with streaming response
    pub async fn generate_streaming(
        &self,
        prompt: &str,
        options: Option<LlmOptions>,
        callback: impl FnMut(String) -> Result<(), Box<dyn std::error::Error>> + Send + 'static,
    ) -> LlmResult<LlmResponse> {
        self.provider
            .generate_streaming(prompt, options, callback)
            .await
    }

    /// Get the provider type
    pub fn provider_type(&self) -> LlmProvider {
        self.provider.provider_type()
    }

    /// Get the default model
    pub fn default_model(&self) -> String {
        self.provider.default_model()
    }

    /// Get the provider name
    pub fn name(&self) -> String {
        self.provider.name()
    }

    /// Check if the provider supports streaming
    pub fn supports_streaming(&self) -> bool {
        self.provider.supports_streaming()
    }

    /// Test the connection to the provider
    pub async fn test_connection(&self) -> LlmResult<()> {
        self.provider.test_connection().await
    }

    /// Get the provider's capabilities
    pub fn capabilities(&self) -> ProviderCapabilities {
        self.provider.capabilities()
    }

    /// Count tokens in a text
    pub fn count_tokens(&self, text: &str) -> usize {
        self.provider.count_tokens(text)
    }
}

/// Manager for multiple LLM providers
pub struct LlmProviderManager {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    default_provider: String,
}

impl LlmProviderManager {
    /// Create a new LLM provider manager
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: "anthropic".to_string(),
        }
    }

    /// Register a provider
    pub fn register_provider(&mut self, name: String, provider: Box<dyn LlmProvider>) {
        info!("Registering LLM provider: {}", name);
        self.providers.insert(name, Arc::new(provider));
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers.get(name).cloned()
    }

    /// Get a provider by type
    pub fn get_provider_by_type(
        &self,
        provider_type: &LlmProvider,
    ) -> Option<Arc<dyn LlmProvider>> {
        match provider_type {
            LlmProvider::Anthropic => self.get_provider("anthropic"),
            LlmProvider::OpenAI => self.get_provider("openai"),
            LlmProvider::Local => self.get_provider("local"),
            LlmProvider::Custom(name) => {
                let custom_name = format!("custom:{}", name);
                self.get_provider(&custom_name)
            }
        }
    }

    /// Get the default provider
    pub fn get_default_provider(&self) -> String {
        self.default_provider.clone()
    }

    /// Set the default provider
    pub fn set_default_provider(&mut self, name: String) {
        debug!("Setting default LLM provider to: {}", name);
        self.default_provider = name;
    }

    /// Check if a provider exists
    pub fn has_provider(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }

    /// Get all provider names
    pub fn get_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Create a client for the default provider
    pub fn get_default_client(&self) -> Option<LlmClient> {
        self.get_provider(&self.default_provider)
            .map(|provider| LlmClient::new(provider))
    }

    /// Create a client for a specific provider
    pub fn create_client(&self, provider_type: &LlmProvider) -> Option<LlmClient> {
        self.get_provider_by_type(provider_type)
            .map(|provider| LlmClient::new(provider))
    }

    /// Get all available provider types
    pub fn available_provider_types(&self) -> Vec<LlmProvider> {
        let mut result = Vec::new();

        for name in self.providers.keys() {
            if name == "anthropic" {
                result.push(LlmProvider::Anthropic);
            } else if name == "openai" {
                result.push(LlmProvider::OpenAI);
            } else if name == "local" {
                result.push(LlmProvider::Local);
            } else if name.starts_with("custom:") {
                let custom_name = name.splitn(2, ':').nth(1).unwrap_or(name).to_string();
                result.push(LlmProvider::Custom(custom_name));
            }
        }

        result
    }

    /// Test all available providers
    pub async fn test_all_providers(&self) -> HashMap<String, Result<(), LlmError>> {
        let mut results = HashMap::new();

        for (name, provider) in &self.providers {
            let result = provider.test_connection().await;
            results.insert(name.clone(), result);
        }

        results
    }
}

impl Default for LlmProviderManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Provider capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Supports streaming responses
    pub streaming: bool,

    /// Supports function calling
    pub function_calling: bool,

    /// Supports vision/multimodal inputs
    pub vision: bool,

    /// Supports embedding generation
    pub embeddings: bool,

    /// Maximum context length in tokens
    pub max_tokens: usize,

    /// Supports system messages
    pub supports_system_messages: bool,
}

/// Request statistics for a provider
#[derive(Debug, Clone, Default)]
pub struct ProviderStats {
    /// Total number of requests
    pub total_requests: usize,

    /// Number of successful requests
    pub successful_requests: usize,

    /// Number of failed requests
    pub failed_requests: usize,

    /// Total tokens used (prompt + completion)
    pub total_tokens: usize,

    /// Total tokens in prompts
    pub prompt_tokens: usize,

    /// Total tokens in completions
    pub completion_tokens: usize,

    /// Total cost in USD
    pub total_cost: f64,

    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// Number of rate limit errors
    pub rate_limit_errors: usize,

    /// Number of timeouts
    pub timeouts: usize,

    /// Number of other errors
    pub other_errors: usize,
}

impl ProviderStats {
    /// Create new provider stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request
    pub fn record_success(
        &mut self,
        prompt_tokens: usize,
        completion_tokens: usize,
        cost: f64,
        response_time_ms: f64,
    ) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens += prompt_tokens + completion_tokens;
        self.total_cost += cost;

        // Update average response time
        let total_response_time =
            self.avg_response_time_ms * (self.successful_requests as f64 - 1.0);
        self.avg_response_time_ms =
            (total_response_time + response_time_ms) / (self.successful_requests as f64);
    }

    /// Record a failed request
    pub fn record_failure(&mut self, error: &LlmError) {
        self.total_requests += 1;
        self.failed_requests += 1;

        match error {
            LlmError::RateLimitExceeded(_, _) => {
                self.rate_limit_errors += 1;
            }
            LlmError::Timeout(_) => {
                self.timeouts += 1;
            }
            _ => {
                self.other_errors += 1;
            }
        }
    }

    /// Get the success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.successful_requests as f64) / (self.total_requests as f64)
        }
    }

    /// Get the average tokens per request
    pub fn avg_tokens_per_request(&self) -> f64 {
        if self.successful_requests == 0 {
            0.0
        } else {
            (self.total_tokens as f64) / (self.successful_requests as f64)
        }
    }

    /// Get the average cost per request
    pub fn avg_cost_per_request(&self) -> f64 {
        if self.successful_requests == 0 {
            0.0
        } else {
            self.total_cost / (self.successful_requests as f64)
        }
    }

    /// Merge with another stats object
    pub fn merge(&mut self, other: &ProviderStats) {
        self.total_requests += other.total_requests;
        self.successful_requests += other.successful_requests;
        self.failed_requests += other.failed_requests;
        self.total_tokens += other.total_tokens;
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_cost += other.total_cost;
        self.rate_limit_errors += other.rate_limit_errors;
        self.timeouts += other.timeouts;
        self.other_errors += other.other_errors;

        // Update average response time
        if self.successful_requests > 0 && other.successful_requests > 0 {
            let total_time = (self.avg_response_time_ms * (self.successful_requests as f64))
                + (other.avg_response_time_ms * (other.successful_requests as f64));
            let total_successful = self.successful_requests + other.successful_requests;
            self.avg_response_time_ms = total_time / (total_successful as f64);
        } else if other.successful_requests > 0 {
            self.avg_response_time_ms = other.avg_response_time_ms;
        }
    }

    /// Reset the stats
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Helper functions for providers
pub mod utils {
    use super::*;

    /// Execute an LLM request with retry logic
    pub async fn execute_with_retry<F, Fut, T>(
        operation: F,
        max_retries: usize,
        initial_delay: Duration,
        max_delay: Duration,
        provider_name: &str,
    ) -> LlmResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = LlmResult<T>>,
    {
        let mut retries = 0;
        let mut delay = initial_delay;

        loop {
            let result = operation().await;

            match &result {
                Ok(_) => return result,
                Err(e) => {
                    if retries >= max_retries {
                        error!(
                            "Maximum retries ({}) reached for {}: {}",
                            max_retries, provider_name, e
                        );
                        return result;
                    }

                    // Only retry on certain errors
                    let should_retry = match e {
                        LlmError::RateLimitExceeded(_, _) => true,
                        LlmError::Timeout(_) => true,
                        LlmError::HttpError(_) => true,
                        LlmError::ModelOverloaded(_) => true,
                        _ => false,
                    };

                    if !should_retry {
                        error!("Non-retriable error for {}: {}", provider_name, e);
                        return result;
                    }

                    // Exponential backoff with jitter
                    let jitter = rand::random::<f64>() * 0.1 * delay.as_millis() as f64;
                    let delay_with_jitter =
                        Duration::from_millis((delay.as_millis() as f64 + jitter) as u64);

                    warn!(
                        "Retrying {} after error (retry {}/{}): {}. Waiting for {:?}",
                        provider_name,
                        retries + 1,
                        max_retries,
                        e,
                        delay_with_jitter
                    );

                    sleep(delay_with_jitter).await;

                    // Increase delay for next retry, up to max_delay
                    delay = std::cmp::min(delay * 2, max_delay);
                    retries += 1;
                }
            }
        }
    }

    /// Helper function to replace empty fields with defaults in LlmOptions
    pub fn apply_option_defaults(options: Option<LlmOptions>, defaults: &LlmOptions) -> LlmOptions {
        let mut result = options.unwrap_or_else(|| defaults.clone());

        if result.temperature.is_none() {
            result.temperature = defaults.temperature;
        }

        if result.top_p.is_none() {
            result.top_p = defaults.top_p;
        }

        if result.max_tokens.is_none() {
            result.max_tokens = defaults.max_tokens;
        }

        if result.stop_sequences.is_none() {
            result.stop_sequences = defaults.stop_sequences.clone();
        }

        if result.timeout_seconds.is_none() {
            result.timeout_seconds = defaults.timeout_seconds;
        }

        if result.stream.is_none() {
            result.stream = defaults.stream;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_provider_enum() {
        assert_eq!(LlmProvider::Anthropic.to_string(), "anthropic");
        assert_eq!(LlmProvider::OpenAI.to_string(), "openai");
        assert_eq!(LlmProvider::Local.to_string(), "local");
        assert_eq!(
            LlmProvider::Custom("test".to_string()).to_string(),
            "custom:test"
        );

        assert_eq!(
            LlmProvider::from_str("anthropic"),
            Some(LlmProvider::Anthropic)
        );
        assert_eq!(
            LlmProvider::from_str("claude"),
            Some(LlmProvider::Anthropic)
        );
        assert_eq!(LlmProvider::from_str("openai"), Some(LlmProvider::OpenAI));
        assert_eq!(LlmProvider::from_str("gpt"), Some(LlmProvider::OpenAI));
        assert_eq!(LlmProvider::from_str("local"), Some(LlmProvider::Local));
        assert_eq!(
            LlmProvider::from_str("custom:test"),
            Some(LlmProvider::Custom("test".to_string()))
        );
        assert_eq!(LlmProvider::from_str("unknown"), None);
    }

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities {
            streaming: true,
            function_calling: false,
            vision: true,
            embeddings: false,
            max_tokens: 16384,
            supports_system_messages: true,
        };

        assert!(caps.streaming);
        assert!(!caps.function_calling);
        assert!(caps.vision);
        assert!(!caps.embeddings);
        assert_eq!(caps.max_tokens, 16384);
        assert!(caps.supports_system_messages);
    }

    #[test]
    fn test_provider_stats() {
        let mut stats = ProviderStats::new();

        // Test initial state
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.success_rate(), 0.0);

        // Record a successful request
        stats.record_success(100, 50, 0.02, 1500.0);
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.prompt_tokens, 100);
        assert_eq!(stats.completion_tokens, 50);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.total_cost, 0.02);
        assert_eq!(stats.avg_response_time_ms, 1500.0);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.avg_tokens_per_request(), 150.0);
        assert_eq!(stats.avg_cost_per_request(), 0.02);

        // Record a failed request
        stats.record_failure(&LlmError::Timeout(30));
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.timeouts, 1);
        assert_eq!(stats.success_rate(), 0.5);

        // Record another successful request
        stats.record_success(200, 100, 0.04, 2500.0);
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.prompt_tokens, 300);
        assert_eq!(stats.completion_tokens, 150);
        assert_eq!(stats.total_tokens, 450);
        assert_eq!(stats.total_cost, 0.06);
        // Average response time: (1500 + 2500) / 2 = 2000
        assert_eq!(stats.avg_response_time_ms, 2000.0);
        assert_eq!(stats.success_rate(), 2.0 / 3.0);
        assert_eq!(stats.avg_tokens_per_request(), 225.0);
        assert_eq!(stats.avg_cost_per_request(), 0.03);

        // Reset stats
        stats.reset();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
    }

    #[test]
    fn test_utils_apply_option_defaults() {
        // Create default options
        let defaults = LlmOptions {
            provider: Some(LlmProvider::Anthropic),
            model: Some("claude-3-sonnet".to_string()),
            temperature: Some(0.7),
            top_p: Some(1.0),
            max_tokens: Some(2000),
            stop_sequences: Some(vec!["STOP".to_string()]),
            timeout_seconds: Some(60),
            stream: Some(false),
            extra_params: None,
        };

        // Test with empty options
        let options = None;
        let result = utils::apply_option_defaults(options, &defaults);
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(2000));
        assert_eq!(result.model, Some("claude-3-sonnet".to_string()));

        // Test with partial options
        let options = Some(LlmOptions {
            temperature: Some(0.5),
            model: Some("different-model".to_string()),
            ..Default::default()
        });
        let result = utils::apply_option_defaults(options, &defaults);
        assert_eq!(result.temperature, Some(0.5)); // From options
        assert_eq!(result.max_tokens, Some(2000)); // From defaults
        assert_eq!(result.model, Some("different-model".to_string())); // From options
    }
}
