//! # Anthropic Claude Integration
//!
//! This module provides integration with Anthropic's Claude models.
//! It implements the `LlmProvider` trait to enable DOMorpher to use
//! Claude models for text generation and analysis.
//!
//! ## Features
//!
//! - Support for Claude 3 Opus, Sonnet, and Haiku models
//! - Streaming response support
//! - Token counting for Claude models
//! - Rate limiting and error handling
//! - Context window management

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::provider::{LlmProvider, ProviderCapabilities, utils};
use super::rate_limiting::{RateLimit, RateLimiter};
use super::{LlmError, LlmOptions, LlmResponse, LlmResult};

/// Anthropic Claude provider
pub struct AnthropicProvider {
    /// API key for Anthropic
    api_key: String,

    /// Default model to use
    default_model: String,

    /// API endpoint
    api_endpoint: String,

    /// Request timeout in seconds
    timeout: u64,

    /// Rate limiter
    rate_limiter: Arc<Mutex<RateLimiter>>,

    /// HTTP client
    client: reqwest::Client,
}

/// Anthropic API response
#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    /// Response ID
    id: String,

    /// Model used
    model: String,

    /// Response type
    #[serde(rename = "type")]
    response_type: String,

    /// Response role
    role: String,

    /// Response content
    content: Vec<AnthropicContent>,

    /// Usage information
    usage: AnthropicUsage,
}

/// Anthropic content block
#[derive(Debug, Clone, Deserialize)]
struct AnthropicContent {
    /// Content type
    #[serde(rename = "type")]
    content_type: String,

    /// Text content
    text: Option<String>,
}

/// Anthropic usage statistics
#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    /// Input tokens
    input_tokens: usize,

    /// Output tokens
    output_tokens: usize,
}

/// Anthropic streaming response
#[derive(Debug, Clone, Deserialize)]
struct AnthropicStreamingResponse {
    /// Response type
    #[serde(rename = "type")]
    response_type: String,

    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<AnthropicDelta>,

    /// Usage information for completion event
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<AnthropicUsage>,
}

/// Anthropic streaming delta
#[derive(Debug, Clone, Deserialize)]
struct AnthropicDelta {
    /// Content type
    #[serde(rename = "type")]
    content_type: String,

    /// Text content
    text: Option<String>,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: &str, default_model: &str, timeout: u64) -> Self {
        // Create HTTP client with appropriate timeout
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout + 5)) // Add 5 seconds buffer
            .build()
            .unwrap_or_else(|_| {
                warn!("Failed to build custom HTTP client for Anthropic provider, using default");
                reqwest::Client::new()
            });

        // Rate limits for Anthropic API
        // See: https://docs.anthropic.com/claude/reference/rate-limits
        let rate_limits = vec![
            // Base rate limit: 50 RPM / 5 per second
            RateLimit::new("base", 50, Duration::from_secs(60)),
            // Spike rate limit: 5 per second
            RateLimit::new("spike", 5, Duration::from_secs(1)),
        ];

        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new("anthropic", rate_limits)));

        Self {
            api_key: api_key.to_string(),
            default_model: default_model.to_string(),
            api_endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            timeout,
            rate_limiter,
            client,
        }
    }

    /// Build a request to the Anthropic API
    async fn build_request(
        &self,
        prompt: &str,
        options: Option<LlmOptions>,
    ) -> LlmResult<serde_json::Value> {
        // Apply defaults
        let defaults = LlmOptions {
            model: Some(self.default_model.clone()),
            temperature: Some(0.7),
            top_p: Some(1.0),
            max_tokens: Some(4096),
            stop_sequences: None,
            timeout_seconds: Some(self.timeout),
            stream: Some(false),
            ..Default::default()
        };

        let options = utils::apply_option_defaults(options, &defaults);

        // Check if the model is specified
        let model = options.model.unwrap_or_else(|| self.default_model.clone());

        // Check if the model is a Claude model
        if !model.contains("claude") {
            return Err(LlmError::ProviderError(format!(
                "Invalid model for Anthropic provider: {}",
                model
            )));
        }

        // Parse the prompt into system prompt and user message
        let (system_prompt, user_message) = self.parse_prompt(prompt);

        // Build messages array
        let mut messages = Vec::new();

        // Add system message if present
        if !system_prompt.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": user_message
        }));

        // Build request
        let mut request = json!({
            "model": model,
            "messages": messages,
            "max_tokens": options.max_tokens.unwrap_or(4096),
        });

        // Add optional parameters
        if let Some(temperature) = options.temperature {
            request["temperature"] = json!(temperature);
        }

        if let Some(top_p) = options.top_p {
            request["top_p"] = json!(top_p);
        }

        if let Some(stop_sequences) = &options.stop_sequences {
            request["stop_sequences"] = json!(stop_sequences);
        }

        if let Some(stream) = options.stream {
            request["stream"] = json!(stream);
        }

        // Add extra parameters
        if let Some(extra_params) = &options.extra_params {
            for (key, value) in extra_params {
                request[key] = value.clone();
            }
        }

        Ok(request)
    }

    /// Parse the prompt into system prompt and user message
    fn parse_prompt(&self, prompt: &str) -> (String, String) {
        // Look for a system prompt marker
        if let Some(system_end) = prompt.find("\n\nHuman:") {
            let system_prompt = prompt[..system_end].trim();
            let user_message = prompt[system_end + 8..].trim();
            return (system_prompt.to_string(), user_message.to_string());
        } else if let Some(system_end) = prompt.find("\n\nUser:") {
            let system_prompt = prompt[..system_end].trim();
            let user_message = prompt[system_end + 7..].trim();
            return (system_prompt.to_string(), user_message.to_string());
        } else if prompt.starts_with("System:") {
            if let Some(system_end) = prompt.find("\n\n") {
                let system_prompt = prompt[7..system_end].trim();
                let user_message = prompt[system_end + 2..].trim();
                return (system_prompt.to_string(), user_message.to_string());
            }
        }

        // No system prompt found, use the entire prompt as user message
        ("".to_string(), prompt.to_string())
    }

    /// Count tokens using Anthropic's tokenizer
    fn count_tokens_anthropic(&self, text: &str) -> usize {
        // This is a simplified approximation as the actual Anthropic tokenizer is not public
        // In a production environment, you'd want to use the Anthropic Tokenizer API or a matching tokenizer

        // Crude approximation: 1 token is roughly 4 characters for English text
        // But add extra tokens for whitespace and special characters
        let char_count = text.chars().count();
        let whitespace_count = text.chars().filter(|c| c.is_whitespace()).count();
        let special_char_count = text
            .chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count();

        let token_estimate = (char_count - whitespace_count - special_char_count) / 4
            + whitespace_count
            + special_char_count * 2;

        // Add a small buffer for safety
        (token_estimate as f64 * 1.1) as usize
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn generate(&self, prompt: &str, options: Option<LlmOptions>) -> LlmResult<LlmResponse> {
        // Acquire rate limit permit
        let mut rate_limiter = self.rate_limiter.lock().await;
        let permit = rate_limiter.acquire().await?;
        drop(rate_limiter);

        let start_time = Instant::now();
        trace!(
            "Generating text with Anthropic Claude, prompt length: {}",
            prompt.len()
        );

        // Build request
        let request = self.build_request(prompt, options.clone()).await?;

        let completion_result = utils::execute_with_retry(
            || async {
                // Check if streaming is requested
                let is_streaming = request["stream"].as_bool().unwrap_or(false);
                if is_streaming {
                    return Err(LlmError::ProviderError(
                        "Streaming requested but using non-streaming API endpoint. Use generate_streaming instead.".to_string()
                    ));
                }

                // Send request to Anthropic API
                let response = self.client
                    .post(&self.api_endpoint)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        LlmError::HttpError(format!("HTTP request to Anthropic API failed: {}", e))
                    })?;

                // Check for rate limit errors
                if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    // Get retry-after header
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(60);

                    return Err(LlmError::RateLimitExceeded("anthropic".to_string(), retry_after));
                }

                // Check for other errors
                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Failed to get error text".to_string());
                    return Err(LlmError::ProviderError(format!(
                        "Anthropic API returned error {}: {}",
                        response.status(),
                        error_text
                    )));
                }

                // Parse response
                let anthropic_response: AnthropicResponse = response
                    .json()
                    .await
                    .map_err(|e| {
                        LlmError::InvalidResponseFormat(format!("Failed to parse Anthropic response: {}", e))
                    })?;

                // Extract text from content
                let content_text = anthropic_response.content
                    .iter()
                    .filter_map(|content| {
                        if content.content_type == "text" {
                            content.text.clone()
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<String>>()
                    .join("");

                // Create LLM response
                let response = LlmResponse {
                    text: content_text,
                    prompt_tokens: Some(anthropic_response.usage.input_tokens),
                    completion_tokens: Some(anthropic_response.usage.output_tokens),
                    total_tokens: Some(anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens),
                    model: Some(anthropic_response.model),
                    metadata: Some(HashMap::from([
                        ("id".to_string(), json!(anthropic_response.id)),
                        ("role".to_string(), json!(anthropic_response.role)),
                    ])),
                    finish_reason: None,
                };

                Ok(response)
            },
            3,
            Duration::from_secs(1),
            Duration::from_secs(30),
            "anthropic"
        ).await;

        // Record metrics
        let elapsed = start_time.elapsed();
        debug!(
            "Anthropic Claude request completed in {:?}, prompt tokens: {}, completion tokens: {}",
            elapsed,
            completion_result
                .as_ref()
                .ok()
                .and_then(|r| r.prompt_tokens)
                .unwrap_or(0),
            completion_result
                .as_ref()
                .ok()
                .and_then(|r| r.completion_tokens)
                .unwrap_or(0)
        );

        // Release rate limit permit
        let mut rate_limiter = self.rate_limiter.lock().await;
        rate_limiter.release(permit);

        completion_result
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        options: Option<LlmOptions>,
        mut callback: impl FnMut(String) -> Result<(), Box<dyn std::error::Error>> + Send + 'static,
    ) -> LlmResult<LlmResponse> {
        // Acquire rate limit permit
        let mut rate_limiter = self.rate_limiter.lock().await;
        let permit = rate_limiter.acquire().await?;
        drop(rate_limiter);

        let start_time = Instant::now();
        trace!(
            "Generating streaming text with Anthropic Claude, prompt length: {}",
            prompt.len()
        );

        // Clone options and force streaming
        let mut streaming_options = options.clone().unwrap_or_default();
        streaming_options.stream = Some(true);

        // Build request
        let request = self.build_request(prompt, Some(streaming_options)).await?;

        let completion_result = utils::execute_with_retry(
            || async {
                // Send request to Anthropic API
                let response = self
                    .client
                    .post(&self.api_endpoint)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        LlmError::HttpError(format!("HTTP request to Anthropic API failed: {}", e))
                    })?;

                // Check for rate limit errors
                if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    // Get retry-after header
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(60);

                    return Err(LlmError::RateLimitExceeded(
                        "anthropic".to_string(),
                        retry_after,
                    ));
                }

                // Check for other errors
                if !response.status().is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Failed to get error text".to_string());
                    return Err(LlmError::ProviderError(format!(
                        "Anthropic API returned error {}: {}",
                        response.status(),
                        error_text
                    )));
                }

                // Process streaming response
                let mut stream = response.bytes_stream();
                let mut full_text = String::new();
                let mut input_tokens = 0;
                let mut output_tokens = 0;
                let mut model = String::new();

                use futures::stream::StreamExt;
                while let Some(chunk_result) = stream.next().await {
                    let chunk = chunk_result.map_err(|e| {
                        LlmError::HttpError(format!("Error reading stream chunk: {}", e))
                    })?;

                    // Each chunk has format "data: {json}\n\n"
                    let chunk_str = String::from_utf8_lossy(&chunk);

                    for line in chunk_str.split('\n') {
                        if line.starts_with("data: ") {
                            let json_str = &line[6..]; // Skip "data: "

                            // Skip empty messages
                            if json_str.trim() == "[DONE]" {
                                continue;
                            }

                            // Parse the JSON
                            let streaming_response: AnthropicStreamingResponse =
                                serde_json::from_str(json_str).map_err(|e| {
                                    LlmError::InvalidResponseFormat(format!(
                                        "Failed to parse streaming response: {}",
                                        e
                                    ))
                                })?;

                            // Process the response based on type
                            match streaming_response.response_type.as_str() {
                                "message_start" => {
                                    // Message start, nothing to do yet
                                }
                                "content_block_start" => {
                                    // Content block start, nothing to do yet
                                }
                                "content_block_delta" => {
                                    // Content delta
                                    if let Some(delta) = streaming_response.delta {
                                        if delta.content_type == "text" {
                                            if let Some(text) = delta.text {
                                                full_text.push_str(&text);
                                                callback(text.clone()).map_err(|e| {
                                                    LlmError::Other(format!(
                                                        "Callback error: {}",
                                                        e
                                                    ))
                                                })?;
                                            }
                                        }
                                    }
                                }
                                "content_block_stop" => {
                                    // Content block stop, nothing to do
                                }
                                "message_delta" => {
                                    // Message delta, usually contains model info
                                    if let Ok(model_info) = serde_json::from_str::<Value>(json_str)
                                    {
                                        if let Some(model_str) =
                                            model_info["delta"]["model"].as_str()
                                        {
                                            model = model_str.to_string();
                                        }
                                    }
                                }
                                "message_stop" => {
                                    // Message stop, contains final usage info
                                    if let Some(usage) = streaming_response.usage {
                                        input_tokens = usage.input_tokens;
                                        output_tokens = usage.output_tokens;
                                    }
                                }
                                _ => {
                                    // Unknown event
                                    debug!(
                                        "Unknown streaming event type: {}",
                                        streaming_response.response_type
                                    );
                                }
                            }
                        }
                    }
                }

                // Create LLM response
                let response = LlmResponse {
                    text: full_text,
                    prompt_tokens: Some(input_tokens),
                    completion_tokens: Some(output_tokens),
                    total_tokens: Some(input_tokens + output_tokens),
                    model: Some(model),
                    metadata: Some(HashMap::new()),
                    finish_reason: None,
                };

                Ok(response)
            },
            3,
            Duration::from_secs(1),
            Duration::from_secs(30),
            "anthropic",
        )
        .await;

        // Record metrics
        let elapsed = start_time.elapsed();
        debug!(
            "Anthropic Claude streaming request completed in {:?}, prompt tokens: {}, completion tokens: {}",
            elapsed,
            completion_result
                .as_ref()
                .ok()
                .and_then(|r| r.prompt_tokens)
                .unwrap_or(0),
            completion_result
                .as_ref()
                .ok()
                .and_then(|r| r.completion_tokens)
                .unwrap_or(0)
        );

        // Release rate limit permit
        let mut rate_limiter = self.rate_limiter.lock().await;
        rate_limiter.release(permit);

        completion_result
    }

    fn provider_type(&self) -> super::provider::LlmProvider {
        super::provider::LlmProvider::Anthropic
    }

    fn default_model(&self) -> String {
        self.default_model.clone()
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.count_tokens_anthropic(text)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_api_endpoint(&self) -> String {
        self.api_endpoint.clone()
    }

    fn get_api_key(&self) -> Option<String> {
        Some(self.api_key.clone())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Capabilities for Claude models
        if self.default_model.contains("claude-3-opus") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                max_tokens: 200_000, // Claude 3 Opus context window
                supports_system_messages: true,
            }
        } else if self.default_model.contains("claude-3-sonnet") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                max_tokens: 200_000, // Claude 3 Sonnet context window
                supports_system_messages: true,
            }
        } else if self.default_model.contains("claude-3-haiku") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                max_tokens: 200_000, // Claude 3 Haiku context window
                supports_system_messages: true,
            }
        } else {
            // Default capabilities for other Claude models
            ProviderCapabilities {
                streaming: true,
                function_calling: false,
                vision: false,
                embeddings: false,
                max_tokens: 100_000,
                supports_system_messages: true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prompt() {
        let provider = AnthropicProvider::new("fake-key", "claude-3-sonnet", 60);

        // Test system + user format with "Human:" marker
        let prompt = "You are an AI assistant. Be concise and helpful.\n\nHuman: What is the capital of France?";
        let (system, user) = provider.parse_prompt(prompt);
        assert_eq!(system, "You are an AI assistant. Be concise and helpful.");
        assert_eq!(user, "What is the capital of France?");

        // Test system + user format with "User:" marker
        let prompt = "You are an AI assistant. Be concise and helpful.\n\nUser: What is the capital of France?";
        let (system, user) = provider.parse_prompt(prompt);
        assert_eq!(system, "You are an AI assistant. Be concise and helpful.");
        assert_eq!(user, "What is the capital of France?");

        // Test "System:" prefix
        let prompt = "System: You are an AI assistant. Be concise and helpful.\n\nWhat is the capital of France?";
        let (system, user) = provider.parse_prompt(prompt);
        assert_eq!(system, "You are an AI assistant. Be concise and helpful.");
        assert_eq!(user, "What is the capital of France?");

        // Test no system prompt
        let prompt = "What is the capital of France?";
        let (system, user) = provider.parse_prompt(prompt);
        assert_eq!(system, "");
        assert_eq!(user, "What is the capital of France?");
    }

    #[test]
    fn test_count_tokens_anthropic() {
        let provider = AnthropicProvider::new("fake-key", "claude-3-sonnet", 60);

        // Test simple text
        let text = "This is a simple test sentence.";
        let count = provider.count_tokens_anthropic(text);
        assert!(count > 0);

        // Test longer text
        let text = "This is a longer test paragraph with multiple sentences. It contains more tokens than the previous example. We want to ensure that the token counting function works correctly for longer texts as well.";
        let count = provider.count_tokens_anthropic(text);
        assert!(count > 0);

        // Test with special characters
        let text = "Text with special characters: !@#$%^&*()_+{}|:<>?~`;',.";
        let count = provider.count_tokens_anthropic(text);
        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_build_request() {
        let provider = AnthropicProvider::new("fake-key", "claude-3-sonnet", 60);

        // Test with minimal options
        let prompt = "What is the capital of France?";
        let options = None;
        let request = provider.build_request(prompt, options).await.unwrap();

        assert_eq!(request["model"], "claude-3-sonnet");
        assert_eq!(request["max_tokens"], 4096);
        assert!(request["messages"].is_array());

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1); // Just the user message
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is the capital of France?");

        // Test with system prompt
        let prompt = "You are a helpful assistant.\n\nHuman: What is the capital of France?";
        let options = None;
        let request = provider.build_request(prompt, options).await.unwrap();

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2); // System + user
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a helpful assistant.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is the capital of France?");

        // Test with custom options
        let prompt = "What is the capital of France?";
        let options = Some(LlmOptions {
            model: Some("claude-3-opus".to_string()),
            temperature: Some(0.2),
            max_tokens: Some(1000),
            stop_sequences: Some(vec!["STOP".to_string()]),
            stream: Some(true),
            ..Default::default()
        });
        let request = provider.build_request(prompt, options).await.unwrap();

        assert_eq!(request["model"], "claude-3-opus");
        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["max_tokens"], 1000);
        assert_eq!(request["stop_sequences"][0], "STOP");
        assert_eq!(request["stream"], true);
    }
}
