//! # OpenAI Integration
//!
//! This module provides integration with OpenAI's models.
//! It implements the `LlmProvider` trait to enable DOMorpher to use
//! GPT-4, GPT-4o, and GPT-3.5 models for text generation and analysis.
//!
//! ## Features
//!
//! - Support for GPT-4o, GPT-4 Turbo, and GPT-3.5 Turbo models
//! - Streaming response support
//! - Token counting for OpenAI models
//! - Rate limiting and error handling
//! - Context window management
//! - Support for function calling

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

/// OpenAI provider
pub struct OpenAiProvider {
    /// API key for OpenAI
    api_key: String,

    /// Default model to use
    default_model: String,

    /// API endpoint
    api_endpoint: String,

    /// Organization ID (optional)
    organization_id: Option<String>,

    /// Request timeout in seconds
    timeout: u64,

    /// Rate limiter
    rate_limiter: Arc<Mutex<RateLimiter>>,

    /// HTTP client
    client: reqwest::Client,
}

/// OpenAI API response
#[derive(Debug, Clone, Deserialize)]
struct OpenAiResponse {
    /// Response ID
    id: String,

    /// Object type
    object: String,

    /// Model used
    model: String,

    /// Created timestamp
    created: u64,

    /// Response choices
    choices: Vec<OpenAiChoice>,

    /// Usage information
    usage: OpenAiUsage,
}

/// OpenAI response choice
#[derive(Debug, Clone, Deserialize)]
struct OpenAiChoice {
    /// Index of the choice
    index: usize,

    /// Message content
    message: OpenAiMessage,

    /// Finish reason
    finish_reason: Option<String>,
}

/// OpenAI message
#[derive(Debug, Clone, Deserialize)]
struct OpenAiMessage {
    /// Message role
    role: String,

    /// Message content
    content: String,
}

/// OpenAI usage statistics
#[derive(Debug, Clone, Deserialize)]
struct OpenAiUsage {
    /// Prompt tokens
    prompt_tokens: usize,

    /// Completion tokens
    completion_tokens: usize,

    /// Total tokens
    total_tokens: usize,
}

/// OpenAI streaming response
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamingResponse {
    /// Response ID
    id: String,

    /// Object type
    object: String,

    /// Model used
    model: Option<String>,

    /// Created timestamp
    created: u64,

    /// Response choices
    choices: Vec<OpenAiStreamingChoice>,
}

/// OpenAI streaming choice
#[derive(Debug, Clone, Deserialize)]
struct OpenAiStreamingChoice {
    /// Index of the choice
    index: usize,

    /// Delta content
    delta: OpenAiDelta,

    /// Finish reason
    finish_reason: Option<String>,
}

/// OpenAI delta content
#[derive(Debug, Clone, Deserialize)]
struct OpenAiDelta {
    /// Message role
    role: Option<String>,

    /// Message content
    content: Option<String>,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: &str, default_model: &str, timeout: u64) -> Self {
        // Extract organization ID if included in the API key
        let (api_key, organization_id) = if api_key.contains(":") {
            let parts: Vec<&str> = api_key.splitn(2, ':').collect();
            (parts[0].to_string(), Some(parts[1].to_string()))
        } else {
            (api_key.to_string(), None)
        };

        // Create HTTP client with appropriate timeout
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout + 5)) // Add 5 seconds buffer
            .build()
            .unwrap_or_else(|_| {
                warn!("Failed to build custom HTTP client for OpenAI provider, using default");
                reqwest::Client::new()
            });

        // Rate limits for OpenAI API (tier 1 defaults)
        // See: https://platform.openai.com/docs/guides/rate-limits
        let rate_limits = vec![
            // Requests per minute (RPM)
            RateLimit::new("base", 3500, Duration::from_secs(60)),
            // Tokens per minute (TPM)
            RateLimit::new("tokens", 350_000, Duration::from_secs(60)),
        ];

        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new("openai", rate_limits)));

        Self {
            api_key,
            default_model: default_model.to_string(),
            api_endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            organization_id,
            timeout,
            rate_limiter,
            client,
        }
    }

    /// Build a request to the OpenAI API
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
            max_tokens: Some(2048),
            stop_sequences: None,
            timeout_seconds: Some(self.timeout),
            stream: Some(false),
            ..Default::default()
        };

        let options = utils::apply_option_defaults(options, &defaults);

        // Check if the model is specified
        let model = options.model.unwrap_or_else(|| self.default_model.clone());

        // Check if the model is an OpenAI model
        if !model.contains("gpt-") {
            return Err(LlmError::ProviderError(format!(
                "Invalid model for OpenAI provider: {}",
                model
            )));
        }

        // Parse the prompt into messages
        let messages = self.parse_prompt(prompt);

        // Build request
        let mut request = json!({
            "model": model,
            "messages": messages,
            "max_tokens": options.max_tokens.unwrap_or(2048),
        });

        // Add optional parameters
        if let Some(temperature) = options.temperature {
            request["temperature"] = json!(temperature);
        }

        if let Some(top_p) = options.top_p {
            request["top_p"] = json!(top_p);
        }

        if let Some(stop_sequences) = &options.stop_sequences {
            request["stop"] = json!(stop_sequences);
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

    /// Parse the prompt into OpenAI messages
    fn parse_prompt(&self, prompt: &str) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // Check for system message format
        if let Some(system_end) = prompt.find("\n\nHuman:") {
            let system_content = prompt[..system_end].trim();
            let user_content = prompt[system_end + 8..].trim();

            messages.push(json!({
                "role": "system",
                "content": system_content
            }));

            messages.push(json!({
                "role": "user",
                "content": user_content
            }));
        } else if let Some(system_end) = prompt.find("\n\nUser:") {
            let system_content = prompt[..system_end].trim();
            let user_content = prompt[system_end + 7..].trim();

            messages.push(json!({
                "role": "system",
                "content": system_content
            }));

            messages.push(json!({
                "role": "user",
                "content": user_content
            }));
        } else if prompt.starts_with("System:") {
            if let Some(system_end) = prompt.find("\n\n") {
                let system_content = prompt[7..system_end].trim();
                let user_content = prompt[system_end + 2..].trim();

                messages.push(json!({
                    "role": "system",
                    "content": system_content
                }));

                messages.push(json!({
                    "role": "user",
                    "content": user_content
                }));
            } else {
                // Just a system message with no user message
                messages.push(json!({
                    "role": "system",
                    "content": prompt[7..].trim()
                }));

                // Add an empty user message to prompt a response
                messages.push(json!({
                    "role": "user",
                    "content": ""
                }));
            }
        } else {
            // No system message, just a user message
            messages.push(json!({
                "role": "user",
                "content": prompt
            }));
        }

        messages
    }

    /// Count tokens using OpenAI's tokenizer
    fn count_tokens_openai(&self, text: &str) -> usize {
        // This is a simplified approximation as the actual OpenAI tokenizer is not public
        // In a production environment, you'd want to use the tiktoken library

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
            + special_char_count;

        // Add a small buffer for safety
        (token_estimate as f64 * 1.05) as usize
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn generate(&self, prompt: &str, options: Option<LlmOptions>) -> LlmResult<LlmResponse> {
        // Acquire rate limit permit
        let mut rate_limiter = self.rate_limiter.lock().await;
        let permit = rate_limiter.acquire().await?;
        drop(rate_limiter);

        let start_time = Instant::now();
        trace!(
            "Generating text with OpenAI, prompt length: {}",
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

                // Send request to OpenAI API
                let mut request_builder = self.client
                    .post(&self.api_endpoint)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json");

                // Add organization header if present
                if let Some(org_id) = &self.organization_id {
                    request_builder = request_builder.header("OpenAI-Organization", org_id);
                }

                let response = request_builder
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        LlmError::HttpError(format!("HTTP request to OpenAI API failed: {}", e))
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

                    return Err(LlmError::RateLimitExceeded("openai".to_string(), retry_after));
                }

                // Check for other errors
                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_else(|_| "Failed to get error text".to_string());
                    return Err(LlmError::ProviderError(format!(
                        "OpenAI API returned error {}: {}",
                        response.status(),
                        error_text
                    )));
                }

                // Parse response
                let openai_response: OpenAiResponse = response
                    .json()
                    .await
                    .map_err(|e| {
                        LlmError::InvalidResponseFormat(format!("Failed to parse OpenAI response: {}", e))
                    })?;

                // Check if we have any choices
                if openai_response.choices.is_empty() {
                    return Err(LlmError::InvalidResponseFormat("No choices in OpenAI response".to_string()));
                }

                // Get the first choice (we only request one)
                let choice = &openai_response.choices[0];

                // Create LLM response
                let response = LlmResponse {
                    text: choice.message.content.clone(),
                    prompt_tokens: Some(openai_response.usage.prompt_tokens),
                    completion_tokens: Some(openai_response.usage.completion_tokens),
                    total_tokens: Some(openai_response.usage.total_tokens),
                    model: Some(openai_response.model),
                    metadata: Some(HashMap::from([
                        ("id".to_string(), json!(openai_response.id)),
                        ("object".to_string(), json!(openai_response.object)),
                        ("created".to_string(), json!(openai_response.created)),
                    ])),
                    finish_reason: choice.finish_reason.clone(),
                };

                Ok(response)
            },
            3,
            Duration::from_secs(1),
            Duration::from_secs(30),
            "openai"
        ).await;

        // Record metrics
        let elapsed = start_time.elapsed();
        debug!(
            "OpenAI request completed in {:?}, prompt tokens: {}, completion tokens: {}",
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
            "Generating streaming text with OpenAI, prompt length: {}",
            prompt.len()
        );

        // Clone options and force streaming
        let mut streaming_options = options.clone().unwrap_or_default();
        streaming_options.stream = Some(true);

        // Build request
        let request = self.build_request(prompt, Some(streaming_options)).await?;

        let completion_result = utils::execute_with_retry(
            || async {
                // Send request to OpenAI API
                let mut request_builder = self
                    .client
                    .post(&self.api_endpoint)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json");

                // Add organization header if present
                if let Some(org_id) = &self.organization_id {
                    request_builder = request_builder.header("OpenAI-Organization", org_id);
                }

                let response = request_builder.json(&request).send().await.map_err(|e| {
                    LlmError::HttpError(format!("HTTP request to OpenAI API failed: {}", e))
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
                        "openai".to_string(),
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
                        "OpenAI API returned error {}: {}",
                        response.status(),
                        error_text
                    )));
                }

                // Process streaming response
                let mut stream = response.bytes_stream();
                let mut full_text = String::new();
                let mut model = String::new();
                let mut finish_reason = None;

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
                            match serde_json::from_str::<OpenAiStreamingResponse>(json_str) {
                                Ok(streaming_response) => {
                                    // Store model info
                                    if let Some(model_str) = streaming_response.model {
                                        model = model_str;
                                    }

                                    // Process choices
                                    if !streaming_response.choices.is_empty() {
                                        let choice = &streaming_response.choices[0];

                                        // Update finish reason
                                        if let Some(reason) = &choice.finish_reason {
                                            finish_reason = Some(reason.clone());
                                        }

                                        // Process delta content
                                        if let Some(content) = &choice.delta.content {
                                            full_text.push_str(content);
                                            callback(content.clone()).map_err(|e| {
                                                LlmError::Other(format!("Callback error: {}", e))
                                            })?;
                                        }
                                    }
                                }
                                Err(e) => {
                                    if !json_str.trim().is_empty() && json_str.trim() != "[DONE]" {
                                        warn!("Failed to parse OpenAI streaming response: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }

                // Approximate token counts
                let prompt_tokens = self.count_tokens_openai(prompt);
                let completion_tokens = self.count_tokens_openai(&full_text);

                // Create LLM response
                let response = LlmResponse {
                    text: full_text,
                    prompt_tokens: Some(prompt_tokens),
                    completion_tokens: Some(completion_tokens),
                    total_tokens: Some(prompt_tokens + completion_tokens),
                    model: Some(model),
                    metadata: Some(HashMap::new()),
                    finish_reason,
                };

                Ok(response)
            },
            3,
            Duration::from_secs(1),
            Duration::from_secs(30),
            "openai",
        )
        .await;

        // Record metrics
        let elapsed = start_time.elapsed();
        debug!(
            "OpenAI streaming request completed in {:?}, prompt tokens: {}, completion tokens: {}",
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
        super::provider::LlmProvider::OpenAI
    }

    fn default_model(&self) -> String {
        self.default_model.clone()
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.count_tokens_openai(text)
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
        // Capabilities based on model
        if self.default_model.contains("gpt-4o") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                max_tokens: 128_000, // GPT-4o context window
                supports_system_messages: true,
            }
        } else if self.default_model.contains("gpt-4-turbo") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                max_tokens: 128_000, // GPT-4 Turbo context window
                supports_system_messages: true,
            }
        } else if self.default_model.contains("gpt-4") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: self.default_model.contains("vision"),
                embeddings: false,
                max_tokens: 8192, // Original GPT-4 context window
                supports_system_messages: true,
            }
        } else if self.default_model.contains("gpt-3.5-turbo") {
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: false,
                embeddings: false,
                max_tokens: 16385, // GPT-3.5 Turbo context window
                supports_system_messages: true,
            }
        } else {
            // Default capabilities
            ProviderCapabilities {
                streaming: true,
                function_calling: true,
                vision: false,
                embeddings: false,
                max_tokens: 4096,
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
        let provider = OpenAiProvider::new("fake-key", "gpt-4o", 60);

        // Test system + user format with "Human:" marker
        let prompt = "You are an AI assistant. Be concise and helpful.\n\nHuman: What is the capital of France?";
        let messages = provider.parse_prompt(prompt);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            "You are an AI assistant. Be concise and helpful."
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is the capital of France?");

        // Test system + user format with "User:" marker
        let prompt = "You are an AI assistant. Be concise and helpful.\n\nUser: What is the capital of France?";
        let messages = provider.parse_prompt(prompt);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            "You are an AI assistant. Be concise and helpful."
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is the capital of France?");

        // Test "System:" prefix
        let prompt = "System: You are an AI assistant. Be concise and helpful.\n\nWhat is the capital of France?";
        let messages = provider.parse_prompt(prompt);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            "You are an AI assistant. Be concise and helpful."
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is the capital of France?");

        // Test no system prompt
        let prompt = "What is the capital of France?";
        let messages = provider.parse_prompt(prompt);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is the capital of France?");
    }

    #[test]
    fn test_count_tokens_openai() {
        let provider = OpenAiProvider::new("fake-key", "gpt-4o", 60);

        // Test simple text
        let text = "This is a simple test sentence.";
        let count = provider.count_tokens_openai(text);
        assert!(count > 0);

        // Test longer text
        let text = "This is a longer test paragraph with multiple sentences. It contains more tokens than the previous example. We want to ensure that the token counting function works correctly for longer texts as well.";
        let count = provider.count_tokens_openai(text);
        assert!(count > 0);

        // Test with special characters
        let text = "Text with special characters: !@#$%^&*()_+{}|:<>?~`;',.";
        let count = provider.count_tokens_openai(text);
        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_build_request() {
        let provider = OpenAiProvider::new("fake-key", "gpt-4o", 60);

        // Test with minimal options
        let prompt = "What is the capital of France?";
        let options = None;
        let request = provider.build_request(prompt, options).await.unwrap();

        assert_eq!(request["model"], "gpt-4o");
        assert_eq!(request["max_tokens"], 2048);
        assert!(request["messages"].is_array());

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1); // Just the user message
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is the capital of France?");

        // Test with custom options
        let prompt = "What is the capital of France?";
        let options = Some(LlmOptions {
            model: Some("gpt-3.5-turbo".to_string()),
            temperature: Some(0.2),
            max_tokens: Some(1000),
            stop_sequences: Some(vec!["STOP".to_string()]),
            stream: Some(true),
            ..Default::default()
        });
        let request = provider.build_request(prompt, options).await.unwrap();

        assert_eq!(request["model"], "gpt-3.5-turbo");
        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["max_tokens"], 1000);
        assert_eq!(request["stop"][0], "STOP");
        assert_eq!(request["stream"], true);
    }
}
