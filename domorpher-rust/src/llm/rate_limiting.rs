//! # Rate Limiting
//!
//! This module provides rate limiting functionality for LLM API requests.
//! It implements configurable rate limits with token bucket and leaky bucket algorithms,
//! supports per-model and per-provider limits, and provides throttling mechanisms.
//!
//! ## Features
//!
//! - Multiple rate limiting algorithms (token bucket, leaky bucket)
//! - Provider-specific rate limits
//! - Model-specific rate limits
//! - Automatic request throttling
//! - Adaptive retry mechanisms
//! - Usage tracking and reporting

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as TokioMutex, RwLock, Semaphore};
use tokio::time::sleep;

/// Rate limiting algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm
    TokenBucket,

    /// Leaky bucket algorithm
    LeakyBucket,

    /// Fixed window algorithm
    FixedWindow,

    /// Sliding window algorithm
    SlidingWindow,
}

impl Default for RateLimitAlgorithm {
    fn default() -> Self {
        Self::TokenBucket
    }
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled
    pub enabled: bool,

    /// Rate limiting algorithm
    pub algorithm: RateLimitAlgorithm,

    /// Requests per minute (RPM)
    pub rpm: u32,

    /// Requests per day (RPD)
    pub rpd: Option<u32>,

    /// Maximum burst size
    pub burst_size: u32,

    /// Maximum number of concurrent requests
    pub max_concurrent_requests: u32,

    /// Timeout for acquiring a permit (seconds)
    pub permit_timeout_seconds: u64,

    /// Retry configuration
    pub retry_config: RetryConfig,

    /// Whether to include waiting time in timeouts
    pub include_waiting_in_timeout: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: RateLimitAlgorithm::default(),
            rpm: 10,
            rpd: Some(10000),
            burst_size: 5,
            max_concurrent_requests: 5,
            permit_timeout_seconds: 60,
            retry_config: RetryConfig::default(),
            include_waiting_in_timeout: true,
        }
    }
}

/// Retry configuration for rate-limited requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,

    /// Base delay for exponential backoff (in milliseconds)
    pub base_delay_ms: u64,

    /// Maximum delay for exponential backoff (in milliseconds)
    pub max_delay_ms: u64,

    /// Jitter factor for randomizing delays (0.0 to 1.0)
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            jitter_factor: 0.1,
        }
    }
}

/// Usage statistics for rate-limited requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Number of successful requests
    pub successful_requests: u64,

    /// Number of throttled requests
    pub throttled_requests: u64,

    /// Number of failed requests
    pub failed_requests: u64,

    /// Total tokens consumed
    pub total_tokens: u64,

    /// Average tokens per request
    pub avg_tokens_per_request: f64,

    /// Average request time (in milliseconds)
    pub avg_request_time_ms: f64,

    /// Maximum request time (in milliseconds)
    pub max_request_time_ms: u64,

    /// Average wait time (in milliseconds)
    pub avg_wait_time_ms: f64,

    /// Maximum wait time (in milliseconds)
    pub max_wait_time_ms: u64,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            successful_requests: 0,
            throttled_requests: 0,
            failed_requests: 0,
            total_tokens: 0,
            avg_tokens_per_request: 0.0,
            avg_request_time_ms: 0.0,
            max_request_time_ms: 0,
            avg_wait_time_ms: 0.0,
            max_wait_time_ms: 0,
        }
    }
}

/// Request tracking information
#[derive(Debug, Clone)]
struct RequestInfo {
    /// Time when the request was made
    timestamp: Instant,

    /// Number of tokens in the request
    tokens: usize,

    /// Request duration
    duration: Option<Duration>,

    /// Wait time before the request was processed
    wait_time: Option<Duration>,
}

/// Rate limiter for API requests
pub struct RateLimiter {
    /// Rate limit configuration
    config: RateLimitConfig,

    /// Per-provider rate limit configurations
    provider_configs: HashMap<String, RateLimitConfig>,

    /// Per-model rate limit configurations
    model_configs: HashMap<String, RateLimitConfig>,

    /// Request history for sliding window
    request_history: Arc<Mutex<VecDeque<RequestInfo>>>,

    /// Token bucket for the token bucket algorithm
    token_bucket: Arc<TokioMutex<TokenBucket>>,

    /// Semaphore for concurrent request limiting
    semaphore: Arc<Semaphore>,

    /// Usage statistics
    stats: Arc<RwLock<UsageStats>>,

    /// Last RPM reset time
    last_rpm_reset: Arc<Mutex<Instant>>,

    /// Last RPD reset time
    last_rpd_reset: Arc<Mutex<Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter with the default configuration
    pub fn new() -> Self {
        Self::with_config(RateLimitConfig::default())
    }

    /// Create a new rate limiter with the given configuration
    pub fn with_config(config: RateLimitConfig) -> Self {
        let now = Instant::now();

        Self {
            token_bucket: Arc::new(TokioMutex::new(TokenBucket::new(
                config.rpm as usize,
                config.burst_size as usize,
            ))),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_requests as usize)),
            request_history: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(UsageStats::default())),
            last_rpm_reset: Arc::new(Mutex::new(now)),
            last_rpd_reset: Arc::new(Mutex::new(now)),
            provider_configs: HashMap::new(),
            model_configs: HashMap::new(),
            config,
        }
    }

    /// Set the configuration for a specific provider
    pub fn set_provider_config(&mut self, provider: String, config: RateLimitConfig) {
        self.provider_configs.insert(provider, config);
    }

    /// Set the configuration for a specific model
    pub fn set_model_config(&mut self, model: String, config: RateLimitConfig) {
        self.model_configs.insert(model, config);
    }

    /// Get the effective configuration for a provider and model
    fn get_effective_config(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> &RateLimitConfig {
        // Check for model-specific config
        if let Some(model_name) = model {
            if let Some(config) = self.model_configs.get(model_name) {
                return config;
            }
        }

        // Check for provider-specific config
        if let Some(provider_name) = provider {
            if let Some(config) = self.provider_configs.get(provider_name) {
                return config;
            }
        }

        // Default to global config
        &self.config
    }

    /// Acquire a permit for making a request
    pub async fn acquire_permit(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<RateLimit, RateLimitError> {
        let config = self.get_effective_config(provider, model);

        if !config.enabled {
            // Rate limiting disabled, permit granted immediately
            return Ok(RateLimit::new(self.clone()));
        }

        // Check if we've exceeded the daily limit
        if let Some(rpd) = config.rpd {
            let mut last_rpd_reset = self
                .last_rpd_reset
                .lock()
                .map_err(|_| RateLimitError::Internal("Failed to acquire RPD lock".to_string()))?;

            let now = Instant::now();
            let elapsed = now.duration_since(*last_rpd_reset);

            if elapsed > Duration::from_secs(24 * 60 * 60) {
                // Reset daily counter
                *last_rpd_reset = now;
            } else {
                // Check daily limit
                let stats = self.stats.read().await;
                let daily_requests = stats.successful_requests;

                if daily_requests >= rpd as u64 {
                    return Err(RateLimitError::DailyLimitExceeded);
                }
            }
        }

        // Try to acquire a semaphore permit for concurrent request limiting
        let semaphore_permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RateLimitError::Internal("Failed to acquire semaphore".to_string()))?;

        // Track wait time
        let wait_start = Instant::now();

        // Apply rate limiting algorithm
        match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                // Try to acquire a token from the bucket
                let mut bucket = self.token_bucket.lock().await;

                if !bucket.acquire() {
                    // No tokens available, need to wait for refill
                    let wait_time = bucket.time_until_next_token();

                    // Check if waiting would exceed the permit timeout
                    if config.include_waiting_in_timeout
                        && wait_time > Duration::from_secs(config.permit_timeout_seconds)
                    {
                        return Err(RateLimitError::PermitTimeoutExceeded);
                    }

                    // Wait for token refill
                    drop(bucket); // Release the lock while waiting
                    sleep(wait_time).await;

                    // Try again
                    let mut bucket = self.token_bucket.lock().await;
                    if !bucket.acquire() {
                        return Err(RateLimitError::RateLimitExceeded);
                    }
                }
            }
            RateLimitAlgorithm::LeakyBucket => {
                // Simple implementation of leaky bucket
                let rpm = config.rpm;
                let interval = Duration::from_secs(60) / rpm;

                // Check last request time
                let mut history = self.request_history.lock().map_err(|_| {
                    RateLimitError::Internal("Failed to acquire history lock".to_string())
                })?;

                if let Some(last_request) = history.back() {
                    let elapsed = wait_start.duration_since(last_request.timestamp);

                    if elapsed < interval {
                        // Need to wait
                        let wait_time = interval - elapsed;

                        // Check if waiting would exceed the permit timeout
                        if config.include_waiting_in_timeout
                            && wait_time > Duration::from_secs(config.permit_timeout_seconds)
                        {
                            return Err(RateLimitError::PermitTimeoutExceeded);
                        }

                        // Wait for the required interval
                        drop(history); // Release the lock while waiting
                        sleep(wait_time).await;
                    }
                }
            }
            RateLimitAlgorithm::FixedWindow => {
                // Simple fixed window rate limiting
                let mut last_rpm_reset = self.last_rpm_reset.lock().map_err(|_| {
                    RateLimitError::Internal("Failed to acquire RPM lock".to_string())
                })?;

                let now = Instant::now();
                let elapsed = now.duration_since(*last_rpm_reset);

                if elapsed > Duration::from_secs(60) {
                    // Reset window
                    *last_rpm_reset = now;

                    // Reset request counter
                    let mut history = self.request_history.lock().map_err(|_| {
                        RateLimitError::Internal("Failed to acquire history lock".to_string())
                    })?;
                    history.clear();
                } else {
                    // Check if we're within the rate limit
                    let history = self.request_history.lock().map_err(|_| {
                        RateLimitError::Internal("Failed to acquire history lock".to_string())
                    })?;

                    if history.len() >= config.rpm as usize {
                        // Exceeded rate limit
                        let wait_time = Duration::from_secs(60) - elapsed;

                        // Check if waiting would exceed the permit timeout
                        if config.include_waiting_in_timeout
                            && wait_time > Duration::from_secs(config.permit_timeout_seconds)
                        {
                            return Err(RateLimitError::PermitTimeoutExceeded);
                        }

                        // Wait until the window resets
                        drop(history); // Release the lock while waiting
                        drop(last_rpm_reset);
                        sleep(wait_time).await;
                    }
                }
            }
            RateLimitAlgorithm::SlidingWindow => {
                // Sliding window rate limiting
                let window_size = Duration::from_secs(60);

                // Remove old requests from the window
                let mut history = self.request_history.lock().map_err(|_| {
                    RateLimitError::Internal("Failed to acquire history lock".to_string())
                })?;

                let now = Instant::now();
                while let Some(request) = history.front() {
                    if now.duration_since(request.timestamp) > window_size {
                        history.pop_front();
                    } else {
                        break;
                    }
                }

                // Check if we're within the rate limit
                if history.len() >= config.rpm as usize {
                    // Exceeded rate limit
                    let oldest = history.front().unwrap();
                    let wait_time = window_size - now.duration_since(oldest.timestamp);

                    // Check if waiting would exceed the permit timeout
                    if config.include_waiting_in_timeout
                        && wait_time > Duration::from_secs(config.permit_timeout_seconds)
                    {
                        return Err(RateLimitError::PermitTimeoutExceeded);
                    }

                    // Wait until the oldest request exits the window
                    drop(history); // Release the lock while waiting
                    sleep(wait_time).await;
                }
            }
        }

        // Calculate wait time
        let wait_time = wait_start.elapsed();

        // Create request info
        let request_info = RequestInfo {
            timestamp: Instant::now(),
            tokens: 0,      // Will be updated later
            duration: None, // Will be updated later
            wait_time: Some(wait_time),
        };

        // Add request to history
        {
            let mut history = self.request_history.lock().map_err(|_| {
                RateLimitError::Internal("Failed to acquire history lock".to_string())
            })?;

            history.push_back(request_info.clone());

            // Limit history size
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Create rate limit object
        Ok(RateLimit {
            rate_limiter: self.clone(),
            _semaphore_permit: semaphore_permit,
            request_info,
            provider: provider.map(String::from),
            model: model.map(String::from),
        })
    }

    /// Update usage statistics
    async fn update_stats(&self, request_info: RequestInfo, success: bool, throttled: bool) {
        let mut stats = self.stats.write().await;

        if success {
            stats.successful_requests += 1;
            stats.total_tokens += request_info.tokens as u64;

            // Update average tokens per request
            if stats.successful_requests > 0 {
                stats.avg_tokens_per_request =
                    stats.total_tokens as f64 / stats.successful_requests as f64;
            }

            // Update request time statistics
            if let Some(duration) = request_info.duration {
                let duration_ms = duration.as_millis() as u64;

                // Update average request time
                let old_avg = stats.avg_request_time_ms;
                let new_avg =
                    old_avg + (duration_ms as f64 - old_avg) / stats.successful_requests as f64;
                stats.avg_request_time_ms = new_avg;

                // Update max request time
                if duration_ms > stats.max_request_time_ms {
                    stats.max_request_time_ms = duration_ms;
                }
            }

            // Update wait time statistics
            if let Some(wait_time) = request_info.wait_time {
                let wait_time_ms = wait_time.as_millis() as u64;

                // Update average wait time
                let old_avg = stats.avg_wait_time_ms;
                let new_avg =
                    old_avg + (wait_time_ms as f64 - old_avg) / stats.successful_requests as f64;
                stats.avg_wait_time_ms = new_avg;

                // Update max wait time
                if wait_time_ms > stats.max_wait_time_ms {
                    stats.max_wait_time_ms = wait_time_ms;
                }
            }
        } else if throttled {
            stats.throttled_requests += 1;
        } else {
            stats.failed_requests += 1;
        }
    }

    /// Get usage statistics
    pub async fn get_stats(&self) -> UsageStats {
        self.stats.read().await.clone()
    }

    /// Reset usage statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = UsageStats::default();
    }

    /// Execute a function with rate limiting
    pub async fn execute<F, Fut, T>(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        f: F,
    ) -> Result<T, RateLimitError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, super::LlmError>>,
    {
        // Acquire permit
        let mut rate_limit = self.acquire_permit(provider, model).await?;

        // Execute the function
        let result = f().await;

        // Update rate limit with result
        match &result {
            Ok(_) => {
                // Success
                rate_limit.set_success(0); // Token count unknown here
            }
            Err(e) => {
                // Check if it was a rate limit error
                let throttled = matches!(e, super::LlmError::RateLimitExceeded(_, _));
                rate_limit.set_failure(throttled);
            }
        }

        // Map error
        result.map_err(|e| match e {
            super::LlmError::RateLimitExceeded(provider, retry_after) => {
                RateLimitError::ExternalRateLimitExceeded {
                    provider: provider.to_string(),
                    retry_after,
                }
            }
            other => RateLimitError::Other(other.to_string()),
        })
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            provider_configs: self.provider_configs.clone(),
            model_configs: self.model_configs.clone(),
            request_history: self.request_history.clone(),
            token_bucket: self.token_bucket.clone(),
            semaphore: self.semaphore.clone(),
            stats: self.stats.clone(),
            last_rpm_reset: self.last_rpm_reset.clone(),
            last_rpd_reset: self.last_rpd_reset.clone(),
        }
    }
}

/// Rate limit permit
pub struct RateLimit {
    /// The rate limiter
    rate_limiter: RateLimiter,

    /// Semaphore permit for concurrent request limiting
    _semaphore_permit: tokio::sync::OwnedSemaphorePermit,

    /// Request information
    request_info: RequestInfo,

    /// Provider name
    provider: Option<String>,

    /// Model name
    model: Option<String>,
}

impl RateLimit {
    /// Create a new rate limit
    fn new(rate_limiter: RateLimiter) -> Self {
        // Create a dummy semaphore permit
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.try_acquire_owned().unwrap();

        Self {
            rate_limiter,
            _semaphore_permit: permit,
            request_info: RequestInfo {
                timestamp: Instant::now(),
                tokens: 0,
                duration: None,
                wait_time: None,
            },
            provider: None,
            model: None,
        }
    }

    /// Mark the request as successful and update statistics
    pub fn set_success(&mut self, tokens: usize) {
        self.request_info.tokens = tokens;
        self.request_info.duration = Some(self.request_info.timestamp.elapsed());

        // Update statistics asynchronously
        let rate_limiter = self.rate_limiter.clone();
        let request_info = self.request_info.clone();

        tokio::spawn(async move {
            rate_limiter.update_stats(request_info, true, false).await;
        });
    }

    /// Mark the request as failed and update statistics
    pub fn set_failure(&mut self, throttled: bool) {
        self.request_info.duration = Some(self.request_info.timestamp.elapsed());

        // Update statistics asynchronously
        let rate_limiter = self.rate_limiter.clone();
        let request_info = self.request_info.clone();

        tokio::spawn(async move {
            rate_limiter
                .update_stats(request_info, false, throttled)
                .await;
        });
    }
}

/// Rate limit error
#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Daily limit exceeded
    #[error("Daily rate limit exceeded")]
    DailyLimitExceeded,

    /// Permit timeout exceeded
    #[error("Permit acquisition timeout exceeded")]
    PermitTimeoutExceeded,

    /// External rate limit exceeded
    #[error(
        "External rate limit exceeded for provider {provider}, retry after {retry_after} seconds"
    )]
    ExternalRateLimitExceeded {
        /// Provider name
        provider: String,

        /// Retry after (in seconds)
        retry_after: u64,
    },

    /// Internal error
    #[error("Internal rate limiting error: {0}")]
    Internal(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

/// Token bucket for token bucket algorithm
struct TokenBucket {
    /// Maximum number of tokens
    capacity: usize,

    /// Current number of tokens
    tokens: usize,

    /// Time of last token refill
    last_refill: Instant,

    /// Tokens per minute
    tokens_per_minute: usize,
}

impl TokenBucket {
    /// Create a new token bucket
    fn new(tokens_per_minute: usize, capacity: usize) -> Self {
        Self {
            capacity,
            tokens: capacity,
            last_refill: Instant::now(),
            tokens_per_minute,
        }
    }

    /// Try to acquire a token
    fn acquire(&mut self) -> bool {
        // Refill tokens based on elapsed time
        self.refill();

        // Check if we have tokens available
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);

        // Calculate tokens to add
        let tokens_to_add = (elapsed.as_secs_f64() * self.tokens_per_minute as f64 / 60.0) as usize;

        if tokens_to_add > 0 {
            // Add tokens, up to capacity
            self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
            self.last_refill = now;
        }
    }

    /// Get time until next token is available
    fn time_until_next_token(&self) -> Duration {
        if self.tokens > 0 {
            // Token already available
            Duration::from_secs(0)
        } else {
            // Calculate time until next token
            let token_interval = Duration::from_secs(60) / self.tokens_per_minute as u32;
            let elapsed = Instant::now().duration_since(self.last_refill);

            if elapsed >= token_interval {
                // Token should be available now
                Duration::from_secs(0)
            } else {
                // Wait for the remainder of the interval
                token_interval - elapsed
            }
        }
    }
}

/// Rate limit middleware trait for adding rate limiting to providers
#[async_trait]
pub trait RateLimitMiddleware {
    /// Execute a function with rate limiting
    async fn with_rate_limit<F, Fut, T>(
        &self,
        rate_limiter: &RateLimiter,
        provider: &str,
        model: &str,
        f: F,
    ) -> super::LlmResult<T>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = super::LlmResult<T>> + Send,
        T: Send;
}

/// Default implementation of rate limit middleware
pub struct DefaultRateLimitMiddleware;

#[async_trait]
impl RateLimitMiddleware for DefaultRateLimitMiddleware {
    async fn with_rate_limit<F, Fut, T>(
        &self,
        rate_limiter: &RateLimiter,
        provider: &str,
        model: &str,
        f: F,
    ) -> super::LlmResult<T>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = super::LlmResult<T>> + Send,
        T: Send,
    {
        // Get effective config for retries
        let config = rate_limiter.get_effective_config(Some(provider), Some(model));
        let retry_config = &config.retry_config;

        // Try to execute with retries
        let mut retry_count = 0;
        let mut last_error = None;

        while retry_count <= retry_config.max_retries {
            // Try to acquire a permit
            match rate_limiter
                .acquire_permit(Some(provider), Some(model))
                .await
            {
                Ok(mut rate_limit) => {
                    // Execute the function
                    match f().await {
                        Ok(result) => {
                            // Success
                            rate_limit.set_success(0); // Token count unknown here
                            return Ok(result);
                        }
                        Err(super::LlmError::RateLimitExceeded(provider_name, retry_after)) => {
                            // External rate limit exceeded
                            rate_limit.set_failure(true);

                            // Check if we should retry
                            if retry_count < retry_config.max_retries {
                                // Calculate retry delay
                                let delay = if retry_after > 0 {
                                    // Use provider's retry-after value
                                    Duration::from_secs(retry_after)
                                } else {
                                    // Use exponential backoff
                                    let base_delay = retry_config.base_delay_ms;
                                    let max_delay = retry_config.max_delay_ms;
                                    let jitter = retry_config.jitter_factor;

                                    // Calculate exponential backoff
                                    let exp_backoff = base_delay * (2u64.pow(retry_count));
                                    let capped_backoff = exp_backoff.min(max_delay);

                                    // Add jitter
                                    let jitter_range = (capped_backoff as f64 * jitter) as u64;
                                    let jitter_amount = fastrand::u64(0..=jitter_range);

                                    Duration::from_millis(capped_backoff + jitter_amount)
                                };

                                // Wait before retrying
                                tokio::time::sleep(delay).await;

                                // Increment retry count
                                retry_count += 1;
                                last_error = Some(super::LlmError::RateLimitExceeded(
                                    provider_name,
                                    retry_after,
                                ));

                                // Continue to next retry
                                continue;
                            } else {
                                // Max retries exceeded
                                return Err(super::LlmError::RateLimitExceeded(
                                    provider_name,
                                    retry_after,
                                ));
                            }
                        }
                        Err(e) => {
                            // Other error
                            rate_limit.set_failure(false);
                            return Err(e);
                        }
                    }
                }
                Err(RateLimitError::RateLimitExceeded) => {
                    // Local rate limit exceeded
                    if retry_count < retry_config.max_retries {
                        // Calculate delay using exponential backoff
                        let base_delay = retry_config.base_delay_ms;
                        let max_delay = retry_config.max_delay_ms;
                        let jitter = retry_config.jitter_factor;

                        // Calculate exponential backoff
                        let exp_backoff = base_delay * (2u64.pow(retry_count));
                        let capped_backoff = exp_backoff.min(max_delay);

                        // Add jitter
                        let jitter_range = (capped_backoff as f64 * jitter) as u64;
                        let jitter_amount = fastrand::u64(0..=jitter_range);

                        let delay = Duration::from_millis(capped_backoff + jitter_amount);

                        // Wait before retrying
                        tokio::time::sleep(delay).await;

                        // Increment retry count
                        retry_count += 1;
                        last_error =
                            Some(super::LlmError::RateLimitExceeded(provider.to_string(), 0));

                        // Continue to next retry
                        continue;
                    } else {
                        // Max retries exceeded
                        return Err(super::LlmError::RateLimitExceeded(provider.to_string(), 0));
                    }
                }
                Err(RateLimitError::DailyLimitExceeded) => {
                    // Daily limit exceeded, no retry
                    return Err(super::LlmError::RateLimitExceeded(
                        provider.to_string(),
                        24 * 60 * 60, // Retry after 24 hours
                    ));
                }
                Err(RateLimitError::ExternalRateLimitExceeded {
                    provider: provider_name,
                    retry_after,
                }) => {
                    // External rate limit exceeded
                    if retry_count < retry_config.max_retries {
                        // Calculate delay
                        let delay = Duration::from_secs(retry_after);

                        // Wait before retrying
                        tokio::time::sleep(delay).await;

                        // Increment retry count
                        retry_count += 1;
                        last_error = Some(super::LlmError::RateLimitExceeded(
                            provider_name.clone(),
                            retry_after,
                        ));

                        // Continue to next retry
                        continue;
                    } else {
                        // Max retries exceeded
                        return Err(super::LlmError::RateLimitExceeded(
                            provider_name,
                            retry_after,
                        ));
                    }
                }
                Err(e) => {
                    // Other error
                    return Err(super::LlmError::Other(e.to_string()));
                }
            }
        }

        // If we get here, all retries failed
        Err(last_error.unwrap_or_else(|| {
            super::LlmError::Other("Rate limit exceeded and all retries failed".to_string())
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let rate_limiter = RateLimiter::new();

        // Check default config
        assert_eq!(rate_limiter.config.enabled, true);
        assert_eq!(
            rate_limiter.config.algorithm,
            RateLimitAlgorithm::TokenBucket
        );
        assert_eq!(rate_limiter.config.rpm, 10);
        assert_eq!(rate_limiter.config.burst_size, 5);
        assert_eq!(rate_limiter.config.max_concurrent_requests, 5);
    }

    #[tokio::test]
    async fn test_rate_limiter_with_config() {
        let config = RateLimitConfig {
            enabled: true,
            algorithm: RateLimitAlgorithm::LeakyBucket,
            rpm: 20,
            rpd: Some(2000),
            burst_size: 10,
            max_concurrent_requests: 8,
            permit_timeout_seconds: 30,
            retry_config: RetryConfig::default(),
            include_waiting_in_timeout: true,
        };

        let rate_limiter = RateLimiter::with_config(config.clone());

        // Check config values
        assert_eq!(rate_limiter.config.enabled, config.enabled);
        assert_eq!(rate_limiter.config.algorithm, config.algorithm);
        assert_eq!(rate_limiter.config.rpm, config.rpm);
        assert_eq!(rate_limiter.config.burst_size, config.burst_size);
        assert_eq!(
            rate_limiter.config.max_concurrent_requests,
            config.max_concurrent_requests
        );
    }

    #[tokio::test]
    async fn test_permit_acquisition() {
        let config = RateLimitConfig {
            enabled: true,
            algorithm: RateLimitAlgorithm::TokenBucket,
            rpm: 100,  // High enough to not interfere with test
            rpd: None, // No daily limit
            burst_size: 10,
            max_concurrent_requests: 5,
            permit_timeout_seconds: 30,
            retry_config: RetryConfig::default(),
            include_waiting_in_timeout: true,
        };

        let rate_limiter = RateLimiter::with_config(config);

        // Acquire permits successfully
        for _ in 0..5 {
            let permit = rate_limiter.acquire_permit(None, None).await;
            assert!(permit.is_ok());
        }

        // Now try to acquire more than max_concurrent_requests
        // This should wait due to the semaphore
        let permit_future = rate_limiter.acquire_permit(None, None);

        // Use a short timeout to avoid waiting for the full permit timeout
        let result = tokio::time::timeout(Duration::from_millis(100), permit_future).await;

        // Expect a timeout since all permits are taken
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stats_update() {
        let rate_limiter = RateLimiter::new();

        // Initial stats should be all zeros
        let initial_stats = rate_limiter.get_stats().await;
        assert_eq!(initial_stats.successful_requests, 0);
        assert_eq!(initial_stats.throttled_requests, 0);
        assert_eq!(initial_stats.failed_requests, 0);

        // Update stats with a successful request
        let request_info = RequestInfo {
            timestamp: Instant::now(),
            tokens: 100,
            duration: Some(Duration::from_millis(50)),
            wait_time: Some(Duration::from_millis(10)),
        };

        rate_limiter.update_stats(request_info, true, false).await;

        // Check updated stats
        let stats = rate_limiter.get_stats().await;
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.throttled_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.total_tokens, 100);
        assert_eq!(stats.avg_tokens_per_request, 100.0);
        assert!(stats.avg_request_time_ms > 0.0);
        assert!(stats.max_request_time_ms > 0);
    }

    #[tokio::test]
    async fn test_execute_with_rate_limiting() {
        let rate_limiter = RateLimiter::new();

        // Define a test function that always succeeds
        let success_fn = || async { Ok::<_, super::super::LlmError>(42) };

        // Execute with rate limiting
        let result = rate_limiter.execute(None, None, success_fn).await;

        // Should succeed
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Check stats
        let stats = rate_limiter.get_stats().await;
        assert_eq!(stats.successful_requests, 1);
    }
}
