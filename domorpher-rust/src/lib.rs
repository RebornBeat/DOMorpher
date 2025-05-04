//! # DOMorpher
//!
//! DOMorpher is an advanced web automation and data extraction framework that combines
//! traditional DOM parsing with large language model intelligence.
//!
//! ## Key Features
//!
//! - **Natural Language Instructions**: Extract data using plain English instead of selectors
//! - **Adaptive Intelligence**: Automatically adjusts to website changes without requiring updates
//! - **Zero-Shot Learning**: Works immediately with any website without prior training
//! - **Semantic Understanding**: Understands meaning and relationships, not just structure
//! - **Autonomous Agents**: Goal-driven web navigation and interaction
//!
//! ## Basic Usage
//!
//! ```rust
//! use domorpher::{extract_from_url, configure};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure API keys
//!     configure(
//!         Some("your-anthropic-api-key"),
//!         Some("your-openai-api-key"),
//!         None,
//!     )?;
//!
//!     // Extract data using natural language
//!     let products = extract_from_url(
//!         "https://example.com/products",
//!         "Extract all products with their names, prices, and descriptions",
//!         None,
//!     ).await?;
//!
//!     println!("{}", serde_json::to_string_pretty(&products)?);
//!     Ok(())
//! }
//! ```
//!
//! ## Advanced Usage
//!
//! For more control, use the `Extractor` struct directly:
//!
//! ```rust
//! use domorpher::{Extractor, ExtractorConfig, LlmProvider, ChunkingStrategy};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create extractor with custom configuration
//!     let config = ExtractorConfig::builder()
//!         .llm_provider(LlmProvider::Anthropic)
//!         .model("claude-3-opus")
//!         .chunking_strategy(ChunkingStrategy::Semantic)
//!         .adaptation_level(AdaptationLevel::Aggressive)
//!         .javascript_support(true)
//!         .build();
//!
//!     let extractor = Extractor::new(config);
//!
//!     // Extract from URL
//!     let results = extractor.extract_from_url(
//!         "https://example.com/products",
//!         "Extract all products with their names, prices, descriptions, and ratings",
//!     ).await?;
//!
//!     println!("{}", serde_json::to_string_pretty(&results)?);
//!     Ok(())
//! }
//! ```
//!
//! ## Autonomous Web Agent
//!
//! For goal-driven web interaction, use the `AutonomousAgent`:
//!
//! ```rust
//! use domorpher::{AutonomousAgent, AgentConfig, NavigationStrategy};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create agent
//!     let config = AgentConfig::builder()
//!         .objective("Find and compare the top 3 best-selling laptops under $1000")
//!         .model("claude-3-opus")
//!         .navigation_strategy(NavigationStrategy::SemanticFirst)
//!         .timeout_seconds(300)
//!         .build();
//!
//!     let agent = AutonomousAgent::new(config);
//!
//!     // Execute agent
//!     let result = agent.execute("https://example.com/electronics", None).await?;
//!
//!     if result.success {
//!         println!("Agent successfully completed the objective!");
//!         println!("{}", serde_json::to_string_pretty(&result.extracted_data)?);
//!     } else {
//!         println!("Agent failed: {}", result.error_message.unwrap_or_default());
//!     }
//!
//!     Ok(())
//! }
//! ```

// Standard library imports
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::{Arc, RwLock};

// External crate imports
use lazy_static::lazy_static;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::Mutex;
use url::Url;

// Module declarations
pub mod agent;
pub mod api;
pub mod browser;
pub mod chunking;
pub mod dom;
pub mod enterprise;
pub mod error;
pub mod execution;
pub mod extension;
pub mod extraction;
pub mod forms;
pub mod instruction;
pub mod interactive;
pub mod llm;
pub mod monitoring;
pub mod parallel;
pub mod pipeline;
pub mod reconciliation;
pub mod schema;
pub mod storage;
pub mod utils;
pub mod visual;

// Re-exports for convenience
pub use agent::agent::{AgentConfig, AgentConfigBuilder, AgentResult, AutonomousAgent};
pub use agent::navigation::{NavigationStrategy, NavigationStrategyBuilder};
pub use api::rest::RestApiClient;
pub use browser::driver::{BrowserDriver, BrowserOptions, BrowserType};
pub use browser::session::{Session, SessionConfig};
pub use chunking::chunk::{Chunk, ChunkTree, EnhancedChunk};
pub use chunking::engine::ChunkingEngine;
pub use chunking::strategies::ChunkingStrategy;
pub use dom::analyzer::DomAnalyzer;
pub use dom::preprocessor::DomPreprocessor;
pub use error::{DOMorpherError, Result};
pub use execution::engine::{AdaptationLevel, ExecutionEngine};
pub use extraction::extractor::{
    Extractor, ExtractorConfig, ExtractorConfigBuilder, extract, extract_from_url,
};
pub use extraction::template::{Template, TemplateBuilder};
pub use llm::provider::{LlmClient, LlmProvider, LlmProviderManager};
pub use pipeline::pipeline::{Pipeline, PipelineInput, PipelineOutput, PipelineStep};
pub use schema::validator::{Schema, ValidationMode, Validator};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
pub const REPOSITORY: &str = "https://github.com/domorpher/domorpher";

// Global configuration
lazy_static! {
    static ref GLOBAL_CONFIG: Arc<RwLock<GlobalConfig>> =
        Arc::new(RwLock::new(GlobalConfig::default()));
    static ref LLM_MANAGER: Arc<Mutex<LlmProviderManager>> =
        Arc::new(Mutex::new(LlmProviderManager::new()));
    static ref CACHE_MANAGER: Arc<RwLock<storage::cache::CacheManager>> =
        Arc::new(RwLock::new(storage::cache::CacheManager::new()));
}

/// Global configuration for DOMorpher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// LLM provider settings
    pub llm: LlmConfig,

    /// Extraction settings
    pub extraction: ExtractionConfig,

    /// Browser settings
    pub browser: BrowserConfig,

    /// HTTP client settings
    pub http: HttpConfig,

    /// JavaScript execution settings
    pub javascript: JavaScriptConfig,

    /// Cache settings
    pub cache: CacheConfig,

    /// Rate limiting settings
    pub rate_limiting: RateLimitingConfig,

    /// Logging settings
    pub logging: LoggingConfig,

    /// Security settings
    pub security: SecurityConfig,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            extraction: ExtractionConfig::default(),
            browser: BrowserConfig::default(),
            http: HttpConfig::default(),
            javascript: JavaScriptConfig::default(),
            cache: CacheConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
            logging: LoggingConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider-specific configurations
    pub providers: HashMap<String, LlmProviderConfig>,

    /// Default provider to use
    pub default_provider: String,

    /// Default timeout for LLM requests in seconds
    pub timeout: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            LlmProviderConfig {
                api_key: None,
                default_model: "claude-3-sonnet".to_string(),
                timeout: 30,
                extra_params: HashMap::new(),
            },
        );
        providers.insert(
            "openai".to_string(),
            LlmProviderConfig {
                api_key: None,
                default_model: "gpt-4o".to_string(),
                timeout: 30,
                extra_params: HashMap::new(),
            },
        );

        Self {
            providers,
            default_provider: "anthropic".to_string(),
            timeout: 30,
        }
    }
}

/// Configuration for specific LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// API key for the provider
    pub api_key: Option<String>,

    /// Default model to use
    pub default_model: String,

    /// Timeout for this provider in seconds
    pub timeout: u64,

    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, Value>,
}

/// Extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Chunking configuration
    pub chunking: ChunkingConfig,

    /// Adaptation configuration
    pub adaptation: AdaptationConfig,

    /// Validation configuration
    pub validation: ValidationConfig,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            chunking: ChunkingConfig::default(),
            adaptation: AdaptationConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Chunking strategy to use
    pub strategy: String,

    /// Maximum chunk size in tokens
    pub max_chunk_size: usize,

    /// Overlap between chunks in tokens
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: "semantic".to_string(),
            max_chunk_size: 8000,
            overlap: 500,
        }
    }
}

/// Adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Adaptation level
    pub level: String,

    /// Maximum retries for failed extractions
    pub max_retries: usize,

    /// Confidence threshold for extraction
    pub confidence_threshold: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            level: "standard".to_string(),
            max_retries: 3,
            confidence_threshold: 0.7,
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation mode
    pub mode: String,

    /// Whether to fail on validation errors
    pub fail_on_error: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            mode: "lenient".to_string(),
            fail_on_error: false,
        }
    }
}

/// Browser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    /// Browser type
    pub browser_type: String,

    /// Whether to run in headless mode
    pub headless: bool,

    /// User agent string
    pub user_agent: String,

    /// Viewport settings
    pub viewport: ViewportConfig,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            browser_type: "chromium".to_string(),
            headless: true,
            user_agent: format!(
                "DOMorpher/{} (+https://github.com/domorpher/domorpher)",
                VERSION
            ),
            viewport: ViewportConfig::default(),
        }
    }
}

/// Viewport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportConfig {
    /// Viewport width
    pub width: u32,

    /// Viewport height
    pub height: u32,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 800,
        }
    }
}

/// HTTP client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Request timeout in seconds
    pub timeout: u64,

    /// Maximum number of redirects to follow
    pub max_redirects: u32,

    /// Number of retry attempts
    pub retry_attempts: u32,

    /// Delay between retries in milliseconds
    pub retry_delay: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout: 30,
            max_redirects: 5,
            retry_attempts: 3,
            retry_delay: 500,
        }
    }
}

/// JavaScript execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JavaScriptConfig {
    /// Whether JavaScript execution is enabled
    pub enabled: bool,

    /// Timeout for JavaScript execution in seconds
    pub timeout: u64,

    /// When to consider navigation complete
    pub wait_until: String,
}

impl Default for JavaScriptConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: 30,
            wait_until: "networkidle".to_string(),
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Caching strategy
    pub strategy: String,

    /// Time-to-live in seconds
    pub ttl: u64,

    /// Maximum cache size in MB
    pub max_size: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            strategy: "memory".to_string(),
            ttl: 3600,
            max_size: 100,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Whether rate limiting is enabled
    pub enabled: bool,

    /// Requests per minute
    pub rpm: u32,

    /// Maximum concurrent requests
    pub concurrent_requests: u32,
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rpm: 10,
            concurrent_requests: 2,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,

    /// Log file path
    pub file: Option<String>,

    /// Whether to log to console
    pub console: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: None,
            console: true,
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Whether to filter PII
    pub filter_pii: bool,

    /// Whether to enable safe browsing checks
    pub safe_browsing: bool,

    /// Whether to prevent cross-site requests
    pub prevent_cross_site: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            filter_pii: true,
            safe_browsing: true,
            prevent_cross_site: false,
        }
    }
}

/// Initialize the library with the given configuration
///
/// This function should be called at the start of your application to ensure all
/// components are properly configured.
///
/// # Arguments
///
/// * `config_path` - Optional path to a configuration file
/// * `env_prefix` - Optional prefix for environment variables (default: "DOMORPHER")
///
/// # Returns
///
/// * `Result<(), DOMorpherError>` - Ok if initialization was successful
pub fn init(config_path: Option<&Path>, env_prefix: Option<&str>) -> Result<()> {
    // Initialize logging
    initialize_logging()?;

    // Load configuration
    load_configuration(config_path, env_prefix)?;

    // Initialize LLM providers
    initialize_llm_providers()?;

    // Initialize cache
    initialize_cache()?;

    info!("DOMorpher {} initialized successfully", VERSION);
    Ok(())
}

/// Configure DOMorpher with API keys
///
/// This is a simplified configuration function that just sets up the necessary API keys.
///
/// # Arguments
///
/// * `anthropic_api_key` - Optional Anthropic API key
/// * `openai_api_key` - Optional OpenAI API key
/// * `custom_api_key` - Optional custom LLM provider API key
///
/// # Returns
///
/// * `Result<(), DOMorpherError>` - Ok if configuration was successful
pub fn configure(
    anthropic_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    custom_api_key: Option<&str>,
) -> Result<()> {
    let mut config = GLOBAL_CONFIG.write().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Update Anthropic API key if provided
    if let Some(key) = anthropic_api_key {
        if let Some(provider_config) = config.llm.providers.get_mut("anthropic") {
            provider_config.api_key = Some(key.to_string());
        } else {
            config.llm.providers.insert(
                "anthropic".to_string(),
                LlmProviderConfig {
                    api_key: Some(key.to_string()),
                    default_model: "claude-3-sonnet".to_string(),
                    timeout: 30,
                    extra_params: HashMap::new(),
                },
            );
        }
    }

    // Update OpenAI API key if provided
    if let Some(key) = openai_api_key {
        if let Some(provider_config) = config.llm.providers.get_mut("openai") {
            provider_config.api_key = Some(key.to_string());
        } else {
            config.llm.providers.insert(
                "openai".to_string(),
                LlmProviderConfig {
                    api_key: Some(key.to_string()),
                    default_model: "gpt-4o".to_string(),
                    timeout: 30,
                    extra_params: HashMap::new(),
                },
            );
        }
    }

    // Update custom API key if provided
    if let Some(key) = custom_api_key {
        if let Some(provider_config) = config.llm.providers.get_mut("custom") {
            provider_config.api_key = Some(key.to_string());
        } else {
            config.llm.providers.insert(
                "custom".to_string(),
                LlmProviderConfig {
                    api_key: Some(key.to_string()),
                    default_model: "custom-model".to_string(),
                    timeout: 30,
                    extra_params: HashMap::new(),
                },
            );
        }
    }

    // Reinitialize LLM providers with the new configuration
    drop(config); // Release the write lock
    initialize_llm_providers()?;

    info!("DOMorpher configured successfully");
    Ok(())
}

/// Configure DOMorpher with a complete configuration object
///
/// # Arguments
///
/// * `config` - The complete configuration
///
/// # Returns
///
/// * `Result<(), DOMorpherError>` - Ok if configuration was successful
pub fn configure_with(config: GlobalConfig) -> Result<()> {
    let mut global_config = GLOBAL_CONFIG.write().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Update the global configuration
    *global_config = config;

    // Release the lock before reinitializing
    drop(global_config);

    // Reinitialize with the new configuration
    initialize_llm_providers()?;
    initialize_cache()?;

    info!("DOMorpher configured successfully with custom configuration");
    Ok(())
}

/// Configure DOMorpher from a JSON file
///
/// # Arguments
///
/// * `path` - Path to the configuration file
///
/// # Returns
///
/// * `Result<(), DOMorpherError>` - Ok if configuration was successful
pub fn configure_from_file(path: &Path) -> Result<()> {
    // Read the file
    let file_content = std::fs::read_to_string(path).map_err(|e| {
        DOMorpherError::ConfigurationError(format!("Failed to read config file: {}", e))
    })?;

    // Parse the JSON
    let config: GlobalConfig = serde_json::from_str(&file_content).map_err(|e| {
        DOMorpherError::ConfigurationError(format!("Failed to parse config file: {}", e))
    })?;

    // Apply the configuration
    configure_with(config)
}

/// Get the current global configuration
///
/// # Returns
///
/// * `Result<GlobalConfig, DOMorpherError>` - The current configuration
pub fn get_config() -> Result<GlobalConfig> {
    let config = GLOBAL_CONFIG.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;
    Ok(config.clone())
}

/// Register a custom LLM provider
///
/// # Arguments
///
/// * `provider_name` - Name of the provider
/// * `provider` - The LLM provider implementation
///
/// # Returns
///
/// * `Result<(), DOMorpherError>` - Ok if registration was successful
pub async fn register_llm_provider(
    provider_name: &str,
    provider: Box<dyn llm::provider::LlmProvider>,
) -> Result<()> {
    let mut manager = LLM_MANAGER.lock().await;
    manager.register_provider(provider_name.to_string(), provider);
    Ok(())
}

/// Get the LLM provider manager
///
/// # Returns
///
/// * `Arc<Mutex<LlmProviderManager>>` - The LLM provider manager
pub fn get_llm_manager() -> Arc<Mutex<LlmProviderManager>> {
    LLM_MANAGER.clone()
}

/// Internal function to initialize logging
fn initialize_logging() -> Result<()> {
    let config = GLOBAL_CONFIG.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Set up logging based on configuration
    let log_level = match config.logging.level.to_lowercase().as_str() {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "info" => log::LevelFilter::Info,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };

    // Set up logging to specified destinations
    let mut builder = env_logger::Builder::new();
    builder.filter_level(log_level);

    // Log to console if enabled
    if config.logging.console {
        builder.target(env_logger::Target::Stdout);
    }

    // Log to file if specified
    if let Some(file_path) = &config.logging.file {
        let file = std::fs::File::create(file_path).map_err(|e| {
            DOMorpherError::LoggingError(format!("Failed to create log file: {}", e))
        })?;

        builder.target(env_logger::Target::Pipe(Box::new(file)));
    }

    // Initialize the logger
    builder
        .try_init()
        .map_err(|e| DOMorpherError::LoggingError(format!("Failed to initialize logger: {}", e)))?;

    debug!("Logging initialized at level: {}", config.logging.level);
    Ok(())
}

/// Internal function to load configuration
fn load_configuration(config_path: Option<&Path>, env_prefix: Option<&str>) -> Result<()> {
    let prefix = env_prefix.unwrap_or("DOMORPHER");

    // Start with default configuration
    let mut config = GlobalConfig::default();

    // Load from file if specified
    if let Some(path) = config_path {
        if path.exists() {
            let file_content = std::fs::read_to_string(path).map_err(|e| {
                DOMorpherError::ConfigurationError(format!("Failed to read config file: {}", e))
            })?;

            config = serde_json::from_str(&file_content).map_err(|e| {
                DOMorpherError::ConfigurationError(format!("Failed to parse config file: {}", e))
            })?;
        } else {
            warn!("Configuration file not found: {:?}", path);
        }
    }

    // Override with environment variables
    override_config_from_env(&mut config, prefix);

    // Update global configuration
    let mut global_config = GLOBAL_CONFIG.write().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;
    *global_config = config;

    debug!("Configuration loaded successfully");
    Ok(())
}

/// Internal function to override configuration from environment variables
fn override_config_from_env(config: &mut GlobalConfig, prefix: &str) {
    // Helper to check and set environment variables
    let get_env = |name: &str| -> Option<String> {
        let env_name = format!("{}_{}", prefix, name).to_uppercase();
        env::var(&env_name).ok()
    };

    // LLM Provider API keys
    if let Some(key) = get_env("ANTHROPIC_API_KEY") {
        if let Some(provider) = config.llm.providers.get_mut("anthropic") {
            provider.api_key = Some(key);
        }
    }

    if let Some(key) = get_env("OPENAI_API_KEY") {
        if let Some(provider) = config.llm.providers.get_mut("openai") {
            provider.api_key = Some(key);
        }
    }

    // Default provider
    if let Some(provider) = get_env("DEFAULT_PROVIDER") {
        config.llm.default_provider = provider;
    }

    // Chunking strategy
    if let Some(strategy) = get_env("CHUNKING_STRATEGY") {
        config.extraction.chunking.strategy = strategy;
    }

    // Adaptation level
    if let Some(level) = get_env("ADAPTATION_LEVEL") {
        config.extraction.adaptation.level = level;
    }

    // Browser type
    if let Some(browser) = get_env("BROWSER_TYPE") {
        config.browser.browser_type = browser;
    }

    // Headless mode
    if let Some(headless) = get_env("BROWSER_HEADLESS") {
        config.browser.headless = headless.to_lowercase() == "true";
    }

    // JavaScript execution
    if let Some(js) = get_env("JAVASCRIPT_ENABLED") {
        config.javascript.enabled = js.to_lowercase() == "true";
    }

    // Cache strategy
    if let Some(strategy) = get_env("CACHE_STRATEGY") {
        config.cache.strategy = strategy;
    }

    // Rate limiting
    if let Some(enabled) = get_env("RATE_LIMIT_ENABLED") {
        config.rate_limiting.enabled = enabled.to_lowercase() == "true";
    }

    if let Some(rpm) = get_env("RATE_LIMIT_RPM") {
        if let Ok(val) = rpm.parse::<u32>() {
            config.rate_limiting.rpm = val;
        }
    }

    // Logging level
    if let Some(level) = get_env("LOG_LEVEL") {
        config.logging.level = level;
    }
}

/// Internal function to initialize LLM providers
fn initialize_llm_providers() -> Result<()> {
    let config = GLOBAL_CONFIG.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Create the LLM provider manager
    let mut manager = llm::provider::LlmProviderManager::new();

    // Initialize each configured provider
    for (provider_name, provider_config) in &config.llm.providers {
        if let Some(api_key) = &provider_config.api_key {
            match provider_name.as_str() {
                "anthropic" => {
                    let provider = llm::anthropic::AnthropicProvider::new(
                        api_key,
                        &provider_config.default_model,
                        provider_config.timeout,
                    );
                    manager.register_provider(provider_name.clone(), Box::new(provider));
                }
                "openai" => {
                    let provider = llm::openai::OpenAiProvider::new(
                        api_key,
                        &provider_config.default_model,
                        provider_config.timeout,
                    );
                    manager.register_provider(provider_name.clone(), Box::new(provider));
                }
                "local" => {
                    if let Some(model_path) = provider_config.extra_params.get("model_path") {
                        let model_path = model_path.as_str().unwrap_or("");
                        let provider = llm::local::LocalModelProvider::new(
                            model_path,
                            &provider_config.default_model,
                            provider_config.timeout,
                        );
                        manager.register_provider(provider_name.clone(), Box::new(provider));
                    }
                }
                _ => {
                    warn!("Unknown LLM provider: {}", provider_name);
                }
            }
        } else {
            debug!(
                "Skipping LLM provider '{}' - no API key configured",
                provider_name
            );
        }
    }

    // Set default provider
    if !manager.has_provider(&config.llm.default_provider) && manager.get_providers().len() > 0 {
        // Use the first available provider as default if configured default is not available
        let first_provider = manager.get_providers()[0].clone();
        warn!(
            "Default provider '{}' not configured, using '{}' as default",
            config.llm.default_provider, first_provider
        );
        manager.set_default_provider(first_provider);
    } else if manager.get_providers().len() > 0 {
        manager.set_default_provider(config.llm.default_provider.clone());
    } else {
        warn!("No LLM providers configured");
    }

    // Update the global LLM manager
    let mut global_manager = LLM_MANAGER.try_lock().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire LLM manager lock".to_string())
    })?;
    *global_manager = manager;

    debug!("LLM providers initialized");
    Ok(())
}

/// Internal function to initialize cache
fn initialize_cache() -> Result<()> {
    let config = GLOBAL_CONFIG.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Create cache manager based on configuration
    let mut cache_manager = storage::cache::CacheManager::new();

    match config.cache.strategy.as_str() {
        "memory" => {
            cache_manager.set_strategy(storage::cache::CacheStrategy::Memory);
            cache_manager.configure_memory_cache(config.cache.max_size, config.cache.ttl);
        }
        "disk" => {
            cache_manager.set_strategy(storage::cache::CacheStrategy::Disk);
            cache_manager.configure_disk_cache(config.cache.max_size, config.cache.ttl);
        }
        "none" => {
            cache_manager.set_strategy(storage::cache::CacheStrategy::None);
        }
        _ => {
            warn!(
                "Unknown cache strategy: {}, using memory cache",
                config.cache.strategy
            );
            cache_manager.set_strategy(storage::cache::CacheStrategy::Memory);
            cache_manager.configure_memory_cache(config.cache.max_size, config.cache.ttl);
        }
    }

    // Update the global cache manager
    let mut global_cache = CACHE_MANAGER.write().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire cache manager lock".to_string())
    })?;
    *global_cache = cache_manager;

    debug!("Cache initialized with strategy: {}", config.cache.strategy);
    Ok(())
}

/// Get the library version
///
/// # Returns
///
/// * `&str` - The version string
pub fn version() -> &'static str {
    VERSION
}

/// Check if the library is properly configured
///
/// # Returns
///
/// * `Result<bool, DOMorpherError>` - Whether the library is configured
pub fn is_configured() -> Result<bool> {
    let config = GLOBAL_CONFIG.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire config lock".to_string())
    })?;

    // Check if at least one LLM provider is configured
    let has_provider = config
        .llm
        .providers
        .iter()
        .any(|(_, provider)| provider.api_key.is_some());

    Ok(has_provider)
}

/// Get cache statistics
///
/// # Returns
///
/// * `Result<CacheStats, DOMorpherError>` - Cache statistics
pub fn get_cache_stats() -> Result<storage::cache::CacheStats> {
    let cache_manager = CACHE_MANAGER.read().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire cache manager lock".to_string())
    })?;
    Ok(cache_manager.get_stats())
}

/// Clear the cache
///
/// # Arguments
///
/// * `cache_type` - Optional cache type to clear (defaults to all)
///
/// # Returns
///
/// * `Result<usize, DOMorpherError>` - Number of items cleared
pub fn clear_cache(cache_type: Option<storage::cache::CacheType>) -> Result<usize> {
    let mut cache_manager = CACHE_MANAGER.write().map_err(|_| {
        DOMorpherError::ConfigurationError("Failed to acquire cache manager lock".to_string())
    })?;
    Ok(cache_manager.clear(cache_type))
}
