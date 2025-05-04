//! # DOMorpher CLI
//!
//! Command-line interface for the DOMorpher web automation and data extraction framework.
//!
//! This binary provides access to DOMorpher's functionality through a user-friendly CLI.

use clap::{Parser, Subcommand};
use colored::Colorize;
use dialoguer::{Confirm, Input, Select};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tokio::io::{self, AsyncWriteExt};

use domorpher::{
    AdaptationLevel, AgentConfig, AgentConfigBuilder, AutonomousAgent, BrowserType,
    ChunkingStrategy, ExtractorConfig, ExtractorConfigBuilder, LlmProvider, NavigationStrategy,
    Pipeline, PipelineStep, Template, VERSION, clear_cache, configure, configure_from_file,
    extract_from_url, get_cache_stats,
};

/// DOMorpher: Intelligent DOM Traversal and Web Extraction with LLM Integration
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Cache directory
    #[arg(long, value_name = "DIRECTORY")]
    cache: Option<PathBuf>,

    /// Set rate limit (requests per minute)
    #[arg(long, value_name = "LIMIT")]
    rate_limit: Option<u32>,

    /// Set request timeout in seconds
    #[arg(long, value_name = "SECONDS")]
    timeout: Option<u64>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract data from HTML or URLs
    Extract {
        /// URL or path to HTML file
        source: String,

        /// Extraction instruction
        #[arg(short, long, value_name = "TEXT")]
        instruction: Option<String>,

        /// Read instruction from file
        #[arg(long, value_name = "FILE")]
        instruction_file: Option<PathBuf>,

        /// Output file (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Output format
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,

        /// Schema file for validation
        #[arg(long, value_name = "FILE")]
        schema: Option<PathBuf>,

        /// Enable JavaScript rendering
        #[arg(long)]
        javascript: bool,

        /// Wait for element to appear
        #[arg(long, value_name = "SELECTOR")]
        wait: Option<String>,

        /// LLM provider to use
        #[arg(long, value_name = "PROVIDER")]
        provider: Option<String>,

        /// LLM model to use
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Chunking strategy
        #[arg(long, value_name = "STRATEGY")]
        chunking: Option<String>,

        /// Adaptation level
        #[arg(long, value_name = "LEVEL")]
        adaptation: Option<String>,
    },

    /// Execute an autonomous agent with a goal
    Execute {
        /// Starting URL for the agent
        url: String,

        /// Agent objective
        #[arg(short, long, value_name = "TEXT")]
        objective: Option<String>,

        /// Read objective from file
        #[arg(long, value_name = "FILE")]
        objective_file: Option<PathBuf>,

        /// Output file (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Output format
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,

        /// Run in headless mode
        #[arg(long)]
        headless: bool,

        /// Browser to use
        #[arg(long, value_name = "BROWSER", default_value = "chromium")]
        browser: String,

        /// Maximum number of actions
        #[arg(long, value_name = "NUMBER")]
        max_actions: Option<u32>,

        /// Execution timeout in seconds
        #[arg(long, value_name = "SECONDS")]
        timeout: Option<u64>,

        /// LLM provider to use
        #[arg(long, value_name = "PROVIDER")]
        provider: Option<String>,

        /// LLM model to use
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Navigation strategy
        #[arg(long, value_name = "STRATEGY")]
        strategy: Option<String>,

        /// Enable visual enhancement
        #[arg(short, long)]
        visual: bool,
    },

    /// Process multiple inputs in batch mode
    Batch {
        #[command(subcommand)]
        batch_command: BatchCommands,
    },

    /// Create and manage extraction templates
    Template {
        #[command(subcommand)]
        template_command: TemplateCommands,
    },

    /// Validate extraction results against schemas
    Validate {
        /// Data file to validate
        data_file: PathBuf,

        /// Schema file
        #[arg(short, long, value_name = "FILE")]
        schema: PathBuf,

        /// Validation mode (strict, lenient)
        #[arg(short, long, value_name = "MODE", default_value = "strict")]
        mode: String,
    },

    /// Configure DOMorpher settings
    Configure {
        /// Create a new configuration file
        #[arg(short, long)]
        interactive: bool,

        /// Test the configuration
        #[arg(short, long)]
        test: bool,

        /// Show current configuration
        #[arg(short, long)]
        show: bool,

        /// Clear cache
        #[arg(long)]
        clear_cache: bool,

        /// Show cache statistics
        #[arg(long)]
        cache_stats: bool,
    },
}

#[derive(Subcommand)]
enum BatchCommands {
    /// Batch extraction from multiple sources
    Extract {
        /// File with URLs or paths (one per line)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Extraction instruction
        #[arg(short, long, value_name = "TEXT")]
        instruction: Option<String>,

        /// Read instruction from file
        #[arg(long, value_name = "FILE")]
        instruction_file: Option<PathBuf>,

        /// Output directory
        #[arg(short, long, value_name = "DIRECTORY")]
        output_dir: PathBuf,

        /// Output format
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,

        /// Schema file for validation
        #[arg(long, value_name = "FILE")]
        schema: Option<PathBuf>,

        /// Number of concurrent processes
        #[arg(long, value_name = "NUMBER", default_value = "1")]
        concurrency: usize,

        /// Delay between requests in milliseconds
        #[arg(long, value_name = "MILLISECONDS", default_value = "0")]
        delay: u64,
    },

    /// Batch execution on multiple URLs
    Execute {
        /// File with URLs (one per line)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,

        /// Agent objective
        #[arg(short, long, value_name = "TEXT")]
        objective: Option<String>,

        /// Read objective from file
        #[arg(long, value_name = "FILE")]
        objective_file: Option<PathBuf>,

        /// Output directory
        #[arg(short, long, value_name = "DIRECTORY")]
        output_dir: PathBuf,

        /// Output format
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,

        /// Number of concurrent processes
        #[arg(long, value_name = "NUMBER", default_value = "1")]
        concurrency: usize,

        /// Delay between requests in milliseconds
        #[arg(long, value_name = "MILLISECONDS", default_value = "0")]
        delay: u64,
    },
}

#[derive(Subcommand)]
enum TemplateCommands {
    /// Create a new template
    Create {
        /// Template name
        name: String,

        /// Extraction instruction
        #[arg(short, long, value_name = "TEXT")]
        instruction: Option<String>,

        /// Read instruction from file
        #[arg(long, value_name = "FILE")]
        instruction_file: Option<PathBuf>,

        /// Schema file
        #[arg(short, long, value_name = "FILE")]
        schema: Option<PathBuf>,

        /// Output file for the template
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
    },

    /// Use an existing template
    Use {
        /// Template file to use
        template_file: PathBuf,

        /// URL or HTML file to extract from
        source: String,

        /// Output file (defaults to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Output format
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,
    },

    /// List available templates
    List {
        /// Templates directory
        #[arg(short, long, value_name = "DIRECTORY")]
        directory: Option<PathBuf>,
    },

    /// Export a template to file
    Export {
        /// Template file to export
        template_file: PathBuf,

        /// Output format (json, yaml)
        #[arg(short, long, value_name = "FORMAT", default_value = "json")]
        format: String,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
    },

    /// Import a template from file
    Import {
        /// Template file to import
        template_file: PathBuf,

        /// Template name
        #[arg(short, long, value_name = "NAME")]
        name: String,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let cli = Cli::parse();

    // Set up logging
    setup_logging(cli.verbose)?;

    // Load configuration
    if let Some(config_path) = &cli.config {
        configure_from_file(config_path)?;
        debug!("Loaded configuration from {:?}", config_path);
    } else {
        // Try to load from default locations
        let default_paths = [
            PathBuf::from("./domorpher.json"),
            PathBuf::from("./domorpher.yaml"),
            PathBuf::from("./domorpher.yml"),
            dirs::config_dir()
                .map(|p| p.join("domorpher/config.json"))
                .unwrap_or_default(),
        ];

        for path in default_paths {
            if path.exists() {
                configure_from_file(&path)?;
                debug!("Loaded configuration from {:?}", path);
                break;
            }
        }
    }

    // Apply command line overrides
    if let Some(cache_dir) = &cli.cache {
        // Set cache directory
        std::env::set_var(
            "DOMORPHER_CACHE_DIRECTORY",
            cache_dir.to_string_lossy().to_string(),
        );
    }

    if let Some(rate_limit) = cli.rate_limit {
        // Set rate limit
        std::env::set_var("DOMORPHER_RATE_LIMIT_RPM", rate_limit.to_string());
    }

    if let Some(timeout) = cli.timeout {
        // Set timeout
        std::env::set_var("DOMORPHER_TIMEOUT", timeout.to_string());
    }

    // Process commands
    match &cli.command {
        Commands::Extract {
            source,
            instruction,
            instruction_file,
            output,
            format,
            schema,
            javascript,
            wait,
            provider,
            model,
            chunking,
            adaptation,
        } => {
            // Get instruction from file or argument
            let instruction_text =
                get_text_from_source_or_file(instruction, instruction_file).await?;

            // Get schema if specified
            let schema_json = if let Some(schema_path) = schema {
                Some(read_json_file(schema_path).await?)
            } else {
                None
            };

            // Build extractor configuration
            let config_builder = ExtractorConfigBuilder::new();
            let config_builder = if let Some(p) = provider {
                config_builder.llm_provider(parse_llm_provider(p))
            } else {
                config_builder
            };

            let config_builder = if let Some(m) = model {
                config_builder.model(m)
            } else {
                config_builder
            };

            let config_builder = if let Some(c) = chunking {
                config_builder.chunking_strategy(parse_chunking_strategy(c))
            } else {
                config_builder
            };

            let config_builder = if let Some(a) = adaptation {
                config_builder.adaptation_level(parse_adaptation_level(a))
            } else {
                config_builder
            };

            let config_builder = config_builder
                .javascript_support(*javascript)
                .schema(schema_json);

            // Add wait selector if specified
            let config_builder = if let Some(w) = wait {
                config_builder.wait_for_selector(w)
            } else {
                config_builder
            };

            let config = config_builder.build();

            // Create progress bar
            let progress_bar = ProgressBar::new_spinner();
            progress_bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap(),
            );
            progress_bar.set_message("Extracting data...");
            progress_bar.enable_steady_tick(std::time::Duration::from_millis(80));

            // Start extraction
            let start_time = Instant::now();
            let result = if source.starts_with("http://") || source.starts_with("https://") {
                // Extract from URL
                progress_bar.set_message(format!("Extracting from URL: {}...", source));
                extract_from_url(source, &instruction_text, Some(config)).await?
            } else {
                // Extract from file
                progress_bar.set_message(format!("Reading file: {}...", source));
                let html = fs::read_to_string(source)?;
                progress_bar.set_message("Extracting from HTML content...");
                domorpher::extract(html, &instruction_text, Some(config)).await?
            };
            let elapsed = start_time.elapsed();

            // Finish progress
            progress_bar.finish_with_message(format!("Extraction completed in {:.2?}", elapsed));

            // Format and output results
            output_result(&result, output, format).await?;
        }

        Commands::Execute {
            url,
            objective,
            objective_file,
            output,
            format,
            headless,
            browser,
            max_actions,
            timeout,
            provider,
            model,
            strategy,
            visual,
        } => {
            // Get objective from file or argument
            let objective_text = get_text_from_source_or_file(objective, objective_file).await?;

            // Build agent configuration
            let config_builder = AgentConfigBuilder::new()
                .objective(&objective_text)
                .browser_options(BrowserOptions {
                    browser_type: parse_browser_type(browser),
                    headless: *headless,
                    ..Default::default()
                });

            let config_builder = if let Some(p) = provider {
                config_builder.llm_provider(parse_llm_provider(p))
            } else {
                config_builder
            };

            let config_builder = if let Some(m) = model {
                config_builder.model(m)
            } else {
                config_builder
            };

            let config_builder = if let Some(s) = strategy {
                config_builder.navigation_strategy(parse_navigation_strategy(s))
            } else {
                config_builder
            };

            let config_builder = if let Some(ma) = max_actions {
                config_builder.max_actions(*ma)
            } else {
                config_builder
            };

            let config_builder = if let Some(t) = timeout {
                config_builder.timeout_seconds(*t)
            } else {
                config_builder
            };

            let config_builder = if *visual {
                config_builder.multimodal_enhancement(json!({
                    "enabled": true,
                    "trigger": "on_dom_ambiguity",
                    "visual_context_level": "minimal"
                }))
            } else {
                config_builder
            };

            let config = config_builder.build();

            // Create agent
            let agent = AutonomousAgent::new(config);

            // Create progress bar
            let progress_bar = ProgressBar::new_spinner();
            progress_bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap(),
            );
            progress_bar.set_message(format!("Executing agent on {}...", url));
            progress_bar.enable_steady_tick(std::time::Duration::from_millis(80));

            // Execute agent
            let start_time = Instant::now();
            let result = agent.execute(url, None).await?;
            let elapsed = start_time.elapsed();

            // Finish progress
            if result.success {
                progress_bar.finish_with_message(format!(
                    "Agent execution completed successfully in {:.2?}",
                    elapsed
                ));
            } else {
                progress_bar.finish_with_message(format!(
                    "Agent execution failed in {:.2?}: {}",
                    elapsed,
                    result.error_message.unwrap_or_default()
                ));
            }

            // Format and output results
            output_result(&result, output, format).await?;
        }

        Commands::Batch { batch_command } => {
            match batch_command {
                BatchCommands::Extract {
                    input,
                    instruction,
                    instruction_file,
                    output_dir,
                    format,
                    schema,
                    concurrency,
                    delay,
                } => {
                    // Get instruction from file or argument
                    let instruction_text =
                        get_text_from_source_or_file(instruction, instruction_file).await?;

                    // Get schema if specified
                    let schema_json = if let Some(schema_path) = schema {
                        Some(read_json_file(schema_path).await?)
                    } else {
                        None
                    };

                    // Read input file
                    let input_content = fs::read_to_string(input)?;
                    let urls: Vec<String> = input_content
                        .lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .collect();

                    // Create output directory if it doesn't exist
                    fs::create_dir_all(output_dir)?;

                    // Setup progress bar
                    let progress_bar = ProgressBar::new(urls.len() as u64);
                    progress_bar.set_style(
                        ProgressStyle::default_bar()
                            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                            .unwrap()
                            .progress_chars("##-"),
                    );

                    // Setup thread pool
                    let pool = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(*concurrency)
                        .build()?;

                    // Process each URL
                    let mut tasks = Vec::new();
                    for (idx, url) in urls.iter().enumerate() {
                        // Clone necessary values for the task
                        let url = url.clone();
                        let instruction = instruction_text.clone();
                        let output_path = output_dir.join(format!("result_{}.{}", idx, format));
                        let schema = schema_json.clone();
                        let progress = progress_bar.clone();
                        let delay_ms = *delay;

                        // Create task
                        let task = pool.spawn(async move {
                            // Add delay if specified
                            if delay_ms > 0 {
                                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms))
                                    .await;
                            }

                            // Create config
                            let config = ExtractorConfig::builder().schema(schema).build();

                            // Extract from URL
                            let result = extract_from_url(&url, &instruction, Some(config)).await;

                            // Report progress
                            match &result {
                                Ok(_) => progress.inc(1),
                                Err(e) => {
                                    progress.inc(1);
                                    progress.println(format!("Error processing {}: {}", url, e));
                                }
                            }

                            // Return result with info
                            (url, output_path, result)
                        });

                        tasks.push(task);
                    }

                    // Wait for all tasks and collect results
                    let mut success_count = 0;
                    let mut error_count = 0;

                    for task in tasks {
                        let (url, output_path, result) = task.await?;

                        match result {
                            Ok(data) => {
                                // Format and write to file
                                let output_str = match *format {
                                    "json" => serde_json::to_string_pretty(&data)?,
                                    "yaml" => serde_yaml::to_string(&data)?,
                                    _ => serde_json::to_string_pretty(&data)?,
                                };

                                fs::write(&output_path, output_str)?;
                                success_count += 1;
                            }
                            Err(e) => {
                                // Write error to file
                                let error_json = json!({
                                    "url": url,
                                    "error": e.to_string(),
                                    "timestamp": chrono::Utc::now().to_rfc3339()
                                });

                                fs::write(
                                    &output_path,
                                    serde_json::to_string_pretty(&error_json)?,
                                )?;
                                error_count += 1;
                            }
                        }
                    }

                    // Finish progress
                    progress_bar.finish_with_message(format!(
                        "Batch processing completed: {} succeeded, {} failed",
                        success_count, error_count
                    ));
                }

                BatchCommands::Execute {
                    input,
                    objective,
                    objective_file,
                    output_dir,
                    format,
                    concurrency,
                    delay,
                } => {
                    // Get objective from file or argument
                    let objective_text =
                        get_text_from_source_or_file(objective, objective_file).await?;

                    // Read input file
                    let input_content = fs::read_to_string(input)?;
                    let urls: Vec<String> = input_content
                        .lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .collect();

                    // Create output directory if it doesn't exist
                    fs::create_dir_all(output_dir)?;

                    // Setup progress bar
                    let progress_bar = ProgressBar::new(urls.len() as u64);
                    progress_bar.set_style(
                        ProgressStyle::default_bar()
                            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                            .unwrap()
                            .progress_chars("##-"),
                    );

                    // Setup semaphore for concurrency control
                    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(*concurrency));

                    // Process each URL
                    let mut tasks = Vec::new();
                    for (idx, url) in urls.iter().enumerate() {
                        // Clone necessary values for the task
                        let url = url.clone();
                        let objective = objective_text.clone();
                        let output_path = output_dir.join(format!("result_{}.{}", idx, format));
                        let progress = progress_bar.clone();
                        let delay_ms = *delay;
                        let semaphore = semaphore.clone();

                        // Create task
                        let task = tokio::spawn(async move {
                            // Acquire semaphore permit
                            let _permit = semaphore.acquire().await.unwrap();

                            // Add delay if specified
                            if delay_ms > 0 {
                                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms))
                                    .await;
                            }

                            // Create agent
                            let config = AgentConfig::builder().objective(&objective).build();

                            let agent = AutonomousAgent::new(config);

                            // Execute agent
                            let result = agent.execute(&url, None).await;

                            // Report progress
                            match &result {
                                Ok(_) => progress.inc(1),
                                Err(e) => {
                                    progress.inc(1);
                                    progress.println(format!("Error processing {}: {}", url, e));
                                }
                            }

                            // Return result with info
                            (url, output_path, result)
                        });

                        tasks.push(task);
                    }

                    // Wait for all tasks and collect results
                    let mut success_count = 0;
                    let mut error_count = 0;

                    for task in futures::future::join_all(tasks).await {
                        match task {
                            Ok((url, output_path, result)) => {
                                match result {
                                    Ok(agent_result) => {
                                        // Format and write to file
                                        let output_str = match *format {
                                            "json" => serde_json::to_string_pretty(&agent_result)?,
                                            "yaml" => serde_yaml::to_string(&agent_result)?,
                                            _ => serde_json::to_string_pretty(&agent_result)?,
                                        };

                                        fs::write(&output_path, output_str)?;

                                        if agent_result.success {
                                            success_count += 1;
                                        } else {
                                            error_count += 1;
                                        }
                                    }
                                    Err(e) => {
                                        // Write error to file
                                        let error_json = json!({
                                            "url": url,
                                            "error": e.to_string(),
                                            "timestamp": chrono::Utc::now().to_rfc3339()
                                        });

                                        fs::write(
                                            &output_path,
                                            serde_json::to_string_pretty(&error_json)?,
                                        )?;
                                        error_count += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Task panicked: {}", e);
                                error_count += 1;
                            }
                        }
                    }

                    // Finish progress
                    progress_bar.finish_with_message(format!(
                        "Batch execution completed: {} succeeded, {} failed",
                        success_count, error_count
                    ));
                }
            }
        }

        Commands::Template { template_command } => {
            match template_command {
                TemplateCommands::Create {
                    name,
                    instruction,
                    instruction_file,
                    schema,
                    output,
                } => {
                    // Get instruction from file or argument
                    let instruction_text =
                        get_text_from_source_or_file(instruction, instruction_file).await?;

                    // Get schema if specified
                    let schema_json = if let Some(schema_path) = schema {
                        Some(read_json_file(schema_path).await?)
                    } else {
                        None
                    };

                    // Create template
                    let template = Template::builder()
                        .name(name)
                        .instruction(&instruction_text)
                        .schema(schema_json)
                        .build();

                    // Save template to file
                    template.save(output)?;

                    println!("Template created and saved to {:?}", output);
                }

                TemplateCommands::Use {
                    template_file,
                    source,
                    output,
                    format,
                } => {
                    // Load template
                    let template = Template::load(template_file)?;

                    // Create progress bar
                    let progress_bar = ProgressBar::new_spinner();
                    progress_bar.set_style(
                        ProgressStyle::default_spinner()
                            .template("{spinner:.green} {msg}")
                            .unwrap(),
                    );
                    progress_bar.set_message("Extracting data using template...");
                    progress_bar.enable_steady_tick(std::time::Duration::from_millis(80));

                    // Extract using template
                    let start_time = Instant::now();
                    let result = if source.starts_with("http://") || source.starts_with("https://")
                    {
                        // Extract from URL
                        progress_bar.set_message(format!("Extracting from URL: {}...", source));
                        template.extract_from_url(&source).await?
                    } else {
                        // Extract from file
                        progress_bar.set_message(format!("Reading file: {}...", source));
                        let html = fs::read_to_string(source)?;
                        progress_bar.set_message("Extracting from HTML content...");
                        template.extract(&html).await?
                    };
                    let elapsed = start_time.elapsed();

                    // Finish progress
                    progress_bar
                        .finish_with_message(format!("Extraction completed in {:.2?}", elapsed));

                    // Format and output results
                    output_result(&result, output, format).await?;
                }

                TemplateCommands::List { directory } => {
                    // Determine templates directory
                    let templates_dir = directory.clone().unwrap_or_else(|| {
                        dirs::config_dir()
                            .map(|p| p.join("domorpher/templates"))
                            .unwrap_or_else(|| PathBuf::from("./templates"))
                    });

                    if !templates_dir.exists() {
                        println!("Templates directory does not exist: {:?}", templates_dir);
                        return Ok(());
                    }

                    // Find template files
                    let mut templates = Vec::new();
                    for entry in fs::read_dir(&templates_dir)? {
                        let entry = entry?;
                        let path = entry.path();

                        if path.is_file()
                            && path
                                .extension()
                                .map_or(false, |ext| ext == "json" || ext == "yaml" || ext == "yml")
                        {
                            // Try to load the template
                            match Template::load(&path) {
                                Ok(template) => {
                                    templates.push((
                                        path.clone(),
                                        template.name().to_string(),
                                        template.description().unwrap_or_default(),
                                    ));
                                }
                                Err(e) => {
                                    warn!("Failed to load template {:?}: {}", path, e);
                                }
                            }
                        }
                    }

                    // Display templates
                    if templates.is_empty() {
                        println!("No templates found in {:?}", templates_dir);
                    } else {
                        println!(
                            "Found {} templates in {:?}:",
                            templates.len(),
                            templates_dir
                        );
                        for (path, name, description) in templates {
                            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                            println!("  - {} ({}): {}", name.cyan(), file_name, description);
                        }
                    }
                }

                TemplateCommands::Export {
                    template_file,
                    format,
                    output,
                } => {
                    // Load template
                    let template = Template::load(template_file)?;

                    // Export to specified format
                    match format.as_str() {
                        "json" => {
                            let json_str = template.to_json()?;
                            fs::write(output, json_str)?;
                        }
                        "yaml" => {
                            let yaml_str = template.to_yaml()?;
                            fs::write(output, yaml_str)?;
                        }
                        _ => {
                            return Err(format!("Unsupported export format: {}", format).into());
                        }
                    }

                    println!("Template exported to {:?} in {} format", output, format);
                }

                TemplateCommands::Import {
                    template_file,
                    name,
                    output,
                } => {
                    // Read template file
                    let file_content = fs::read_to_string(template_file)?;

                    // Determine format based on extension
                    let extension = template_file
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .unwrap_or("json");

                    // Parse and create template
                    let template = match extension {
                        "json" => Template::from_json(&file_content)?,
                        "yaml" | "yml" => Template::from_yaml(&file_content)?,
                        _ => {
                            return Err(
                                format!("Unsupported template format: {}", extension).into()
                            );
                        }
                    };

                    // Rename template if specified
                    let template = if !name.is_empty() {
                        template.with_name(name)
                    } else {
                        template
                    };

                    // Save template
                    template.save(output)?;

                    println!("Template imported and saved to {:?}", output);
                }
            }
        }

        Commands::Validate {
            data_file,
            schema,
            mode,
        } => {
            // Read data and schema files
            let data = read_json_file(data_file).await?;
            let schema_json = read_json_file(schema).await?;

            // Parse validation mode
            let validation_mode = match mode.to_lowercase().as_str() {
                "strict" => domorpher::schema::validator::ValidationMode::Strict,
                "lenient" => domorpher::schema::validator::ValidationMode::Lenient,
                _ => domorpher::schema::validator::ValidationMode::Strict,
            };

            // Create validator
            let validator =
                domorpher::schema::validator::Validator::new(schema_json, validation_mode)?;

            // Validate data
            let validation_result = validator.validate(&data)?;

            if validation_result.is_valid {
                println!("{}", "✓ Validation successful".green());
                println!(
                    "Validated {} items with {} warnings",
                    validation_result.validated_count,
                    validation_result.warnings.len()
                );

                if !validation_result.warnings.is_empty() {
                    println!("\nWarnings:");
                    for warning in &validation_result.warnings {
                        println!("  - {}", warning.yellow());
                    }
                }
            } else {
                println!("{}", "✗ Validation failed".red());
                println!(
                    "Found {} errors and {} warnings",
                    validation_result.errors.len(),
                    validation_result.warnings.len()
                );

                if !validation_result.errors.is_empty() {
                    println!("\nErrors:");
                    for error in &validation_result.errors {
                        println!("  - {}", error.red());
                    }
                }

                if !validation_result.warnings.is_empty() {
                    println!("\nWarnings:");
                    for warning in &validation_result.warnings {
                        println!("  - {}", warning.yellow());
                    }
                }
            }
        }

        Commands::Configure {
            interactive,
            test,
            show,
            clear_cache,
            cache_stats,
        } => {
            if *interactive {
                // Interactive configuration
                let config = create_interactive_config().await?;

                // Save configuration
                let config_dir = dirs::config_dir()
                    .map(|p| p.join("domorpher"))
                    .unwrap_or_else(|| PathBuf::from("."));

                fs::create_dir_all(&config_dir)?;
                let config_path = config_dir.join("config.json");

                fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

                println!("Configuration saved to {:?}", config_path);

                // Apply configuration
                configure_with(config)?;
            }

            if *test {
                // Test configuration
                println!("Testing DOMorpher configuration...");

                // Test LLM providers
                println!("Testing LLM providers...");
                let llm_manager = domorpher::get_llm_manager();
                let mut manager = llm_manager.try_lock().unwrap();

                for provider_name in manager.get_providers() {
                    print!("  Testing {}... ", provider_name);
                    io::stdout().flush().await?;

                    match manager.get_provider(&provider_name) {
                        Some(provider) => match provider.test_connection().await {
                            Ok(_) => println!("{}", "OK".green()),
                            Err(e) => println!("{} ({})", "FAILED".red(), e),
                        },
                        None => println!("{} (provider not found)", "FAILED".red()),
                    }
                }

                // Test cache
                println!("Testing cache...");
                let cache_stats = get_cache_stats()?;
                println!(
                    "  Cache status: {}",
                    if cache_stats.enabled {
                        "Enabled".green()
                    } else {
                        "Disabled".yellow()
                    }
                );
                println!("  Cache type: {}", cache_stats.strategy);
                println!("  Items: {}", cache_stats.item_count);
                println!("  Size: {:.2} MB", cache_stats.size_mb);
            }

            if *show {
                // Show current configuration
                let config = domorpher::get_config()?;
                println!("{}", serde_json::to_string_pretty(&config)?);
            }

            if *clear_cache {
                // Clear cache
                let cleared = clear_cache(None)?;
                println!("Cleared {} cache items", cleared);
            }

            if *cache_stats {
                // Show cache statistics
                let stats = get_cache_stats()?;
                println!("Cache statistics:");
                println!(
                    "  Enabled: {}",
                    if stats.enabled {
                        "Yes".green()
                    } else {
                        "No".yellow()
                    }
                );
                println!("  Strategy: {}", stats.strategy);
                println!("  Item count: {}", stats.item_count);
                println!("  Size: {:.2} MB", stats.size_mb);
                println!("  Hits: {}", stats.hits);
                println!("  Misses: {}", stats.misses);
                println!(
                    "  Hit ratio: {:.1}%",
                    if stats.hits + stats.misses > 0 {
                        (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0
                    } else {
                        0.0
                    }
                );
            }

            // If no specific action was requested, show help
            if !*interactive && !*test && !*show && !*clear_cache && !*cache_stats {
                let mut cmd = Cli::command();
                cmd.find_subcommand_mut("configure").unwrap().print_help()?;
            }
        }
    }

    Ok(())
}

/// Setup logging based on verbosity
fn setup_logging(verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp(Some(env_logger::TimestampPrecision::Seconds))
        .format_module_path(false)
        .init();

    Ok(())
}

/// Get text from either a direct source or a file
async fn get_text_from_source_or_file(
    source: &Option<String>,
    file: &Option<PathBuf>,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(text) = source {
        Ok(text.clone())
    } else if let Some(path) = file {
        Ok(fs::read_to_string(path)?)
    } else {
        let text: String = Input::new()
            .with_prompt("Enter instruction")
            .interact_text()?;
        Ok(text)
    }
}

/// Read a JSON file
async fn read_json_file(path: &PathBuf) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let value = serde_json::from_str(&content)?;
    Ok(value)
}

/// Output result to file or stdout
async fn output_result(
    result: &impl serde::Serialize,
    output_path: &Option<PathBuf>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Format result
    let output_str = match format.to_lowercase().as_str() {
        "json" => serde_json::to_string_pretty(result)?,
        "yaml" => serde_yaml::to_string(result)?,
        "csv" => {
            // Convert to CSV (simplified implementation)
            let value = serde_json::to_value(result)?;

            // Check if it's an array
            if let serde_json::Value::Array(items) = &value {
                let mut wtr = csv::Writer::from_writer(vec![]);

                for item in items {
                    if let serde_json::Value::Object(map) = item {
                        let record: std::collections::HashMap<_, _> = map
                            .iter()
                            .map(|(k, v)| (k.clone(), v.to_string()))
                            .collect();
                        wtr.serialize(record)?;
                    }
                }

                String::from_utf8(wtr.into_inner()?)?
            } else {
                return Err("CSV output requires an array of objects".into());
            }
        }
        "xml" => {
            // Convert to XML (simplified implementation)
            let value = serde_json::to_value(result)?;
            let mut xml = String::new();
            xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

            // Check if it's an array
            if let serde_json::Value::Array(items) = &value {
                xml.push_str("<results>\n");

                for item in items {
                    if let serde_json::Value::Object(map) = item {
                        xml.push_str("  <item>\n");

                        for (key, value) in map {
                            xml.push_str(&format!("    <{}>{}</{}>\n", key, value, key));
                        }

                        xml.push_str("  </item>\n");
                    }
                }

                xml.push_str("</results>\n");
                xml
            } else if let serde_json::Value::Object(map) = &value {
                xml.push_str("<result>\n");

                for (key, value) in map {
                    xml.push_str(&format!("  <{}>{}</{}>\n", key, value, key));
                }

                xml.push_str("</result>\n");
                xml
            } else {
                return Err("XML output requires an object or array of objects".into());
            }
        }
        _ => serde_json::to_string_pretty(result)?,
    };

    // Output result
    if let Some(path) = output_path {
        fs::write(path, output_str)?;
        println!("Output written to {:?}", path);
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

/// Parse LLM provider
fn parse_llm_provider(provider: &str) -> LlmProvider {
    match provider.to_lowercase().as_str() {
        "anthropic" => LlmProvider::Anthropic,
        "openai" => LlmProvider::OpenAI,
        "local" => LlmProvider::Local,
        _ => LlmProvider::Anthropic,
    }
}

/// Parse chunking strategy
fn parse_chunking_strategy(strategy: &str) -> ChunkingStrategy {
    match strategy.to_lowercase().as_str() {
        "size" => ChunkingStrategy::Size,
        "semantic" => ChunkingStrategy::Semantic,
        "hierarchical" => ChunkingStrategy::Hierarchical,
        "adaptive" => ChunkingStrategy::Adaptive,
        _ => ChunkingStrategy::Semantic,
    }
}

/// Parse adaptation level
fn parse_adaptation_level(level: &str) -> AdaptationLevel {
    match level.to_lowercase().as_str() {
        "minimal" => AdaptationLevel::Minimal,
        "standard" => AdaptationLevel::Standard,
        "aggressive" => AdaptationLevel::Aggressive,
        _ => AdaptationLevel::Standard,
    }
}

/// Parse browser type
fn parse_browser_type(browser: &str) -> BrowserType {
    match browser.to_lowercase().as_str() {
        "chromium" => BrowserType::Chromium,
        "firefox" => BrowserType::Firefox,
        "webkit" => BrowserType::Webkit,
        _ => BrowserType::Chromium,
    }
}

/// Parse navigation strategy
fn parse_navigation_strategy(strategy: &str) -> NavigationStrategy {
    match strategy.to_lowercase().as_str() {
        "semantic_first" => NavigationStrategy::SemanticFirst,
        "structure_first" => NavigationStrategy::StructureFirst,
        "balanced" => NavigationStrategy::Balanced,
        "aggressive" => NavigationStrategy::Aggressive,
        _ => NavigationStrategy::SemanticFirst,
    }
}

/// Create interactive configuration
async fn create_interactive_config() -> Result<domorpher::GlobalConfig, Box<dyn std::error::Error>>
{
    println!("{}", "DOMorpher Interactive Configuration".cyan().bold());
    println!("This wizard will help you create a configuration file for DOMorpher.");
    println!();

    // Start with default configuration
    let mut config = domorpher::GlobalConfig::default();

    // LLM provider configuration
    println!("{}", "LLM Provider Configuration".cyan());

    // Anthropic
    let use_anthropic = Confirm::new()
        .with_prompt("Configure Anthropic (Claude) API?")
        .default(true)
        .interact()?;

    if use_anthropic {
        let api_key: String = Input::new()
            .with_prompt("Enter Anthropic API key")
            .interact_text()?;

        let model: String = Input::new()
            .with_prompt("Default Anthropic model")
            .default("claude-3-sonnet".to_string())
            .interact_text()?;

        if let Some(provider) = config.llm.providers.get_mut("anthropic") {
            provider.api_key = Some(api_key);
            provider.default_model = model;
        }
    }

    // OpenAI
    let use_openai = Confirm::new()
        .with_prompt("Configure OpenAI API?")
        .default(true)
        .interact()?;

    if use_openai {
        let api_key: String = Input::new()
            .with_prompt("Enter OpenAI API key")
            .interact_text()?;

        let model: String = Input::new()
            .with_prompt("Default OpenAI model")
            .default("gpt-4o".to_string())
            .interact_text()?;

        if let Some(provider) = config.llm.providers.get_mut("openai") {
            provider.api_key = Some(api_key);
            provider.default_model = model;
        }
    }

    // Default provider
    if use_anthropic && use_openai {
        let provider_options = vec!["anthropic", "openai"];
        let default_provider = Select::new()
            .with_prompt("Select default LLM provider")
            .default(0)
            .items(&provider_options)
            .interact()?;

        config.llm.default_provider = provider_options[default_provider].to_string();
    } else if use_anthropic {
        config.llm.default_provider = "anthropic".to_string();
    } else if use_openai {
        config.llm.default_provider = "openai".to_string();
    }

    // Browser configuration
    println!();
    println!("{}", "Browser Configuration".cyan());

    let browser_options = vec!["chromium", "firefox", "webkit"];
    let browser_type = Select::new()
        .with_prompt("Select default browser")
        .default(0)
        .items(&browser_options)
        .interact()?;

    config.browser.browser_type = browser_options[browser_type].to_string();

    config.browser.headless = Confirm::new()
        .with_prompt("Run browser in headless mode by default?")
        .default(true)
        .interact()?;

    // Cache configuration
    println!();
    println!("{}", "Cache Configuration".cyan());

    let cache_options = vec!["memory", "disk", "none"];
    let cache_strategy = Select::new()
        .with_prompt("Select cache strategy")
        .default(0)
        .items(&cache_options)
        .interact()?;

    config.cache.strategy = cache_options[cache_strategy].to_string();

    if cache_options[cache_strategy] != "none" {
        config.cache.ttl = Input::new()
            .with_prompt("Cache TTL in seconds")
            .default(3600)
            .interact()?;

        config.cache.max_size = Input::new()
            .with_prompt("Maximum cache size in MB")
            .default(100)
            .interact()?;
    }

    // Rate limiting
    println!();
    println!("{}", "Rate Limiting Configuration".cyan());

    config.rate_limiting.enabled = Confirm::new()
        .with_prompt("Enable rate limiting?")
        .default(true)
        .interact()?;

    if config.rate_limiting.enabled {
        config.rate_limiting.rpm = Input::new()
            .with_prompt("Requests per minute")
            .default(10)
            .interact()?;

        config.rate_limiting.concurrent_requests = Input::new()
            .with_prompt("Maximum concurrent requests")
            .default(2)
            .interact()?;
    }

    // JavaScript execution
    println!();
    println!("{}", "JavaScript Configuration".cyan());

    config.javascript.enabled = Confirm::new()
        .with_prompt("Enable JavaScript execution by default?")
        .default(false)
        .interact()?;

    if config.javascript.enabled {
        config.javascript.timeout = Input::new()
            .with_prompt("JavaScript execution timeout in seconds")
            .default(30)
            .interact()?;

        let wait_options = vec!["domcontentloaded", "load", "networkidle"];
        let wait_until = Select::new()
            .with_prompt("When to consider navigation complete")
            .default(2)
            .items(&wait_options)
            .interact()?;

        config.javascript.wait_until = wait_options[wait_until].to_string();
    }

    println!();
    println!("{}", "Configuration complete!".green());

    Ok(config)
}
