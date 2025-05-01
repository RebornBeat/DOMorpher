# DOMorpher Technical Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
   1. [Architectural Overview](#architectural-overview)
   2. [Component Interactions](#component-interactions)
   3. [Data Flow](#data-flow)

2. [Core Components](#core-components)
   1. [DOM Preprocessor](#dom-preprocessor)
   2. [Chunking Engine](#chunking-engine)
   3. [LLM Integration Layer](#llm-integration-layer)
   4. [Instruction Parser](#instruction-parser)
   5. [Adaptive Execution Engine](#adaptive-execution-engine)
   6. [Result Reconciliation](#result-reconciliation)
   7. [Schema Enforcement](#schema-enforcement)

3. [API Reference](#api-reference)
   1. [Python API](#python-api)
   2. [Rust API](#rust-api)
   3. [Command-Line Interface](#command-line-interface)
   4. [Configuration Options](#configuration-options)

4. [Implementation Details](#implementation-details)
   1. [DOM Parsing Strategy](#dom-parsing-strategy)
   2. [LLM Prompt Engineering](#llm-prompt-engineering)
   3. [Chunking Algorithms](#chunking-algorithms)
   4. [Context Management](#context-management)
   5. [Result Validation](#result-validation)

5. [Integration Guides](#integration-guides)
   1. [Python Integration](#python-integration)
   2. [Rust Integration](#rust-integration)
   3. [CI/CD Pipeline Integration](#cicd-pipeline-integration)
   4. [Error Handling and Logging](#error-handling-and-logging)

6. [Advanced Features](#advanced-features)
   1. [Extraction Templates](#extraction-templates)
   2. [Schema Validation](#schema-validation)
   3. [Incremental Training](#incremental-training)
   4. [JavaScript Execution](#javascript-execution)
   5. [Multi-Modal Extraction](#multi-modal-extraction)
   6. [DOM Navigation Strategies](#dom-navigation-strategies)
   7. [Autonomous Web Agents](#autonomous-web-agents)

7. [Performance Considerations](#performance-considerations)
   1. [Memory Usage Optimization](#memory-usage-optimization)
   2. [Rate Limiting](#rate-limiting)
   3. [Caching Strategies](#caching-strategies)
   4. [Parallel Processing](#parallel-processing)

8. [Security Considerations](#security-considerations)
   1. [API Key Management](#api-key-management)
   2. [Data Privacy](#data-privacy)
   3. [Content Safety](#content-safety)

9. [Customization and Extension](#customization-and-extension)
   1. [Custom Extractors](#custom-extractors)
   2. [LLM Provider Plugins](#llm-provider-plugins)
   3. [Custom Validation Rules](#custom-validation-rules)
   4. [Pipeline Extensions](#pipeline-extensions)

10. [Troubleshooting and FAQs](#troubleshooting-and-faqs)
    1. [Common Issues](#common-issues)
    2. [Performance Tuning](#performance-tuning)
    3. [Extraction Quality](#extraction-quality)
    4. [Handling Site Changes](#handling-site-changes)

## System Architecture

### Architectural Overview

DOMorpher employs a layered architecture designed to combine the precision of traditional DOM parsing with the semantic understanding capabilities of large language models. The system consists of seven primary layers:

1. **Input Layer**: Handles HTML acquisition from various sources (URLs, files, strings)
2. **Preprocessing Layer**: Normalizes and optimizes HTML for further processing
3. **Analysis Layer**: Combines traditional parsing with LLM-based understanding
4. **Execution Layer**: Implements extraction strategies and adapts to challenges
5. **Validation Layer**: Ensures extracted data meets expected schemas and quality standards
6. **Reconciliation Layer**: Combines and normalizes results from multiple extraction passes
7. **Output Layer**: Formats and delivers extraction results in the requested format

Each layer operates both independently and cohesively, allowing for step-by-step processing while maintaining the context required for intelligent extraction.

The architecture follows these design principles:
- **Separation of concerns**: Each component has a well-defined responsibility
- **Pluggable components**: All major components can be swapped with custom implementations
- **Progressive enhancement**: The system can fall back to simpler methods when advanced strategies fail
- **Contextual awareness**: Information is shared between components to enhance extraction quality

### Component Interactions

Components interact through well-defined interfaces, with the following primary workflows:

**Standard Extraction Flow**:
1. The Input Layer acquires and validates HTML content
2. The Preprocessing Layer normalizes and optimizes the DOM
3. The Instruction Parser converts natural language to executable strategies
4. The Chunking Engine segments the content when necessary
5. The LLM Integration Layer processes each chunk with contextual awareness
6. The Adaptive Execution Engine applies extraction strategies with real-time feedback
7. The Result Reconciliation component combines partial results
8. The Schema Enforcement component validates and normalizes the output
9. The Output Layer delivers the formatted result

**Feedback Loop**:
1. Extraction results are monitored for quality
2. Failed or low-confidence extractions trigger alternative strategies
3. Successful strategies are recorded for optimization
4. User feedback is incorporated into future extraction attempts

**Learning System**:
1. Extraction patterns are recorded in an anonymized knowledge base
2. Site-specific optimizations are developed from repeated interactions
3. New extraction templates are suggested based on common patterns
4. Internal prompt engineering is refined continuously

### Data Flow

Data flows through the system in the following stages:

1. **Initial HTML**: Raw HTML enters the system from the input source
2. **Preprocessed HTML**: Normalized and optimized HTML with enhanced accessibility for LLM analysis
3. **Chunked Content**: Segmented HTML when content exceeds context limits
4. **Extraction Instructions**: Parsed and formalized version of natural language instructions
5. **LLM Prompts**: Specialized prompts combining instructions with HTML content and context
6. **Raw Extraction Results**: Initial extraction output from the LLM
7. **Validated Results**: Extraction results after schema validation and normalization
8. **Reconciled Results**: Combined results from multiple extraction passes
9. **Formatted Output**: Final results in the requested format (JSON, CSV, etc.)

Each stage includes metadata to maintain context and inform downstream processing, including:
- Confidence scores for extracted elements
- Relationships between elements in different chunks
- Extraction strategies attempted
- Processing time and resource usage

## Core Components

### DOM Preprocessor

The DOM Preprocessor optimizes HTML content for LLM processing by:

1. **Normalizing HTML**: Correcting common HTML issues and standardizing structure
2. **Enhancing Semantic Structure**: Adding context clues for improved LLM understanding
3. **Removing Irrelevant Content**: Eliminating boilerplate and non-essential elements
4. **Simplifying Complex Structures**: Converting intricate layouts to more interpretable forms

**Key Functions**:

- `normalize_html(html: str) -> str`: Standardizes HTML structure and fixes common issues
- `enhance_semantics(html: str) -> str`: Adds semantic markers for improved LLM understanding
- `remove_boilerplate(html: str) -> str`: Eliminates headers, footers, ads, and other non-essential content
- `simplify_nested_structures(html: str) -> str`: Flattens deeply nested elements while preserving relationships

**Customization Options**:

- Preprocessing intensity (minimal, standard, aggressive)
- Content type hints (e-commerce, news, forum, documentation)
- Element preservation rules (elements that should never be removed)
- Structure simplification thresholds

### Chunking Engine

The Chunking Engine divides large HTML documents into processable segments while preserving context:

1. **Size-Based Chunking**: Divides content based on token limits
2. **Semantic Chunking**: Creates chunks based on content meaning and structure
3. **Hierarchical Chunking**: Maintains parent-child relationships between chunks
4. **Overlap Strategy**: Includes contextual overlap between adjacent chunks

**Key Functions**:

- `chunk_by_size(html: str, max_tokens: int) -> List[Chunk]`: Divides based on token count
- `chunk_semantically(html: str) -> List[Chunk]`: Divides based on content meaning
- `create_chunk_hierarchy(chunks: List[Chunk]) -> ChunkTree`: Establishes relationships between chunks
- `add_chunk_context(chunks: List[Chunk], overlap_strategy: str) -> List[EnhancedChunk]`: Adds context to each chunk

**Chunk Structure**:

Each chunk contains:
- HTML content
- Position information (index, parent-child relationships)
- Context summary (what came before/after)
- Element paths for rejoining results
- Metadata for special handling

### LLM Integration Layer

The LLM Integration Layer manages interactions with language model providers:

1. **Provider Abstraction**: Unified interface for multiple LLM providers
2. **Prompt Management**: Generation and optimization of LLM prompts
3. **Response Processing**: Parsing and validation of LLM responses
4. **Fallback Management**: Handling rate limits, timeouts, and service disruptions

**Supported LLM Providers**:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku)
- Cohere (Command, Command Light)
- Local models (llama.cpp, Phi models)
- Custom API endpoints

**Key Functions**:

- `create_prompt(html: str, instruction: str, context: Dict) -> str`: Generates optimized LLM prompts
- `process_chunk(chunk: EnhancedChunk, instruction: str) -> ExtractedData`: Processes a chunk with the LLM
- `handle_llm_response(response: str) -> Dict`: Parses and structures LLM responses
- `manage_rate_limits(provider: str) -> None`: Implements intelligent request scheduling

### Instruction Parser

The Instruction Parser converts natural language extraction instructions into executable strategies:

1. **Instruction Analysis**: Parsing and understanding natural language instructions
2. **Strategy Formulation**: Converting instructions to extraction strategies
3. **Ambiguity Resolution**: Clarifying unclear or ambiguous instructions
4. **Instruction Enhancement**: Adding implicit details based on content type

**Key Functions**:

- `parse_instruction(instruction: str) -> ParsedInstruction`: Converts natural language to structured instructions
- `identify_extraction_targets(instruction: str) -> List[ExtractionTarget]`: Identifies what should be extracted
- `create_extraction_strategy(parsed_instruction: ParsedInstruction) -> ExtractionStrategy`: Creates execution plan
- `enhance_instruction(instruction: str, content_type: str) -> EnhancedInstruction`: Adds implicit context

### Adaptive Execution Engine

The Adaptive Execution Engine implements extraction strategies with real-time adaptation:

1. **Strategy Execution**: Applying planned extraction approaches
2. **Real-time Monitoring**: Evaluating extraction success
3. **Strategy Adaptation**: Adjusting approaches based on results
4. **Multi-Pass Extraction**: Applying different strategies for challenging content

**Key Functions**:

- `execute_strategy(strategy: ExtractionStrategy, chunk: EnhancedChunk) -> ExtractedData`: Applies extraction strategy
- `evaluate_extraction(data: ExtractedData, expectations: Dict) -> EvaluationResult`: Assesses extraction quality
- `adapt_strategy(strategy: ExtractionStrategy, evaluation: EvaluationResult) -> ExtractionStrategy`: Adjusts strategy
- `perform_multi_pass_extraction(chunk: EnhancedChunk, strategies: List[ExtractionStrategy]) -> ExtractedData`: Tries multiple approaches

**Adaptation Mechanisms**:

- Confidence threshold adjustment
- Alternative selector strategies
- Extraction scope narrowing/widening
- Fallback to different LLM models
- Template-based approaches for known patterns

### Result Reconciliation

The Result Reconciliation component combines and normalizes results from multiple extraction passes:

1. **Result Merging**: Combining partial results from different chunks
2. **Conflict Resolution**: Resolving contradictory extractions
3. **Relationship Reconstruction**: Rebuilding hierarchical relationships
4. **Consistency Enforcement**: Ensuring consistent data formats

**Key Functions**:

- `merge_chunk_results(results: List[ExtractedData]) -> ExtractedData`: Combines results from all chunks
- `resolve_conflicts(conflicting_data: List[DataPoint]) -> DataPoint`: Handles contradictory extractions
- `reconstruct_hierarchies(flat_data: ExtractedData) -> HierarchicalData`: Rebuilds nested structures
- `ensure_consistency(data: ExtractedData) -> NormalizedData`: Standardizes data formats

### Schema Enforcement

The Schema Enforcement component validates and normalizes extracted data:

1. **Schema Validation**: Checking data against expected schemas
2. **Type Conversion**: Converting data to appropriate types
3. **Format Standardization**: Ensuring consistent data formats
4. **Missing Data Handling**: Addressing incomplete extractions

**Key Functions**:

- `validate_against_schema(data: Dict, schema: Schema) -> ValidationResult`: Checks data against schema
- `convert_data_types(data: Dict, schema: Schema) -> Dict`: Converts data to correct types
- `standardize_formats(data: Dict, format_rules: Dict) -> Dict`: Ensures consistent formatting
- `handle_missing_data(data: Dict, schema: Schema) -> Dict`: Fills in or marks missing values

**Schema Definition Options**:

- JSON Schema
- Pydantic models
- Custom validation rules
- Required vs. optional fields
- Type coercion rules

## API Reference

### Python API

#### Core Classes

**`Extractor` Class**:

```python
class Extractor:
    """Main extraction class for retrieving structured data from HTML."""

    def __init__(
        self,
        instruction: Optional[str] = None,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-sonnet",
        chunking_strategy: str = "auto",
        adaptation_level: str = "standard",
        schema: Optional[dict] = None,
        javascript_support: bool = False,
        cache_strategy: Optional[str] = None,
        rate_limiting: Optional[dict] = None,
        **kwargs
    ):
        """Initialize an extractor with the given parameters.

        Args:
            instruction: Natural language instruction for extraction
            llm_provider: LLM provider to use ("anthropic", "openai", etc.)
            llm_model: Specific model to use
            chunking_strategy: "auto", "size", "semantic", or "hierarchical"
            adaptation_level: "minimal", "standard", "aggressive"
            schema: Optional schema for validating extracted data
            javascript_support: Whether to render JavaScript
            cache_strategy: Caching strategy ("memory", "persist", "none")
            rate_limiting: Rate limiting configuration
            **kwargs: Additional configuration options
        """
        pass

    def extract(
        self,
        html: str,
        instruction: Optional[str] = None,
        schema: Optional[dict] = None
    ) -> Union[dict, list]:
        """Extract data from HTML using the given instruction.

        Args:
            html: HTML content to extract from
            instruction: Override the default instruction
            schema: Override the default schema

        Returns:
            Extracted data as a dictionary or list
        """
        pass

    def extract_from_url(
        self,
        url: str,
        instruction: Optional[str] = None,
        schema: Optional[dict] = None,
        headers: Optional[dict] = None,
        cookies: Optional[dict] = None,
        **kwargs
    ) -> Union[dict, list]:
        """Extract data from a URL using the given instruction.

        Args:
            url: URL to extract from
            instruction: Override the default instruction
            schema: Override the default schema
            headers: Optional HTTP headers
            cookies: Optional HTTP cookies
            **kwargs: Additional request parameters

        Returns:
            Extracted data as a dictionary or list
        """
        pass

    def provide_feedback(
        self,
        html_section: str,
        expected_result: Union[dict, list],
        instruction: Optional[str] = None
    ) -> None:
        """Provide feedback to improve extraction accuracy.

        Args:
            html_section: HTML section that was incorrectly processed
            expected_result: The correct extraction result
            instruction: The instruction associated with this feedback
        """
        pass
```

**`AutonomousAgent` Class**:

```python
class AutonomousAgent:
    """Autonomous agent for goal-driven web navigation and extraction."""

    def __init__(
        self,
        objective: str = None,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-opus",
        browser_options: Optional[dict] = None,
        navigation_strategy: Optional[dict] = None,
        timeout_seconds: int = 300,
        max_actions: int = 50,
        multimodal_enhancement: Optional[dict] = None,
        **kwargs
    ):
        """Initialize an autonomous agent with the given parameters.

        Args:
            objective: Natural language objective to accomplish
            llm_provider: LLM provider to use ("anthropic", "openai", etc.)
            llm_model: Specific model to use
            browser_options: Browser configuration options
            navigation_strategy: Strategy for DOM navigation
            timeout_seconds: Maximum execution time in seconds
            max_actions: Maximum number of actions to attempt
            multimodal_enhancement: Configuration for visual analysis
            **kwargs: Additional configuration options
        """
        pass

    def execute(
        self,
        url: str,
        objective: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """Execute the objective on the given URL.

        Args:
            url: Starting URL for the agent
            objective: Override the default objective
            **kwargs: Additional execution parameters

        Returns:
            AgentResult containing execution status and extracted data
        """
        pass

    def on_element_appear(
        self,
        selector: str,
        callback: Callable[[WebElement], None]
    ) -> None:
        """Register a callback for when an element appears.

        Args:
            selector: CSS selector for the element
            callback: Function to call when the element appears
        """
        pass

    def on_element_state_change(
        self,
        selector: str,
        state: str,
        callback: Callable[[WebElement, str], None]
    ) -> None:
        """Register a callback for element state changes.

        Args:
            selector: CSS selector for the element
            state: State to monitor ("disabled", "checked", etc.)
            callback: Function to call when the state changes
        """
        pass

    def on_url_change(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """Register a callback for URL changes.

        Args:
            callback: Function to call when the URL changes
        """
        pass

    def visualize_journey(
        self,
        output_path: str
    ) -> None:
        """Generate a visual representation of the agent's journey.

        Args:
            output_path: Path to save the visualization
        """
        pass
```

**`Template` Class**:

```python
class Template:
    """Reusable extraction template for common scenarios."""

    def __init__(
        self,
        instruction: str,
        schema: Optional[dict] = None,
        site_optimizations: Optional[dict] = None,
        **kwargs
    ):
        """Initialize a template with the given parameters.

        Args:
            instruction: Natural language instruction for extraction
            schema: Optional schema for validating extracted data
            site_optimizations: Site-specific optimization hints
            **kwargs: Additional configuration options
        """
        pass

    def extract(
        self,
        html: str,
        **kwargs
    ) -> Union[dict, list]:
        """Extract data from HTML using this template.

        Args:
            html: HTML content to extract from
            **kwargs: Override default template parameters

        Returns:
            Extracted data as a dictionary or list
        """
        pass

    def extract_from_url(
        self,
        url: str,
        **kwargs
    ) -> Union[dict, list]:
        """Extract data from a URL using this template.

        Args:
            url: URL to extract from
            **kwargs: Override default template parameters

        Returns:
            Extracted data as a dictionary or list
        """
        pass

    def save(
        self,
        path: str
    ) -> None:
        """Save template to a file.

        Args:
            path: File path to save the template
        """
        pass

    @classmethod
    def load(
        cls,
        path: str
    ) -> 'Template':
        """Load template from a file.

        Args:
            path: File path to load the template from

        Returns:
            Template instance
        """
        pass
```

**`Pipeline` Class**:

```python
class Pipeline:
    """Processing pipeline for complex extraction workflows."""

    def __init__(
        self,
        steps: List[PipelineStep],
        error_handling: str = "stop",
        **kwargs
    ):
        """Initialize a pipeline with the given steps.

        Args:
            steps: List of processing steps
            error_handling: How to handle errors ("stop", "continue", "retry")
            **kwargs: Additional configuration options
        """
        pass

    def process(
        self,
        input_data: Union[str, dict],
        **kwargs
    ) -> Any:
        """Process input data through the pipeline.

        Args:
            input_data: Input data (HTML, URL, or intermediate result)
            **kwargs: Override default pipeline parameters

        Returns:
            Processed result from the final pipeline step
        """
        pass

    def add_step(
        self,
        step: PipelineStep,
        position: Optional[int] = None
    ) -> None:
        """Add a step to the pipeline.

        Args:
            step: Pipeline step to add
            position: Optional position to insert the step (defaults to end)
        """
        pass

    def remove_step(
        self,
        index: int
    ) -> PipelineStep:
        """Remove a step from the pipeline.

        Args:
            index: Index of step to remove

        Returns:
            Removed pipeline step
        """
        pass
```

**`Session` Class**:

```python
class Session:
    """Session for stateful web interactions."""

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        **kwargs
    ):
        """Initialize a session.

        Args:
            browser_type: Type of browser to use ("chromium", "firefox", "webkit")
            headless: Whether to run in headless mode
            **kwargs: Additional browser options
        """
        pass

    def execute(
        self,
        url: Optional[str] = None,
        objective: str = None,
        **kwargs
    ) -> AgentResult:
        """Execute an objective within this session.

        Args:
            url: URL to navigate to (optional if already on a page)
            objective: Natural language objective to accomplish
            **kwargs: Additional execution parameters

        Returns:
            AgentResult containing execution status and extracted data
        """
        pass

    def navigate(
        self,
        url: str
    ) -> None:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
        """
        pass

    def screenshot(
        self,
        path: Optional[str] = None
    ) -> Optional[bytes]:
        """Take a screenshot of the current page.

        Args:
            path: Optional path to save the screenshot

        Returns:
            Screenshot as bytes if path is None
        """
        pass

    def close(
        self
    ) -> None:
        """Close the session and release resources."""
        pass
```

#### Module Functions

```python
def extract(
    html: str,
    instruction: str,
    **kwargs
) -> Union[dict, list]:
    """Quick extraction function for simple cases.

    Args:
        html: HTML content to extract from
        instruction: Natural language instruction for extraction
        **kwargs: Additional configuration options

    Returns:
        Extracted data as a dictionary or list
    """
    pass

def extract_from_url(
    url: str,
    instruction: str,
    **kwargs
) -> Union[dict, list]:
    """Quick extraction function for URLs.

    Args:
        url: URL to extract from
        instruction: Natural language instruction for extraction
        **kwargs: Additional configuration options

    Returns:
        Extracted data as a dictionary or list
    """
    pass

def execute(
    url: str,
    objective: str,
    **kwargs
) -> AgentResult:
    """Quick execution function for autonomous web tasks.

    Args:
        url: URL to start from
        objective: Natural language objective to accomplish
        **kwargs: Additional configuration options

    Returns:
        AgentResult containing execution status and extracted data
    """
    pass

def parallel_execute(
    urls: List[str],
    objective: str,
    max_concurrent: int = 3,
    **kwargs
) -> List[AgentResult]:
    """Execute the same objective on multiple URLs in parallel.

    Args:
        urls: List of URLs to execute on
        objective: Natural language objective to accomplish
        max_concurrent: Maximum number of concurrent executions
        **kwargs: Additional configuration options

    Returns:
        List of AgentResult objects
    """
    pass

def configure(
    **kwargs
) -> None:
    """Configure global DOMorpher settings.

    Args:
        **kwargs: Configuration options (API keys, default providers, etc.)
    """
    pass
```

### Rust API

#### Core Structures

**`Extractor` Struct**:

```rust
pub struct Extractor {
    // Internal fields omitted
}

impl Extractor {
    /// Create a new extractor with the given configuration
    pub fn new(config: ExtractorConfig) -> Self;

    /// Extract data from HTML using the given instruction
    pub async fn extract(
        &self,
        html: &str,
        instruction: &str,
    ) -> Result<serde_json::Value, Error>;

    /// Extract data from a URL using the given instruction
    pub async fn extract_from_url(
        &self,
        url: &str,
        instruction: &str,
    ) -> Result<serde_json::Value, Error>;

    /// Provide feedback to improve extraction accuracy
    pub async fn provide_feedback(
        &mut self,
        html_section: &str,
        expected_result: &serde_json::Value,
        instruction: Option<&str>,
    ) -> Result<(), Error>;
}
```

**`AutonomousAgent` Struct**:

```rust
pub struct AutonomousAgent {
    // Internal fields omitted
}

impl AutonomousAgent {
    /// Create a new agent with the given configuration
    pub fn new(config: AgentConfig) -> Self;

    /// Execute the agent with the given objective
    pub async fn execute(
        &self,
        url: &str,
        objective: Option<&str>,
    ) -> Result<AgentResult, Error>;

    /// Register a callback for element appearance
    pub fn on_element_appear<F>(
        &mut self,
        selector: &str,
        callback: F,
    ) where F: Fn(&WebElement) + Send + Sync + 'static;

    /// Register a callback for element state changes
    pub fn on_element_state_change<F>(
        &mut self,
        selector: &str,
        state: &str,
        callback: F,
    ) where F: Fn(&WebElement, &str) + Send + Sync + 'static;

    /// Register a callback for URL changes
    pub fn on_url_change<F>(
        &mut self,
        callback: F,
    ) where F: Fn(&str, &str) + Send + Sync + 'static;
}
```

**`ExtractorConfig` Struct**:

```rust
pub struct ExtractorConfig {
    // Internal fields omitted
}

impl ExtractorConfig {
    /// Create a new configuration builder
    pub fn builder() -> ExtractorConfigBuilder;
}

pub struct ExtractorConfigBuilder {
    // Internal fields omitted
}

impl ExtractorConfigBuilder {
    /// Set the LLM provider
    pub fn llm_provider(mut self, provider: LlmProvider) -> Self;

    /// Set the model name
    pub fn model(mut self, model: &str) -> Self;

    /// Set the chunking strategy
    pub fn chunking_strategy(mut self, strategy: ChunkingStrategy) -> Self;

    /// Set the adaptation level
    pub fn adaptation_level(mut self, level: AdaptationLevel) -> Self;

    /// Enable JavaScript support
    pub fn javascript_support(mut self, enabled: bool) -> Self;

    /// Set cache strategy
    pub fn cache_strategy(mut self, strategy: CacheStrategy) -> Self;

    /// Build the configuration
    pub fn build(self) -> ExtractorConfig;
}
```

**`AgentConfig` Struct**:

```rust
pub struct AgentConfig {
    // Internal fields omitted
}

impl AgentConfig {
    /// Create a new configuration builder
    pub fn builder() -> AgentConfigBuilder;
}

pub struct AgentConfigBuilder {
    // Internal fields omitted
}

impl AgentConfigBuilder {
    /// Set the objective
    pub fn objective(mut self, objective: &str) -> Self;

    /// Set the LLM provider
    pub fn llm_provider(mut self, provider: LlmProvider) -> Self;
    
    /// Set the model name
    pub fn model(mut self, model: &str) -> Self;
    
    /// Set browser options
    pub fn browser_options(mut self, options: BrowserOptions) -> Self;
    
    /// Set navigation strategy
    pub fn navigation_strategy(mut self, strategy: NavigationStrategy) -> Self;
    
    /// Set timeout in seconds
    pub fn timeout_seconds(mut self, timeout: u64) -> Self;
    
    /// Set maximum actions
    pub fn max_actions(mut self, max: u32) -> Self;
    
    /// Enable multimodal enhancement
    pub fn multimodal_enhancement(mut self, config: MultimodalConfig) -> Self;
    
    /// Build the configuration
    pub fn build(self) -> AgentConfig;
}
```

**`Template` Struct**:

```rust
pub struct Template {
    // Internal fields omitted
}

impl Template {
    /// Create a new template
    pub fn new(
        instruction: &str,
        schema: Option<&Schema>,
        site_optimizations: Option<HashMap<String, String>>,
    ) -> Self;

    /// Extract data from HTML using this template
    pub async fn extract(
        &self,
        html: &str,
    ) -> Result<serde_json::Value, Error>;

    /// Extract data from a URL using this template
    pub async fn extract_from_url(
        &self,
        url: &str,
    ) -> Result<serde_json::Value, Error>;

    /// Save template to file
    pub fn save(&self, path: &str) -> Result<(), Error>;
    
    /// Load template from file
    pub fn load(path: &str) -> Result<Self, Error>;
}
```

**`Pipeline` Struct**:

```rust
pub struct Pipeline {
    // Internal fields omitted
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new(
        steps: Vec<Box<dyn PipelineStep>>,
        error_handling: ErrorHandling,
    ) -> Self;

    /// Process input data through the pipeline
    pub async fn process(
        &self,
        input: PipelineInput,
    ) -> Result<PipelineOutput, Error>;

    /// Add a step to the pipeline
    pub fn add_step(
        &mut self,
        step: Box<dyn PipelineStep>,
        position: Option<usize>,
    );

    /// Remove a step from the pipeline
    pub fn remove_step(
        &mut self,
        index: usize,
    ) -> Option<Box<dyn PipelineStep>>;
}
```

**`Session` Struct**:

```rust
pub struct Session {
    // Internal fields omitted
}

impl Session {
    /// Create a new session
    pub fn new(
        browser_type: BrowserType,
        headless: bool,
        options: Option<BrowserOptions>,
    ) -> Result<Self, Error>;

    /// Execute an objective in this session
    pub async fn execute(
        &self,
        url: Option<&str>,
        objective: &str,
    ) -> Result<AgentResult, Error>;

    /// Navigate to a URL
    pub async fn navigate(&self, url: &str) -> Result<(), Error>;

    /// Take a screenshot
    pub async fn screenshot(&self, path: Option<&str>) -> Result<Option<Vec<u8>>, Error>;

    /// Close the session
    pub async fn close(self) -> Result<(), Error>;
}
```

#### Module Functions

```rust
/// Quick extraction function for simple cases
pub async fn extract(
    html: &str,
    instruction: &str,
    config: Option<ExtractorConfig>,
) -> Result<serde_json::Value, Error>;

/// Quick extraction function for URLs
pub async fn extract_from_url(
    url: &str,
    instruction: &str,
    config: Option<ExtractorConfig>,
) -> Result<serde_json::Value, Error>;

/// Quick execution function for autonomous web tasks
pub async fn execute(
    url: &str,
    objective: &str,
    config: Option<AgentConfig>,
) -> Result<AgentResult, Error>;

/// Execute the same objective on multiple URLs in parallel
pub async fn parallel_execute(
    urls: &[String],
    objective: &str,
    max_concurrent: usize,
    config: Option<AgentConfig>,
) -> Result<Vec<AgentResult>, Error>;

/// Configure global DOMorpher settings
pub fn configure(config: GlobalConfig) -> Result<(), Error>;
```

### Command-Line Interface

DOMorpher provides a comprehensive command-line interface for extraction tasks:

```
USAGE:
    domorpher [OPTIONS] <SUBCOMMAND>

OPTIONS:
    -v, --verbose               Enable verbose output
    --config <FILE>             Use custom configuration file
    --cache <DIRECTORY>         Specify cache directory
    --rate-limit <LIMIT>        Set rate limit (requests per minute)
    --timeout <SECONDS>         Set request timeout
    -h, --help                  Print help information
    -V, --version               Print version information

SUBCOMMANDS:
    extract         Extract data from HTML or URLs
    execute         Execute an autonomous agent with a goal
    batch           Process multiple inputs in batch mode
    template        Create and manage extraction templates
    validate        Validate extraction results against schemas
    configure       Configure DOMorpher settings
    help            Print this message or the help of the given subcommand(s)
```

**Extract Subcommand**:

```
USAGE:
    domorpher extract [OPTIONS] <SOURCE>

ARGS:
    <SOURCE>    URL or path to HTML file

OPTIONS:
    -i, --instruction <TEXT>       Extraction instruction
    --instruction-file <FILE>      Read instruction from file
    -o, --output <FILE>            Output file (defaults to stdout)
    -f, --format <FORMAT>          Output format [default: json] [possible values: json, csv, yaml, xml]
    --schema <FILE>                Schema file for validation
    --javascript                   Enable JavaScript rendering
    --wait <SELECTOR>              Wait for element to appear
    --provider <PROVIDER>          LLM provider to use
    --model <MODEL>                LLM model to use
    --chunking <STRATEGY>          Chunking strategy
    --adaptation <LEVEL>           Adaptation level
    -h, --help                     Print help information
```

**Execute Subcommand**:

```
USAGE:
    domorpher execute [OPTIONS] <URL>

ARGS:
    <URL>    Starting URL for the agent

OPTIONS:
    -o, --objective <TEXT>         Agent objective
    --objective-file <FILE>        Read objective from file
    -o, --output <FILE>            Output file (defaults to stdout)
    -f, --format <FORMAT>          Output format [default: json]
    --headless                     Run in headless mode
    --browser <BROWSER>            Browser to use [default: chromium] [possible values: chromium, firefox, webkit]
    --max-actions <NUMBER>         Maximum number of actions
    --timeout <SECONDS>            Execution timeout in seconds
    --provider <PROVIDER>          LLM provider to use
    --model <MODEL>                LLM model to use
    --strategy <STRATEGY>          Navigation strategy [possible values: semantic_first, structure_first, balanced, aggressive]
    -v, --visual                   Enable visual enhancement
    -h, --help                     Print help information
```

**Batch Subcommand**:

```
USAGE:
    domorpher batch [OPTIONS] <SUBCOMMAND>

SUBCOMMANDS:
    extract    Batch extraction from multiple sources
    execute    Batch execution on multiple URLs

OPTIONS:
    -i, --input <FILE>             File with URLs or paths (one per line)
    -o, --output-dir <DIRECTORY>   Output directory
    -f, --format <FORMAT>          Output format [default: json]
    --concurrency <NUMBER>         Number of concurrent processes
    --delay <MILLISECONDS>         Delay between requests
    -h, --help                     Print help information
```

**Template Subcommand**:

```
USAGE:
    domorpher template <SUBCOMMAND>

SUBCOMMANDS:
    create      Create a new template
    use         Use an existing template
    list        List available templates
    export      Export a template to file
    import      Import a template from file
    help        Print this message or the help of the given subcommand(s)
```

### Configuration Options

DOMorpher can be configured through a configuration file, environment variables, or programmatically:

**Configuration File (JSON/YAML)**:

```json
{
  "llm": {
    "providers": {
      "anthropic": {
        "api_key": "${ANTHROPIC_API_KEY}",
        "default_model": "claude-3-sonnet",
        "timeout": 30
      },
      "openai": {
        "api_key": "${OPENAI_API_KEY}",
        "default_model": "gpt-4o",
        "timeout": 30
      }
    },
    "default_provider": "anthropic"
  },
  "extraction": {
    "chunking": {
      "strategy": "semantic",
      "max_chunk_size": 8000,
      "overlap": 500
    },
    "adaptation": {
      "level": "standard",
      "max_retries": 3,
      "confidence_threshold": 0.7
    }
  },
  "autonomous_agent": {
    "default_model": "claude-3-opus",
    "timeout_seconds": 300,
    "max_actions": 50,
    "navigation_strategies": {
      "default": "semantic_first",
      "exploration_level": "balanced",
      "patience": "medium"
    },
    "multimodal_enhancement": {
      "enabled": false,
      "trigger": "on_failure",
      "visual_context_level": "minimal"
    }
  },
  "browser": {
    "type": "chromium",
    "headless": true,
    "user_agent": "DOMorpher/0.1.0",
    "viewport": {
      "width": 1280,
      "height": 800
    }
  },
  "http": {
    "timeout": 10,
    "max_redirects": 5,
    "retry_attempts": 3,
    "retry_delay": 500
  },
  "javascript": {
    "enabled": false,
    "timeout": 30,
    "wait_until": "networkidle"
  },
  "cache": {
    "strategy": "memory",
    "ttl": 3600,
    "max_size": 100
  },
  "rate_limiting": {
    "enabled": true,
    "rpm": 10,
    "concurrent_requests": 2
  }
}
```

**Environment Variables**:

```
DOMORPHER_ANTHROPIC_API_KEY=your_api_key
DOMORPHER_OPENAI_API_KEY=your_api_key
DOMORPHER_DEFAULT_PROVIDER=anthropic
DOMORPHER_DEFAULT_MODEL=claude-3-sonnet
DOMORPHER_CACHE_STRATEGY=memory
DOMORPHER_RATE_LIMIT=10
DOMORPHER_JAVASCRIPT_ENABLED=false
DOMORPHER_ADAPTATION_LEVEL=standard
DOMORPHER_DEFAULT_AGENT_MODEL=claude-3-opus
DOMORPHER_BROWSER_TYPE=chromium
DOMORPHER_BROWSER_HEADLESS=true
```

**Programmatic Configuration**:

```python
import domorpher

domorpher.configure(
    anthropic_api_key="your_api_key",
    openai_api_key="your_api_key",
    default_provider="anthropic",
    default_model="claude-3-sonnet",
    cache_strategy="memory",
    cache_ttl=3600,
    rate_limit=10,
    javascript_enabled=False,
    adaptation_level="standard",
    default_agent_model="claude-3-opus",
    browser_type="chromium",
    browser_headless=True
)
```

## Implementation Details

### DOM Parsing Strategy

DOMorpher uses a hybrid DOM parsing approach that combines traditional traversal with semantic understanding:

**Preprocessing Pipeline**:

1. **HTML Normalization**: Using a standard HTML parser to correct and normalize HTML
   - Fixing unclosed tags
   - Normalizing attributes
   - Removing invalid elements
   - Converting to UTF-8

2. **Semantic Enhancement**: Adding context clues for LLM understanding
   - Identifying main content areas
   - Marking navigation, headers, footers
   - Annotating list structures
   - Identifying interactive elements

3. **Visual Layout Analysis**: Incorporating visual layout information
   - Adding positional attributes
   - Marking visible vs. hidden elements
   - Annotating grid/table structures
   - Identifying visual breaks and sections

4. **Structure Simplification**: Flattening complex nested structures
   - Removing unnecessary wrapper divs
   - Simplifying deeply nested elements
   - Flattening redundant structures
   - Preserving semantic relationships

**DOM Representation Formats**:

DOMorpher provides different DOM representations for LLM processing:

- **Full HTML**: Complete HTML with preprocessing enhancements
- **Simplified HTML**: Reduced HTML focusing on content elements
- **DOM Tree Text**: Textual representation of the DOM tree
- **Semantic Outline**: Hierarchical content outline
- **Element Table**: Tabular representation of key elements

The system selects the most appropriate representation based on:
- Content type (e-commerce, news, documentation, etc.)
- Extraction requirements
- LLM model capabilities
- Processing resource constraints

**DOM-to-Semantic-Markdown Conversion**:

For token efficiency, DOMorpher can convert HTML to a semantic markdown representation:

```python
# Example usage:
semantic_content = dom_to_markdown.convert(html, {
    "extractMainContent": True,  # Focus on the main content
    "enableTableColumnTracking": True,  # Preserve table structures
    "includeMetaData": "extended"  # Capture all available metadata
})

# Format for LLM (token-optimized)
llm_prompt = f"""
Analyzing the following webpage content:

```markdown
{semantic_content}
```

Your objective is to: {objective}
Based on this content, what elements would you interact with and why?
"""
```

### LLM Prompt Engineering

DOMorpher uses specialized prompt templates for different extraction scenarios:

**Base Prompt Template**:

```
You are an expert web data extraction system called DOMorpher. You analyze HTML and extract specific information based on instructions.

HTML Content:
```html
{html_content}
```

Instructions:
{instruction}

Extract the requested information and return it as a valid JSON object with the following structure:
{expected_structure}

Additional Guidelines:
1. Be precise and accurate in your extraction
2. Follow the exact structure requested
3. Return null for any fields you cannot find
4. Provide confidence scores (0-1) for each extracted field
5. Include any relationships between extracted items
```

**Specialized Prompt Variations**:

1. **Hierarchical Extraction**:
   - Emphasizes parent-child relationships
   - Includes guidance for handling nested structures
   - Provides examples of correctly formatted hierarchical data

2. **Table Extraction**:
   - Specialized guidance for extracting tabular data
   - Instructions for handling merged cells, headers, and footers
   - Row/column relationship preservation

3. **Multi-Item Extraction**:
   - Guidance for extracting repeated elements (product listings, search results)
   - Pagination handling instructions
   - Consistency enforcement across items

4. **Text-Heavy Content**:
   - Instructions for maintaining text formatting
   - Paragraph and section relationship preservation
   - List detection and formatting

**Autonomous Agent Prompt Template**:

```
You are an autonomous web navigation agent with the objective:
"{objective}"

Current page context:
```markdown
{current_context}
```

Relevant action history:
{action_context}

Progress so far:
{progress_summary}

What action should I take next to achieve my objective?
Provide your reasoning in detail, then specify the exact action in the format:

ACTION: [action type]
TARGET: [element description or selector]
VALUE: [any additional parameters]
```

**Prompt Chaining Strategy**:

For complex extractions, DOMorpher uses a multi-step prompt chain:

1. **Analysis Prompt**: Assesses overall structure and content categories
2. **Planning Prompt**: Determines the optimal extraction strategy
3. **Extraction Prompt**: Performs the actual data extraction
4. **Validation Prompt**: Verifies extraction quality and completeness
5. **Refinement Prompt**: Enhances or corrects extracted data as needed

Each step uses the results from previous steps to improve extraction quality.

### Chunking Algorithms

DOMorpher employs several chunking algorithms to process large HTML documents:

**Size-Based Chunking**:

1. Tokenizes HTML content to estimate token count
2. Creates chunks based on token limit (default: 8,000 tokens)
3. Ensures clean breaks at appropriate HTML boundaries
4. Includes overlap between chunks for context continuity

**Semantic Chunking**:

1. Analyzes document structure to identify logical sections
2. Creates chunks based on semantic boundaries:
   - Main content sections
   - Article divisions
   - Product groups
   - Comment threads
3. Ensures related content stays together
4. Adds contextual markers for relationships between chunks

**Hierarchical Chunking**:

1. Creates a tree structure representing document hierarchy
2. Processes top-level containers first
3. Recursively processes child elements as needed
4. Preserves parent-child relationships between chunks

**Adaptive Semantic Chunking**:

The system also uses semantic importance-based chunking:

```python
def create_semantic_chunks(self, dom):
    # Analyze DOM to identify semantic sections
    sections = self.identify_semantic_sections(dom)
    
    # Assign importance scores to each section based on:
    for section in sections:
        # 1. Semantic hierarchy (headers, main content, etc.)
        hierarchy_score = self.assess_hierarchy_importance(section)
        
        # 2. Content density and relevance
        content_score = self.assess_content_density(section)
        
        # 3. Interactive element concentration
        interactivity_score = self.count_interactive_elements(section) / section["size"]
        
        # Calculate overall importance
        section["importance"] = (hierarchy_score * 0.4 + 
                                content_score * 0.4 + 
                                interactivity_score * 0.2)
    
    # Create chunks prioritizing important sections while maintaining context
    chunks = self.create_prioritized_chunks(sections)
    
    return chunks
```

**Chunk Context Management**:

Each chunk includes contextual information:
- Location within the document
- Parent-child relationships
- Brief summaries of adjacent chunks
- Global document metadata
- Path information for result reconciliation

### Context Management

DOMorpher maintains context throughout the extraction process:

**Global Context**:

- Document metadata (title, URL, etc.)
- Site-specific information
- Extraction goals and requirements
- Overall document structure

**Chunk Context**:

- Position within document
- Relationship to other chunks
- Content type and category
- Element paths for reconstruction

**Extraction Context**:

- Previously extracted data
- Confidence scores for extracted elements
- Alternative extraction attempts
- Intermediate processing results

**Agent Action Context**:

- Previous actions and their results
- Navigation path taken
- Goal progress indicators
- Failed attempts and reasons

**Context Propagation Mechanisms**:

1. **Context Headers**: Each chunk includes a context summary
2. **Progressive Refinement**: Later chunks learn from earlier chunks
3. **Global State**: Maintains document-wide extraction state
4. **Cross-Reference Markers**: Special tokens that link related content

**Context Application Examples**:

1. **Product Listings**: Earlier chunks identify the product structure, later chunks use this pattern
2. **Article Content**: Article metadata from headers is propagated to content chunks
3. **Forum Threads**: Parent-child relationships between posts are maintained
4. **Documentation**: Type definitions from earlier sections inform method interpretation
5. **Navigation**: Previous page insights inform understanding of current page

### Result Validation

DOMorpher implements comprehensive validation to ensure extraction quality:

**Schema Validation**:

1. **Type Checking**: Verifying data types match expected schema
2. **Required Field Validation**: Ensuring all required fields are present
3. **Format Validation**: Checking data formats (dates, prices, etc.)
4. **Constraint Validation**: Enforcing value constraints (min/max, patterns)

**Semantic Validation**:

1. **Consistency Checking**: Verifying related data is consistent
2. **Relationship Validation**: Ensuring parent-child relationships make sense
3. **Value Range Verification**: Checking values are within reasonable ranges
4. **Context Compliance**: Verifying data makes sense in context

**Statistical Validation**:

1. **Outlier Detection**: Identifying unusual values
2. **Pattern Compliance**: Ensuring data follows expected patterns
3. **Distribution Analysis**: Verifying numeric distributions
4. **Frequency Analysis**: Checking text value distributions

**LLM-Based Validation**:

1. **Coherence Checking**: Using LLMs to verify data makes sense
2. **Completeness Assessment**: LLM analysis of extraction completeness
3. **Context Verification**: Ensuring extracted data matches source context
4. **Plausibility Checking**: Verifying data values are plausible

## Integration Guides

### Python Integration

**Basic Integration**:

```python
import domorpher

# Configure API keys
domorpher.configure(
    anthropic_api_key="your_api_key",
    openai_api_key="your_api_key"
)

# Simple extraction from URL
results = domorpher.extract_from_url(
    "https://example.com/products",
    "Extract all product names, prices, and descriptions"
)

# Process results
for product in results:
    print(f"Product: {product['name']}")
    print(f"Price: {product['price']}")
    print(f"Description: {product['description']}")
```

**Autonomous Agent Integration**:

```python
import domorpher

# Configure API keys
domorpher.configure(
    anthropic_api_key="your_api_key"
)

# Create an autonomous agent
agent = domorpher.AutonomousAgent(
    objective="Find the best-rated smartphone under $500 and extract its specifications",
    llm_model="claude-3-opus",
    navigation_strategy={
        "approach": "semantic_first",
        "patience": "high"
    }
)

# Execute the agent
result = agent.execute("https://www.techreview.com/smartphones")

# Process results
if result.success:
    phone = result.extracted_data
    print(f"Best phone: {phone['name']}")
    print(f"Price: ${phone['price']}")
    print(f"Rating: {phone['rating']}/5")
    print(f"Specs:")
    for key, value in phone['specifications'].items():
        print(f"  {key}: {value}")
else:
    print(f"Failed: {result.error_message}")
```

**Advanced Integration with Pydantic**:

```python
import domorpher
from pydantic import BaseModel
from typing import List, Optional

# Define a schema
class Product(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    rating: Optional[float] = None
    in_stock: bool = True

# Create an extractor with custom configuration
extractor = domorpher.Extractor(
    llm_provider="anthropic",
    llm_model="claude-3-opus",
    chunking_strategy="semantic",
    adaptation_level="aggressive",
    schema=List[Product],
    javascript_support=True,
    cache_strategy="persist",
    rate_limiting={"rpm": 10}
)

# Extract from URL
products = extractor.extract_from_url(
    "https://example.com/products",
    "Extract all products with their names, prices, descriptions, ratings, and availability"
)

# Products are already validated against the schema
for product in products:
    if product.rating and product.rating > 4.0:
        print(f"Highly rated product: {product.name} (${product.price})")
```

**Pipeline Integration**:

```python
import domorpher
from domorpher.steps import (
    FetchURL,
    PreprocessHTML,
    ExtractMainContent,
    HandlePagination,
    CustomExtractor,
    SchemaValidator,
    OutputFormatter
)

# Define a schema file (or use a Pydantic model)
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "content": {"type": "string"}
        },
        "required": ["title", "author", "date"]
    }
}

# Create a pipeline
pipeline = domorpher.Pipeline([
    FetchURL(headers={"User-Agent": "Mozilla/5.0 ..."}),
    PreprocessHTML(remove_scripts=True, remove_styles=True),
    ExtractMainContent(),
    HandlePagination(max_pages=5),
    CustomExtractor("Extract all articles with their titles, authors, dates, and full content"),
    SchemaValidator(schema=schema),
    OutputFormatter(format="json")
])

# Process a URL through the pipeline
results = pipeline.process("https://example.com/blog")

# Save results
with open("articles.json", "w") as f:
    f.write(results)
```

**Session-Based Integration**:

```python
import domorpher

# Create a session for stateful interactions
session = domorpher.Session(browser_type="chromium", headless=False)

try:
    # Log in to the site
    login_result = session.execute(
        "https://example.com/login",
        "Log in with username 'testuser' and password 'password123'"
    )

    if login_result.success:
        # Navigate to the dashboard
        dashboard_result = session.execute(
            objective="Go to my dashboard and extract my account balance"
        )

        print(f"Account balance: ${dashboard_result.extracted_data['balance']}")

        # Take a screenshot
        session.screenshot("dashboard.png")

        # Execute another objective in the same session
        order_result = session.execute(
            "https://example.com/orders",
            "Find my most recent order and extract its status"
        )

        print(f"Recent order: {order_result.extracted_data['order_id']}")
        print(f"Status: {order_result.extracted_data['status']}")
finally:
    # Always close the session
    session.close()
```

### Rust Integration

**Basic Integration**:

```rust
use domorpher::{extract_from_url, Error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with environment variables
    // DOMORPHER_ANTHROPIC_API_KEY and DOMORPHER_OPENAI_API_KEY

    // Simple extraction
    let results = extract_from_url(
        "https://example.com/products",
        "Extract all product names, prices, and descriptions",
        None, // Use default configuration
    ).await?;

    // Process results
    if let serde_json::Value::Array(products) = results {
        for product in products {
            println!("Product: {}", product["name"].as_str().unwrap_or("Unknown"));
            println!("Price: {}", product["price"].as_str().unwrap_or("Unknown"));
            println!("Description: {}", product["description"].as_str().unwrap_or(""));
            println!("-----------------------");
        }
    }

    Ok(())
}
```

**Autonomous Agent Integration**:

```rust
use domorpher::{execute, AgentConfig, Error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple agent execution
    let result = execute(
        "https://www.techreview.com/smartphones",
        "Find the best-rated smartphone under $500 and extract its specifications",
        None, // Use default configuration
    ).await?;

    if result.success {
        if let Some(data) = result.extracted_data {
            println!("Best phone: {}", data["name"].as_str().unwrap_or("Unknown"));
            println!("Price: ${}", data["price"].as_f64().unwrap_or(0.0));
            println!("Rating: {}/5", data["rating"].as_f64().unwrap_or(0.0));
            
            if let Some(specs) = data["specifications"].as_object() {
                println!("Specs:");
                for (key, value) in specs {
                    println!("  {}: {}", key, value.as_str().unwrap_or("Unknown"));
                }
            }
        }
    } else {
        println!("Failed: {}", result.error_message.unwrap_or_default());
    }

    Ok(())
}
```

**Advanced Integration**:

```rust
use domorpher::{AutonomousAgent, AgentConfig, AgentConfigBuilder, LlmProvider, NavigationStrategy, Error};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Phone {
    name: String,
    price: f64,
    rating: Option<f64>,
    specifications: std::collections::HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the agent
    let config = AgentConfig::builder()
        .objective("Find the best-rated smartphone under $500 and extract its specifications")
        .llm_provider(LlmProvider::Anthropic)
        .model("claude-3-opus")
        .navigation_strategy(NavigationStrategy::SemanticFirst)
        .timeout_seconds(300)
        .max_actions(50)
        .build();

    let agent = AutonomousAgent::new(config);

    // Execute the agent
    let result = agent.execute("https://www.techreview.com/smartphones", None).await?;

    if result.success {
        // Parse result into strongly typed struct
        if let Some(data) = result.extracted_data {
            let phone: Phone = serde_json::from_value(data)?;
            
            println!("Best phone: {}", phone.name);
            println!("Price: ${:.2}", phone.price);
            if let Some(rating) = phone.rating {
                println!("Rating: {}/5", rating);
            }
            
            println!("Specs:");
            for (key, value) in phone.specifications {
                println!("  {}: {}", key, value);
            }
        }
    } else {
        println!("Failed: {}", result.error_message.unwrap_or_default());
    }

    Ok(())
}
```

**Session-Based Integration**:

```rust
use domorpher::{Session, BrowserType, Error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a session for stateful interactions
    let session = Session::new(BrowserType::Chromium, false, None)?;

    // Error handling with proper cleanup
    let result = async {
        // Log in to the site
        let login_result = session.execute(
            Some("https://example.com/login"),
            "Log in with username 'testuser' and password 'password123'"
        ).await?;

        if login_result.success {
            // Navigate to the dashboard
            let dashboard_result = session.execute(
                None, // Use current page
                "Go to my dashboard and extract my account balance"
            ).await?;

            if let Some(data) = dashboard_result.extracted_data {
                println!("Account balance: ${}", data["balance"].as_str().unwrap_or("Unknown"));
            }

            // Take a screenshot
            session.screenshot(Some("dashboard.png")).await?;

            // Execute another objective in the same session
            let order_result = session.execute(
                Some("https://example.com/orders"),
                "Find my most recent order and extract its status"
            ).await?;

            if let Some(data) = order_result.extracted_data {
                println!("Recent order: {}", data["order_id"].as_str().unwrap_or("Unknown"));
                println!("Status: {}", data["status"].as_str().unwrap_or("Unknown"));
            }
        }

        Ok::<(), Error>(())
    }.await;

    // Always close the session
    session.close().await?;

    // Handle any errors from the session operations
    if let Err(e) = result {
        eprintln!("Error during session: {}", e);
    }

    Ok(())
}
```

### CI/CD Pipeline Integration

DOMorpher can be integrated into CI/CD pipelines for automated data extraction:

**GitHub Actions Example**:

```yaml
name: Extract Product Data

on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight
  workflow_dispatch:      # Allow manual triggering

jobs:
  extract:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install domorpher

    - name: Run extraction
      env:
        DOMORPHER_ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        python -c "
        import domorpher
        import json

        results = domorpher.extract_from_url(
            'https://example.com/products',
            'Extract all products with their names, prices, and availability'
        )

        with open('products.json', 'w') as f:
            json.dump(results, f, indent=2)
        "

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: product-data
        path: products.json
```

**Jenkins Pipeline Example**:

```groovy
pipeline {
    agent any

    triggers {
        cron('0 0 * * *')  // Run daily at midnight
    }

    environment {
        DOMORPHER_ANTHROPIC_API_KEY = credentials('anthropic-api-key')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install domorpher'
            }
        }

        stage('Extract Data') {
            steps {
                sh '''
                python3 -c "
                import domorpher
                import json

                urls = [
                    'https://example.com/page1',
                    'https://example.com/page2',
                    'https://example.com/page3'
                ]

                extractor = domorpher.Extractor(
                    instruction='Extract all pricing information',
                    javascript_support=True
                )

                all_results = []
                for url in urls:
                    results = extractor.extract_from_url(url)
                    all_results.extend(results)

                with open('data.json', 'w') as f:
                    json.dump(all_results, f, indent=2)
                "
                '''
            }
        }

        stage('Process Data') {
            steps {
                sh 'python3 process_data.py'
            }
        }

        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'data.json', fingerprint: true
            }
        }
    }
}
```

**Autonomous Agent in CI/CD**:

```yaml
name: Automated Web Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-website:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install domorpher pytest

    - name: Run automated web tests
      env:
        DOMORPHER_ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        python -c "
        import domorpher
        import json
        import sys

        SCENARIOS = [
            {
                'name': 'Product Search',
                'url': 'https://example.com',
                'objective': 'Search for \"wireless headphones\" and verify at least 5 results appear'
            },
            {
                'name': 'Account Creation',
                'url': 'https://example.com/register',
                'objective': 'Try to create an account with invalid email format and verify error message appears'
            },
            {
                'name': 'Shopping Cart',
                'url': 'https://example.com/products',
                'objective': 'Add any product to cart, then verify it appears in the cart with correct price'
            }
        ]

        results = []
        for scenario in SCENARIOS:
            print(f'Running scenario: {scenario[\"name\"]}')
            
            agent = domorpher.AutonomousAgent(objective=scenario['objective'])
            result = agent.execute(scenario['url'])
            
            success = result.success
            results.append({
                'scenario': scenario['name'],
                'success': success,
                'message': result.error_message if not success else 'Passed'
            })
            
            print(f'Result: {\"PASS\" if success else \"FAIL\"}')
            
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Exit with error if any scenario failed
        if not all(r['success'] for r in results):
            sys.exit(1)
        "

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test_results.json
```

### Error Handling and Logging

DOMorpher provides comprehensive error handling and logging capabilities:

**Python Error Handling**:

```python
import domorpher
import logging
from domorpher.exceptions import (
    ExtractionError,
    LlmProviderError,
    ValidationError,
    RateLimitError,
    ParsingError,
    NavigationError,
    TimeoutError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("domorpher.log"),
        logging.StreamHandler()
    ]
)

# Create extractor with logging
extractor = domorpher.Extractor(
    instruction="Extract all product information",
    error_handler=lambda e: logging.error(f"Extraction error: {e}")
)

try:
    results = extractor.extract_from_url("https://example.com/products")
except ExtractionError as e:
    logging.error(f"Extraction failed: {e}")
    # Try fallback strategy
    try:
        extractor = domorpher.Extractor(
            instruction="Extract all product information",
            adaptation_level="aggressive",
            llm_model="claude-3-opus"  # Use more powerful model as fallback
        )
        results = extractor.extract_from_url("https://example.com/products")
    except Exception as e:
        logging.critical(f"Fallback extraction failed: {e}")
        results = []
except LlmProviderError as e:
    logging.error(f"LLM provider error: {e}")
    # Try alternative provider
    try:
        extractor = domorpher.Extractor(
            instruction="Extract all product information",
            llm_provider="openai"  # Switch provider
        )
        results = extractor.extract_from_url("https://example.com/products")
    except Exception as e:
        logging.critical(f"Alternative provider failed: {e}")
        results = []
except RateLimitError as e:
    logging.warning(f"Rate limit exceeded: {e}")
    # Implement exponential backoff
    import time
    for attempt in range(5):
        time.sleep(2 ** attempt)
        try:
            results = extractor.extract_from_url("https://example.com/products")
            break
        except RateLimitError:
            logging.warning(f"Rate limit still exceeded (attempt {attempt + 1})")
    else:
        logging.error("All retry attempts failed")
        results = []
except Exception as e:
    logging.exception(f"Unexpected error: {e}")
    results = []
```

**Handling Autonomous Agent Errors**:

```python
import domorpher
import logging
from domorpher.exceptions import (
    NavigationError,
    ExecutionTimeoutError,
    MaxActionsExceededError,
    BrowserError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# Create agent with error handling
agent = domorpher.AutonomousAgent(
    objective="Complete checkout process with test credit card",
    timeout_seconds=120,
    max_actions=30
)

try:
    result = agent.execute("https://example.com/cart")
    
    if result.success:
        logger.info(f"Checkout completed successfully. Order ID: {result.extracted_data.get('order_id')}")
    else:
        logger.warning(f"Checkout completed but with issues: {result.warning_message}")
        
except NavigationError as e:
    logger.error(f"Navigation error: {e}")
    logger.info("Current page state saved to error_screenshot.png")
    agent.screenshot("error_screenshot.png")
    
except ExecutionTimeoutError as e:
    logger.error(f"Execution timed out after {e.seconds} seconds")
    logger.info("Partial results may be available")
    if e.partial_results:
        logger.info(f"Progress made: {e.progress_summary}")
        
except MaxActionsExceededError as e:
    logger.error(f"Exceeded maximum actions ({e.max_actions})")
    logger.info(f"Last action attempted: {e.last_action}")
    
except BrowserError as e:
    logger.error(f"Browser error: {e}")
    # Try with different browser
    try:
        alternate_agent = domorpher.AutonomousAgent(
            objective="Complete checkout process with test credit card",
            browser_options={"browser_type": "firefox"}  # Try Firefox instead
        )
        result = alternate_agent.execute("https://example.com/cart")
        logger.info("Successfully completed with alternate browser")
    except Exception as fallback_error:
        logger.critical(f"Fallback browser also failed: {fallback_error}")
        
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    
finally:
    # Save execution log regardless of outcome
    agent.save_execution_log("checkout_execution.log")
```

**Rust Error Handling**:

```rust
use domorpher::{Extractor, ExtractorConfig, Error};
use domorpher::error::{
    ExtractionError,
    LlmProviderError,
    ValidationError,
    RateLimitError,
    ParsingError
};
use log::{info, warn, error};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    // Create extractor
    let config = ExtractorConfig::builder()
        .instruction("Extract all product information")
        .build();

    let extractor = Extractor::new(config);

    // Attempt extraction with error handling
    match extractor.extract_from_url("https://example.com/products").await {
        Ok(results) => {
            info!("Extraction successful");
            // Process results
            // ...
        },
        Err(Error::Extraction(err)) => {
            error!("Extraction error: {}", err);

            // Try fallback strategy
            info!("Trying fallback strategy...");
            let fallback_config = ExtractorConfig::builder()
                .instruction("Extract all product information")
                .adaptation_level(AdaptationLevel::Aggressive)
                .model("claude-3-opus")
                .build();

            let fallback_extractor = Extractor::new(fallback_config);

            match fallback_extractor.extract_from_url("https://example.com/products").await {
                Ok(results) => {
                    info!("Fallback extraction successful");
                    // Process results
                    // ...
                },
                Err(err) => {
                    error!("Fallback extraction failed: {}", err);
                    // Handle failure
                    // ...
                }
            }
        },
        Err(Error::LlmProvider(err)) => {
            error!("LLM provider error: {}", err);

            // Try alternative provider
            info!("Trying alternative provider...");
            let alt_config = ExtractorConfig::builder()
                .instruction("Extract all product information")
                .llm_provider(LlmProvider::OpenAI)
                .build();

            let alt_extractor = Extractor::new(alt_config);

            match alt_extractor.extract_from_url("https://example.com/products").await {
                Ok(results) => {
                    info!("Alternative provider extraction successful");
                    // Process results
                    // ...
                },
                Err(err) => {
                    error!("Alternative provider failed: {}", err);
                    // Handle failure
                    // ...
                }
            }
        },
        Err(Error::RateLimit(err)) => {
            warn!("Rate limit exceeded: {}", err);

            // Implement exponential backoff
            for attempt in 0..5 {
                warn!("Retrying in {} seconds (attempt {})", 2u64.pow(attempt), attempt + 1);
                sleep(Duration::from_secs(2u64.pow(attempt))).await;

                match extractor.extract_from_url("https://example.com/products").await {
                    Ok(results) => {
                        info!("Retry successful");
                        // Process results
                        // ...
                        break;
                    },
                    Err(Error::RateLimit(_)) => {
                        warn!("Rate limit still exceeded");
                        continue;
                    },
                    Err(err) => {
                        error!("Error during retry: {}", err);
                        // Handle other errors
                        // ...
                        break;
                    }
                }
            }
        },
        Err(err) => {
            error!("Unexpected error: {}", err);
            // Handle other errors
            // ...
        }
    }

    Ok(())
}
```

## Advanced Features

### Extraction Templates

Templates allow reusing extraction patterns across multiple sites:

**Template Definition**:

```python
import domorpher

# Define a product extraction template
product_template = domorpher.Template(
    """
    For each product on the page, extract:
    - Product name
    - Brand
    - Regular price
    - Sale price (if available)
    - Discount percentage (if available)
    - Rating (numeric value and number of reviews)
    - Available colors and sizes
    - Whether the item is in stock
    - Shipping information
    - Product description (truncated to 200 characters)
    - Product features (as a list)
    - Image URL
    """,
    schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "brand": {"type": "string"},
                "regular_price": {"type": "number"},
                "sale_price": {"type": "number", "nullable": True},
                "discount_percentage": {"type": "number", "nullable": True},
                "rating": {"type": "number", "nullable": True},
                "review_count": {"type": "integer", "nullable": True},
                "colors": {"type": "array", "items": {"type": "string"}},
                "sizes": {"type": "array", "items": {"type": "string"}},
                "in_stock": {"type": "boolean"},
                "shipping_info": {"type": "string", "nullable": True},
                "description": {"type": "string", "nullable": True},
                "features": {"type": "array", "items": {"type": "string"}},
                "image_url": {"type": "string", "nullable": True}
            },
            "required": ["name", "regular_price", "in_stock"]
        }
    },
    site_optimizations={
        "amazon.com": {
            "selectors": {
                "product_container": ".s-result-item",
                "name": "h2 a span",
                "price": ".a-price .a-offscreen"
            },
            "wait_for": ".s-result-item",
            "additional_instructions": "Look for prime eligibility in the shipping information"
        },
        "walmart.com": {
            "selectors": {
                "product_container": ".product-tile",
                "name": ".product-title-link span",
                "price": ".price-main"
            },
            "wait_for": ".product-tile",
            "additional_instructions": "Extract 'pickup available' information"
        }
    }
)

# Save template for reuse
product_template.save("product_template.json")
```

**Template Usage**:

```python
import domorpher

# Load saved template
product_template = domorpher.Template.load("product_template.json")

# Use template across different sites
amazon_products = product_template.extract_from_url("https://www.amazon.com/s?k=laptop")
walmart_products = product_template.extract_from_url("https://www.walmart.com/browse/electronics/laptops/")
best_buy_products = product_template.extract_from_url("https://www.bestbuy.com/site/computer-tablets/laptops/")

# Compare prices across sites
products_by_name = {}

for product in amazon_products:
    products_by_name.setdefault(product["name"], {})["amazon"] = product["regular_price"]

for product in walmart_products:
    products_by_name.setdefault(product["name"], {})["walmart"] = product["regular_price"]

for product in best_buy_products:
    products_by_name.setdefault(product["name"], {})["best_buy"] = product["regular_price"]

# Find best deals
for name, prices in products_by_name.items():
    if len(prices) > 1:  # Product found on multiple sites
        best_site = min(prices.keys(), key=lambda site: prices[site])
        print(f"{name}: Best price ${prices[best_site]} at {best_site}")
```

**Template Sharing**:

```python
import domorpher
import json
import requests

# Load template
product_template = domorpher.Template.load("product_template.json")

# Export template as JSON
template_json = product_template.to_json()

# Share template
with open("shared_template.json", "w") as f:
    f.write(template_json)

# Template repository integration
response = requests.post(
    "https://template-repo.example.com/api/templates",
    json=json.loads(template_json),
    headers={"Authorization": "Bearer your_api_key"}
)

# Collaborative template improvement
improved_template = domorpher.Template.from_json(template_json)
improved_template.add_site_optimization("target.com", {
    "selectors": {
        "product_container": ".product-card",
        "name": ".product-title",
        "price": ".current-price"
    },
    "wait_for": ".product-card",
    "additional_instructions": "Extract RedCard discount information"
})
improved_template.save("improved_template.json")
```

### Schema Validation

DOMorpher supports comprehensive schema validation:

**JSON Schema**:

```python
import domorpher

# Define schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "description": {"type": "string", "nullable": True},
            "rating": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": 0, "maximum": 5},
                    "count": {"type": "integer", "minimum": 0}
                },
                "required": ["value"]
            },
            "in_stock": {"type": "boolean"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["name", "price", "in_stock"]
    }
}

# Create extractor with schema
extractor = domorpher.Extractor(
    instruction="Extract all products with their details",
    schema=schema,
    validation_mode="strict"  # Options: strict, lenient, none
)

# Extract with validation
products = extractor.extract_from_url("https://example.com/products")

# Products are guaranteed to match the schema
for product in products:
    print(f"Product: {product['name']}, Price: ${product['price']}")
    if product.get("rating"):
        print(f"Rating: {product['rating']['value']}/5 ({product['rating'].get('count', 'No')} reviews)")
```

**Pydantic Models**:

```python
import domorpher
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

# Define Pydantic models
class Rating(BaseModel):
    value: float = Field(..., ge=0, le=5)
    count: Optional[int] = Field(None, ge=0)

    @validator("value")
    def round_rating(cls, v):
        return round(v * 2) / 2  # Round to nearest 0.5

class Product(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    rating: Optional[Rating] = None
    in_stock: bool
    categories: List[str] = []
    metadata: Dict[str, Any] = {}

    @validator("price")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v

# Create extractor with Pydantic schema
extractor = domorpher.Extractor(
    instruction="Extract all products with their details",
    schema=List[Product]
)

# Extract with validation
products = extractor.extract_from_url("https://example.com/products")

# Products are Pydantic model instances
for product in products:
    print(f"Product: {product.name}, Price: ${product.price}")
    if product.rating:
        print(f"Rating: {product.rating.value}/5 ({product.rating.count or 'No'} reviews)")
```

**Custom Validation Rules**:

```python
import domorpher

# Define custom validation functions
def validate_prices(data):
    """Ensure sale prices are lower than regular prices."""
    if isinstance(data, list):
        for item in data:
            validate_prices(item)
    elif isinstance(data, dict):
        if "regular_price" in data and "sale_price" in data:
            if data["sale_price"] is not None and data["regular_price"] < data["sale_price"]:
                raise ValueError(f"Sale price ${data['sale_price']} should be lower than regular price ${data['regular_price']}")

        for value in data.values():
            if isinstance(value, (dict, list)):
                validate_prices(value)

def validate_stock_consistency(data):
    """Ensure stock status is consistent with available sizes/colors."""
    if isinstance(data, list):
        for item in data:
            validate_stock_consistency(item)
    elif isinstance(data, dict):
        if "in_stock" in data and data["in_stock"] is False:
            if "available_sizes" in data and data["available_sizes"]:
                raise ValueError("Product marked as out of stock but has available sizes")
            if "available_colors" in data and data["available_colors"]:
                raise ValueError("Product marked as out of stock but has available colors")

        for value in data.values():
            if isinstance(value, (dict, list)):
                validate_stock_consistency(value)

# Create validation ruleset
validation_rules = [
    validate_prices,
    validate_stock_consistency
]

# Create extractor with custom validation
extractor = domorpher.Extractor(
    instruction="Extract all products with their details",
    custom_validators=validation_rules
)

# Extract with validation
try:
    products = extractor.extract_from_url("https://example.com/products")
    print(f"Successfully extracted {len(products)} products")
except domorpher.exceptions.ValidationError as e:
    print(f"Validation error: {e}")
    # Handle validation failure
```

### Incremental Training

DOMorpher supports incremental training to improve extraction accuracy:

**Feedback Collection**:

```python
import domorpher

# Create extractor with feedback collection
extractor = domorpher.Extractor(
    instruction="Extract job listings with company, title, salary, and location",
    feedback_collection=True
)

# Initial extraction
jobs = extractor.extract_from_url("https://example.com/jobs")

# Provide feedback on incorrect extractions
extractor.provide_feedback(
    html_section='<div class="job-card">Senior Software Engineer at Acme Inc. - $80K-$120K - Remote</div>',
    expected_result={
        "company": "Acme Inc.",
        "title": "Senior Software Engineer",
        "salary_range": {"min": 80000, "max": 120000},
        "location": "Remote"
    }
)

extractor.provide_feedback(
    html_section='<div class="job-card">Data Scientist (Contract) - TechCorp - $50/hr - New York</div>',
    expected_result={
        "company": "TechCorp",
        "title": "Data Scientist",
        "contract_type": "Contract",
        "hourly_rate": 50,
        "location": "New York"
    }
)

# Save feedback for future use
extractor.save_feedback("job_extraction_feedback.json")
```

**Feedback Application**:

```python
import domorpher

# Create extractor with saved feedback
extractor = domorpher.Extractor(
    instruction="Extract job listings with company, title, salary, and location",
    feedback_file="job_extraction_feedback.json"
)

# Extraction now benefits from past feedback
improved_jobs = extractor.extract_from_url("https://example.com/jobs")

# Feedback can be applied selectively by site
site_specific_extractor = domorpher.Extractor(
    instruction="Extract job listings with company, title, salary, and location",
    feedback_file="job_extraction_feedback.json",
    feedback_application={
        "site_patterns": ["example.com", "jobs.example.org"],
        "apply_threshold": 0.8  # Minimum similarity for applying feedback
    }
)

# Extract from different site
other_jobs = site_specific_extractor.extract_from_url("https://anotherjobsite.com/listings")
```

**Continuous Learning**:

```python
import domorpher
import json
from datetime import datetime

class FeedbackCollector:
    def __init__(self, template_name, feedback_file):
        self.template_name = template_name
        self.feedback_file = feedback_file
        self.feedback_data = self.load_feedback()

    def load_feedback(self):
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"template": self.template_name, "examples": []}

    def save_feedback(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

    def add_feedback(self, html, expected, actual):
        self.feedback_data["examples"].append({
            "html": html,
            "expected": expected,
            "actual": actual,
            "timestamp": datetime.now().isoformat()
        })
        self.save_feedback()

    def create_extractor(self):
        return domorpher.Extractor(
            instruction=product_template.instruction,
            feedback_data=self.feedback_data
        )

# Initialize feedback collector
collector = FeedbackCollector("product_extraction", "product_feedback.json")

# Run extraction
extractor = collector.create_extractor()
products = extractor.extract_from_url("https://example.com/products")

# Let user review results and provide feedback
for i, product in enumerate(products):
    print(f"Product {i+1}:")
    print(f"Name: {product.get('name')}")
    print(f"Price: {product.get('price')}")

    is_correct = input("Is this extraction correct? (y/n): ").lower() == 'y'
    if not is_correct:
        # Get source HTML for this product
        product_html = extractor.get_source_html(i)

        # Get corrected data from user
        corrected_name = input(f"Correct name (was: {product.get('name')}): ") or product.get('name')
        corrected_price = float(input(f"Correct price (was: {product.get('price')}): ") or product.get('price'))

        corrected_data = {
            "name": corrected_name,
            "price": corrected_price
        }

        # Add feedback
        collector.add_feedback(product_html, corrected_data, product)

        print("Feedback recorded!")

# Future extractions will use the accumulated feedback
improved_extractor = collector.create_extractor()
improved_products = improved_extractor.extract_from_url("https://example.com/products")
```

### JavaScript Execution

DOMorpher can handle JavaScript-heavy websites:

**Basic JavaScript Support**:

```python
import domorpher

# Create extractor with JavaScript support
extractor = domorpher.Extractor(
    instruction="Extract product information from this interactive catalog",
    javascript_support=True,
    wait_for_selector=".product-grid.loaded"
)

# Extract from JavaScript-heavy page
products = extractor.extract_from_url("https://example.com/interactive-catalog")
```

**Advanced JavaScript Handling**:

```python
import domorpher

# Create extractor with advanced JavaScript options
extractor = domorpher.Extractor(
    instruction="Extract all reviews from this product page",
    javascript={
        "enabled": True,
        "wait_until": "networkidle",  # wait for network to be idle
        "timeout": 30,  # seconds
        "user_agent": "Mozilla/5.0 ...",
        "viewport": {"width": 1280, "height": 800},
        "scroll_to_bottom": True,
        "click_selectors": [".load-more-reviews", ".show-all-reviews"],
        "wait_for_selectors": [".review-list.fully-loaded"],
        "execute_scripts": [
            "window.scrollTo(0, document.body.scrollHeight);",
            "document.querySelector('.load-more-reviews')?.click();"
        ]
    }
)

# Extract from complex interactive page
reviews = extractor.extract_from_url("https://example.com/product/reviews")
```

**Interaction Scripting**:

```python
import domorpher

# Define interaction script
interaction_script = """
// Wait for page to load
await page.waitForSelector('.product-filters');

// Select category from dropdown
await page.click('.category-dropdown');
await page.waitForSelector('.category-dropdown-menu');
await page.click('.category-dropdown-menu .category-item:nth-child(3)');

// Wait for products to load
await page.waitForSelector('.product-grid.loaded');

// Sort by price
await page.click('.sort-dropdown');
await page.waitForSelector('.sort-dropdown-menu');
await page.click('.sort-dropdown-menu .sort-price-low');

// Wait for sorted products
await page.waitForSelector('.product-grid.sorted');

// Apply price filter
await page.click('.price-filter-min');
await page.keyboard.type('100');
await page.click('.price-filter-max');
await page.keyboard.type('500');
await page.click('.apply-filters');

// Wait for filtered products
await page.waitForTimeout(2000);
await page.waitForSelector('.product-grid.filtered');
"""

# Create extractor with interaction script
extractor = domorpher.Extractor(
    instruction="Extract all products between $100 and $500 in the Electronics category, sorted by price",
    javascript={
        "enabled": True,
        "interaction_script": interaction_script,
        "take_screenshot": True,
        "screenshot_path": "filtered_products.png"
    }
)

# Extract from interactive page with complex navigation
products = extractor.extract_from_url("https://example.com/products")

# Get interaction debugging information
js_logs = extractor.get_javascript_logs()
console_messages = extractor.get_console_messages()
errors = extractor.get_javascript_errors()

if errors:
    print("JavaScript errors occurred:")
    for error in errors:
        print(f"  - {error}")
```

### Multi-Modal Extraction

DOMorpher can extract information from multiple types of content:

**Text and Image Analysis**:

```python
import domorpher

# Create extractor with multi-modal capabilities
extractor = domorpher.Extractor(
    instruction="""
    For each product, extract:
    - Product name and price from the text
    - Main colors visible in the product image
    - Number of items shown in the image
    - Whether the product image shows the item in use/being worn
    """,
    capabilities=["text", "vision"],
    vision_model="claude-3-opus"  # Supports multi-modal analysis
)

# Extract from page with product images
products = extractor.extract_from_url("https://example.com/products")

# Results include image-derived information
for product in products:
    print(f"Product: {product['name']}")
    print(f"Price: ${product['price']}")
    print(f"Colors: {', '.join(product['colors'])}")
    print(f"Items shown: {product['item_count']}")
    print(f"Shown in use: {'Yes' if product['shown_in_use'] else 'No'}")
```

**Table Extraction**:

```python
import domorpher
import pandas as pd

# Create extractor focused on tabular data
extractor = domorpher.Extractor(
    instruction="""
    Extract the complete specification table for this product, including:
    - All specification categories and their values
    - Technical specifications with units
    - Compatibility information
    """,
    format="table"
)

# Extract table data
table_data = extractor.extract_from_url("https://example.com/product/specs")

# Convert to pandas DataFrame
specs_df = pd.DataFrame(table_data)

# Analyze specifications
numeric_specs = specs_df[specs_df['value'].str.contains(r'\d')].copy()
numeric_specs['numeric_value'] = numeric_specs['value'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

# Find products with highest specifications
highest_specs = {}
for category in numeric_specs['category'].unique():
    category_specs = numeric_specs[numeric_specs['category'] == category]
    highest = category_specs.loc[category_specs['numeric_value'].idxmax()]
    highest_specs[category] = {
        'value': highest['value'],
        'product': highest['product'] if 'product' in highest else 'This product'
    }

print("Highest specifications:")
for category, info in highest_specs.items():
    print(f"{category}: {info['value']} ({info['product']})")
```

**Interactive Element Analysis**:

```python
import domorpher

# Create extractor for interactive elements
extractor = domorpher.Extractor(
    instruction="""
    For the product configurator, extract:
    - All available configuration options and their values
    - Price impact of each configuration choice
    - Compatibility between different options
    - Default selected options
    - Currently selected options
    """,
    javascript={
        "enabled": True,
        "interaction_mode": "analyze_interactive"
    }
)

# Extract from interactive product configurator
configuration = extractor.extract_from_url("https://example.com/product/configure")

# Access configuration options
options = configuration["options"]
for category, choices in options.items():
    print(f"{category} options:")
    for choice in choices:
        price_impact = choice.get("price_impact", 0)
        price_str = f"+${price_impact}" if price_impact > 0 else f"-${abs(price_impact)}" if price_impact < 0 else "No change"
        print(f"  - {choice['name']}: {price_str}")

        if choice.get("incompatible_with"):
            print(f"    Incompatible with: {', '.join(choice['incompatible_with'])}")
```

### DOM Navigation Strategies

DOMorpher provides configurable navigation strategies for autonomous agents:

**Strategy Configuration**:

```python
import domorpher

# Create agent with specific navigation strategy
agent = domorpher.AutonomousAgent(
    objective="Find and compare the top 3 best-selling laptops",
    navigation_strategy={
        # Overall approach
        "approach": "semantic_first",  # Focus on semantic understanding over DOM structure
        
        # Exploration depth
        "depth": "comprehensive",  # Explore deeply into the site's content
        
        # Exploration width
        "breadth": "focused",  # Focus on most relevant paths, not all possibilities
        
        # Navigation patience
        "patience": "high",  # Try multiple approaches before giving up on a goal
        
        # Error recovery
        "recovery": "aggressive",  # Actively recover from navigation errors
        
        # Exploration pattern
        "pattern": "goal_oriented"  # Prioritize paths that lead toward the objective
    }
)

# Execute with strategy
result = agent.execute("https://example.com/electronics")
```

**Semantic-First vs. Structure-First**:

```python
# Semantic-first (understands meaning over structure)
semantic_agent = domorpher.AutonomousAgent(
    objective="Find laptops with at least 16GB RAM under $1000",
    navigation_strategy={
        "approach": "semantic_first",
        "semantic_recognition": {
            "prioritize_text": True,   # Focus on text content
            "context_aware": True,     # Consider surrounding context
            "category_detection": True  # Identify product categories by meaning
        }
    }
)

# Structure-first (understands DOM patterns)
structure_agent = domorpher.AutonomousAgent(
    objective="Find laptops with at least 16GB RAM under $1000",
    navigation_strategy={
        "approach": "structure_first",
        "structural_recognition": {
            "prioritize_hierarchy": True,  # Focus on DOM hierarchy
            "pattern_matching": True,      # Identify repeated DOM patterns
            "element_mapping": True        # Map visual layout to DOM structure
        }
    }
)
```

**Navigation Patterns**:

```python
# Different exploration patterns
breadth_first_agent = domorpher.AutonomousAgent(
    objective="Research electric vehicles",
    navigation_strategy={
        "exploration": "breadth_first",  # Explore all options at current level before going deeper
        "max_breadth": 5,                # Maximum number of parallel paths to explore
        "min_confidence": 0.3            # Minimum confidence to explore a path
    }
)

depth_first_agent = domorpher.AutonomousAgent(
    objective="Research electric vehicles",
    navigation_strategy={
        "exploration": "depth_first",   # Follow most promising path deeply before trying alternatives
        "max_depth": 5,                 # Maximum click depth to explore
        "backtracking": True,           # Return to previous points to try alternative paths
        "min_confidence": 0.6           # Minimum confidence to explore a path
    }
)

hybrid_agent = domorpher.AutonomousAgent(
    objective="Research electric vehicles",
    navigation_strategy={
        "exploration": "adaptive",      # Adapt strategy based on site structure and results
        "priority_threshold": 0.7,      # Threshold for prioritizing depth vs breadth
        "exploration_timeout": 120,     # Maximum seconds to spend exploring
        "result_threshold": 5           # Stop after finding this many relevant results
    }
)
```

**Interactive Element Recognition**:

```python
# Configure how the agent recognizes and prioritizes interactive elements
agent = domorpher.AutonomousAgent(
    objective="Complete the checkout process",
    navigation_strategy={
        "element_recognition": {
            "prioritize": [
                "buttons",         # Focus on buttons first
                "inputs",          # Then form inputs
                "select",          # Then dropdown selects
                "links"            # Then regular links
            ],
            "text_match_threshold": 0.8,  # Semantic similarity threshold for text matching
            "visual_priority": "high",    # Consider visual prominence when selecting elements
            "form_recognition": "deep"    # Deeply understand form structure and relationships
        }
    }
)
```

### Autonomous Web Agents

DOMorpher's autonomous web agents leverage LLM reasoning to navigate and interact with web pages:

**Agent Architecture**:

The autonomous agent architecture consists of five core components:

1. **DOM Intelligence Engine**: Analyzes the DOM to understand page structure and content
2. **Action Planning System**: Determines what actions to take based on the objective
3. **DOM Execution Layer**: Executes actions on the page by interacting with elements
4. **Observation System**: Monitors page changes and action results
5. **Memory Manager**: Maintains context and history across multiple pages

**Perception and Reasoning Loop**:

```python
class DOMPerception:
    def analyze_current_state(self):
        # Capture the current DOM state
        html = self.page.content()

        # Get a list of all interactive elements
        interactive_elements = self.page.evaluate("""
            () => {
                const elements = [];
                // Find all potentially interactive elements
                const interactives = document.querySelectorAll(
                    'button, a, input, select, [role="button"], [tabindex]:not([tabindex="-1"])'
                );

                // Gather details about each element
                interactives.forEach((el, index) => {
                    // Element details extraction...
                });
                return elements;
            }
        """)

        return {
            "html": html,
            "interactive_elements": interactive_elements,
            "url": self.page.url(),
            "title": self.page.title()
        }

class LLMReasoning:
    def determine_next_action(self, page_state, user_objective, context):
        # Create prompt for the LLM
        prompt = f"""
        You are an autonomous web agent with the objective: "{user_objective}"

        The current webpage contains the following interactive elements:
        {self._format_elements(page_state["interactive_elements"])}

        Based on the objective and the available elements, what action should I take next?

        Options include:
        1. Click an element (specify which one by index or description)
        2. Input text into a field (specify which field and what text)
        3. Select an option from a dropdown (specify which dropdown and what option)
        4. Scroll the page (specify direction and amount)
        5. Wait for an element to appear (specify what to look for)
        6. Navigate to a different URL (specify the URL)
        7. Extract information from the current page (specify what information)
        8. Complete the task (if the objective has been achieved)
        """

        # Get LLM's decision
        response = self.llm.generate(prompt)

        # Parse the action from the response
        return self._parse_action(response)
```

**Action Execution**:

```python
class WebActionExecutor:
    async def execute_action(self, action):
        if action["type"] == "click":
            await self._execute_click(action["target"])
        elif action["type"] == "input":
            await self._execute_input(action["target"], action["value"])
        elif action["type"] == "select":
            await self._execute_select(action["target"], action["value"])
        elif action["type"] == "scroll":
            await self._execute_scroll(action["direction"], action["amount"])
        elif action["type"] == "wait":
            await self._execute_wait(action["target"])
        elif action["type"] == "navigate":
            await self._execute_navigate(action["url"])
        elif action["type"] == "extract":
            return await self._execute_extract(action["data_points"])
        elif action["type"] == "complete":
            return {"status": "complete", "message": "Task completed successfully"}

    async def _execute_click(self, target):
        if isinstance(target, int):
            # Click by element index
            await self.page.evaluate(f"""
                (index) => {{
                    const elements = document.querySelectorAll(
                        'button, a, input, select, [role="button"], [tabindex]:not([tabindex="-1"])'
                    );
                    if (elements[index]) elements[index].click();
                }}
            """, target)
        else:
            # Click by description (using text content)
            await self.page.evaluate(f"""
                (text) => {{
                    const elements = Array.from(document.querySelectorAll(
                        'button, a, input, select, [role="button"], [tabindex]:not([tabindex="-1"])'
                    ));
                    const element = elements.find(el =>
                        el.innerText.includes(text) ||
                        el.value?.includes(text) ||
                        el.placeholder?.includes(text)
                    );
                    if (element) element.click();
                }}
            """, target)
```

**Memory and Context Management**:

```python
class AgentMemory:
    def __init__(self):
        self.action_history = []
        self.page_states = []
        self.extracted_data = {}
        self.navigation_path = []
        self.task_progress = {}

    def store_action(self, action, result):
        self.action_history.append({
            "action": action,
            "result": result,
            "timestamp": time.time()
        })

    def store_page_state(self, state):
        # Keep a limited history to save memory
        if len(self.page_states) > 10:
            self.page_states.pop(0)
        self.page_states.append(state)
        
        # Track navigation path
        self.navigation_path.append({
            "url": state["url"],
            "title": state["title"],
            "timestamp": time.time()
        })

    def update_task_progress(self, progress_info):
        self.task_progress.update(progress_info)

    def get_context_for_llm(self):
        # Create a rich context from history that can help the LLM make better decisions
        return {
            "recent_actions": self.action_history[-5:],
            "task_progress": self.task_progress,
            "navigation_path": self.navigation_path[-5:]
        }
```

**Main Agent Loop**:

```python
class AutonomousWebAgent:
    async def execute(self, start_url, objective, max_steps=50):
        # Initialize browser page
        await self.page.goto(start_url)

        # Main interaction loop
        step_count = 0
        while step_count < max_steps:
            # 1. Perceive the current state
            page_state = await self.perception.analyze_current_state()
            self.memory.store_page_state(page_state)

            # 2. Determine the next action
            context = self.memory.get_context_for_llm()
            action = await self.reasoning.determine_next_action(
                page_state,
                objective,
                context
            )

            # 3. Execute the action
            result = await self.executor.execute_action(action)
            self.memory.store_action(action, result)

            # 4. Check if task is complete
            if action["type"] == "complete":
                return {
                    "success": True,
                    "steps_taken": step_count,
                    "extracted_data": self.memory.extracted_data,
                    "navigation_path": self.memory.navigation_path
                }

            # 5. Handle any extracted data
            if action["type"] == "extract" and result.get("data"):
                self.memory.extracted_data.update(result["data"])

            # Allow time for page to respond to action
            await self.page.waitForTimeout(1000)
            step_count += 1

        # If we reach max steps without completion
        return {
            "success": False,
            "reason": "Maximum step count reached without completing the task",
            "steps_taken": step_count,
            "extracted_data": self.memory.extracted_data,
            "navigation_path": self.memory.navigation_path
        }
```

**Handling Complex Scenarios**:

```python
# Example of an agent handling multi-step workflows
checkout_agent = domorpher.AutonomousAgent(
    objective="""
    Complete the checkout process with these details:
    - Add any laptop under $1000 to cart
    - Proceed to checkout
    - Fill shipping address as:
        - Name: John Smith
        - Address: 123 Main St
        - City: Boston
        - State: MA
        - Zip: 02108
        - Phone: 555-123-4567
    - Select standard shipping
    - Use test credit card 4111 1111 1111 1111, exp 12/25, CVV 123
    - Place the order
    - Extract order confirmation number
    """,
    navigation_strategy={
        "approach": "semantic_first",
        "form_recognition": "deep",
        "error_recovery": "aggressive"
    },
    browser_options={
        "headless": False,  # To see the process
        "slow_mo": 100      # Slow down actions for visibility
    }
)

result = checkout_agent.execute("https://example.com/laptops")

if result.success:
    print(f"Order placed successfully!")
    print(f"Confirmation number: {result.extracted_data.get('confirmation_number')}")
    print(f"Order total: ${result.extracted_data.get('order_total')}")
else:
    print(f"Failed to complete checkout: {result.reason}")
    print(f"Progress made: {len(result.navigation_path)} pages navigated")
```

**Visual Enhancement Option**:

```python
# Agent with visual enhancement for complex pages
visual_agent = domorpher.AutonomousAgent(
    objective="Find and book the cheapest flight from NYC to London for next month",
    multimodal_enhancement={
        "enabled": True,                 # Enable visual analysis
        "trigger": "on_dom_ambiguity",   # Only use when DOM is ambiguous
        "visual_context_level": "full",  # Full page screenshots
        "element_highlighting": True,    # Highlight elements for better identification
        "visual_reasoning": "deep"       # Use deep visual reasoning
    }
)

result = visual_agent.execute("https://example.com/flights")
```

## Performance Considerations

### Memory Usage Optimization

DOMorpher implements several memory optimization strategies:

**Chunked Processing**:

```python
import domorpher

# Configure memory-optimized extractor
extractor = domorpher.Extractor(
    instruction="Extract all products from this large catalog",
    memory_optimization={
        "enabled": True,
        "chunk_size": 5000,  # Characters per chunk
        "max_memory_mb": 500,  # Target maximum memory usage
        "cleanup_threshold": 0.8,  # Trigger cleanup at 80% of max memory
        "incremental_results": True  # Return results as they become available
    }
)

# Process large page
for partial_results in extractor.extract_from_url_streaming("https://example.com/large-catalog"):
    # Process results in smaller batches
    for product in partial_results:
        process_product(product)

    # Allow garbage collection between chunks
    import gc
    gc.collect()
```

**Resource Monitoring**:

```python
import domorpher
import psutil
import time

class ResourceMonitor:
    def __init__(self, memory_threshold_mb=1000):
        self.memory_threshold_mb = memory_threshold_mb
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.get_memory_usage()
        self.peak_memory = self.start_memory

    def get_memory_usage(self):
        return self.process.memory_info().rss / (1024 * 1024)  # MB

    def check(self):
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)

        # Check if memory usage exceeds threshold
        if current_memory > self.memory_threshold_mb:
            return False
        return True

    def report(self):
        elapsed = time.time() - self.start_time
        current_memory = self.get_memory_usage()

        return {
            "elapsed_seconds": elapsed,
            "start_memory_mb": self.start_memory,
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": current_memory - self.start_memory
        }

# Create monitor
monitor = ResourceMonitor(memory_threshold_mb=2000)

# Create extractor with monitoring
extractor = domorpher.Extractor(
    instruction="Extract all articles from this news archive",
    on_chunk_processed=lambda chunk_idx: monitor.check()
)

try:
    # Extract data
    results = extractor.extract_from_url("https://example.com/news-archive")

    # Report resource usage
    usage = monitor.report()
    print(f"Extraction completed in {usage['elapsed_seconds']:.2f} seconds")
    print(f"Peak memory usage: {usage['peak_memory_mb']:.2f} MB")
    print(f"Memory increase: {usage['memory_increase_mb']:.2f} MB")

except MemoryError:
    # Handle memory limit exceeded
    print("Memory limit exceeded. Trying with more aggressive chunking...")

    # Create more memory-efficient extractor
    extractor = domorpher.Extractor(
        instruction="Extract all articles from this news archive",
        memory_optimization={
            "enabled": True,
            "chunk_size": 2000,  # Smaller chunks
            "aggressive_gc": True
        }
    )

    # Try again with optimized settings
    results = list(extractor.extract_from_url_streaming("https://example.com/news-archive"))
```

**Optimized Processing Pipeline**:

```python
import domorpher
from domorpher.steps import (
    FetchURL,
    PreprocessHTML,
    ExtractMainContent,
    ChunkContent,
    ProcessChunks,
    MergeResults
)

# Create memory-optimized pipeline
pipeline = domorpher.Pipeline([
    # Fetch URL with streaming enabled
    FetchURL(streaming=True),

    # Preprocess while streaming
    PreprocessHTML(streaming=True, minimal=True),

    # Extract main content to reduce processing load
    ExtractMainContent(),

    # Chunk content for efficient processing
    ChunkContent(
        strategy="adaptive",
        min_chunk_size=2000,
        max_chunk_size=8000,
        overlap=200
    ),

    # Process chunks in parallel with controlled concurrency
    ProcessChunks(
        extractor=domorpher.Extractor(
            instruction="Extract article titles, authors, and publication dates"
        ),
        max_concurrency=2,  # Control parallelism
        timeout=30,  # Seconds per chunk
        error_handling="skip"  # Continue even if some chunks fail
    ),

    # Merge results with duplicate removal
    MergeResults(
        remove_duplicates=True,
        sort_by="confidence",
        post_processing=lambda results: results[:100]  # Limit final results
    )
])

# Process large site with limited memory
results = pipeline.process("https://example.com/large-news-archive")
```

### Rate Limiting

DOMorpher provides comprehensive rate limiting capabilities:

**Basic Rate Limiting**:

```python
import domorpher

# Configure rate limiting
domorpher.configure(
    rate_limiting={
        "enabled": True,
        "rpm": 10,  # Requests per minute
        "max_concurrent": 2,
        "retry_count": 3,
        "retry_delay": 5,  # Seconds
        "provider_limits": {
            "anthropic": {"rpm": 5},
            "openai": {"rpm": 20}
        }
    }
)

# Create extractor with rate limiting
extractor = domorpher.Extractor(
    instruction="Extract product information",
    rate_limiting={
        "rpm": 5,  # Override global setting
        "backoff_factor": 2.0  # Exponential backoff
    }
)

# Extract from multiple URLs with rate limiting
urls = [
    "https://example.com/products/page1",
    "https://example.com/products/page2",
    "https://example.com/products/page3",
    # ...more URLs
]

all_products = []
for url in urls:
    # Rate limiting is automatically applied
    products = extractor.extract_from_url(url)
    all_products.extend(products)
```

**Advanced Rate Limiting**:

```python
import domorpher
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class RateLimitConfig:
    rpm: int
    burst: int = 2
    concurrent: int = 1
    cooldown_period: float = 5.0

class RateLimiter:
    def __init__(self, config: Dict[str, RateLimitConfig]):
        self.config = config
        self.request_times: Dict[str, List[float]] = {}
        self.active_requests: Dict[str, int] = {}

    async def acquire(self, provider: str) -> bool:
        if provider not in self.config:
            return True

        config = self.config[provider]

        # Initialize if needed
        if provider not in self.request_times:
            self.request_times[provider] = []
            self.active_requests[provider] = 0

        # Check concurrent limit
        if self.active_requests[provider] >= config.concurrent:
            return False

        # Clean up old request times
        now = time.time()
        min_time = now - 60.0  # 1 minute window
        self.request_times[provider] = [t for t in self.request_times[provider] if t >= min_time]

        # Check rate limit
        if len(self.request_times[provider]) >= config.rpm:
            # Check if burst capacity is available
            if len(self.request_times[provider]) < config.rpm + config.burst:
                # Check cooldown period
                cooldown_end = self.request_times[provider][0] + config.cooldown_period
                if now < cooldown_end:
                    return False
            else:
                return False

        # Acquire permit
        self.request_times[provider].append(now)
        self.active_requests[provider] += 1
        return True

    def release(self, provider: str) -> None:
        if provider in self.active_requests:
            self.active_requests[provider] = max(0, self.active_requests[provider] - 1)

# Create rate limiter
rate_limiter = RateLimiter({
    "anthropic": RateLimitConfig(rpm=5, burst=2, concurrent=1),
    "openai": RateLimitConfig(rpm=20, burst=5, concurrent=3),
    "website": RateLimitConfig(rpm=10, burst=0, concurrent=2)
})

# Create custom executor with rate limiting
async def rate_limited_extract(extractor, urls):
    results = []

    for url in urls:
        # Wait for website rate limit permit
        while not await rate_limiter.acquire("website"):
            await asyncio.sleep(1)

        try:
            # Extract from URL (this will also use the LLM rate limits internally)
            result = await extractor.extract_from_url_async(url)
            results.append(result)
        finally:
            # Release permit
            rate_limiter.release("website")

    return results

# Create extractor
extractor = domorpher.Extractor(
    instruction="Extract product information",
    rate_limiter=rate_limiter  # Use custom rate limiter
)

# Extract from multiple URLs with rate limiting
urls = ["https://example.com/products/1", "https://example.com/products/2", "https://example.com/products/3"]
all_products = asyncio.run(rate_limited_extract(extractor, urls))
```

### Caching Strategies

DOMorpher implements several caching strategies:

**Basic Caching**:

```python
import domorpher

# Configure caching
domorpher.configure(
    cache={
        "enabled": True,
        "strategy": "disk",
        "directory": "./domorpher_cache",
        "ttl": 3600,  # Seconds
        "max_size": 1024  # MB
    }
)

# Create extractor with caching
extractor = domorpher.Extractor(
    instruction="Extract product information",
    cache={
        "html": True,  # Cache HTML content
        "results": True,  # Cache extraction results
        "llm": True,  # Cache LLM responses
        "ttl": 86400  # Override TTL for this extractor (24 hours)
    }
)

# Extract with caching
# First call will be slow, subsequent calls will be fast
products1 = extractor.extract_from_url("https://example.com/products")

# This will use cached results
products2 = extractor.extract_from_url("https://example.com/products")

# Force refresh cache
products3 = extractor.extract_from_url("https://example.com/products", cache_bypass=True)
```

**Advanced Caching**:

```python
import domorpher
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class SmartCache:
    def __init__(self, base_dir: str, ttl: int = 3600):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_key(self, url: str, instruction: str, options: Dict[str, Any]) -> str:
        """Generate a cache key based on URL, instruction, and options."""
        key_data = {
            "url": url,
            "instruction": instruction,
            "options": {k: v for k, v in options.items() if k != "cache_bypass"}
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _get_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.base_dir / f"{key}.json"

    def get(self, url: str, instruction: str, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result if available and valid."""
        if options.get("cache_bypass", False):
            return None

        key = self._get_key(url, instruction, options)
        path = self._get_path(key)

        if not path.exists():
            return None

        # Check if cache is expired
        modified_time = path.stat().st_mtime
        if time.time() - modified_time > self.ttl:
            return None

        try:
            with open(path, "r") as f:
                cached_data = json.load(f)

            # Validate cache structure
            if not isinstance(cached_data, dict) or "timestamp" not in cached_data or "results" not in cached_data:
                return None

            return cached_data["results"]
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, url: str, instruction: str, options: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Cache extraction results."""
        key = self._get_key(url, instruction, options)
        path = self._get_path(key)

        cache_data = {
            "timestamp": time.time(),
            "url": url,
            "instruction": instruction,
            "results": results
        }

        with open(path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def clear(self, url: Optional[str] = None, older_than: Optional[int] = None) -> int:
        """Clear cache entries.

        Args:
            url: Optional URL to clear. If None, clears all entries.
            older_than: Optional age in seconds. If provided, only clears entries older than this.

        Returns:
            Number of entries cleared.
        """
        count = 0
        now = time.time()

        for path in self.base_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    cached_data = json.load(f)

                if url is not None and cached_data.get("url") != url:
                    continue

                if older_than is not None:
                    timestamp = cached_data.get("timestamp", 0)
                    if now - timestamp < older_than:
                        continue

                path.unlink()
                count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                # If there's any error, delete the file
                path.unlink(missing_ok=True)
                count += 1

        return count

# Create smart cache
cache = SmartCache("./smart_cache", ttl=86400)

# Create extractor with custom cache
extractor = domorpher.Extractor(
    instruction="Extract product information",
    custom_cache=cache
)

# Extract with custom caching
products = extractor.extract_from_url("https://example.com/products")

# Clear old cache entries
cleared_count = cache.clear(older_than=7*86400)  # Clear entries older than 7 days
print(f"Cleared {cleared_count} old cache entries")
```

### Parallel Processing

DOMorpher supports parallel processing for improved performance:

**Basic Parallelism**:

```python
import domorpher
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure parallelism
domorpher.configure(
    parallelism={
        "max_workers": 4,
        "chunk_processing": True,
        "url_fetching": True
    }
)

# Create extractor with parallel processing
extractor = domorpher.Extractor(
    instruction="Extract product information",
    parallelism={
        "chunks": True,  # Process chunks in parallel
        "max_chunk_concurrency": 2  # Maximum parallel chunk processing
    }
)

# Extract from multiple URLs in parallel
async def extract_all(urls):
    tasks = [extractor.extract_from_url_async(url) for url in urls]
    return await asyncio.gather(*tasks)

urls = [
    "https://example.com/products/page1",
    "https://example.com/products/page2",
    "https://example.com/products/page3",
    "https://example.com/products/page4"
]

# Run async extraction
all_products = asyncio.run(extract_all(urls))

# Flatten results
products = [item for sublist in all_products for item in sublist]
```

**Parallel Agent Execution**:

```python
import domorpher
import asyncio

# Create an objective to run across multiple sites
objective = "Find and compare the top 3 best-selling smartphones under $500"

# List of electronics retailers to check
sites = [
    "https://www.retailer1.com/electronics",
    "https://www.retailer2.com/mobile",
    "https://www.retailer3.com/smartphones",
    "https://www.retailer4.com/cell-phones"
]

# Function to run parallel agents
async def run_parallel_agents(sites, objective, max_concurrent=2):
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(site):
        async with semaphore:
            agent = domorpher.AutonomousAgent(objective=objective)
            return await agent.execute_async(site)
    
    # Create tasks for each site
    tasks = [execute_with_semaphore(site) for site in sites]
    
    # Run all tasks and gather results
    return await asyncio.gather(*tasks)

# Run parallel execution
results = asyncio.run(run_parallel_agents(sites, objective, max_concurrent=2))

# Process results from all sites
all_phones = []
for i, result in enumerate(results):
    if result.success:
        site_name = sites[i].split('/')[2]
        for phone in result.extracted_data:
            phone['source'] = site_name
            all_phones.append(phone)

# Find the best deals across all sites
if all_phones:
    # Sort by price
    all_phones.sort(key=lambda p: p.get('price', float('inf')))
    
    print("Top 3 smartphones under $500 across all retailers:")
    for i, phone in enumerate(all_phones[:3]):
        print(f"{i+1}. {phone['name']} - ${phone['price']} at {phone['source']}")
        print(f"   Rating: {phone.get('rating', 'N/A')}")
        print(f"   Key features: {', '.join(phone.get('features', []))}")
        print()
else:
    print("No results found across any retailers.")
```

**Process Pool Execution**:

```python
import domorpher
import concurrent.futures
import os

def process_url(url_data):
    """Process a single URL with DOMorpher."""
    url, instruction = url_data
    try:
        results = domorpher.extract_from_url(url, instruction)
        return {
            "url": url,
            "success": True,
            "results": results,
            "error": None
        }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "results": None,
            "error": str(e)
        }

# List of URLs to process with their instructions
url_instructions = [
    ("https://example.com/page1", "Extract all product information"),
    ("https://example.com/page2", "Extract all product information"),
    ("https://example.com/page3", "Extract all product information"),
    ("https://example.com/page4", "Extract all product information"),
    ("https://example.com/page5", "Extract all product information"),
]

# Number of worker processes (default to CPU count)
num_workers = os.cpu_count() or 4

# Use ProcessPoolExecutor for truly parallel execution
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Submit all URLs for processing
    future_to_url = {executor.submit(process_url, url_data): url_data[0] for url_data in url_instructions}
    
    # Process results as they complete
    results = []
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"Successfully processed {url}: {len(result['results'])} items extracted")
            else:
                print(f"Failed to process {url}: {result['error']}")
        except Exception as e:
            print(f"Exception while processing {url}: {e}")
            
# Combine all successful results
all_data = []
for result in results:
    if result["success"]:
        all_data.extend(result["results"])

print(f"Total items extracted: {len(all_data)}")
```

## Security Considerations

### API Key Management

DOMorpher provides several methods for secure API key management:

**Environment Variables**:

```python
# Set environment variables before running
# export DOMORPHER_ANTHROPIC_API_KEY=your_api_key
# export DOMORPHER_OPENAI_API_KEY=your_api_key

import domorpher
import os

# Keys will be loaded automatically from environment variables
extractor = domorpher.Extractor(
    instruction="Extract product information",
    llm_provider="anthropic"  # Will use DOMORPHER_ANTHROPIC_API_KEY
)

# Manually check if keys are available
if not os.environ.get("DOMORPHER_ANTHROPIC_API_KEY"):
    print("Warning: Anthropic API key not found in environment variables")
```

**Configuration File**:

```python
import domorpher

# Load configuration from file with environment variable interpolation
domorpher.configure_from_file("config.json")

# Example config.json:
# {
#   "llm": {
#     "providers": {
#       "anthropic": {
#         "api_key": "${ANTHROPIC_API_KEY}",
#         "default_model": "claude-3-sonnet"
#       }
#     }
#   }
# }
```

**Secret Manager Integration**:

```python
import domorpher
from some_secret_manager import get_secret

# Load keys from a secret manager
anthropic_key = get_secret("anthropic-api-key")
openai_key = get_secret("openai-api-key")

# Configure with retrieved keys
domorpher.configure(
    anthropic_api_key=anthropic_key,
    openai_api_key=openai_key
)
```

**Key Rotation**:

```python
import domorpher
import threading
import time

# Set up key rotation
def rotate_api_keys():
    while True:
        try:
            # Get new keys from secret manager
            new_anthropic_key = get_secret("anthropic-api-key")
            new_openai_key = get_secret("openai-api-key")
            
            # Update configuration with new keys
            domorpher.configure(
                anthropic_api_key=new_anthropic_key,
                openai_api_key=new_openai_key
            )
            
            print("API keys rotated successfully")
        except Exception as e:
            print(f"Error rotating API keys: {e}")
        
        # Sleep for 24 hours
        time.sleep(24 * 60 * 60)

# Start key rotation in background thread
rotation_thread = threading.Thread(target=rotate_api_keys, daemon=True)
rotation_thread.start()
```

### Data Privacy

DOMorpher includes features to ensure data privacy:

**Content Filtering**:

```python
import domorpher

# Create extractor with privacy filters
extractor = domorpher.Extractor(
    instruction="Extract contact information",
    privacy_settings={
        "filter_pii": True,  # Filter personally identifiable information
        "filter_types": [
            "credit_card",
            "ssn",
            "passport",
            "phone",
            "email"
        ],
        "redaction_mode": "mask",  # Options: "mask", "remove", "replace"
        "replacement_token": "[REDACTED]"
    }
)

# Extract with privacy filtering
results = extractor.extract_from_url("https://example.com/contacts")

# Results will have PII filtered according to settings
```

**Local Processing Options**:

```python
import domorpher

# Configure local extraction for privacy-sensitive data
extractor = domorpher.Extractor(
    instruction="Extract medical information from patient records",
    llm_provider="local",  # Use local model
    llm_model="local/phi-3-mini",  # Local model path
    data_privacy={
        "local_only": True,  # Process everything locally
        "no_external_requests": True,  # No external network requests
        "no_cache": True  # Don't cache any data
    }
)

# Extract sensitive data without sending it to external APIs
results = extractor.extract(sensitive_html)
```

**Data Retention Controls**:

```python
import domorpher

# Configure with strict data retention
domorpher.configure(
    data_retention={
        "html_content": "none",  # Don't store HTML content
        "extracted_data": "minimal",  # Store minimal extraction results
        "llm_requests": "none",  # Don't store LLM requests
        "llm_responses": "none",  # Don't store LLM responses
        "browser_data": "none"  # Don't store browser data
    }
)

# Use with data retention settings
extractor = domorpher.Extractor(
    instruction="Extract financial information"
)

# No sensitive data will be retained beyond the current session
results = extractor.extract_from_url("https://example.com/financial")
```

### Content Safety

DOMorpher implements content safety measures:

**Content Filters**:

```python
import domorpher

# Configure with content safety
domorpher.configure(
    content_safety={
        "enabled": True,
        "block_unsafe_content": True,
        "log_safety_issues": True,
        "safety_thresholds": {
            "hate": 0.7,
            "harassment": 0.7,
            "self_harm": 0.8,
            "sexual": 0.8,
            "violence": 0.8
        }
    }
)

# Extract with safety checks
try:
    results = domorpher.extract_from_url(
        "https://example.com/forum",
        "Extract all forum posts and their content"
    )
except domorpher.exceptions.ContentSafetyError as e:
    print(f"Content safety issue detected: {e}")
    # Handle the error appropriately
```

**Safe Browsing Integration**:

```python
import domorpher

# Create agent with safe browsing
agent = domorpher.AutonomousAgent(
    objective="Research competitive products",
    safe_browsing={
        "enabled": True,
        "check_url_reputation": True,
        "block_malicious_sites": True,
        "site_category_restrictions": [
            "malware",
            "phishing",
            "adult"
        ]
    }
)

# Execute with safe browsing
try:
    result = agent.execute("https://example.com/products")
except domorpher.exceptions.SafeBrowsingError as e:
    print(f"Safe browsing issue detected: {e.reason}")
    print(f"URL: {e.url}")
    print(f"Category: {e.category}")
```

**Execution Sandboxing**:

```python
import domorpher

# Create agent with sandboxed execution
agent = domorpher.AutonomousAgent(
    objective="Test website functionality",
    sandbox_settings={
        "enabled": True,
        "isolation_level": "high",
        "network_restrictions": "same_domain",  # Only allow same-domain requests
        "javascript_execution": "limited",
        "cookie_access": "restricted",
        "resource_limits": {
            "memory_mb": 500,
            "cpu_percent": 50,
            "execution_time_seconds": 60
        }
    }
)

# Execute in sandbox
result = agent.execute("https://example.com")
```

## Customization and Extension

### Custom Extractors

DOMorpher allows you to create custom extractors:

**Creating a Custom Extractor**:

```python
import domorpher
from domorpher.extractors import BaseExtractor

class CustomProductExtractor(BaseExtractor):
    """Custom extractor for product information."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.product_types = kwargs.get("product_types", [])
        
    def preprocess_html(self, html):
        """Custom HTML preprocessing."""
        # Apply specialized preprocessing for product pages
        preprocessed = super().preprocess_html(html)
        
        # Add custom enhancement for product data
        if "<div class='product-specs'" in preprocessed:
            # Convert product specs table to a more structured format
            # ...
            pass
            
        return preprocessed
        
    def create_extraction_prompt(self, html, instruction):
        """Customize the LLM prompt for product extraction."""
        # Start with the base prompt
        base_prompt = super().create_extraction_prompt(html, instruction)
        
        # Add product-specific instructions
        product_instructions = """
        Pay special attention to:
        1. Product variants (sizes, colors, etc.)
        2. Technical specifications in tables
        3. Pricing tiers and volume discounts
        4. Availability information
        """
        
        # Combine prompts
        return f"{base_prompt}\n\nAdditional Product Guidance:\n{product_instructions}"
        
    def postprocess_results(self, results):
        """Custom post-processing for product data."""
        # Apply product-specific transformations
        for product in results:
            # Normalize prices
            if "price" in product and isinstance(product["price"], str):
                product["price"] = self._convert_price_to_number(product["price"])
                
            # Handle product variants
            if "variants" in product:
                product["variants"] = self._normalize_variants(product["variants"])
                
        return results
        
    def _convert_price_to_number(self, price_str):
        """Convert price strings to numbers."""
        # Remove currency symbols and commas
        clean_price = price_str.replace("$", "").replace(",", "").strip()
        try:
            return float(clean_price)
        except ValueError:
            return None
            
    def _normalize_variants(self, variants):
        """Normalize product variants."""
        # Implement variant normalization logic
        # ...
        return variants

# Use the custom extractor
extractor = CustomProductExtractor(
    llm_provider="anthropic",
    llm_model="claude-3-sonnet",
    product_types=["electronics", "appliances"]
)

# Extract with custom extractor
results = extractor.extract_from_url(
    "https://example.com/products/laptop",
    "Extract detailed product information"
)
```

**Custom Extraction Pipeline Step**:

```python
import domorpher
from domorpher.steps import PipelineStep

class ProductCategorizer(PipelineStep):
    """Pipeline step to categorize products."""
    
    def __init__(self, category_rules=None):
        self.category_rules = category_rules or {}
        
    async def process(self, data, context):
        """Process the data by categorizing products."""
        if not isinstance(data, list):
            # Not a list of products
            return data
            
        for product in data:
            # Skip non-dict items
            if not isinstance(product, dict):
                continue
                
            # Determine category based on product attributes
            product["category"] = self._determine_category(product)
            
        return data
        
    def _determine_category(self, product):
        """Determine the category of a product."""
        name = product.get("name", "").lower()
        description = product.get("description", "").lower()
        
        # Apply category rules
        for category, patterns in self.category_rules.items():
            for pattern in patterns:
                if pattern in name or pattern in description:
                    return category
                    
        # Apply heuristics for common categories
        if any(term in name for term in ["laptop", "desktop", "computer"]):
            return "computers"
        elif any(term in name for term in ["phone", "smartphone"]):
            return "phones"
        # ... more categories
        
        return "other"

# Create a pipeline with the custom step
pipeline = domorpher.Pipeline([
    domorpher.steps.FetchURL(),
    domorpher.steps.PreprocessHTML(),
    domorpher.steps.CustomExtractor("Extract all products with their names, prices, and descriptions"),
    ProductCategorizer(category_rules={
        "gaming": ["gaming", "game", "player"],
        "business": ["business", "professional", "enterprise"],
        "student": ["student", "education", "school", "college"]
    }),
    domorpher.steps.OutputFormatter()
])

# Use the pipeline
results = pipeline.process("https://example.com/products")
```

### LLM Provider Plugins

DOMorpher supports custom LLM provider plugins:

**Creating a Custom LLM Provider**:

```python
import domorpher
from domorpher.llm import BaseLlmProvider
import requests

class CustomLlmProvider(BaseLlmProvider):
    """Custom LLM provider implementation."""
    
    def __init__(self, api_key, model="default-model", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.model = model
        self.api_url = kwargs.get("api_url", "https://api.custom-llm.com/generate")
        self.timeout = kwargs.get("timeout", 30)
        
    async def generate(self, prompt, **kwargs):
        """Generate text from the LLM."""
        # Override parameters with kwargs
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make request
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result["text"]
            
        except requests.RequestException as e:
            raise domorpher.exceptions.LlmProviderError(f"Error calling custom LLM: {e}")
            
    def get_token_count(self, text):
        """Estimate the number of tokens in the text."""
        # Simple estimation: average 4 characters per token
        return len(text) // 4

# Register the custom provider
domorpher.register_llm_provider("custom", CustomLlmProvider)

# Use the custom provider
domorpher.configure(
    custom_api_key="your-custom-api-key",
    custom_api_url="https://your-custom-llm-api.com/generate"
)

extractor = domorpher.Extractor(
    instruction="Extract product information",
    llm_provider="custom",
    llm_model="custom-model-name"
)

# Use the extractor
results = extractor.extract_from_url("https://example.com/products")
```

**Multi-Provider Fallback**:

```python
import domorpher
from domorpher.llm import LlmProviderManager
import random

class FallbackLlmManager(LlmProviderManager):
    """LLM manager with advanced fallback logic."""
    
    def __init__(self, providers_config):
        super().__init__(providers_config)
        self.provider_stats = {name: {"success": 0, "failure": 0} for name in self.providers}
        
    async def generate_with_fallback(self, prompt, **kwargs):
        """Generate text with intelligent fallback between providers."""
        # Get preferred provider
        preferred = kwargs.get("preferred_provider")
        
        # Get all available providers
        available_providers = list(self.providers.keys())
        
        # Order providers by preference and past success rate
        if preferred and preferred in available_providers:
            # Move preferred to front
            available_providers.remove(preferred)
            ordered_providers = [preferred] + available_providers
        else:
            # Order by success rate
            ordered_providers = sorted(
                available_providers,
                key=lambda p: self._get_success_rate(p),
                reverse=True
            )
            
        # Try providers in order
        last_error = None
        for provider_name in ordered_providers:
            provider = self.providers[provider_name]
            try:
                result = await provider.generate(prompt, **kwargs)
                
                # Update stats
                self.provider_stats[provider_name]["success"] += 1
                
                return {
                    "text": result,
                    "provider": provider_name,
                    "fallback_used": provider_name != preferred
                }
            except Exception as e:
                # Update stats
                self.provider_stats[provider_name]["failure"] += 1
                
                # Store error and try next provider
                last_error = e
                continue
                
        # All providers failed
        raise domorpher.exceptions.LlmProviderError(
            f"All LLM providers failed. Last error: {last_error}"
        )
        
    def _get_success_rate(self, provider_name):
        """Calculate success rate for a provider."""
        stats = self.provider_stats[provider_name]
        total = stats["success"] + stats["failure"]
        
        if total == 0:
            return 0.5  # Default for no data
            
        return stats["success"] / total

# Configure with multiple providers
domorpher.configure(
    anthropic_api_key="your-anthropic-key",
    openai_api_key="your-openai-key",
    custom_api_key="your-custom-key",
    fallback_manager=FallbackLlmManager
)

# Use with fallback capability
extractor = domorpher.Extractor(
    instruction="Extract product information",
    llm_provider="anthropic",  # Primary provider
    llm_fallbacks=["openai", "custom"]  # Fallback order
)

# Extract with fallback capability
results = extractor.extract_from_url("https://example.com/products")
```

### Custom Validation Rules

DOMorpher supports custom validation rules:

**Creating Custom Validators**:

```python
import domorpher
from domorpher.validation import BaseValidator
import re

class E-commerceDataValidator(BaseValidator):
    """Custom validator for e-commerce data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_fields = kwargs.get("required_fields", ["name", "price"])
        self.min_price = kwargs.get("min_price", 0.01)
        self.sku_pattern = kwargs.get("sku_pattern", r'^[A-Z0-9]{6,12})
        
    def validate(self, data):
        """Validate the extracted data."""
        if not isinstance(data, list):
            raise domorpher.exceptions.ValidationError("Data must be a list of products")
            
        validated_data = []
        errors = []
        
        for i, product in enumerate(data):
            try:
                validated_product = self._validate_product(product)
                validated_data.append(validated_product)
            except Exception as e:
                errors.append({
                    "index": i,
                    "product": product,
                    "error": str(e)
                })
                
        return {
            "valid_data": validated_data,
            "errors": errors,
            "is_valid": len(errors) == 0
        }
        
    def _validate_product(self, product):
        """Validate a single product."""
        # Check if product is a dictionary
        if not isinstance(product, dict):
            raise ValueError("Product must be a dictionary")
            
        # Check required fields
        for field in self.required_fields:
            if field not in product:
                raise ValueError(f"Missing required field: {field}")
                
        # Validate and normalize price
        if "price" in product:
            price = self._validate_price(product["price"])
            product["price"] = price
            
        # Validate SKU if present
        if "sku" in product:
            self._validate_sku(product["sku"])
            
        # Validate stock status
        if "in_stock" in product:
            product["in_stock"] = bool(product["in_stock"])
            
        return product
        
    def _validate_price(self, price):
        """Validate and normalize price."""
        # Convert string to float if needed
        if isinstance(price, str):
            try:
                price = float(price.replace("$", "").replace(",", ""))
            except ValueError:
                raise ValueError(f"Invalid price format: {price}")
                
        # Check if price is a number
        if not isinstance(price, (int, float)):
            raise ValueError(f"Price must be a number, got {type(price)}")
            
        # Check minimum price
        if price < self.min_price:
            raise ValueError(f"Price must be at least {self.min_price}")
            
        return price
        
    def _validate_sku(self, sku):
        """Validate SKU format."""
        if not isinstance(sku, str):
            raise ValueError("SKU must be a string")
            
        if not re.match(self.sku_pattern, sku):
            raise ValueError(f"SKU does not match required pattern: {sku}")
            
        return sku

# Use the custom validator
validator = E-commerceDataValidator(
    required_fields=["name", "price", "brand"],
    min_price=1.00,
    sku_pattern=r'^[A-Z]{2}\d{6}
)

# Create extractor with custom validator
extractor = domorpher.Extractor(
    instruction="Extract all products with their details",
    custom_validator=validator
)

# Extract and validate
results = extractor.extract_from_url("https://example.com/products")

# Check validation results
if not results["is_valid"]:
    print(f"Found {len(results['errors'])} validation errors:")
    for error in results["errors"]:
        print(f"  Product {error['index']}: {error['error']}")
        
# Use only valid data
valid_products = results["valid_data"]
```

### Pipeline Extensions

DOMorpher allows custom pipeline extensions:

**Creating Custom Pipeline Steps**:

```python
import domorpher
from domorpher.steps import PipelineStep
import hashlib
import csv
import io

class ProductDeduplicate(PipelineStep):
    """Pipeline step to deduplicate products based on various criteria."""
    
    def __init__(self, dedup_keys=None, similarity_threshold=0.9):
        self.dedup_keys = dedup_keys or ["name", "brand", "price"]
        self.similarity_threshold = similarity_threshold
        
    async def process(self, data, context):
        """Process data by deduplicating products."""
        if not isinstance(data, list):
            return data
            
        # Generate fingerprints for each product
        products_with_fingerprints = []
        for product in data:
            fingerprint = self._generate_fingerprint(product)
            products_with_fingerprints.append((product, fingerprint))
            
        # Group by similarity
        unique_products = []
        used_fingerprints = set()
        
        for product, fingerprint in products_with_fingerprints:
            # Skip if this exact fingerprint is already used
            if fingerprint in used_fingerprints:
                continue
                
            # Check for similar fingerprints
            is_duplicate = False
            for existing_fp in used_fingerprints:
                similarity = self._calculate_similarity(fingerprint, existing_fp)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_products.append(product)
                used_fingerprints.add(fingerprint)
                
        return unique_products
        
    def _generate_fingerprint(self, product):
        """Generate a fingerprint for the product based on key fields."""
        # Create a consistent representation of key fields
        values = []
        for key in self.dedup_keys:
            if key in product:
                # Normalize the value
                value = product[key]
                if isinstance(value, (int, float)):
                    value = str(round(value, 2))
                elif isinstance(value, str):
                    value = value.lower().strip()
                    
                values.append(f"{key}:{value}")
                
        # Create a hash of the combined values
        fingerprint = hashlib.md5("|".join(values).encode()).hexdigest()
        return fingerprint
        
    def _calculate_similarity(self, fp1, fp2):
        """Calculate similarity between two fingerprints."""
        # Simple similarity based on matching characters
        matching = sum(a == b for a, b in zip(fp1, fp2))
        return matching / len(fp1)

class CsvExporter(PipelineStep):
    """Pipeline step to export data as CSV."""
    
    def __init__(self, filename=None, dialect="excel", encoding="utf-8"):
        self.filename = filename
        self.dialect = dialect
        self.encoding = encoding
        
    async def process(self, data, context):
        """Process data by exporting to CSV."""
        if not isinstance(data, list) or not data:
            return data
            
        # Get field names from first item
        first_item = data[0]
        if not isinstance(first_item, dict):
            return data
            
        fieldnames = list(first_item.keys())
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, dialect=self.dialect)
        
        # Write header and rows
        writer.writeheader()
        for item in data:
            writer.writerow(item)
            
        # Get CSV content
        csv_content = output.getvalue()
        
        # Write to file if filename is provided
        if self.filename:
            with open(self.filename, "w", encoding=self.encoding) as f:
                f.write(csv_content)
                
        # Add CSV to context
        context["csv_content"] = csv_content
        
        # Return original data
        return data

# Create a pipeline with custom steps
pipeline = domorpher.Pipeline([
    domorpher.steps.FetchURL(),
    domorpher.steps.PreprocessHTML(),
    domorpher.steps.CustomExtractor("Extract all products with their details"),
    ProductDeduplicate(dedup_keys=["name", "brand", "price"]),
    domorpher.steps.SchemaValidator(schema=product_schema),
    CsvExporter(filename="products.csv")
])

# Process with the pipeline
results = pipeline.process("https://example.com/products")

# Access the CSV content
csv_content = pipeline.context.get("csv_content")
print(f"Generated CSV with {len(results)} products")
```

## Troubleshooting and FAQs

### Common Issues

**Issue: Rate Limiting**

```
Error: API rate limit exceeded (error code: rate_limit_exceeded)
```

**Solution**:

```python
import domorpher
import time

# Configure with rate limiting
domorpher.configure(
    rate_limiting={
        "enabled": True,
        "rpm": 5,  # Lower requests per minute
        "retry_count": 3,
        "retry_delay": 10  # Longer delay between retries
    }
)

# Manual retry logic for rate limiting
def extract_with_retry(url, instruction, max_retries=5, backoff_factor=2):
    retries = 0
    delay = 1
    
    while retries < max_retries:
        try:
            return domorpher.extract_from_url(url, instruction)
        except domorpher.exceptions.RateLimitError as e:
            retries += 1
            if retries >= max_retries:
                raise
                
            print(f"Rate limit hit. Retrying in {delay} seconds... ({retries}/{max_retries})")
            time.sleep(delay)
            delay *= backoff_factor
```

**Issue: Extraction Quality**

**Solution**:

```python
import domorpher

# Improve extraction quality
extractor = domorpher.Extractor(
    instruction="Extract product information with high precision",
    llm_model="claude-3-opus",  # Use more powerful model
    adaptation_level="aggressive",  # Try harder to get good results
    chunking_strategy="semantic",  # Better chunking for complex pages
    post_processing={
        "confidence_threshold": 0.7,  # Only keep high-confidence results
        "consistency_check": True,  # Check for consistency across results
        "refinement": True  # Apply refinement pass for better quality
    }
)

# Extract with detailed instruction
results = extractor.extract_from_url(
    "https://example.com/products",
    """
    Extract all products on the page with the following details:
    - Product name (look for h1 or h2 elements with the main product title)
    - Brand name (usually near the product name or in the product metadata)
    - Current price (look for elements with price formatting, may include currency symbols)
    - Original price if discounted (often shown as crossed-out text near the current price)
    - Discount percentage (may be shown as a percentage or badge near the price)
    - Average rating (typically shown as stars, look for numeric value out of 5)
    - Number of reviews (often in parentheses near the rating)
    - Available colors (look for color selection options)
    - Available sizes (look for size selection options)
    - In-stock status (look for text indicating "in stock" or "out of stock")
    - Shipping information (delivery time estimates, shipping costs)
    - SKU or product code (often in small text or product metadata)
    
    For each field, provide a confidence score between 0 and 1.
    """
)
```

**Issue: JavaScript Rendering**

**Solution**:

```python
import domorpher

# Improve JavaScript rendering
extractor = domorpher.Extractor(
    instruction="Extract product information",
    javascript={
        "enabled": True,
        "wait_until": "networkidle0",  # Wait until network is completely idle
        "timeout": 60,  # Longer timeout
        "wait_for_selector": ".product-container",  # Wait for specific element
        "wait_for_function": "window.productsLoaded === true",  # Wait for JS variable
        "viewport": {"width": 1920, "height": 1080},  # Larger viewport
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",  # Modern user agent
        "execute_beforehand": [
            "window.scrollTo(0, document.body.scrollHeight);",  # Scroll to bottom
            "document.querySelector('.load-more-button')?.click();"  # Click load more
        ]
    }
)

# Extract with JavaScript rendering
results = extractor.extract_from_url("https://example.com/products")
```

**Issue: Large Pages**

**Solution**:

```python
import domorpher

# Handle large pages
extractor = domorpher.Extractor(
    instruction="Extract all articles",
    chunking_strategy="adaptive",
    chunking_config={
        "min_chunk_size": 1000,
        "max_chunk_size": 5000,
        "overlap": 200,
        "max_chunks": 50,  # Limit total chunks
        "important_selectors": [
            "article", ".news-item", ".story"
        ],
        "preserve_selectors": True,  # Don't break inside important elements
        "incremental_processing": True  # Process chunks incrementally
    }
)

# Use streaming extraction for large pages
for chunk_results in extractor.extract_from_url_streaming("https://example.com/news-archive"):
    # Process each chunk of results as they become available
    for article in chunk_results:
        process_article(article)
```

### Performance Tuning

**Memory Optimization**:

```python
import domorpher
import gc
import psutil

# Monitor and optimize memory usage
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

# Configure for low memory usage
domorpher.configure(
    memory_optimization={
        "enabled": True,
        "aggressive_gc": True,
        "stream_processing": True,
        "max_memory_mb": 1000,
        "max_context_items": 10
    }
)

# Use context manager for automatic cleanup
with domorpher.OptimizedSession(max_memory_mb=500) as session:
    extractor = session.create_extractor(
        instruction="Extract product information",
        chunking_strategy="size",
        chunking_config={"max_chunk_size": 3000}
    )
    
    # Process in batches
    batch_size = 10
    urls = [f"https://example.com/products/page{i}" for i in range(1, 101)]
    
    all_products = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        
        batch_products = []
        for url in batch_urls:
            products = extractor.extract_from_url(url)
            batch_products.extend(products)
            
            # Force garbage collection after each URL
            gc.collect()
            
        # Process batch results
        all_products.extend(process_batch(batch_products))
        
        # Clear batch data
        batch_products = None
        gc.collect()
        
        # Check memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"Memory usage after batch {i//batch_size + 1}: {current_memory:.2f} MB")

final_memory = process.memory_info().rss / (1024 * 1024)  # MB
print(f"Memory change: {final_memory - initial_memory:.2f} MB")
```

**Throughput Optimization**:

```python
import domorpher
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Measure and optimize throughput
async def measure_throughput(urls, concurrency_levels):
    results = {}
    
    for concurrency in concurrency_levels:
        print(f"Testing concurrency level: {concurrency}")
        
        # Configure for this concurrency level
        domorpher.configure(
            parallelism={
                "max_workers": concurrency,
                "chunk_processing": True,
                "url_fetching": True
            },
            rate_limiting={
                "enabled": True,
                "rpm": 10 * concurrency,  # Scale with concurrency
                "max_concurrent": concurrency
            }
        )
        
        # Create shared extractor
        extractor = domorpher.Extractor(
            instruction="Extract product information",
            cache_strategy="memory"  # Use memory cache for benchmark
        )
        
        # Create pool
        pool = ThreadPoolExecutor(max_workers=concurrency)
        
        # Measure extraction time
        start_time = time.time()
        
        futures = []
        for url in urls:
            futures.append(pool.submit(extractor.extract_from_url, url))
            
        # Wait for all to complete
        all_results = []
        for future in futures:
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error during extraction: {e}")
                
        end_time = time.time()
        
        # Calculate metrics
        duration = end_time - start_time
        throughput = len(urls) / duration
        
        results[concurrency] = {
            "duration": duration,
            "throughput": throughput,
            "success_rate": len(all_results) / len(urls)
        }
        
        print(f"Concurrency {concurrency}: {throughput:.2f} URLs/sec, duration: {duration:.2f}s")
        
        # Clean up
        pool.shutdown()
        gc.collect()
        
    # Find optimal concurrency
    optimal = max(results.items(), key=lambda x: x[1]["throughput"])
    print(f"Optimal concurrency: {optimal[0]} with throughput of {optimal[1]['throughput']:.2f} URLs/sec")
    
    return results

# Test URLs
test_urls = [f"https://example.com/products/{i}" for i in range(1, 21)]

# Test concurrency levels
concurrency_levels = [1, 2, 4, 8, 16]

# Run throughput test
throughput_results = asyncio.run(measure_throughput(test_urls, concurrency_levels))

# Configure with optimal settings
optimal_concurrency = max(throughput_results.items(), key=lambda x: x[1]["throughput"])[0]

domorpher.configure(
    parallelism={
        "max_workers": optimal_concurrency,
        "chunk_processing": True,
        "url_fetching": True
    }
)
```

### Extraction Quality

**Improving Extraction Accuracy**:

```python
import domorpher

# Strategies for improving extraction quality
extractor = domorpher.Extractor(
    instruction="Extract product information",
    quality_settings={
        "model_tier": "high",  # Use highest quality model
        "multi_pass": True,  # Use multiple extraction passes
        "verification": True,  # Verify extraction results
        "adaptation": "aggressive",  # Aggressively adapt to content structure
        "confidence_threshold": 0.7,  # Minimum confidence score
        "refinement_passes": 2  # Number of refinement passes
    }
)

# Two-phase extraction for better quality
# Phase 1: Structure analysis
structure_analysis = extractor.analyze_structure(
    "https://example.com/products",
    analysis_type="detailed"
)

# Phase 2: Targeted extraction with structure insights
extraction_prompt = f"""
Extract all products on the page with their details.

Page analysis showed the following structure:
{structure_analysis['summary']}

Product containers are identified by: {structure_analysis['containers']['products']}
Price elements are usually found in: {structure_analysis['elements']['price']}
Product names are found in: {structure_analysis['elements']['name']}

Extract each product with:
- Name
- Price
- Description
- Rating
- Availability
"""

# Extract with enhanced prompt
products = extractor.extract_from_url(
    "https://example.com/products",
    extraction_prompt
)

# Verify results
if products:
    print(f"Extracted {len(products)} products")
    
    # Check confidence scores
    low_confidence = [p for p in products if p.get("_confidence", 1.0) < 0.8]
    if low_confidence:
        print(f"Warning: {len(low_confidence)} products have low confidence scores")
        
    # Check expected fields
    missing_fields = []
    for product in products:
        for field in ["name", "price"]:
            if field not in product or not product[field]:
                missing_fields.append((product.get("name", "Unknown"), field))
                
    if missing_fields:
        print(f"Warning: {len(missing_fields)} missing fields detected")
```

**Handling Edge Cases**:

```python
import domorpher
import json
import requests

# Load template
product_template = domorpher.Template.load("product_template.json")

# Export template as JSON
template_json = product_template.to_json()

# Share template
with open("shared_template.json", "w") as f:
    f.write(template_json)

# Template repository integration
response = requests.post(
    "https://template-repo.example.com/api/templates",
    json=json.loads(template_json),
    headers={"Authorization": "Bearer your_api_key"}
)

# Collaborative template improvement
improved_template = domorpher.Template.from_json(template_json)
improved_template.add_site_optimization("target.com", {
    "selectors": {
        "product_container": ".product-card",
        "name": ".product-title",
        "price": ".current-price"
    },
    "wait_for": ".product-card",
    "additional_instructions": "Extract RedCard discount information"
})
improved_template.save("improved_template.json")
```

### Schema Validation

DOMorpher supports comprehensive schema validation:

**JSON Schema**:

```python
import domorpher

# Define schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "description": {"type": "string", "nullable": True},
            "rating": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": 0, "maximum": 5},
                    "count": {"type": "integer", "minimum": 0}
                },
                "required": ["value"]
            },
            "in_stock": {"type": "boolean"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["name", "price", "in_stock"]
    }
}

# Create extractor with schema
extractor = domorpher.Extractor(
    instruction="Extract all products with their details",
    schema=schema,
    validation_mode="strict"  # Options: strict, lenient, none
)

# Extract with validation
products = extractor.extract_from_url("https://example.com/products")

# Products are guaranteed to match the schema
for product in products:
    print(f"Product: {product['name']}, Price: ${product['price']}")
    if product.get("rating"):
        print(f"Rating: {product['rating']['value']}/5 ({product['rating'].get('count', 'No')} reviews)")
```
