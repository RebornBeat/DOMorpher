# DOMorpher

## Intelligent DOM Navigation and Web Extraction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

DOMorpher revolutionizes web automation and data extraction by combining DOM parsing precision with large language model intelligence. Unlike traditional tools that rely on brittle selectors or rigid scripts, DOMorpher uses natural language instructions and zero-shot learning to understand and interact with web pages intelligently—no training required.

## The DOM-First Revolution

Traditional web automation requires pre-defined scripts and selectors that break when websites change. DOMorpher introduces a fundamentally different paradigm:

1. **Natural Language Objectives**: Tell the system what you want to accomplish in plain English
2. **Direct DOM Understanding**: The LLM directly interprets the DOM structure to understand page content and functionality
3. **Autonomous Decision Making**: The system decides how to navigate and interact based on context and goal
4. **Zero-Shot Intelligence**: No training or examples needed—works immediately with any website
5. **Adaptable Execution**: Automatically adjusts to site changes, new designs, and unexpected states

## How DOMorpher Works

```python
import domorpher

# Create an autonomous agent with just an objective
agent = domorpher.AutonomousAgent(
    objective="Find laptops with RTX 4080 graphics cards under $2000 and extract their specifications"
)

# Launch the agent on a starting URL
results = agent.execute("https://www.techstore.com")

# Print structured results
for laptop in results.extracted_data:
    print(f"Model: {laptop['model']}")
    print(f"Price: ${laptop['price']}")
    print(f"Specs: {laptop['specifications']}")
    print(f"URL: {laptop['url']}")
    print("-" * 40)
```

Behind the scenes, DOMorpher will:

1. Navigate to the URL and analyze the DOM
2. Identify navigation elements and interaction paths
3. Determine and execute the optimal sequence of actions (clicks, inputs, scrolls)
4. Locate relevant product information in the DOM
5. Extract and structure the requested data
6. Return comprehensive results

## Core Features

### DOM-Native Intelligence

DOMorpher directly processes and understands the Document Object Model:

```python
# Traditional approach with Selenium
driver.find_element(By.CSS_SELECTOR, ".login-button").click()
driver.find_element(By.ID, "username").send_keys("user@example.com")
driver.find_element(By.ID, "password").send_keys("password123")
driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

# With DOMorpher
agent = domorpher.AutonomousAgent(
    objective="Log in using email user@example.com and password password123"
)
agent.execute("https://example.com/login")
```

### Adaptive Extraction

DOMorpher understands page content semantically, not just structurally:

```python
# Extract structured data with natural language instructions
results = domorpher.extract(
    html,
    "Find all products on the page and extract their names, prices, ratings, and availability"
)
```

### Autonomous Navigation

Define complex goals and let DOMorpher figure out how to achieve them:

```python
# Works across completely different websites with the same objective
agent = domorpher.AutonomousAgent(
    objective="Find the most popular article and extract its title and author"
)

# Different sites, same instruction works for all
cnn_results = agent.execute("https://cnn.com")
nyt_results = agent.execute("https://nytimes.com")
medium_results = agent.execute("https://medium.com")
```

### Complex Reasoning

DOMorpher can handle scenarios requiring sophisticated reasoning:

```python
# Complex reasoning about form completion
agent = domorpher.AutonomousAgent(
    objective="""
    Fill out the mortgage application form with these details:
    - Income: $120,000
    - Credit score: Excellent
    - Loan amount: $400,000
    - Property type: Single-family home
    - Down payment: 20%
    - Term: 30 years
    Make sure to check if we qualify before submitting.
    """
)
result = agent.execute("https://mortgage-calculator.com")
```

### Hierarchical Understanding

DOMorpher understands complex DOM relationships and nested structures:

```python
# Extract hierarchical data with nested relationships
agent = domorpher.AutonomousAgent(
    objective="""
    Extract the complete forum thread structure including:
    - Main post content and author
    - All replies with author names and timestamps
    - Nested reply relationships (which comment replies to which)
    - Upvote counts for each post
    """
)
thread_data = agent.execute("https://forum.example.com/thread/123")
```

## Installation

### Python

```bash
pip install domorpher

# With optional extras
pip install "domorpher[playwright,transformers,cache,all]"
```

### JavaScript/TypeScript

```bash
npm install domorpher
```

### Rust

```toml
# In Cargo.toml
[dependencies]
domorpher = "0.1.0"
```

## API Key Configuration

DOMorpher requires API keys for LLM providers:

```python
import domorpher

# Configure API keys
domorpher.configure(
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-...",
    # Or use environment variables DOMORPHER_ANTHROPIC_API_KEY, etc.
)
```

## Usage Examples

### E-commerce Product Extraction

```python
import domorpher

# Extract product details from an e-commerce site
products = domorpher.extract_from_url(
    "https://example.com/products",
    """
    Extract all products on the page with the following information:
    - Product name
    - Price
    - Rating (out of 5 stars)
    - Number of reviews
    - Whether the product is in stock
    - All available colors
    - All available sizes
    - Shipping information
    """
)

# Save results
import json
with open("products.json", "w") as f:
    json.dump(products, f, indent=2)
```

### Authenticated Interaction

```python
import domorpher

# Create a session for authenticated interactions
session = domorpher.Session()

# Log in to a site
login_result = session.execute(
    url="https://example.com/login",
    objective="Log in with username 'testuser' and password 'password123'"
)

if login_result.success:
    # Perform authenticated actions
    profile_data = session.execute(
        objective="Go to my profile page and extract my account information"
    )

    print(f"Account type: {profile_data['account_type']}")
    print(f"Member since: {profile_data['join_date']}")
    print(f"Subscription status: {profile_data['subscription_status']}")
```

### Interactive Form Completion

```python
import domorpher

# Complete a complex form with validation
result = domorpher.execute(
    url="https://someform.example.com",
    objective="""
    Fill out the job application form with the following information:
    - Full name: John Doe
    - Email: john.doe@example.com
    - Phone: 555-123-4567
    - Resume: Upload the file from /path/to/resume.pdf
    - Cover letter: Paste the text from /path/to/cover_letter.txt
    - Years of experience: 5
    - Desired salary: $80,000
    - Available start date: Select "Immediately"
    - Where did you hear about us?: Select "Job Board"

    After filling everything out, review the application before submitting.
    """
)

print(f"Form submission successful: {result.success}")
if result.success:
    print(f"Confirmation number: {result.confirmation_id}")
else:
    print(f"Error: {result.error_message}")
```

## Technical Architecture

DOMorpher employs a modular architecture with seven core components:

1. **DOM Preprocessor**: Normalizes and optimizes HTML for LLM processing
2. **Chunking Engine**: Divides large HTML documents into processable segments while preserving context
3. **LLM Integration Layer**: Handles interactions with language model providers
4. **Instruction Parser**: Converts natural language instructions into executable strategies
5. **Adaptive Execution Engine**: Implements extraction strategies with real-time adaptation
6. **Result Reconciliation**: Combines and normalizes results from multiple extraction passes
7. **Schema Enforcement**: Validates and normalizes extracted data

This layered approach allows for flexibility in both deployment and integration scenarios.

## Comparison with Traditional Tools

| Feature | Traditional Scrapers | Traditional Automation | DOMorpher |
|---------|---------------------|------------------------|-----------|
| **Setup Complexity** | Complex selectors & rules | Detailed scripts | Simple natural language |
| **Adaptability** | Breaks on site changes | Breaks on site changes | Adapts automatically |
| **DOM Understanding** | Structure only | Structure only | Semantic understanding |
| **Maintenance** | Constant updates needed | Frequent fixes required | Self-maintaining |
| **Intelligence** | None | Limited | Full LLM reasoning |
| **Learning Curve** | High (selector expertise) | High (programming) | Low (natural language) |
| **Cross-site Reuse** | Low (site-specific) | Low (site-specific) | High (goal-oriented) |

## Advanced Features

### Extraction Templates

Create reusable extraction templates for common scenarios:

```python
# Define a template
product_template = domorpher.Template(
    """
    For each product on the page, extract:
    - Product name
    - Brand
    - Price (current and original if discounted)
    - Rating (numeric value and number of reviews)
    - Available colors and sizes
    - Whether the item is in stock
    """
)

# Use the template across different sites
amazon_products = product_template.extract(amazon_html)
walmart_products = product_template.extract(walmart_html)
target_products = product_template.extract(target_html)
```

### DOM Navigation Strategies

Configure how DOMorpher approaches DOM navigation:

```python
agent = domorpher.AutonomousAgent(
    objective="Research electric vehicles and compare features",
    navigation_strategy={
        "approach": "semantic_first",  # Focus on meaning over structure
        "depth": "comprehensive",      # Explore deeply into the site
        "patience": "high",            # Try multiple approaches if needed
        "exploration": "breadth_first" # Explore breadth before depth
    }
)
```

### Schema Validation

Define expected output schemas and validate extraction results:

```python
import pydantic
from typing import List, Optional

class Product(pydantic.BaseModel):
    name: str
    price: float
    original_price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    in_stock: bool

# Extract with schema validation
products = domorpher.extract(
    html,
    "Extract all products with their names, prices, and availability",
    schema=List[Product]  # Will ensure extraction matches this schema
)
```

### JavaScript Execution

Extract data from JavaScript-rendered pages:

```python
# Configure with JavaScript rendering support
extractor = domorpher.Extractor(
    "Extract prices from the interactive pricing table",
    javascript_support=True,
    wait_for_selector=".pricing-table.loaded"
)

# Extract from JavaScript-heavy page
results = extractor.extract_from_url("https://example.com/interactive-pricing")
```

### Parallel Processing

Accelerate multi-page workflows with parallel processing:

```python
# Process multiple pages simultaneously
results = domorpher.parallel_execute(
    urls=["https://store1.com", "https://store2.com", "https://store3.com"],
    objective="Find the cheapest 65-inch 4K TV with at least 4-star rating",
    max_concurrent=3
)

# Find the best deal across all stores
best_deal = min(results, key=lambda x: x.get('price', float('inf')))
print(f"Best deal: {best_deal['model']} at {best_deal['store']} for ${best_deal['price']}")
```

## Documentation

For more detailed information, check out our comprehensive documentation:

- [API Reference](https://domorpher.readthedocs.io/en/latest/api/)
- [Architecture Guide](https://domorpher.readthedocs.io/en/latest/architecture/)
- [Integration Tutorials](https://domorpher.readthedocs.io/en/latest/tutorials/)
- [Advanced Usage](https://domorpher.readthedocs.io/en/latest/advanced/)
- [Performance Optimization](https://domorpher.readthedocs.io/en/latest/performance/)

## Roadmap

- [x] Core DOM analysis and navigation
- [x] Autonomous goal-driven execution
- [x] Multi-step workflow handling
- [ ] Complex form handling improvements
- [ ] Interactive element pattern library
- [ ] Session and authentication management
- [ ] Optimized parallel processing
- [ ] Advanced failure recovery strategies
- [ ] User feedback incorporation
- [ ] Browser extension interface
- [ ] Visual enhancement for edge cases
- [ ] Enterprise integration APIs

## Community and Support

- [GitHub Issues](https://github.com/domorpher/domorpher/issues)
- [Discord Community](https://discord.gg/domorpher)
- [Documentation](https://domorpher.readthedocs.io/)
- [Examples Repository](https://github.com/domorpher/examples)

## License

DOMorpher is released under the MIT License. See [LICENSE](LICENSE) for details.

---

DOMorpher: Intelligent web interaction powered by LLMs.
