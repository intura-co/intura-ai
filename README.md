![intura-banner](./assets/images/intura.jpg)

# Intura-AI: Intelligent Research and Experimentation AI

[![PyPI version](https://badge.fury.io/py/intura-ai.svg)](https://badge.fury.io/py/intura-ai) 
[![LangChain Compatible](https://img.shields.io/badge/LangChain-Compatible-blue)](https://python.langchain.com/docs/get_started/introduction.html)

`intura-ai` is a Python package designed to streamline Large Language Model (LLM) experimentation and production. It provides tools for logging LLM usage and managing experiment predictions, with seamless LangChain compatibility.

## ⚠️ Beta Status

**IMPORTANT**: Intura AI is currently in **BETA** and under active development. While we're working hard to ensure stability, you may encounter:

- API changes without prior notice
- Incomplete features
- Bugs and performance issues

We welcome your feedback and contributions to help improve the library!

# Getting Started with Intura

This guide will help you start experimenting with Large Language Models (LLMs) using Intura in under 5 minutes. We'll walk you through setting up your first experiment using either our SDK or the Intura Dashboard.

## Quick Start Options

* **SDK-Based Approach:**
  * Use the Intura AI SDK for programmatic experiment creation and management, offering flexibility and integration into your existing workflows.
* **Intura Dashboard:**
  * Start experimenting immediately with our user-friendly dashboard at [Intura Dashboard](https://dashboard.intura.co/). This option is perfect for quickly exploring Intura's capabilities without writing code.

### Prerequisites

Before you begin, ensure you have:

* **Python 3.10 or Later:**
  * Download and install Python from [python.org/downloads](https://www.python.org/downloads/)
  * During installation, select the option to add Python to your system's PATH
  * This ensures seamless SDK functionality

## Installation and Setup

### Step 1: Install the Intura AI SDK

Open your terminal or command prompt and run:

```bash
pip install intura-ai
```

### Step 2: Obtain Your Intura API Key

Your API key authenticates your access to the Intura platform. You can:
* Find it in the [Intura Dashboard](https://dashboard.intura.co/) 
* Or contact `developer@intura.co`

Store this key securely, as it grants access to your Intura resources.

## Creating Your First Experiment

Now you can define an experiment to compare different LLMs or prompting strategies.

### Step 1: Define Your Experiment

```python
import os
from intura_ai.platform import DashboardPlatform
from intura_ai.platform.domain import ExperimentModel, ExperimentTreatmentModel

# Initialize the platform client with your Intura API key
client = DashboardPlatform(intura_api_key=os.environ.get("INTURA_API_KEY", "<YOUR_API_KEY>"))

# Create an experiment with multiple treatment variations
experiment_id = client.create_experiment(ExperimentModel(
    experiment_name="Motivation Messages Comparison",
    treatment_list=[
        # Treatment 1: Using Gemini model
        ExperimentTreatmentModel(
            treatment_model_name="gemini-1.5-flash",
            treatment_model_provider="Google",
            prompt="Act as a motivational coach providing inspiring daily messages"
        ),
        # Treatment 2: Using Claude model
        ExperimentTreatmentModel(
            treatment_model_name="claude-3-5-sonnet-20240620",
            treatment_model_provider="Anthropic",
            prompt="Act as a motivational coach providing inspiring daily messages"
        ),
    ]
))

print(f"Experiment created with ID: {experiment_id}")
```

In this example:
- We create an experiment comparing two different LLMs (Gemini and Claude)
- Both use the same prompt, allowing us to compare model performance
- You could also test different prompts with the same model

### Step 2: Run Your Experiment

After creating your experiment, you can run it and collect results:

```python
import os
from intura_ai.experiments import ChatModelExperiment

# Initialize the experiment client
chat_client = ChatModelExperiment(
    intura_api_key=os.environ.get("INTURA_API_KEY", "<YOUR_API_KEY>")
)

# Invoke the experiment chain with user-specific features
result = chat_client.invoke(
    experiment_id=experiment_id,  # Use the ID from the experiment you created
    features={
        "user_id": "user123",     # User identifier (example feature)
        "subscription_tier": "FREE",  # Can be used for segmentation (example feature)
        "user_type": "FULL_TIME", # (example feature)
        "location": "US"          # Any custom features you want to track (example feature)
    },
    messages=[{
        "role": "human",
        "content": "I'm feeling unmotivated today. Can you help me get back on track?"
    }]
)

print(result)
```

### Step 3: View and Analyze Results

After running your experiment with multiple users or queries, you can:

1. Log into the [Intura Dashboard](https://dashboard.intura.co/)
2. Navigate to your experiment
3. View performance metrics, including:
   - Response times
   - User segments and their interactions
   - Cost analysis
   - Response quality comparisons

## Advanced Usage

### Multi-Turn Conversations

For multi-turn conversations, you can add to the messages array:

```python
result = chat_client.build(
    experiment_id=experiment_id,
    features={"user_id": "user123"},
    messages=[
        {"role": "human", "content": "Help me plan a healthy diet"},
        {"role": "assistant", "content": "I'd be happy to help you plan a healthy diet! What are your dietary preferences or restrictions?"},
        {"role": "human", "content": "I'm vegetarian and allergic to nuts"}
    ]
)
```

## Troubleshooting

If you encounter issues:

1. **API Key Errors**: Verify your Intura API key and LLM provider API keys are correct
2. **Installation Problems**: Ensure you're using Python 3.10+ and have installed the correct packages
3. **Model Unavailability**: Check that you have access to the specific models in your treatments
4. **Request Failures**: Verify your internet connection and that the LLM providers' services are operational

For further assistance, contact support at `developer@intura.co`

## Contributing

We welcome contributions to Intura-AI! Please feel free to:
- Submit pull requests
- Open issues for bug reports
- Suggest feature enhancements
- Improve documentation

See our [Contributing Guidelines](https://github.com/intura-io/intura-ai/blob/main/CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/intura-io/intura-ai/blob/main/LICENSE) file for details.