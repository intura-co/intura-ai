# Intura-AI: Intelligent Research and Experimentation AI

[![PyPI version](https://badge.fury.io/py/intura-ai.svg)](https://badge.fury.io/py/intura-ai) 
[![LangChain Compatible](https://img.shields.io/badge/LangChain-Compatible-blue)](https://python.langchain.com/docs/get_started/introduction.html)


`intura-ai` is a Python package designed to streamline LLM experimentation and production. It provides tools for logging LLM usage and managing experiment predictions, with seamless LangChain compatibility.

# Getting Started with Intura

Dive into the world of Intura and begin experimenting with Large Language Models (LLMs) in under 5 minutes. This guide will walk you through the essential steps to set up your first experiment, either through our SDK or directly via the Intura Dashboard.

## Quick Start Options

* **SDK-Based Experiment Creation:**
    * Utilize the Intura AI SDK for programmatic experiment creation and management, offering flexibility and integration into your existing workflows.
* **Intura Dashboard:**
    * Jump straight into experimentation with our user-friendly dashboard, accessible at [intura.dashboard](https://intura-dashboard-566556985624.asia-southeast2.run.app/). This option is perfect for quickly exploring Intura's capabilities.

### Prerequisites

Before you begin, ensure you have the following:

* **Python 3.10 or Later:**
    * Download and install Python from [python.org/downloads](https://www.python.org/downloads/). During installation, select the option to add Python to your system's PATH and install all necessary dependencies. This ensures seamless SDK functionality.

## SDK Initialization and Setup

To leverage the Intura AI SDK, you'll need to install it and configure your environment.

1.  **Install the Intura AI SDK:**
    * Open your terminal or command prompt and execute the following command:

    ```bash
    pip install intura-ai
    ```

2.  **Obtain Your API Key:**
    * Your API key is essential for authenticating with the Intura platform. You can find it within the [Intura Dashboard](https://intura-dashboard-566556985624.asia-southeast2.run.app/) or by contacting `admin@intura.co`. Store this key securely, as it grants access to your Intura resources.

## Creating Your First Experiment

With the SDK installed and your API key ready, you can now define your experiment.

1.  **Experiment Definition:**
    * Use the `DashboardPlatform` functions from the `intura_ai.platform` module to operate your experiment. You'll also need to import and use `ExperimentModel` and `ExperimentTreatmentModel` from `intura_ai.platform.domain` to define your experiment and its variations.

    ```python
    from intura_ai.platform import DashboardPlatform
    from intura_ai.platform.domain import ExperimentModel, ExperimentTreatmentModel
    
    client = DashboardPlatform(intura_api_key=os.environ.get("INTURA_API_KEY", "<INTURA_API_KEY>"))
    experiment_id = client.create_experiment(ExperimentModel(
        experiment_name="Example Experiment",
        treatment_list=[
            ExperimentTreatmentModel(
                treatment_model_name="gemini-1.5-flash",
                treatment_model_provider="Google",
                prompt="Act as personal assistant"
            ),
            ExperimentTreatmentModel(
                treatment_model_name="claude-3-5-sonnet-20240620",
                treatment_model_provider="Anthropic",
                prompt="Act as personal assistant"
            ),
        ]
    ))
    ```
    
    
## Running Your Experiment

After defining your experiment, you can run it and analyze the results.

1.  **Initialize the Experiment Client:**
    * Use the `ChatModelExperiment` class to create a client that interacts with your experiment.

    ```python
    import os
    from intura_ai.experiments import ChatModelExperiment

    client = ChatModelExperiment(
        intura_api_key=os.environ.get("INTURA_API_KEY", "<INTURA_API_KEY>")
    )

    choiced_model, model_config, chat_prompts = client.build(
        experiment_id=experiment_id,
        features={
            "user_id": "Rama12345",
            "membership": "FREE",
            "employment_type": "FULL_TIME",
            "feature_x": "your custom features"
        }
    )
    ```

2.  **Craft the Final Prompt:**
    * Add your final user prompt to the chat prompts list.

    ```python
    chat_prompts.append(('human', 'give me motivation for today'))
    ```

3.  **Set Up and Invoke the Model:**
    * Initialize the chosen LLM model with its configuration and API key, then invoke it with the crafted prompts.

    ```python
    import os

    # Set your LLM API keys as environment variables
    os.environ["GOOGLE_API_KEY"] = "xxx"
    os.environ["ANTHROPIC_API_KEY"] = "xxx"
    os.environ["DEEPSEEK_API_KEY"] = "xxx"
    os.environ["OPENAI_API_KEY"] = "xxx"
    os.environ["ANOTHER_LLM_KEY"] = "xxx"

    model = choiced_model(**model_config)

    # Or set LLM API keys as parameters
    # model = choiced_model(**model_config, api_key="<YOUR_LLM_API_KEY>")

    response = model.invoke(chat_prompts)
    print(response)
    ```

By following these steps, you can quickly set up and run your first experiment with Intura, exploring the power of LLMs and optimizing their performance for your specific needs.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports or feature requests.

## License
This project is licensed under the MIT License.