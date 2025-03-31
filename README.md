# Intura-AI: Intelligent Research and Experimentation AI

[![PyPI version](https://badge.fury.io/py/intura-ai.svg)](https://badge.fury.io/py/intura-ai) 
[![LangChain Compatible](https://img.shields.io/badge/LangChain-Compatible-blue)](https://python.langchain.com/docs/get_started/introduction.html)


`intura-ai` is a Python package designed to streamline LLM experimentation and production. It provides tools for logging LLM usage and managing experiment predictions, with seamless LangChain compatibility.

**Dashboard:** [dashboard.intura.co](https://intura-dashboard-566556985624.asia-southeast2.run.app)

## Features

* **Experiment Prediction:**
    * `ChatModelExperiment`: Facilitates the selection and execution of LangChain models based on experiment configurations.
* **LangChain Compatibility:**
    * Designed to integrate smoothly with LangChain workflows.

## Installation

```bash
pip install intura-ai
```

## Usage

### Experiment Prediction
Use `ChatModelExperiment` to fetch and execute pre-configured LangChain models.

```python
from intura_ai.experiments import ChatModelExperiment

EXPERIMENT_ID = "..."
INTURA_API_KEY = "..."
client = ChatModelExperiment(
    intura_api_key=INTURA_API_KEY
)

choiced_model, model_config, chat_prompts = client.build(
    experiment_id=EXPERIMENT_ID,
    features={
        "user_id": "Rama12345", 
        "membership": "FREE", 
        "employment_type": "FULL_TIME",
        "feature_x": "your custom features"
    }
)
chat_prompts.append(('human', 'give me today quote for programmer'))

print(client.choiced_model) # Your choiced model for instance: claude-3-5-sonnet-20240620

# Set api_key as environment 
import os

os.environ["GOOGLE_API_KEY"] = "xxx"
os.environ["ANTHROPIC_API_KEY"] = "xxx"
os.environ["DEEPSEEK_API_KEY"] = "xxx"
os.environ["OPENAI_API_KEY"] = "xxx"

model = choiced_model(**model_config)

# Or set api_key as params

model = choiced_model(**model_config, api_key="<YOUR_API_KEY>")

# Inference

model.invoke(chat_prompts)
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports or feature requests.

## License
This project is licensed under the MIT License.