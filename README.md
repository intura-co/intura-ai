# Intura-AI: Intelligent Research and Experimentation AI

[![PyPI version](https://badge.fury.io/py/intura-ai.svg)](https://badge.fury.io/py/intura-ai) 
[![LangChain Compatible](https://img.shields.io/badge/LangChain-Compatible-blue)](https://python.langchain.com/docs/get_started/introduction.html)


`intura-ai` is a Python package designed to streamline LLM experimentation and production. It provides tools for logging LLM usage and managing experiment predictions, with seamless LangChain compatibility.

**Dashboard:** [dashboard.intura.co](https://intura-dashboard-566556985624.asia-southeast2.run.app)

## Features

* **Callbacks:**
    * `UsageTrackCallback`: Log LLM usage details for analysis and monitoring.
* **Experiment Prediction:**
    * `ChatModelExperiment`: Facilitates the selection and execution of LangChain models based on experiment configurations.
* **LangChain Compatibility:**
    * Designed to integrate smoothly with LangChain workflows.

## Installation

```bash
pip install intura-ai
```

## Usage

### Initialization
Before using `intura-ai`, you need to initialize the client with your API key.
```python
import os
from intura_ai.client import intura_initialization

INTURA_API_KEY = "..."
intura_initialization(INTURA_API_KEY)
```

### Experiment Prediction
Use `ChatModelExperiment` to fetch and execute pre-configured LangChain models.

```python
from intura_ai.experiments import ChatModelExperiment

EXPERIMENT_ID = "..."
client = ChatModelExperiment(EXPERIMENT_ID)

choiced_model, model_config, chat_prompts = client.build(
    features={
        "user_id": "Rama12345", 
        "membership": "FREE", 
        "employment_type": "FULL_TIME",
        "feature_x": "your custom features"
    }
)
chat_prompts.append(('human', 'give me today quote for programmer'))

print(client.choiced_model) # Your choiced model for instance: claude-3-5-sonnet-20240620
model = choiced_model(**model_config, api_key="<YOUR_API_KEY | Set by Environment>")
model.invoke(chat_prompts)
```

### Usage Tracking Callback
Integrate `UsageTrackCallback` to log LLM usage during execution.

```python
from intura_ai.callbacks import UsageTrackCallback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

EXPERIMENT_ID = "..."
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    max_tokens=300,
    timeout=None,
    max_retries=2,
    callbacks=[
        UsageTrackCallback(EXPERIMENT_ID)
    ]
)

messages = [HumanMessage(content="What is the capital of France?")]
llm.invoke(messages)
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports or feature requests.

## License
This project is licensed under the MIT License.