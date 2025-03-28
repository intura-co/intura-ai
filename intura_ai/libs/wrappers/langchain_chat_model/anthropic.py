from langchain_anthropic import ChatAnthropic
from .intura import InturaChatModel

class InturaChatAnthropic(InturaChatModel, ChatAnthropic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    