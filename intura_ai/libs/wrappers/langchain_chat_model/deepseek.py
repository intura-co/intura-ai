from langchain_deepseek import ChatDeepSeek
from .intura import InturaChatModel

class InturaChatDeepSeek(InturaChatModel, ChatDeepSeek):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        