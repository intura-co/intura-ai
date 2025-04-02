from langchain_ollama import ChatOllama
from .intura import InturaChatModel

class InturaChatOllama(InturaChatModel, ChatOllama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        