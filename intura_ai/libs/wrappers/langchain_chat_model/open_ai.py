from langchain_openai import ChatOpenAI
from .intura import InturaChatModel

class InturaChatOpenAI(InturaChatModel, ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)