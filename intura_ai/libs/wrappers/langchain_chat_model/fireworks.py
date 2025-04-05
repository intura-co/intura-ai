from langchain_fireworks import ChatFireworks
from .intura import InturaChatModel

class InturaChatFireworks(InturaChatModel, ChatFireworks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    