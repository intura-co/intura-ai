from langchain_together import ChatTogether
from .intura import InturaChatModel

class InturaChatTogether(InturaChatModel, ChatTogether):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    