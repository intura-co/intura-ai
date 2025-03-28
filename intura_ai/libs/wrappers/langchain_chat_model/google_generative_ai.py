from langchain_google_genai import ChatGoogleGenerativeAI
from .intura import InturaChatModel

class InturaChatGoogleGenerativeAI(InturaChatModel, ChatGoogleGenerativeAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        