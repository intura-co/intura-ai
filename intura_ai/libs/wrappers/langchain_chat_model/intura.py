from langchain.schema.runnable import Runnable

class InturaChatModel(Runnable):
        
    def invoke(self, input, config=None, **kwargs):
        ai_msg = super().invoke(input, config, **kwargs)
        return ai_msg
    
    async def ainvoke(self, input, config = None, **kwargs):
        ai_msg = await super().ainvoke(input, config, **kwargs)
        return ai_msg