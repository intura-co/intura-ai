import time
import asyncio

from uuid import uuid4
from langchain.schema.runnable import Runnable
from intura_ai.client import get_intura_client

class InturaChatModel(Runnable):
        
    def invoke(self, input, config=None, **kwargs):
        start_time = time.perf_counter() 
        ai_msg = super().invoke(input, config, **kwargs)
        asyncio.create_task(
            self._insert_intura_inference(
                ai_msg, 
                metadata=self.metadata, 
                start_time=start_time, 
            )
        )
        return ai_msg
    
    async def _insert_intura_inference(self, ai_msg, **kwargs):
        intura_api = get_intura_client()
        end_time = time.perf_counter()
        input_token = ai_msg.usage_metadata["input_tokens"]
        output_token = ai_msg.usage_metadata["output_tokens"]
        latency = (end_time - kwargs["start_time"]) * 1000
        payload = {
            "session_id": kwargs["metadata"]["session_id"],
            "experiment_id": kwargs["metadata"]["experiment_id"],
            "latency": latency,
            "result": [
                    {
                "treatment_id": kwargs["metadata"]["treatment_id"],
                "treatment_name": kwargs["metadata"]["treatment_name"],
                "prediction_id": str(uuid4()),
                "predictions": {
                    "result": ai_msg.content,
                    "cost": {
                        "total_tokens": input_token + output_token,
                        "output_tokens": output_token,
                        "input_tokens": input_token,
                        "cached_tokens": None
                    },
                    "latency": latency
                },
                "prediction_attribute": {
                    "source": "SDK"
                }
            }]
        }
        intura_api.insert_log_inference(payload=payload)
    