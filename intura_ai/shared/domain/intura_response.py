from typing import Any, Dict
from pydantic import BaseModel, Field

class InturaResponseModel(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str
    status_code: int
    success: bool
    message: str