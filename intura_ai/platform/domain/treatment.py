import uuid
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, Union

class ExperimentTreatmentModel(BaseModel):
    treatment_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier.")
    treatment_model_name: str = Field(description="Model name (e.g., 'gpt-4', 'claude-haiku').")
    treatment_model_provider: str = Field(description="Model provider (e.g., 'Google', 'Anthropic').")
    treatment_name: Optional[str] = Field(default=None, description="Treatment name.")
    treatment_description: Optional[str] = Field(default=None, description="Treatment description.")
    treatment_model_configuration: Optional[Union[Dict[str, Any], str, None]] = Field(default={}, description="Model configuration.")
    prompt: str = Field(description="Prompt text.")
    feature_flag: str = Field(default="LIVE_MODE", description="Feature flag (e.g., 'LIVE_MODE', 'LOG_MODE', 'OFF_MODE').")

    @field_validator('treatment_name', mode='before')
    def set_treatment_name(cls, v, info):
        # If treatment_name is not provided, use treatment_model_name
        if v is None:
            values = info.data
            if 'treatment_model_name' in values:
                return values['treatment_model_name']
        return v