import uuid
from pydantic import BaseModel, Field
from typing import List, Optional
from .treatment import ExperimentTreatmentModel

class ExperimentConfigurationModel(BaseModel):
    explore_exploit_ratio: float = Field(0.5, description="The explore-exploit ratio for decision-making.")
    
class ExperimentReward(BaseModel):
    key: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier.")
    name: str = Field(..., description="Reward name (e.g., 'click', 'token usage', 'latency').")
    value: float = Field(..., description="Value must be between 0 and 1.")
    

class ExperimentModel(BaseModel):
    experiment_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier.")
    experiment_name: str = Field(..., description="Experiment name.")
    experiment_type: str = Field("CHAT_EXPERIMENT", description="LLM experiment type (e.g., 'CHAT', 'MULTIMODAL', 'CONTENT').")
    experiment_description: Optional[str] = Field(None, description="Experiment description.")
    experiment_status: str = Field("LIVE", description="Experiment status (e.g., 'LIVE', 'DRAFT', 'EXPIRED', 'PENDING').")
    experiment_config: ExperimentConfigurationModel = Field(ExperimentConfigurationModel(), description="Experiment configuration.")
    is_active: bool = Field(True, description="Indicates whether the experiment is active.")
    reward_formula: List[ExperimentReward] = Field(
        [
            ExperimentReward(name="total_tokens", value=0.5),
            ExperimentReward(name="internal_like_event", value=0.5),
        ],
        description="Reward formula; the sum of values must equal 1.",
    )
    treatment_list: List[ExperimentTreatmentModel] = Field(..., description="List of treatments.")