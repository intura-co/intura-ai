from pydantic import BaseModel

class Treatment(BaseModel):
    treatment_name: str