
from pydantic import BaseModel, Field
from typing import Optional


class PredictionInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    sex: str
    age: Optional[float] = Field(default=None, ge=0)
    sibsp: int = Field(default=0, ge=0)
    parch: int = Field(default=0, ge=0)
    fare: float = Field(default=0.0, ge=0)
    embarked: Optional[str] = None
    who: str
    adult_male: bool
    deck: Optional[str] = None
    embark_town: Optional[str] = None
    alone: bool


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
