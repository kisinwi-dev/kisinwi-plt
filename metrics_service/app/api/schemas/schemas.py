from pydantic import BaseModel
from typing import Dict, List, Union

class MetricUpdate(BaseModel):
    task_id: str 
    metric_name: str
    value: Union[float, int, List[float]]
    step: int

class MetricData(BaseModel):
    steps: List[int] = []
    values: List[Union[float, int, List[float]]] = [] 

class TaskMetrics(BaseModel):
    task_id: str
    metrics: Dict[str, MetricData] = {} 

class MetricsResponse(BaseModel):
    task_id: str
    metrics: Dict[str, MetricData]