"""
Workload models for AI training and inference patterns.

This module defines the data structures for representing different
types of AI workloads and their characteristics.
"""

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class WorkloadType(str, Enum):
    """Enumeration for different types of workloads."""
    TRAINING = "training"
    INFERENCE = "inference"

class Workload(BaseModel):
    """Represents a single workload or job to be processed."""
    id: str
    workload_type: WorkloadType
    required_tflops: float = Field(..., description="The computational requirement for the job in TFLOPS.")
    duration_hours: float = Field(..., description="The estimated duration of the job in hours.")
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    assigned_gpu_id: str | None = None 