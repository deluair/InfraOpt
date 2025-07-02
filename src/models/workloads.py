"""
Workload models for AI training and inference patterns.

This module defines the data structures for representing different
types of AI workloads and their characteristics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class WorkloadType(Enum):
    """Types of AI workloads."""
    TRAINING = "training"
    INFERENCE = "inference"
    RESEARCH = "research"
    COMMERCIAL = "commercial"


class ModelType(Enum):
    """Types of AI models."""
    LLM = "large_language_model"
    VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    AUTONOMOUS = "autonomous_systems"
    MULTIMODAL = "multimodal"


@dataclass
class WorkloadPattern:
    """Base class for workload patterns."""
    
    id: str
    name: str
    workload_type: WorkloadType
    model_type: ModelType
    priority: int  # 1-10 scale
    sla_requirements: Dict[str, Any]
    
    # Resource requirements
    compute_intensity: float  # TFLOPS required
    memory_requirement: float  # GB required
    storage_requirement: float  # TB required
    network_bandwidth: float  # Gbps required
    
    # Timing characteristics
    duration_hours: float
    frequency_per_day: float
    time_critical: bool
    
    def get_resource_score(self) -> float:
        """Calculate overall resource intensity score."""
        return (self.compute_intensity * 0.4 + 
                self.memory_requirement * 0.3 + 
                self.storage_requirement * 0.2 + 
                self.network_bandwidth * 0.1)


@dataclass
class TrainingJob(WorkloadPattern):
    """Represents AI model training workloads."""
    
    model_size_billions: float  # Billions of parameters
    training_data_size_tb: float
    convergence_epochs: int
    checkpoint_frequency: int  # Epochs between checkpoints
    distributed_training: bool
    gradient_accumulation_steps: int = 1
    
    # Training-specific metrics
    learning_rate: float = 1e-4
    batch_size: int = 32
    mixed_precision: bool = True
    
    def get_training_efficiency(self) -> float:
        """Calculate training efficiency based on configuration."""
        efficiency = 1.0
        
        # Efficiency factors
        if self.mixed_precision:
            efficiency *= 1.2  # 20% improvement with mixed precision
        
        if self.distributed_training:
            efficiency *= 1.3  # 30% improvement with distributed training
        
        # Penalty for large model size
        if self.model_size_billions > 100:
            efficiency *= 0.9
        
        return efficiency
    
    def estimate_training_time(self, available_tflops: float) -> float:
        """Estimate training time in hours."""
        # Simplified training time estimation
        base_time = self.model_size_billions * self.convergence_epochs * 0.1
        effective_tflops = available_tflops * self.get_training_efficiency()
        return base_time / effective_tflops


@dataclass
class InferenceRequest(WorkloadPattern):
    """Represents AI model inference workloads."""
    
    requests_per_second: float
    latency_requirement_ms: float
    throughput_requirement: float
    model_loading_time_seconds: float
    
    # Inference-specific characteristics
    batch_processing: bool = True
    dynamic_batching: bool = True
    model_quantization: bool = False
    
    def get_inference_efficiency(self) -> float:
        """Calculate inference efficiency based on configuration."""
        efficiency = 1.0
        
        if self.batch_processing:
            efficiency *= 1.15  # 15% improvement with batching
        
        if self.dynamic_batching:
            efficiency *= 1.1  # 10% improvement with dynamic batching
        
        if self.model_quantization:
            efficiency *= 1.25  # 25% improvement with quantization
        
        return efficiency
    
    def calculate_throughput(self, available_tflops: float) -> float:
        """Calculate achievable throughput in requests per second."""
        effective_tflops = available_tflops * self.get_inference_efficiency()
        return effective_tflops / self.compute_intensity


@dataclass
class ResearchWorkload(WorkloadPattern):
    """Represents experimental research workloads."""
    
    experimental_phase: str  # "exploration", "validation", "production"
    success_probability: float  # 0-1 scale
    iteration_count: int
    parallel_experiments: int
    
    # Research-specific characteristics
    requires_specialized_hardware: bool = False
    data_exploration_intensive: bool = True
    rapid_prototyping: bool = True
    
    def get_research_efficiency(self) -> float:
        """Calculate research efficiency based on characteristics."""
        efficiency = 1.0
        
        if self.parallel_experiments > 1:
            efficiency *= 0.9  # 10% penalty for parallel experiments
        
        if self.requires_specialized_hardware:
            efficiency *= 0.8  # 20% penalty for specialized requirements
        
        return efficiency


@dataclass
class CommercialWorkload(WorkloadPattern):
    """Represents production commercial workloads."""
    
    revenue_per_request: float  # $ per request
    sla_penalty_cost: float  # $ per SLA violation
    peak_load_multiplier: float  # Peak vs average load
    seasonal_variation: Dict[str, float]  # Monthly variation factors
    
    # Commercial-specific characteristics
    auto_scaling: bool = True
    load_balancing: bool = True
    disaster_recovery: bool = True
    
    def calculate_revenue_potential(self, requests_per_day: float) -> float:
        """Calculate potential daily revenue."""
        return requests_per_day * self.revenue_per_request
    
    def get_sla_risk_cost(self, availability: float) -> float:
        """Calculate SLA violation risk cost."""
        sla_target = self.sla_requirements.get("availability", 0.99)
        if availability < sla_target:
            violation_rate = sla_target - availability
            return violation_rate * self.sla_penalty_cost
        return 0.0


# Predefined workload templates
WORKLOAD_TEMPLATES = {
    "llm_training": TrainingJob(
        id="llm_training_1b",
        name="Large Language Model Training (1B parameters)",
        workload_type=WorkloadType.TRAINING,
        model_type=ModelType.LLM,
        priority=8,
        sla_requirements={"completion_time": 168},  # 7 days
        compute_intensity=500.0,  # TFLOPS
        memory_requirement=640.0,  # GB
        storage_requirement=10.0,  # TB
        network_bandwidth=100.0,  # Gbps
        duration_hours=168.0,
        frequency_per_day=0.1,  # Once every 10 days
        time_critical=True,
        model_size_billions=1.0,
        training_data_size_tb=100.0,
        convergence_epochs=3,
        checkpoint_frequency=1,
        distributed_training=True
    ),
    
    "vision_inference": InferenceRequest(
        id="vision_inference_realtime",
        name="Real-time Computer Vision Inference",
        workload_type=WorkloadType.INFERENCE,
        model_type=ModelType.VISION,
        priority=9,
        sla_requirements={"latency_ms": 100, "availability": 0.999},
        compute_intensity=50.0,  # TFLOPS
        memory_requirement=32.0,  # GB
        storage_requirement=1.0,  # TB
        network_bandwidth=10.0,  # Gbps
        duration_hours=24.0,
        frequency_per_day=1.0,
        time_critical=True,
        requests_per_second=1000.0,
        latency_requirement_ms=100.0,
        throughput_requirement=1000.0,
        model_loading_time_seconds=30.0,
        batch_processing=True,
        dynamic_batching=True
    ),
    
    "recommendation_system": CommercialWorkload(
        id="recommendation_production",
        name="Production Recommendation System",
        workload_type=WorkloadType.COMMERCIAL,
        model_type=ModelType.RECOMMENDATION,
        priority=7,
        sla_requirements={"latency_ms": 50, "availability": 0.995},
        compute_intensity=20.0,  # TFLOPS
        memory_requirement=128.0,  # GB
        storage_requirement=5.0,  # TB
        network_bandwidth=50.0,  # Gbps
        duration_hours=24.0,
        frequency_per_day=1.0,
        time_critical=True,
        revenue_per_request=0.01,  # $0.01 per recommendation
        sla_penalty_cost=1000.0,  # $1000 per SLA violation
        peak_load_multiplier=3.0,
        seasonal_variation={
            "jan": 0.8, "feb": 0.9, "mar": 1.0, "apr": 1.1,
            "may": 1.2, "jun": 1.3, "jul": 1.2, "aug": 1.1,
            "sep": 1.0, "oct": 1.1, "nov": 1.2, "dec": 1.4
        }
    ),
    
    "autonomous_research": ResearchWorkload(
        id="autonomous_research_exp",
        name="Autonomous Systems Research",
        workload_type=WorkloadType.RESEARCH,
        model_type=ModelType.AUTONOMOUS,
        priority=5,
        sla_requirements={"flexibility": "high"},
        compute_intensity=100.0,  # TFLOPS
        memory_requirement=256.0,  # GB
        storage_requirement=20.0,  # TB
        network_bandwidth=25.0,  # Gbps
        duration_hours=48.0,
        frequency_per_day=0.5,  # Every 2 days
        time_critical=False,
        experimental_phase="exploration",
        success_probability=0.3,
        iteration_count=10,
        parallel_experiments=3,
        requires_specialized_hardware=True
    )
}


# Workload scheduling patterns
SCHEDULING_PATTERNS = {
    "batch_processing": {
        "description": "Non-time-critical batch processing",
        "priority_range": (1, 5),
        "time_critical": False,
        "resource_preemption": True
    },
    "interactive": {
        "description": "Interactive workloads with moderate latency requirements",
        "priority_range": (6, 8),
        "time_critical": True,
        "resource_preemption": False
    },
    "real_time": {
        "description": "Real-time workloads with strict latency requirements",
        "priority_range": (9, 10),
        "time_critical": True,
        "resource_preemption": False
    }
} 