"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DetectedClaim(BaseModel):
    """Represents a claim detected by the agent.
    
    Attributes:
        claim_text: The text of the detected claim
        label: Classification - "factual", "hallucinated", or "unverifiable"
        reason: Explanation for the label assignment
        corrected_fact: Correction if hallucinated, None otherwise
    """
    claim_text: str = Field(..., description="The text of the detected claim")
    label: str = Field(
        ..., 
        description="Classification: 'factual', 'hallucinated', or 'unverifiable'"
    )
    reason: str = Field(..., description="Explanation for the label")
    corrected_fact: Optional[str] = Field(
        None, 
        description="Correction if hallucinated"
    )
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label is one of the allowed values."""
        valid_labels = {"factual", "hallucinated", "unverifiable"}
        if v not in valid_labels:
            raise ValueError(
                f"Invalid label '{v}'. Must be one of {valid_labels}"
            )
        return v


class DetectionOutput(BaseModel):
    """Agent's detection output containing all detected claims.
    
    Attributes:
        detected_claims: List of claims detected by the agent
    """
    detected_claims: List[DetectedClaim] = Field(
        ..., 
        description="List of detected claims with labels and corrections"
    )


class Observation(BaseModel):
    """Observation provided to the agent at the start of an episode.
    
    Attributes:
        generated_text: The LLM-generated text to analyze for hallucinations
        task_instruction: Instructions for the detection task
    """
    generated_text: str = Field(
        ..., 
        description="The LLM output to analyze"
    )
    task_instruction: str = Field(
        ..., 
        description="Instructions for detection task"
    )


class Action(BaseModel):
    """Action submitted by the agent.
    
    Attributes:
        detection_output: The agent's detection results
    """
    detection_output: DetectionOutput = Field(
        ..., 
        description="Detection results from the agent"
    )


class StepResult(BaseModel):
    """Result returned after an agent takes a step.
    
    Attributes:
        observation: Next observation (empty for single-turn episodes)
        reward: Reward score for the agent's detection output
        done: Whether the episode is complete
        info: Additional information including metrics and metadata
    """
    observation: Observation = Field(
        ..., 
        description="Next observation"
    )
    reward: float = Field(
        ..., 
        description="Reward score"
    )
    done: bool = Field(
        ..., 
        description="Whether episode is complete"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional info: difficulty_level, source_dataset, metrics"
    )
