"""
Data models module.

Pydantic models for API requests/responses and internal data structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlanStatus(str, Enum):
    """Analysis plan status."""
    
    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class HypothesisModel(BaseModel):
    """Hypothesis data model."""
    
    id: str
    statement: str
    rationale: str
    status: str = "pending"
    evidence: list[str] = Field(default_factory=list)


class StepModel(BaseModel):
    """Analysis step model."""
    
    step_number: int
    description: str
    action_type: str
    details: dict[str, Any] = Field(default_factory=dict)


class PlanModel(BaseModel):
    """Analysis plan model."""
    
    question: str
    hypotheses: list[HypothesisModel]
    steps: list[StepModel]
    version: int = 1
    status: PlanStatus = PlanStatus.DRAFT


class InsightModel(BaseModel):
    """Data insight model."""
    
    id: str
    type: str
    title: str
    description: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class ReportModel(BaseModel):
    """Final analysis report model."""
    
    session_id: str
    question: str
    summary: str
    hypotheses: list[HypothesisModel]
    insights: list[InsightModel]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
