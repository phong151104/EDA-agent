"""
Hypothesis Log module.

Shared state for tracking plan and hypothesis evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class PlanVersion:
    """A version of the analysis plan."""
    
    version: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    plan_data: dict[str, Any] = field(default_factory=dict)
    critic_feedback: str | None = None
    approval_score: float | None = None


@dataclass
class HypothesisRecord:
    """Record of a hypothesis through its lifecycle."""
    
    id: str
    statement: str
    rationale: str
    status_history: list[tuple[str, datetime]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    final_verdict: str | None = None  # "validated", "invalidated", "inconclusive"
    
    def update_status(self, new_status: str) -> None:
        """Update hypothesis status."""
        self.status_history.append((new_status, datetime.utcnow()))


class HypothesisLog:
    """
    Log for tracking plan and hypothesis evolution.
    
    Maintains history of all plan versions and hypothesis
    status changes throughout the analysis session.
    """
    
    def __init__(self, session_id: str | None = None):
        """
        Initialize hypothesis log.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or str(uuid4())
        self.plan_versions: list[PlanVersion] = []
        self.hypotheses: dict[str, HypothesisRecord] = {}
        self.debate_log: list[dict[str, Any]] = []
    
    def add_plan_version(
        self,
        plan_data: dict[str, Any],
        critic_feedback: str | None = None,
        approval_score: float | None = None,
    ) -> int:
        """
        Add a new plan version.
        
        Args:
            plan_data: The plan data
            critic_feedback: Feedback from critic
            approval_score: Approval score
            
        Returns:
            Version number
        """
        version = len(self.plan_versions) + 1
        self.plan_versions.append(
            PlanVersion(
                version=version,
                plan_data=plan_data,
                critic_feedback=critic_feedback,
                approval_score=approval_score,
            )
        )
        return version
    
    def add_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        rationale: str,
    ) -> HypothesisRecord:
        """
        Add or update a hypothesis.
        
        Args:
            hypothesis_id: Unique hypothesis ID
            statement: Hypothesis statement
            rationale: Reasoning behind hypothesis
            
        Returns:
            HypothesisRecord
        """
        if hypothesis_id in self.hypotheses:
            record = self.hypotheses[hypothesis_id]
        else:
            record = HypothesisRecord(
                id=hypothesis_id,
                statement=statement,
                rationale=rationale,
            )
            record.update_status("pending")
            self.hypotheses[hypothesis_id] = record
        
        return record
    
    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: str,
        evidence: str | None = None,
    ) -> None:
        """
        Update hypothesis status.
        
        Args:
            hypothesis_id: Hypothesis to update
            new_status: New status
            evidence: Optional supporting evidence
        """
        if hypothesis_id in self.hypotheses:
            record = self.hypotheses[hypothesis_id]
            record.update_status(new_status)
            if evidence:
                record.evidence.append(evidence)
    
    def log_debate_round(
        self,
        iteration: int,
        planner_output: dict[str, Any],
        critic_output: dict[str, Any],
    ) -> None:
        """
        Log a Planner-Critic debate round.
        
        Args:
            iteration: Debate iteration number
            planner_output: Output from Planner
            critic_output: Output from Critic
        """
        self.debate_log.append({
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "planner": planner_output,
            "critic": critic_output,
        })
    
    def get_current_plan(self) -> PlanVersion | None:
        """Get the latest plan version."""
        return self.plan_versions[-1] if self.plan_versions else None
    
    def get_hypothesis_summary(self) -> dict[str, list[str]]:
        """
        Get summary of hypotheses by status.
        
        Returns:
            Dictionary mapping status to list of hypothesis IDs
        """
        summary: dict[str, list[str]] = {
            "validated": [],
            "invalidated": [],
            "pending": [],
            "inconclusive": [],
        }
        
        for h_id, record in self.hypotheses.items():
            if record.final_verdict:
                status = record.final_verdict
            elif record.status_history:
                status = record.status_history[-1][0]
            else:
                status = "pending"
            
            if status in summary:
                summary[status].append(h_id)
            else:
                summary["pending"].append(h_id)
        
        return summary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert log to dictionary."""
        return {
            "sessionId": self.session_id,
            "planVersions": [
                {
                    "version": v.version,
                    "createdAt": v.created_at.isoformat(),
                    "approvalScore": v.approval_score,
                }
                for v in self.plan_versions
            ],
            "hypotheses": {
                h_id: {
                    "statement": r.statement,
                    "statusHistory": [
                        {"status": s, "timestamp": t.isoformat()}
                        for s, t in r.status_history
                    ],
                    "finalVerdict": r.final_verdict,
                }
                for h_id, r in self.hypotheses.items()
            },
            "debateRounds": len(self.debate_log),
        }
