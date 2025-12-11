"""
Plan Verifier - 3-layer verification for Critic Agent.

Layer 1: Data Availability (Neo4j query)
Layer 2: Logical Consistency (Rules-based)
Layer 3: Business Logic (Optional LLM)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """A verification issue found in the plan."""
    
    layer: str  # "data", "logic", "business"
    severity: str  # "error", "warning", "info"
    step_id: str  # Which step has the issue
    message: str
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "severity": self.severity,
            "step_id": self.step_id,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class VerificationResult:
    """Result of plan verification."""
    
    passed: bool
    layer1_passed: bool = True  # Data availability
    layer2_passed: bool = True  # Logical consistency
    layer3_passed: bool = True  # Business logic
    issues: List[VerificationIssue] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "layer1_passed": self.layer1_passed,
            "layer2_passed": self.layer2_passed,
            "layer3_passed": self.layer3_passed,
            "issues": [i.to_dict() for i in self.issues],
        }


class PlanVerifier:
    """
    3-layer verification for analysis plans.
    
    Layer 1: Data Availability - Check if referenced data exists in Neo4j
    Layer 2: Logical Consistency - Check step dependencies and structure
    Layer 3: Business Logic - Optional LLM check for business sense
    """
    
    def __init__(self, domain: str = "vnfilm_ticketing"):
        self.domain = domain
    
    async def verify(
        self,
        plan: Dict[str, Any],
        run_layer3: bool = False,
    ) -> VerificationResult:
        """
        Run all verification layers on the plan.
        
        Args:
            plan: The analysis plan to verify
            run_layer3: Whether to run LLM-based business logic check
            
        Returns:
            VerificationResult with all issues found
        """
        issues = []
        
        # Layer 1: Data Availability
        logger.info("[Critic] ═══════════════════════════════════════════════")
        logger.info("[Critic] Layer 1: DATA AVAILABILITY CHECK")
        logger.info("[Critic] ───────────────────────────────────────────────")
        
        layer1_issues = await self._verify_data_availability(plan)
        issues.extend(layer1_issues)
        
        layer1_passed = not any(i.severity == "error" for i in layer1_issues)
        if layer1_passed:
            logger.info("[Critic] ✅ Layer 1 PASSED - All data requirements available")
        else:
            logger.warning(f"[Critic] ❌ Layer 1 FAILED - {len(layer1_issues)} issues found")
        
        # Layer 2: Logical Consistency
        logger.info("[Critic] ───────────────────────────────────────────────")
        logger.info("[Critic] Layer 2: LOGICAL CONSISTENCY CHECK")
        logger.info("[Critic] ───────────────────────────────────────────────")
        
        layer2_issues = self._verify_logical_consistency(plan)
        issues.extend(layer2_issues)
        
        layer2_passed = not any(i.severity == "error" for i in layer2_issues)
        if layer2_passed:
            logger.info("[Critic] ✅ Layer 2 PASSED - Plan structure is valid")
        else:
            logger.warning(f"[Critic] ❌ Layer 2 FAILED - {len(layer2_issues)} issues found")
        
        # Layer 3: Business Logic (optional)
        layer3_passed = True
        if run_layer3 and layer1_passed and layer2_passed:
            logger.info("[Critic] ───────────────────────────────────────────────")
            logger.info("[Critic] Layer 3: BUSINESS LOGIC CHECK (LLM)")
            logger.info("[Critic] ───────────────────────────────────────────────")
            
            layer3_issues = await self._verify_business_logic(plan)
            issues.extend(layer3_issues)
            
            layer3_passed = not any(i.severity == "error" for i in layer3_issues)
            if layer3_passed:
                logger.info("[Critic] ✅ Layer 3 PASSED - Business logic is sound")
            else:
                logger.warning(f"[Critic] ❌ Layer 3 FAILED - {len(layer3_issues)} issues found")
        
        # Overall result
        overall_passed = layer1_passed and layer2_passed and layer3_passed
        
        logger.info("[Critic] ═══════════════════════════════════════════════")
        logger.info(f"[Critic] VERIFICATION RESULT: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        logger.info(f"[Critic] Layer 1 (Data): {'✅' if layer1_passed else '❌'}")
        logger.info(f"[Critic] Layer 2 (Logic): {'✅' if layer2_passed else '❌'}")
        logger.info(f"[Critic] Layer 3 (Biz): {'✅' if layer3_passed else '⏭️ Skipped' if not run_layer3 else '❌'}")
        logger.info("[Critic] ═══════════════════════════════════════════════")
        
        return VerificationResult(
            passed=overall_passed,
            layer1_passed=layer1_passed,
            layer2_passed=layer2_passed,
            layer3_passed=layer3_passed,
            issues=issues,
        )
    
    # =========================================================================
    # Layer 1: Data Availability
    # =========================================================================
    
    async def _verify_data_availability(
        self,
        plan: Dict[str, Any],
    ) -> List[VerificationIssue]:
        """Check if referenced data exists in Neo4j."""
        from src.validation import MetadataStore
        
        issues = []
        steps = plan.get("steps", [])
        
        with MetadataStore(domain=self.domain) as store:
            for step in steps:
                step_id = step.get("id", f"s{step.get('step_number', '?')}")
                reqs = step.get("requirements", {})
                
                # Check tables_hint
                tables_hint = reqs.get("tables_hint", [])
                for table in tables_hint:
                    logger.info(f"[Critic]   Checking table: {table}")
                    
                    if not store.get_table(table):
                        similar = store.find_similar_table(table)
                        issues.append(VerificationIssue(
                            layer="data",
                            severity="error",
                            step_id=step_id,
                            message=f"Table '{table}' không tồn tại",
                            suggestion=f"Có thể dùng '{similar}'?" if similar else None,
                        ))
                        logger.warning(f"[Critic]     ❌ Table '{table}' NOT FOUND")
                    else:
                        logger.info(f"[Critic]     ✅ Table '{table}' exists")
                
                # Check data_needed → can we find relevant columns?
                data_needed = reqs.get("data_needed", [])
                for data in data_needed:
                    logger.info(f"[Critic]   Checking data: '{data}'")
                    
                    # Try to find matching column by name or concept
                    found = False
                    for table in tables_hint or store.get_all_tables():
                        cols = store.get_columns(table)
                        for col in cols:
                            col_name = col.get("column_name", "")
                            col_desc = col.get("description", "")
                            col_biz = col.get("business_name", "")
                            
                            # Simple matching
                            if any(keyword in data.lower() for keyword in [
                                col_name.lower(),
                                col_biz.lower() if col_biz else "",
                            ]):
                                found = True
                                logger.info(f"[Critic]     ✅ Mapped '{data}' → {table}.{col_name}")
                                break
                        if found:
                            break
                    
                    if not found:
                        # Not an error, just a warning - Code Agent might figure it out
                        issues.append(VerificationIssue(
                            layer="data",
                            severity="warning",
                            step_id=step_id,
                            message=f"Chưa map được '{data}' với column cụ thể",
                            suggestion="Code Agent sẽ xác định column phù hợp",
                        ))
                        logger.info(f"[Critic]     ⚠️ '{data}' - no exact match, Code Agent will resolve")
        
        return issues
    
    # =========================================================================
    # Layer 2: Logical Consistency
    # =========================================================================
    
    def _verify_logical_consistency(
        self,
        plan: Dict[str, Any],
    ) -> List[VerificationIssue]:
        """Check step dependencies and structure."""
        issues = []
        
        hypotheses = plan.get("hypotheses", [])
        steps = plan.get("steps", [])
        
        hypothesis_ids = {h.get("id") for h in hypotheses}
        step_ids = {s.get("id", f"s{s.get('step_number', i)}") for i, s in enumerate(steps)}
        
        logger.info(f"[Critic]   Hypotheses: {list(hypothesis_ids)}")
        logger.info(f"[Critic]   Steps: {list(step_ids)}")
        
        # Check 1: Every step has valid hypothesis_id
        for step in steps:
            step_id = step.get("id", f"s{step.get('step_number', '?')}")
            hypo_id = step.get("hypothesis_id", "")
            
            if hypo_id and hypo_id not in hypothesis_ids:
                issues.append(VerificationIssue(
                    layer="logic",
                    severity="warning",
                    step_id=step_id,
                    message=f"Step tham chiếu hypothesis '{hypo_id}' không tồn tại",
                ))
                logger.warning(f"[Critic]   ⚠️ Step {step_id} references unknown hypothesis {hypo_id}")
        
        # Check 2: Dependencies exist
        for step in steps:
            step_id = step.get("id", f"s{step.get('step_number', '?')}")
            depends_on = step.get("depends_on", [])
            
            for dep in depends_on:
                if dep not in step_ids:
                    issues.append(VerificationIssue(
                        layer="logic",
                        severity="error",
                        step_id=step_id,
                        message=f"Dependency '{dep}' không tồn tại",
                    ))
                    logger.warning(f"[Critic]   ❌ Step {step_id} depends on missing step {dep}")
                else:
                    logger.info(f"[Critic]   ✅ Step {step_id} → {dep} (valid dependency)")
        
        # Check 3: Visualization steps should depend on query
        for step in steps:
            step_id = step.get("id", f"s{step.get('step_number', '?')}")
            action_type = step.get("action_type", "")
            depends_on = step.get("depends_on", [])
            
            if action_type == "visualization" and not depends_on:
                issues.append(VerificationIssue(
                    layer="logic",
                    severity="warning",
                    step_id=step_id,
                    message="Visualization step nên có dữ liệu từ query trước",
                    suggestion="Thêm depends_on tới step query tương ứng",
                ))
                logger.warning(f"[Critic]   ⚠️ Visualization {step_id} has no data dependency")
        
        # Check 4: At least one step per hypothesis
        steps_by_hypo = {}
        for step in steps:
            hypo_id = step.get("hypothesis_id", "")
            if hypo_id:
                steps_by_hypo.setdefault(hypo_id, []).append(step)
        
        for hypo_id in hypothesis_ids:
            if hypo_id not in steps_by_hypo:
                issues.append(VerificationIssue(
                    layer="logic",
                    severity="warning",
                    step_id="plan",
                    message=f"Hypothesis '{hypo_id}' không có step nào để validate",
                ))
                logger.warning(f"[Critic]   ⚠️ Hypothesis {hypo_id} has no validation steps")
            else:
                logger.info(f"[Critic]   ✅ Hypothesis {hypo_id} has {len(steps_by_hypo[hypo_id])} steps")
        
        # Check 5: No circular dependencies (simple check)
        visited = set()
        for step in steps:
            step_id = step.get("id")
            deps = step.get("depends_on", [])
            if step_id in deps:
                issues.append(VerificationIssue(
                    layer="logic",
                    severity="error",
                    step_id=step_id,
                    message="Step tự phụ thuộc vào chính nó (circular dependency)",
                ))
                logger.error(f"[Critic]   ❌ Step {step_id} has circular dependency")
        
        return issues
    
    # =========================================================================
    # Layer 3: Business Logic (Optional LLM)
    # =========================================================================
    
    async def _verify_business_logic(
        self,
        plan: Dict[str, Any],
    ) -> List[VerificationIssue]:
        """Check business logic with LLM."""
        # TODO: Implement LLM-based business logic validation
        # For now, just return empty - this layer is optional
        logger.info("[Critic]   LLM business logic check - TODO")
        return []
    
    def format_issues(self, issues: List[VerificationIssue]) -> str:
        """Format issues as feedback for Planner."""
        if not issues:
            return ""
        
        lines = ["## Critic Feedback\n"]
        
        # Group by layer
        by_layer = {}
        for issue in issues:
            by_layer.setdefault(issue.layer, []).append(issue)
        
        layer_names = {
            "data": "📊 Data Availability Issues",
            "logic": "🔗 Logical Consistency Issues",
            "business": "💼 Business Logic Issues",
        }
        
        for layer, layer_issues in by_layer.items():
            lines.append(f"\n### {layer_names.get(layer, layer)}\n")
            
            errors = [i for i in layer_issues if i.severity == "error"]
            warnings = [i for i in layer_issues if i.severity == "warning"]
            
            if errors:
                for i, issue in enumerate(errors, 1):
                    lines.append(f"❌ **[{issue.step_id}]** {issue.message}")
                    if issue.suggestion:
                        lines.append(f"   → {issue.suggestion}")
            
            if warnings:
                for i, issue in enumerate(warnings, 1):
                    lines.append(f"⚠️ **[{issue.step_id}]** {issue.message}")
                    if issue.suggestion:
                        lines.append(f"   → {issue.suggestion}")
        
        return "\n".join(lines)
