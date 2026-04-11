"""
core/governance.py
-------------------
Model lifecycle governance layer.

Ensures trained models are evaluated, reviewed, and controlled before
reaching production. Implements human-in-the-loop checkpoints and
automated quality gates.

Governance Pipeline:
    train → [evaluate] → [quality_gate] → [review_gate] → promote/reject

Quality Gates:
  - CLIP score must exceed threshold
  - Aesthetic score must exceed threshold
  - FID score must be below threshold
  - Must not regress vs baseline

Review Gates:
  - auto: all gates pass → auto-promote
  - manual: write approval request to disk, wait for human confirmation
  - strict: always require human confirmation

Governance log persisted to governance/decisions.jsonl for audit.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Gate thresholds (configurable) ───────────────────────────

_DEFAULT_GATES = {
    "min_clip_score":        25.0,   # CLIP score floor
    "min_aesthetic_score":    4.0,   # aesthetic score floor
    "max_fid_score":         50.0,   # FID ceiling (None = no check)
    "max_clip_regression":   -3.0,   # max allowed drop from baseline
    "max_aesthetic_regression": -0.5,
}


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return {"gate": self.gate_name, "passed": self.passed,
                "reason": self.reason, "value": self.value,
                "threshold": self.threshold}


@dataclass
class GovernanceDecision:
    decision_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_type:     str   = "diffusion"
    version:        str   = ""
    timestamp:      float = field(default_factory=time.time)
    action:         str   = "pending"  # promote | reject | pending | escalate
    reason:         str   = ""
    gates:          List[GateResult] = field(default_factory=list)
    baseline_version: Optional[str] = None
    reviewer:       str   = "auto"
    override:       bool  = False

    @property
    def all_gates_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "model_type": self.model_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "action": self.action,
            "reason": self.reason,
            "gates_passed": self.all_gates_passed,
            "gates": [g.to_dict() for g in self.gates],
            "baseline_version": self.baseline_version,
            "reviewer": self.reviewer,
            "override": self.override,
        }


class GovernanceGate:
    """Evaluates a single quality gate."""

    def __init__(self, name: str, threshold: Optional[float],
                 higher_is_better: bool = True):
        self.name = name
        self.threshold = threshold
        self.higher_is_better = higher_is_better

    def evaluate(self, value: Optional[float]) -> GateResult:
        if value is None:
            return GateResult(self.name, True, "no data, skipping", None, self.threshold)
        if self.threshold is None:
            return GateResult(self.name, True, "no threshold set", value, None)
        if self.higher_is_better:
            passed = value >= self.threshold
            reason = (f"{value:.3f} >= {self.threshold:.3f}" if passed
                      else f"{value:.3f} < {self.threshold:.3f} (FAIL)")
        else:
            passed = value <= self.threshold
            reason = (f"{value:.3f} <= {self.threshold:.3f}" if passed
                      else f"{value:.3f} > {self.threshold:.3f} (FAIL)")
        return GateResult(self.name, passed, reason, value, self.threshold)


class ModelGovernance:
    """
    Controls model promotion pipeline with configurable gates and review modes.

    Modes:
      auto   — promote automatically if all gates pass
      manual — write approval request, block until file confirmed
      strict — always escalate to human review

    Usage:
        gov = ModelGovernance(registry, versioning, cfg)
        decision = gov.evaluate_and_decide("diffusion", "0.2.0",
                                           eval_clip=32.4, eval_aes=5.2)
        if decision.action == "promote":
            registry.promote("diffusion", "0.2.0")
    """

    def __init__(
        self,
        registry,                  # ModelRegistry
        versioning_hub,            # ModelVersioningHub
        governance_dir: str = "outputs/governance",
        mode: str = "auto",        # auto | manual | strict
        gate_config: Optional[dict] = None,
    ):
        self.registry       = registry
        self.versioning_hub = versioning_hub
        self.governance_dir = Path(governance_dir)
        self.governance_dir.mkdir(parents=True, exist_ok=True)
        self.mode           = mode
        self._gate_cfg      = {**_DEFAULT_GATES, **(gate_config or {})}
        self._decision_log  = self.governance_dir / "decisions.jsonl"

    def _gates(self) -> List[GovernanceGate]:
        return [
            GovernanceGate("clip_score",
                           self._gate_cfg.get("min_clip_score"), True),
            GovernanceGate("aesthetic_score",
                           self._gate_cfg.get("min_aesthetic_score"), True),
            GovernanceGate("fid_score",
                           self._gate_cfg.get("max_fid_score"), False),
        ]

    def _regression_gates(
        self, baseline_clip: Optional[float], baseline_aes: Optional[float],
        current_clip: Optional[float], current_aes: Optional[float],
    ) -> List[GateResult]:
        results = []
        if baseline_clip is not None and current_clip is not None:
            delta = current_clip - baseline_clip
            threshold = self._gate_cfg.get("max_clip_regression", -3.0)
            passed = delta >= threshold
            results.append(GateResult(
                "clip_regression", passed,
                f"delta={delta:+.3f} (threshold={threshold:+.3f})",
                delta, threshold,
            ))
        if baseline_aes is not None and current_aes is not None:
            delta = current_aes - baseline_aes
            threshold = self._gate_cfg.get("max_aesthetic_regression", -0.5)
            passed = delta >= threshold
            results.append(GateResult(
                "aesthetic_regression", passed,
                f"delta={delta:+.3f} (threshold={threshold:+.3f})",
                delta, threshold,
            ))
        return results

    def evaluate_and_decide(
        self,
        model_type: str,
        version: str,
        eval_clip_score: Optional[float] = None,
        eval_aesthetic_score: Optional[float] = None,
        eval_fid_score: Optional[float] = None,
        reviewer: str = "auto",
        force_action: Optional[str] = None,   # override: "promote" | "reject"
    ) -> GovernanceDecision:
        """
        Evaluate a candidate model version against quality gates and
        make a promotion decision.

        Returns GovernanceDecision with action: promote | reject | escalate | pending
        """
        decision = GovernanceDecision(
            model_type=model_type,
            version=version,
            reviewer=reviewer,
        )

        # Manual override
        if force_action in ("promote", "reject"):
            decision.action   = force_action
            decision.override = True
            decision.reason   = f"Manual override by {reviewer}"
            self._log_decision(decision)
            if force_action == "promote":
                self._execute_promotion(model_type, version, decision, reviewer)
            return decision

        # ── Quality gates ──────────────────────────────────────
        for gate in self._gates():
            val = {"clip_score": eval_clip_score,
                   "aesthetic_score": eval_aesthetic_score,
                   "fid_score": eval_fid_score}.get(gate.name)
            decision.gates.append(gate.evaluate(val))

        # ── Regression gates vs production baseline ────────────
        prod = self.registry.get_production(model_type)
        if prod:
            decision.baseline_version = prod.version
            reg_gates = self._regression_gates(
                prod.eval_clip_score, prod.eval_aesthetic_score,
                eval_clip_score, eval_aesthetic_score,
            )
            decision.gates.extend(reg_gates)

        failed_gates = [g for g in decision.gates if not g.passed]

        # ── Decision logic ─────────────────────────────────────
        if self.mode == "strict":
            decision.action = "escalate"
            decision.reason = "Strict mode: requires human review"

        elif self.mode == "manual":
            if failed_gates:
                decision.action = "reject"
                decision.reason = (
                    f"Failed gates: {', '.join(g.gate_name for g in failed_gates)}"
                )
            else:
                decision.action = "escalate"
                decision.reason = "Manual mode: awaiting human confirmation"
                self._write_approval_request(decision)

        else:  # auto
            if failed_gates:
                decision.action = "reject"
                decision.reason = (
                    f"Failed quality gates: "
                    f"{', '.join(g.gate_name for g in failed_gates)}"
                )
            else:
                decision.action = "promote"
                decision.reason = f"All {len(decision.gates)} gates passed"

        # ── Execute ────────────────────────────────────────────
        if decision.action == "promote":
            self._execute_promotion(model_type, version, decision, reviewer)

        elif decision.action == "reject":
            self.registry.mark_failed(model_type, version, decision.reason)
            logger.warning(
                f"REJECTED {model_type} v{version}: {decision.reason}"
            )

        self._log_decision(decision)
        self._print_decision(decision)
        return decision

    def _execute_promotion(
        self, model_type: str, version: str,
        decision: GovernanceDecision, reviewer: str,
    ) -> None:
        try:
            self.registry.promote(
                model_type, version,
                promoted_by=reviewer,
                require_eval=False,
            )
            # Bump model version
            self.versioning_hub.tracker(model_type).add_version(
                version, training_event="promote",
                notes=decision.reason,
            ) if version not in {
                n.version for n in
                self.versioning_hub.tracker(model_type).all_versions()
            } else None
            logger.info(f"PROMOTED {model_type} v{version}")
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            decision.action = "failed"
            decision.reason = str(e)

    def _write_approval_request(self, decision: GovernanceDecision) -> None:
        req_path = self.governance_dir / f"approval_request_{decision.decision_id}.json"
        req_path.write_text(json.dumps({
            "request": "APPROVAL REQUIRED",
            "decision_id": decision.decision_id,
            "model_type": decision.model_type,
            "version": decision.version,
            "gates": [g.to_dict() for g in decision.gates],
            "instructions": (
                "Review the gates above. To approve: rename this file to "
                f"approval_request_{decision.decision_id}.APPROVED.json. "
                "To reject: rename to .REJECTED.json"
            ),
        }, indent=2))
        logger.info(f"Approval request written → {req_path}")

    def _log_decision(self, decision: GovernanceDecision) -> None:
        with open(self._decision_log, "a") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    def _print_decision(self, decision: GovernanceDecision) -> None:
        icon = {"promote": "✓", "reject": "✗",
                "escalate": "?", "pending": "…"}.get(decision.action, "?")
        print(f"\n  [{icon}] GOVERNANCE: {decision.model_type} v{decision.version} "
              f"→ {decision.action.upper()}")
        print(f"      Reason: {decision.reason}")
        for g in decision.gates:
            flag = "✓" if g.passed else "✗"
            print(f"      [{flag}] {g.gate_name:<30} {g.reason}")
        print()

    def wait_for_approval(
        self, decision: GovernanceDecision, timeout_s: float = 3600.0,
        poll_s: float = 30.0,
    ) -> str:
        """
        Block until a manual approval request is resolved.
        Returns "approved" | "rejected" | "timeout".
        """
        import time as _time
        req_base = str(self.governance_dir /
                       f"approval_request_{decision.decision_id}")
        t0 = _time.time()
        while _time.time() - t0 < timeout_s:
            if Path(f"{req_base}.APPROVED.json").exists():
                logger.info("Manual approval received")
                return "approved"
            if Path(f"{req_base}.REJECTED.json").exists():
                logger.info("Manual rejection received")
                return "rejected"
            _time.sleep(poll_s)
        logger.warning(f"Approval timed out after {timeout_s:.0f}s")
        return "timeout"

    def audit_log(self, n: int = 20) -> List[dict]:
        """Return last N governance decisions."""
        if not self._decision_log.exists():
            return []
        lines = self._decision_log.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def update_gates(self, new_config: dict) -> None:
        """Dynamically update gate thresholds."""
        self._gate_cfg.update(new_config)
        logger.info(f"Governance gates updated: {new_config}")

    def print_audit_log(self, n: int = 10) -> None:
        log = self.audit_log(n)
        print(f"\n  Governance Audit Log (last {len(log)} decisions)")
        print(f"  {'='*55}")
        for d in log:
            icon = "✓" if d["action"] == "promote" else "✗"
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(d["timestamp"]))
            print(f"  [{icon}] {ts}  {d['model_type']:<12} v{d['version']:<12} "
                  f"→ {d['action']:<10}  {d['reason'][:40]}")
        print()
