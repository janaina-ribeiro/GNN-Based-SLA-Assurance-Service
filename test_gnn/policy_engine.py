from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple


@dataclass
class DecisionResult:
    alert_triggered: int
    reason: str


class PolicyEngine:
    """Simple risk policy with per-link cooldown and optional global budget.

    Thresholds are configured per horizon (in minutes), for example:
    {3: 0.55, 10: 0.45, 30: 0.35}
    """

    def __init__(
        self,
        thresholds: Dict[int, float],
        cooldown_minutes: int = 10,
        max_alerts_per_hour: Optional[int] = None,
    ) -> None:
        self.thresholds = {int(k): float(v) for k, v in thresholds.items()}
        self.cooldown = timedelta(minutes=int(cooldown_minutes))
        self.max_alerts_per_hour = (
            int(max_alerts_per_hour) if max_alerts_per_hour is not None else None
        )

        self._last_alert_by_link: Dict[str, datetime] = {}
        self._global_alert_window: Deque[datetime] = deque()

    def _cleanup_global_window(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=1)
        while self._global_alert_window and self._global_alert_window[0] < cutoff:
            self._global_alert_window.popleft()

    def _check_budget(self, now: datetime) -> bool:
        if self.max_alerts_per_hour is None:
            return True
        self._cleanup_global_window(now)
        return len(self._global_alert_window) < self.max_alerts_per_hour

    def _check_threshold(self, horizon_minutes: int, risk: float) -> Tuple[bool, str]:
        if horizon_minutes not in self.thresholds:
            return False, f"missing_threshold_h{horizon_minutes}"
        threshold = self.thresholds[horizon_minutes]
        if risk >= threshold:
            return True, "threshold_pass"
        return False, "below_threshold"

    def _check_cooldown(self, link: str, now: datetime) -> Tuple[bool, str]:
        last = self._last_alert_by_link.get(link)
        if last is None:
            return True, "cooldown_pass"
        if now - last >= self.cooldown:
            return True, "cooldown_pass"
        return False, "cooldown_block"

    def decide(self, *, timestamp: datetime, link: str, horizon_minutes: int, risk: float) -> DecisionResult:
        threshold_ok, threshold_reason = self._check_threshold(horizon_minutes, risk)
        if not threshold_ok:
            return DecisionResult(alert_triggered=0, reason=threshold_reason)

        cooldown_ok, cooldown_reason = self._check_cooldown(link, timestamp)
        if not cooldown_ok:
            return DecisionResult(alert_triggered=0, reason=cooldown_reason)

        budget_ok = self._check_budget(timestamp)
        if not budget_ok:
            return DecisionResult(alert_triggered=0, reason="budget_block")

        self._last_alert_by_link[link] = timestamp
        self._global_alert_window.append(timestamp)
        return DecisionResult(alert_triggered=1, reason="alert")
