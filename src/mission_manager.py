from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MissionPhase(str, Enum):
    INITIALIZATION = "INITIALIZATION"
    RENDEZVOUS = "RENDEZVOUS"
    HOVER = "HOVER"
    TAG_DESCENT = "TAG_DESCENT"
    TAG_CONTACT = "TAG_CONTACT"
    DEPARTURE = "DEPARTURE"
    SAFE = "SAFE"
    ABORT = "ABORT"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class MissionThresholds:
    rendezvous_range_m: float = 10.0
    rendezvous_speed_mps: float = 0.10


@dataclass(frozen=True)
class MissionStatus:
    range_m: float
    relative_speed_mps: float
    hover_stable: bool = False
    tag_authorized: bool = False
    contact_detected: bool = False
    tag_complete: bool = False
    departure_complete: bool = False
    abort_requested: bool = False


class MissionManager:
    def __init__(self, thresholds: MissionThresholds | None = None) -> None:
        self.thresholds = thresholds or MissionThresholds()
        self.phase = MissionPhase.INITIALIZATION

    def start(self) -> MissionPhase:
        if self.phase != MissionPhase.INITIALIZATION:
            raise RuntimeError("Mission can only start from INITIALIZATION.")

        self.phase = MissionPhase.RENDEZVOUS
        return self.phase

    def enter_safe_mode(self) -> MissionPhase:
        self.phase = MissionPhase.SAFE
        return self.phase

    def update(self, status: MissionStatus) -> MissionPhase:
        if status.abort_requested:
            self.phase = MissionPhase.ABORT
            return self.phase

        if self.phase == MissionPhase.RENDEZVOUS:
            close_enough = status.range_m <= self.thresholds.rendezvous_range_m
            slow_enough = (
                status.relative_speed_mps <= self.thresholds.rendezvous_speed_mps
            )
            if close_enough and slow_enough:
                self.phase = MissionPhase.HOVER

        elif self.phase == MissionPhase.HOVER:
            if status.hover_stable and status.tag_authorized:
                self.phase = MissionPhase.TAG_DESCENT

        elif self.phase == MissionPhase.TAG_DESCENT:
            if status.contact_detected:
                self.phase = MissionPhase.TAG_CONTACT

        elif self.phase == MissionPhase.TAG_CONTACT:
            if status.tag_complete:
                self.phase = MissionPhase.DEPARTURE

        elif self.phase == MissionPhase.DEPARTURE:
            if status.departure_complete:
                self.phase = MissionPhase.COMPLETE

        return self.phase

