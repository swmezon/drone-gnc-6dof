from dataclasses import dataclass
import numpy as np


@dataclass
class WaypointGuidance:
    """
    Time-parameterized waypoint guidance.

    Converts discrete waypoints into a smooth reference trajectory.
    Each segment uses quintic time scaling so desired position,
    velocity, and acceleration transition smoothly.
    """

    waypoints: np.ndarray
    acceptance_radius: float = 0.30
    cruise_speed: float = 1.0
    index: int = 0

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, dtype=float)

        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")

        if len(self.waypoints) < 1:
            raise ValueError("at least one waypoint is required")

        if self.acceptance_radius <= 0.0:
            raise ValueError("acceptance_radius must be positive")

        if self.cruise_speed <= 0.0:
            raise ValueError("cruise_speed must be positive")

        self.segment_start = None
        self.segment_elapsed = 0.0
        self.segment_duration = 0.0

        self.acc_ref = np.zeros(3)

        self.reached_count = 0
        self.complete = False

    def reset(self, initial_position: np.ndarray):
        self.index = 0

        self.segment_start = np.asarray(
            initial_position,
            dtype=float,
        ).copy()

        self.segment_elapsed = 0.0

        self.acc_ref = np.zeros(3)

        self.reached_count = 0
        self.complete = False

        self._configure_segment()

    def _configure_segment(self):
        target = self.waypoints[self.index]

        length = float(
            np.linalg.norm(
                target - self.segment_start
            )
        )

        if length < 1e-12:
            self.segment_duration = 0.0
        else:
            # Quintic time scaling reaches a peak normalized
            # velocity of 1.875. Multiplying by 1.875 here
            # keeps the peak physical reference speed at or
            # below cruise_speed.
            self.segment_duration = (
                1.875
                * length
                / self.cruise_speed
            )

    def _segment_reference(self):
        target = self.waypoints[self.index]

        delta = target - self.segment_start

        if self.segment_duration <= 1e-12:
            return (
                target.copy(),
                np.zeros(3),
                np.zeros(3),
                True,
            )

        tau = np.clip(
            self.segment_elapsed
            / self.segment_duration,
            0.0,
            1.0,
        )

        # Quintic trajectory:
        #
        # s(0) = 0
        # s(1) = 1
        #
        # velocity = 0 at both ends
        # acceleration = 0 at both ends

        s = (
            10.0 * tau**3
            - 15.0 * tau**4
            + 6.0 * tau**5
        )

        ds_dt = (
            30.0 * tau**2
            - 60.0 * tau**3
            + 30.0 * tau**4
        ) / self.segment_duration

        d2s_dt2 = (
            60.0 * tau
            - 180.0 * tau**2
            + 120.0 * tau**3
        ) / (self.segment_duration**2)

        pos_ref = (
            self.segment_start
            + s * delta
        )

        vel_ref = ds_dt * delta

        acc_ref = d2s_dt2 * delta

        reference_complete = (
            tau >= 1.0 - 1e-12
        )

        return (
            pos_ref,
            vel_ref,
            acc_ref,
            reference_complete,
        )

    def update(
        self,
        position: np.ndarray,
        dt: float = 0.01,
    ):
        position = np.asarray(
            position,
            dtype=float,
        )

        if self.segment_start is None:
            self.reset(position)

        if not self.complete:
            self.segment_elapsed += max(
                float(dt),
                0.0,
            )

        (
            pos_ref,
            vel_ref,
            acc_ref,
            reference_complete,
        ) = self._segment_reference()

        target = self.waypoints[self.index]

        dist = float(
            np.linalg.norm(
                target - position
            )
        )

        # A waypoint counts as reached only when:
        #
        # 1. The scheduled reference has arrived there.
        # 2. The actual vehicle is within the acceptance radius.

        if (
            reference_complete
            and dist <= self.acceptance_radius
            and not self.complete
        ):
            self.reached_count = max(
                self.reached_count,
                self.index + 1,
            )

            if self.index < len(self.waypoints) - 1:

                completed_target = target.copy()

                self.index += 1

                self.segment_start = (
                    completed_target
                )

                self.segment_elapsed = 0.0

                self._configure_segment()

                pos_ref = (
                    self.segment_start.copy()
                )

                vel_ref = np.zeros(3)
                acc_ref = np.zeros(3)

                target = self.waypoints[
                    self.index
                ]

                dist = float(
                    np.linalg.norm(
                        target - position
                    )
                )

            else:
                self.complete = True

                pos_ref = target.copy()
                vel_ref = np.zeros(3)
                acc_ref = np.zeros(3)

        self.acc_ref = acc_ref.copy()

        # Keep yaw fixed during this benchmark.
        yaw_ref = 0.0

        return (
            pos_ref.copy(),
            vel_ref.copy(),
            yaw_ref,
            self.index,
            dist,
        )