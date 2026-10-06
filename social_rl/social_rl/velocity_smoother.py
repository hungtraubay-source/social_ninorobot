"""Rate-limit commands identically during RL training and deployment."""

import math


class VelocitySmoother:
    """Limit changes in the velocity command emitted at one control tick.

    The policy selects a target velocity, not an instantaneous wheel command.
    Keeping this stateful limiter inside both train and deployment makes that
    actuator constraint part of the environment the policy learns.  Linear
    acceleration is in m/s² and angular acceleration is in rad/s².
    """

    def __init__(self, max_linear_acceleration: float,
                 max_angular_acceleration: float, enabled: bool = True):
        self.max_linear_acceleration = float(max_linear_acceleration)
        self.max_angular_acceleration = float(max_angular_acceleration)
        self.enabled = bool(enabled)
        if self.max_linear_acceleration <= 0.0:
            raise ValueError(
                'max_linear_acceleration must be greater than zero')
        if self.max_angular_acceleration <= 0.0:
            raise ValueError(
                'max_angular_acceleration must be greater than zero')
        self.reset()

    def reset(self):
        """Forget the last command after an episode, goal, or hard stop."""
        self._linear = 0.0
        self._angular = 0.0

    @staticmethod
    def _limit(target: float, previous: float, maximum_change: float) -> float:
        return previous + max(-maximum_change,
                              min(maximum_change, target - previous))

    def step(self, target_linear: float, target_angular: float,
             period: float) -> tuple:
        """Return the next executable ``(linear, angular)`` command.

        ``period`` is the policy control period in seconds.  A non-finite
        target would otherwise become a non-finite Twist and is rejected
        before it can reach the base.
        """
        target_linear = float(target_linear)
        target_angular = float(target_angular)
        period = float(period)
        if (not math.isfinite(target_linear)
                or not math.isfinite(target_angular)):
            raise ValueError('velocity targets must be finite')
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError('period must be finite and greater than zero')
        if not self.enabled:
            self._linear = target_linear
            self._angular = target_angular
            return self._linear, self._angular

        self._linear = self._limit(
            target_linear, self._linear,
            self.max_linear_acceleration * period)
        self._angular = self._limit(
            target_angular, self._angular,
            self.max_angular_acceleration * period)
        return self._linear, self._angular
